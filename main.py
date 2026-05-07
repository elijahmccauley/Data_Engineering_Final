import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re

from multimodal_autoddg import (
    profiling,
    description_generation,
    text_processing,
    image_processing,
    evaluation,
)


load_dotenv(".secrets")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATASET_NAME = "Amazon ecommerce"

def scan_dataset_directory(base_path):
    """
    Scans a directory to identify csvs and image folders
    """
    dataset_assets = {
        'csv_files': [],
        'image_folders': []
    }
    valid_image_extensions = ('.png', '.jpg', '.jpeg')
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                dataset_assets['csv_files'].append(os.path.join(root, file))
                
            elif file.lower().endswith(valid_image_extensions):
                if root not in dataset_assets['image_folders']:
                    dataset_assets['image_folders'].append(root)
                    
    return dataset_assets

def run_pipeline(df: pd.DataFrame, image_folders: list = None, dataset_name: str = DATASET_NAME, use_koesten_prompt: bool = False, persona: str = "general"):
    """
    Master pipeline that dynamically routes data through the appropriate profilers
    and generates the highest-fidelity description possible.
    """
    print(f"\n--- Running Pipeline for: {dataset_name} ---")
    
    # STEP 1: Tabular Profiling (Always runs)
    print("1. Profiling Tabular Data...")
    full_profile = profiling.build_dataset_profile(df)
    compact_profile = profiling.build_compact_profile(full_profile)
    
    # STEP 2: Text Profiling (If semantic text exists)
    print("2. Detecting & Profiling Semantic Text...")
    text_cols = text_processing.detect_semantic_text_columns(df)
    text_semantic_summary = ""
    text_samples = {}
    
    if text_cols:
        text_samples = text_processing.sample_semantic_text(df, text_cols)
        text_semantic_summary = text_processing.generate_text_semantic_summary(
            dataset_name=dataset_name, 
            text_samples=text_samples, 
            client=client
        )
    
    # STEP 3: Image Profiling (Handles Local Folders AND URL Columns)
    image_semantic_summary = ""
    image_captions = []
    
    # 3a. Check for local image folders
    if image_folders:
        print(f"3a. Processing Images across {len(image_folders)} local folder(s) via BLIP...")
        for folder in image_folders:
            if os.path.exists(folder):
                raw_captions = image_processing.generate_image_captions(folder, sample_size=50)
                image_captions.extend([cap[0] for cap in raw_captions])
                
    # 3b. Check the DataFrame for image URL columns
    url_pattern = re.compile(r'^https?://.*\.(?:jpg|jpeg|png|gif|webp).*$', re.IGNORECASE)
    url_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            valid_sample = df[col].dropna().astype(str)
            # If more than 50% of the column matches an image URL, tag it
            if len(valid_sample) > 0 and valid_sample.str.match(url_pattern).mean() > 0.5:
                url_columns.append(col)
                
    if url_columns:
        print(f"3b. Processing Image URLs from column(s) {url_columns} via BLIP...")
        for col in url_columns:
            # generate_image_url_captions returns a list of strings directly
            url_caps = image_processing.generate_image_url_captions(df, url_column=col, sample_size=50)
            image_captions.extend(url_caps)

    # 3c. Compress everything into a semantic summary
    if image_captions:
        print(f"   -> Compressing {len(image_captions)} total visual samples into a semantic summary...")
        image_semantic_summary = image_processing.generate_image_semantic_summary(
            dataset_name=dataset_name,
            image_captions=image_captions,
            client=client
        )
    else:
        print("3. No valid image folders or URL columns detected. Skipping vision pipeline.")

    # STEP 4: Routing to the correct Generator
    print(f"4. Generating Final Description with Persona: {persona}, (Koesten Prompt: {use_koesten_prompt})...")
    
    if image_captions and text_cols:
        desc = description_generation.generate_multimodal_description(
            dataset_name, compact_profile, text_semantic_summary, 
            text_samples, image_semantic_summary, image_captions, client=client, use_koesten_prompt=use_koesten_prompt, persona=persona
        )
    elif image_captions:
        desc = description_generation.generate_multimodal_description(
            dataset_name, compact_profile, 
            text_semantic_summary="No semantic text columns available in the base tabular data.", 
            text_samples={}, 
            image_semantic_summary=image_semantic_summary, 
            image_captions=image_captions, 
            client=client, use_koesten_prompt=use_koesten_prompt, persona=persona
        )
    elif text_cols:
        desc = description_generation.generate_tabular_text_description(
            dataset_name, compact_profile, text_semantic_summary, text_samples, client=client, use_koesten_prompt=use_koesten_prompt
        )
    else:
        desc = description_generation.generate_tabular_only_description(
            dataset_name, compact_profile, client=client, use_koesten_prompt=use_koesten_prompt
        )
        
    return desc


def run_ablation_study(csv_path: str, image_folders: list, persona: str = "general"):
    """
    Executes the 4-part test.
    """
    print("\n" + "="*50)
    print("STARTING ABLATION STUDY")
    print("="*50)
    
    base_df = pd.read_csv(csv_path)
    
    # Create the ablated dataset (mimicking a CSV with bad/missing text)
    # We use errors='ignore' so it doesn't crash if a column is already missing
    text_columns_to_drop = text_processing.detect_semantic_text_columns(base_df)
    ablated_df = base_df.drop(columns=text_columns_to_drop, errors='ignore')
    
    descriptions = {}
    '''
    # 1. Baseline AutoDDG (Full DF, No Images)
    print("\n>>> TEST 1: Baseline AutoDDG (Full Text, No Images)")
    descriptions["AutoDDG_Baseline"] = run_pipeline(base_df, image_folders=None, dataset_name=f"{DATASET_NAME} (Baseline)")
    
    # 2. Text-Ablated AutoDDG (Ablated DF, No Images)
    print("\n>>> TEST 2: Text-Ablated AutoDDG (No Semantic Text, No Images)")
    descriptions["AutoDDG_Ablated"] = run_pipeline(ablated_df, image_folders=None, dataset_name=f"{DATASET_NAME} (Ablated)")
    
    # 3. Multimodal Baseline (Full DF, With Images)
    print("\n>>> TEST 3: Multimodal Baseline (Full Text, WITH Images)")
    descriptions["Multimodal_Baseline"] = run_pipeline(base_df, image_folders=image_folders, dataset_name=f"{DATASET_NAME} (Multimodal)", persona=persona)
    
    # 4. Multimodal Baseline (KOESTEN PROMPT) <-- THE NEW TEST
    print("\n>>> TEST 4: Multimodal Koesten (Full Text, WITH Images, KOESTEN PROMPT)")
    descriptions["Multimodal_Koesten"] = run_pipeline(base_df, image_folders=image_folders, dataset_name=f"{DATASET_NAME} (Multimodal + Koesten)", use_koesten_prompt=True, persona=persona)
    '''
    # 5. Multimodal Text-Ablated (Ablated DF, With Images)
    print("\n>>> TEST 5: Multimodal Ablated (No Semantic Text, WITH Images)")
    descriptions["Multimodal_Ablated"] = run_pipeline(ablated_df, image_folders=image_folders, dataset_name=f"{DATASET_NAME} (Multimodal Ablated)", use_koesten_prompt=True, persona=persona)
    
    # STEP 5: Run Yuheng's Evaluator on the results
    print("\n" + "="*50)
    print("RUNNING LLM JUDGE (Pointwise Evaluation)")
    print("="*50)
    
    # We use the full profile as the ground truth for the judge
    ground_truth_profile = profiling.build_compact_profile(profiling.build_dataset_profile(base_df))
    
    eval_results = evaluation.evaluate_pointwise(
        dataset_name=DATASET_NAME,
        compact_profile=ground_truth_profile,
        descriptions=descriptions,
        client=client
    )
    
    # Print the final scorecard
    for test_name, text in descriptions.items():
        print(f"\n--- {test_name.upper()} DESCRIPTION ---")
        print(text)
        print("\n--- SCORES ---")
        scores = eval_results.get(test_name, {})
        for metric, score in scores.items():
            print(f"{metric.capitalize()}: {score}")
        print("-" * 40)


if __name__ == "__main__":
    # --- CONFIGURATION FOR YOUR LOCAL MACHINE ---
    TARGET_DIRECTORY = "./e-commerce"
    assets = scan_dataset_directory(TARGET_DIRECTORY)
    if not assets['csv_files']:
        if not assets['image_folders']:
            print("Error: No CSV or Image files found in the target directory.")
            exit(1)
            
        print("\n--- IMAGE-ONLY DATASET DETECTED ---")
        print("No CSV found. Generating captions to construct a synthetic 1-column CSV...")
        
        all_captions = []
        for folder in assets['image_folders']:
            raw_caps = image_processing.generate_image_captions(folder, sample_size=50)
            all_captions.extend([cap[0] for cap in raw_caps])
            
        # Treat the captions as a 1-column DataFrame
        base_df = pd.DataFrame({'generated_image_caption': all_captions})
        
        # Run the standard pipeline
        # (We pass image_folders=None because the images are already consumed into the text profile)
        final_desc = run_pipeline(base_df, image_folders=None, dataset_name=f"{DATASET_NAME} (Image-Only)")
        
        print("\n--- FINAL IMAGE-ONLY DESCRIPTION ---")
        print(final_desc)
        
        # We safely exit here because an ablation study (dropping text columns) 
        # doesn't mathematically apply to a synthetic 1-column dataset.
        exit(0)
        
    target_csv_path = assets['csv_files'][0]
    detected_image_folders = assets['image_folders']
    
    print(f"Found base tabular data: {target_csv_path}")
    print(f"Found {len(detected_image_folders)} image folder(s).")
    
    run_ablation_study(
        csv_path=target_csv_path, 
        image_folders=detected_image_folders, persona="general"  # You can switch to "general" if you want a more neutral description
    )