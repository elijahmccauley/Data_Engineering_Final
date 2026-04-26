import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from modules import (
    profiling,
    description_generation,
    text_processing,
    image_processing,
    evaluation,
)


load_dotenv(".secrets")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATASET_NAME = "Amazon E-commerce"

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

def run_pipeline(df: pd.DataFrame, image_folder: str = None, dataset_name: str = DATASET_NAME):
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
    
    # STEP 3: Image Profiling (If image folder provided)
    image_semantic_summary = ""
    image_captions = []
    
    if image_folder and os.path.exists(image_folder):
        print(f"3. Processing Images in {image_folder} via BLIP...")
        # Get raw captions [(caption, filename), ...]
        raw_captions = image_processing.generate_image_captions(image_folder, sample_size=10)
        image_captions = [cap[0] for cap in raw_captions]
        
        # Compress into semantic summary
        image_semantic_summary = image_processing.generate_image_semantic_summary(
            dataset_name=dataset_name,
            image_captions=image_captions,
            client=client
        )
    else:
        print("3. No valid image folder provided. Skipping vision pipeline.")

    # STEP 4: Routing to the correct Generator
    print("4. Generating Final Description...")
    
    if image_captions and text_cols:
        # 1. Full Multimodal (Tabular + Text + Image)
        desc = description_generation.generate_multimodal_description(
            dataset_name, compact_profile, text_semantic_summary, 
            text_samples, image_semantic_summary, image_captions, client=client
        )
    elif image_captions:
        # 2. Tabular + Image (NO Semantic Text)
        desc = description_generation.generate_multimodal_description(
            dataset_name, compact_profile, 
            text_semantic_summary="No semantic text columns available in the base tabular data.", 
            text_samples={}, 
            image_semantic_summary=image_semantic_summary, 
            image_captions=image_captions, 
            client=client
        )
    elif text_cols:
        # 3. Tabular + Text 
        desc = description_generation.generate_tabular_text_description(
            dataset_name, compact_profile, text_semantic_summary, text_samples, client=client
        )
    else:
        # 4. Tabular Only (Original AutoDDG)
        desc = description_generation.generate_tabular_only_description(
            dataset_name, compact_profile, client=client
        )
        
    return desc


def run_ablation_study(csv_path: str, image_folder: str, text_columns_to_drop: list):
    """
    Executes the 4-part test.
    """
    print("\n" + "="*50)
    print("STARTING ABLATION STUDY")
    print("="*50)
    
    base_df = pd.read_csv(csv_path)
    
    # Create the ablated dataset (mimicking a CSV with bad/missing text)
    # We use errors='ignore' so it doesn't crash if a column is already missing
    ablated_df = base_df.drop(columns=text_columns_to_drop, errors='ignore')
    
    descriptions = {}
    
    # 1. Baseline AutoDDG (Full DF, No Images)
    print("\n>>> TEST 1: Baseline AutoDDG (Full Text, No Images)")
    descriptions["AutoDDG_Baseline"] = run_pipeline(base_df, image_folder=None, dataset_name=f"{DATASET_NAME} (Baseline)")
    
    # 2. Text-Ablated AutoDDG (Ablated DF, No Images)
    print("\n>>> TEST 2: Text-Ablated AutoDDG (No Semantic Text, No Images)")
    descriptions["AutoDDG_Ablated"] = run_pipeline(ablated_df, image_folder=None, dataset_name=f"{DATASET_NAME} (Ablated)")
    
    # 3. Multimodal Baseline (Full DF, With Images)
    print("\n>>> TEST 3: Multimodal Baseline (Full Text, WITH Images)")
    descriptions["Multimodal_Baseline"] = run_pipeline(base_df, image_folder=image_folder, dataset_name=f"{DATASET_NAME} (Multimodal)")
    
    # 4. Multimodal Text-Ablated (Ablated DF, With Images)
    print("\n>>> TEST 4: Multimodal Ablated (No Semantic Text, WITH Images)")
    descriptions["Multimodal_Ablated"] = run_pipeline(ablated_df, image_folder=image_folder, dataset_name=f"{DATASET_NAME} (Multimodal Ablated)")
    
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
        image_folders=detected_image_folders
    )