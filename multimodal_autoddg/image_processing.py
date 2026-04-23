import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO
import config
from utils.openai_utils import call_openai

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_captions(image_folder, sample_size=50):
    """
    Takes a folder of images, generates captions for a random sample, and returns a list of captions.
    """
    captions = []
    image_files = os.listdir(image_folder)[:sample_size]  # Limit to sample size
    
    for file in image_files:
        #print(file)
        raw_image = Image.open(os.path.join(image_folder, file)).convert("RGB")
        
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        captions.append((caption, file))
        
    return captions

def generate_image_url_captions(df, url_column, sample_size=50):
    """
    Takes a pandas DataFrame and a column of image URLs, fetches a random sample,
    and returns a list of generated captions.
    """
    captions = []
    valid_df = df.dropna(subset=[url_column])
    sample_df = valid_df.sample(n=min(sample_size, len(valid_df)))
    
    for index, row in sample_df.iterrows():
        url = row[url_column]
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            raw_image = Image.open(BytesIO(response.content)).convert('RGB')
            
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            captions.append(caption)
        except Exception as e:
            print(f"Skipped image at index {index} due to error: {e}")
            continue
        
    return captions

def generate_image_semantic_summary(
    dataset_name: str,
    image_captions: list[str],
    client=None,
    model: str = config.DEFAULT_MODEL,
) -> str:
    """
    Call the LLM to produce a short semantic summary of the generated image captions.
    """
    if not image_captions:
        return "No visual data detected."
        
    lines = [f"  - {cap}" for cap in image_captions]
    
    prompt = f"""You are analysing a dataset and summarising the semantic meaning of its visual fields based on generated image captions.

Dataset name: {dataset_name}

Sampled image captions:
{chr(10).join(lines)}

Write a concise semantic summary (2-3 sentences) covering:
1. What kinds of products, objects, or visual items appear in the data.
2. Common visual themes, colors, or prominent features.
3. What analytical or computer vision tasks this image data could support.

Do NOT invent visual information not visible in the captions."""

    return call_openai(
        prompt=prompt,
        system_message=(
            "You are a careful data analyst who summarises the visual content "
            "of datasets. Be factual and concise."
        ),
        model=model,
        temperature=0.2,
        client=client,
    )