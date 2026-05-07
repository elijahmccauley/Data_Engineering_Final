# Data Engineering Final: Multimodal Dataset Description Generation

This repository contains the codebase for our Multimodal Dataset Description Generator. Our pipeline extends the original AutoDDG framework by incorporating visual data (via BLIP image captioning) and text-based semantic profiling to dynamically generate persona-driven dataset descriptions using Large Language Models (LLMs).

## System Prerequisites
- **Python:** Python 3.10 or higher (3.10 - 3.11 is recommended for PyTorch/Transformers stability).
- **Git:** To clone the repository.

## 1. Installation & Setup

First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/elijahmccauley/Data_Engineering_Final.git
cd Data_Engineering_Final
```

Next, install the required third-party Python packages:
```bash
pip install -r requirements.txt
```

### API Key Configuration
In order to run the text semantic profiler and description generators, you will need an OpenAI API key. 
1. We have provided a `.secrets.example` file in the repository.
2. Rename this file to `.secrets` (or make a copy).
3. Open the file and replace `"your_api_key_here"` with your actual OpenAI API key.

## 2. Data Acquisition
Due to size constraints, the datasets are not included in this repository. You must download them manually from the following sources:

1. **Amazon Clothing:** [Fashion Images Dataset](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images)
2. **Amazon Tech:** [Amazon Sales Dataset EDA](https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda/notebook)
3. **Gingivitis Dental Data:** [Mendeley Dental Dataset](https://data.mendeley.com/datasets/3253gj88rr/1)

### Handling the Dental Dataset (.rar file)
The Gingivitis dataset downloads as a `.rar` file. You will need third-party software to extract it depending on your operating system:
* **Windows:** Download and use [7-Zip](https://www.7-zip.org/) or WinRAR. Right-click the file and select "Extract Here".
* **Mac:** Download [The Unarchiver](https://theunarchiver.com/) from the App Store, or use the terminal command `brew install unrar` followed by `unrar x <filename>.rar`.
* **Linux:** Use the terminal command `sudo apt-get install unrar` followed by `unrar x <filename>.rar`.

## 3. Expected Directory Structure
Once downloaded and extracted, organize the data folders at the root of your repository so they exactly match this structure:

```text
/repo_root
├── amazon_clothes/
│   ├── Apparel/
│   │   ├── Boys/Images/images_with_product_ids/
│   │   └── Girls/Images/images_with_product_ids/
│   ├── Footwear/
│   │   ├── Men/Images/images_with_product_ids/
│   │   └── Women/Images/images_with_product_ids/
│   └── fashion.csv
├── amazon_tech/
│   └── amazon_tech.csv
├── dental_dataset/
│   └── Dataset/
│       ├── Test/
│       ├── Training/
│       │   ├── Images/
│       │   ├── Labels/
│       │   └── Train_captions.csv
│       └── Validation/
├── multimodal_autoddg/
│   ├── description_generation.py
│   ├── evaluation.py
│   ├── image_processing.py
│   ├── profiling.py
│   └── text_processing.py
├── utils/
│   └── openai_utils.py
├── .gitignore
├── .secrets              
├── README.md
├── autoddg.ipynb
├── autoddg_workflow.png
├── config.py
├── main.py
└── requirements.txt
```

## 4. Usage

To run the pipeline, you will manually configure the execution variables inside `main.py`. Open `main.py` and update the following variables at the top of the execution block to match the dataset and test case you want to run:

```python
# In main.py, update these variables before running:
DATASET_NAME = "Amazon Clothing"    # Valid: "Amazon Clothing", "Amazon Tech", "Gingivitis"
DATA_PATH = "./amazon_clothes"      # Ensure your local data path matches this
PERSONA = "general"                 # Valid: "general", "sales"
USE_KOESTEN_PROMPT = True           # Valid: True, False
```

Once configured, simply execute the script:
```bash
python main.py
```

## Note on Reproducibility and LLMs
This project uses a Large Language Model (GPT-4o-mini) to synthesize the final descriptions. While we provide strict, specific prompts and configurations, please note that LLM outputs can be stochastic. Minor variations in the generated text may occur, and exact wording will not be perfectly identical on subsequent runs, though the structural formatting and semantic insights will remain consistent.