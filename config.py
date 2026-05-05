# --- LLM Configuration ---
# Used across description_generation.py, evaluation.py, text_processing.py, and image_processing.py
DEFAULT_MODEL = "gpt-4o-mini" 

# --- Profiling Configuration ---
# Used in profiling.py
TOP_K_VALUES = 5 
IMPORTANT_PROFILE_COLUMNS = [
    'product_id', 'product_name', 'category', 
    'discounted_price', 'actual_price', 'discount_percentage', 
    'rating', 'rating_count'
]

# --- Text Processing Configuration ---
# Used in text_processing.py
TEXT_MIN_AVG_LENGTH = 50          # Minimum avg string length to be considered semantic text
TEXT_MAX_SAMPLES = 10             # Number of unique values to sample for the LLM
TEXT_SAMPLE_RANDOM_STATE = 27     # For reproducible sampling