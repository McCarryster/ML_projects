import os
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer

# Set the environment variable to disable the warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

model_name = "meta-llama/Llama-3.2-1B"
script_dir = os.path.dirname(__file__)
cache_dir = os.path.join(script_dir, '../models/llama3.2/1B_automodel')

# Download the model
AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)