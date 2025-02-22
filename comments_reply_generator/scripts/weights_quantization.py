import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

# Set environment variable to disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Model name and local directories
model_name = "meta-llama/Llama-3.2-1B"
script_dir = os.path.dirname(__file__)
cache_dir = os.path.join(script_dir, '../models/llama3.2/1B')
quantized_model_dir = os.path.join(script_dir, '../models/llama3.2/1B_quantized')

# Configuration for 4-bit quantization using BitsAndBytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # specifies the type of 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,    # specifies the compute data type
    bnb_4bit_use_double_quant=True,          # for nested quantization
)

# Load the model with the quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=quantization_config,
    device_map="auto"                        # automatically manage the allocation of layers to available devices
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Save the quantized model and tokenizer to the local directory
model.save_pretrained(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir)

print(f"Quantized model saved to: {quantized_model_dir}")
