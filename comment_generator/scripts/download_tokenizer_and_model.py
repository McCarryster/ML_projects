from transformers import AutoModelForCausalLM, AutoTokenizer
import os


model_name = "model_hf_repo"
cache_dir = "/cache/folder"
os.makedirs(cache_dir, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)