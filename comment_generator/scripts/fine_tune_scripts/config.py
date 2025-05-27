import torch
import os


def get_env_int(var_name, default):
    try:
        return int(os.getenv(var_name, default))
    except ValueError:
        print(f"Warning: Invalid integer for {var_name}, using default {default}")
        return default

def get_env_float(var_name, default):
    try:
        return float(os.getenv(var_name, default))
    except ValueError:
        print(f"Warning: Invalid float for {var_name}, using default {default}")
        return default

def get_env_bool(var_name, default):
    val_str = os.getenv(var_name, str(default))
    return val_str.lower() == 'true'


stuff_and_paths = {
    'model_name': "Vikhr-Llama3.1-8B-Instruct", # Base model name
    'base_model_hf_id': "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24",
    'saving_to_hf': get_env_bool('SAVING_TO_HF', True), # Pushing to HF logic
    'hf_repo_id': 'mccarryster/com-gen-llama3.1-8B', # Where to push
    'load_from_hf': get_env_bool('LOAD_FROM_HF', False), # Load from HF or local
    'checkpoint_num': get_env_int('CHECKPOINT_NUM', 0), # What checkpoint to load
    'training_results_dir': '/training_results', # Where to save checkpoints, final model and logs
    'model_dir': '/app/model/models--Vikhrmodels--Vikhr-Llama3.1-8B-Instruct-R-21-09-24/snapshots/3d76e0f58bd5db2f6ced2b3744dd042852dec903', # Base model dir
    'train_data_dir': '/app/data/comment_only_train_set.json', # Directory with train data
    'val_data_dir': '/app/data/comment_only_val_set.json' # Directory with validation data
}

train_args = {
    'num_epochs': get_env_int('NUM_EPOCHS', 1),
    'batch_size': get_env_int('BATCH_SIZE', 3),
    'learning_rate': get_env_float('LR_RATE', 3e-5),
    'weight_decay': get_env_float('WEIGHT_DECAY', 0.01),
    'make_lora': get_env_bool('MAKE_LORA', False),
    'resume_from_checkpoint': get_env_bool('RESUME_FROM_CHECKPOINT', False),
    'gradient_clip': get_env_float('GRADIENT_CLIP', 1.0),
    'mixed_precision': get_env_bool('MIXED_PRECISION', False),
    'early_stopping': get_env_bool('EARLY_STOPPING', False),
    'early_stopping_patience': get_env_int('EARLY_STOPPING_PATIENCE', 3),
    'grad_checkpoint': get_env_bool('GRAD_CHECKPOINT', True),
    'save_steps': get_env_int('SAVE_STEPS', 10000),
    'logging_steps': get_env_int('LOGGING_STEPS', 100),
    'validation_steps': get_env_int('VALIDATION_STEPS', 5000),
    'max_seq_length': get_env_int('MAX_SEQ_LENGTH', 512),
    'gradient_accumulation': get_env_bool('GRADIENT_ACCUMULATION', True),
    'accumulation_steps': get_env_int('ACCUMULATION_STEPS', 16),
    'full_validation': get_env_bool('FULL_VALIDATION', False),
    'val_set_size': get_env_int('VAL_SET_SIZE', 3000),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

generation_args = {
    'max_new_tokens': 128,     # Comments typically are not long
    'num_beams': 1,
    'temperature': 0.5,
    'repetition_penalty': 1.2,
    'no_repeat_ngram_size': 2,
    'length_penalty': 1.0,
    'do_sample': True,
    'early_stopping': False,
    'top_k': 50, 
    'top_p': 1.0,
}