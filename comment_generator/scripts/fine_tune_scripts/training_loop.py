import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from train_evaluate_logic import *
from config import *
import os
import json
from logger_setup import logger, log_separator
import sys
import math
from huggingface_hub import snapshot_download


# PIPELINE:
#     0. Specify paths and other things
#     1. Load Model
#     2. Initialize Tokenizer
#     3. Load and tokenize data then make dataloader
#     4. Define optimizer, scheduler and scaler
#     5. Define writer and save setup information
#     6. Training loop
#     7. Evalation loop
#     8. Model saving
#     9. Try to generate something


# 0. Specify paths and other things
model_save_dir = os.path.join(stuff_and_paths['training_results_dir'], f"fine_tuned_{stuff_and_paths['model_name']}")
checkpoints_path = os.path.join(stuff_and_paths['training_results_dir'], 'checkpoints')
log_dir = os.path.join(stuff_and_paths['training_results_dir'], 'logs')

os.makedirs(stuff_and_paths['training_results_dir'], exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

logger.info(f"Using {train_args['device']}")
print(f"Using {train_args['device']}")


log_separator()
print('#'*100)


print('Using those train_args:')
logger.info('Using those train_args:')
for key, value in train_args.items():
    print(f"{key} = {value}")
    logger.info(f"{key} = {value}")


log_separator()
print('#'*100)


print('Using those stuff_and_paths:')
logger.info('Using those stuff_and_paths:')
for key, value in stuff_and_paths.items():
    print(f"{key} = {value}")
    logger.info(f"{key} = {value}")


log_separator()
print('#'*100)


# 1. Load Model
# Load checkpoint from HF or local
if train_args['resume_from_checkpoint']:
    # Download checkpoint folder from HF repo using branch revision and allow_patterns
    if stuff_and_paths['load_from_hf']:
        hf_checkpoint_path = snapshot_download(
            repo_id=stuff_and_paths['hf_repo_id'],
            revision="main",
            allow_patterns=[f"checkpoints/checkpoint_{stuff_and_paths['checkpoint_num']}/*"],
            local_dir=stuff_and_paths['training_results_dir'],  # Where to save the checkpoint locally
        )
    else:
        print('Loading local checkpoint')
        logger.info('Loading local checkpoint')

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(stuff_and_paths['model_dir'], torch_dtype=torch.bfloat16, local_files_only=True)

# Handle LoRA configuration
if train_args['make_lora']:
    if train_args['resume_from_checkpoint']:
        model = PeftModel.from_pretrained(
            base_model,
            f"{stuff_and_paths['training_results_dir']}/checkpoints/checkpoint_{stuff_and_paths['checkpoint_num']}",  # checkpoint folder with adapter weights
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        print('LoRA adapter loaded from checkpoint')
        logger.info('LoRA adapter loaded from checkpoint')
    else:
        # Initialize new LoRA config
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

        print('LoRA adapter applied to base model')
        logger.info('LoRA adapter applied to base model')
else:
    model = base_model
    print('Using base model without LoRA')
    logger.info('Using base model without LoRA')

# Load training states if resuming
if train_args['resume_from_checkpoint']:
    checkpoint_data = torch.load(f"{stuff_and_paths['training_results_dir']}/checkpoints/checkpoint_{stuff_and_paths['checkpoint_num']}/checkpoint_step_{stuff_and_paths['checkpoint_num']}.pth",
                                 map_location='cpu', weights_only=False)

model.to(train_args['device'])
print(f"Model loaded - torch.cuda.memory_allocated ({torch.cuda.memory_allocated(0)/1024/1024/1024} GB)")
logger.info(f"Model loaded - torch.cuda.memory_allocated ({torch.cuda.memory_allocated(0)/1024/1024/1024} GB)")


log_separator()
print('#'*100)


# 2. Initialize Tokenizer
if train_args['resume_from_checkpoint']:
    tokenizer = AutoTokenizer.from_pretrained(f"{stuff_and_paths['training_results_dir']}/checkpoints/checkpoint_{stuff_and_paths['checkpoint_num']}")
else:
    tokenizer = AutoTokenizer.from_pretrained(stuff_and_paths['model_dir'])
tokenizer.padding_side = "left" # left because they said so in documentation
special_tokens = ["<|start_header_id|>", "<|end_header_id|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
print(tokenizer.special_tokens_map)
logger.info(tokenizer.special_tokens_map)


log_separator()
print('#'*100)


# 3. Load and tokenize data then make dataloader
class CommentDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=train_args['max_seq_length']):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def _format_text(self, item):
        reactions = ", ".join([f"{k}:{v}" for k, v in item['input']['post_reactions'].items()])
        desired_reactions = ", ".join([f"{k}:{v}" for k, v in item['input']['desired_reactions'].items()]) or "None"
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a social media assistant. Generate a comment. Consider the Reaction of users and the tone of the Post.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Post: {item['input']['post']}
Reactions: {reactions}
Position: {item['input']['position']}
Desired reactions: {desired_reactions}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{item['output']}<|eot_id|>"""

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = self._format_text(item)
        
        # Tokenize without adding special tokens
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids[0]
        labels = torch.full_like(input_ids, -100)
        
        # Find where label starts
        cls_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        header_count = 0
        start_idx = None
        for idx, token in enumerate(input_ids):
            if token == cls_id:
                header_count += 1
                if header_count == 3:
                    start_idx = idx + 1
                    break

        # Replace -100 with label from start_idx to :
        labels[start_idx:] = input_ids[start_idx:]

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized.attention_mask[0],
            "labels": labels
        }

# Create datasets and then dataloaders
train_dataset = CommentDataset(stuff_and_paths['train_data_dir'], tokenizer)
val_dataset = CommentDataset(stuff_and_paths['val_data_dir'], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=train_args['batch_size'], shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=train_args['batch_size'], shuffle=False)

for i, batch in enumerate(train_dataloader):
    print(i, f"train_dataloader {len(train_dataloader)}: input_ids shape: {batch['input_ids'].shape}, attention_mask shape: {batch['attention_mask'].shape}, labels shape: {batch['labels'].shape}")
    logger.info(i, f"train_dataloader {len(train_dataloader)}: input_ids shape: {batch['input_ids'].shape}, attention_mask shape: {batch['attention_mask'].shape}, labels shape: {batch['labels'].shape}")
    decoded_input = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
    labels = batch['labels'][0].clone()
    labels[labels == -100] = tokenizer.pad_token_id # Replace -100 with pad_token_id for decoding
    decoded_output = tokenizer.decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=True)

    # Check decoded input
    print(f"[Decoded INPUT]: {decoded_input}")
    logger.info(f"[Decoded INPUT]: {decoded_input}")
    print("-" * 50)
    logger.info("-" * 50)
    # Check attention mask
    print(f"[Attention MASK]: {batch['attention_mask'][0]}")
    logger.info(f"[Attention MASK]: {batch['attention_mask'][0]}")
    print("-" * 50)
    logger.info("-" * 50)
    # Check decoded output
    print(f"[Decoded OUTPUT]: {decoded_output}")
    logger.info(f"[Decoded OUTPUT]: {decoded_output}")

    log_separator()
    print('#'*100)

    # Check input tensor
    print(f"[tensor INPUT]: {batch['input_ids'][0]}")
    logger.info(f"[tensor INPUT]: {batch['input_ids'][0]}")
    print("-" * 50)
    logger.info("-" * 50)
    # Check output tensor
    print(f"[tensor OUTPUT]: {batch['labels'][0]}")
    logger.info(f"[tensor OUTPUT]: {batch['labels'][0]}")

    break


log_separator()
print('#'*100)


# 4. Define optimizer, scheduler and scaler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_args['learning_rate'],
    weight_decay=train_args['weight_decay'],
    betas=(0.9, 0.999)
)

if train_args['gradient_accumulation']:
    num_training_steps = math.ceil(len(train_dataloader) / train_args['accumulation_steps']) * train_args['num_epochs']
else:
    num_training_steps = len(train_dataloader) * train_args['num_epochs']
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
    num_training_steps=num_training_steps
)

scaler = torch.amp.GradScaler() if train_args['mixed_precision'] else None

if train_args['resume_from_checkpoint']:
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint_data:
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])

print("Optimizer, scheduler and scaler are defined")
logger.info("Optimizer, scheduler and scaler are defined")


log_separator()
print('#'*100)


# 5. Define writer and save setup information
writer = SummaryWriter(log_dir=log_dir)
model_things = {
    'model_name': stuff_and_paths['model_name'],
    'optimizer': optimizer.__class__.__name__,
    'num_training_steps': num_training_steps,
    'scheduler': scheduler.__class__.__name__,
    'train_args': {k: str(v) if isinstance(v, torch.device) else v for k, v in train_args.items()},
    'generation_args': {k: str(v) if isinstance(v, torch.device) else v for k, v in generation_args.items()},
    'train_dataloader_len': len(train_dataloader),
    'val_dataloader_len': len(val_dataloader)
}
with open(os.path.join(log_dir, 'model_metadata.json'), 'w') as f:
    json.dump(model_things, f, indent=4)


# 6. Training loop:
def main():
    start_time = datetime.now()


    if train_args['resume_from_checkpoint']:
        train(model, tokenizer, optimizer, scaler, scheduler, writer, train_dataloader, val_dataloader, checkpoints_path, model_save_dir, checkpoint_data=checkpoint_data)
    else:
        train(model, tokenizer, optimizer, scaler, scheduler, writer, train_dataloader, val_dataloader, checkpoints_path, model_save_dir)


    log_separator()
    print('#'*100)
    logger.info('Model trained, evaluated and saved successfully!')
    print('Model trained, evaluated, saved and pushed to HF repo successfully!')
    log_separator()
    print('#'*100)

    end_time = datetime.now()
    
    logger.info(f"Training took: {end_time - start_time} seconds")
    print(f"Training took: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        raise ValueError("Please set the environment variable HUGGING_FACE_HUB_TOKEN before running.")
    main()
