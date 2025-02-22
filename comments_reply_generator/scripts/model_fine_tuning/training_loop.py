import torch
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from train_evaluate_logic import *
from config import *
import json


# PIPELINE:
#     0. Specify paths and other things
#     1. Load Model and Tokenizer
#     2. Load and tokenize data then make dataloader
#     3. Define optimizer
#     4. Training loop:
#         1. Forward pass: compute prediction
#         2. Backward pass: gradients
#         3. Update weights
#         4. Evaluate the model on validation set
#     5. Try to generate something


# 0. Specify paths and other things
script_dir = os.path.dirname(__file__)
model_name = 'llama3.2_1B'
model_dir = "/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/models/llama3.2/1B/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
model_save_dir = os.path.join(script_dir, '../../models/fine_tuned_llama3.2_1B/model')
check_points_path = os.path.join(script_dir, '../../models/fine_tuned_llama3.2_1B/checkpoints')
data_dir = os.path.join(script_dir, '../../data/data_types/processed_data/head_data.json')
log_dir = os.path.join(script_dir, '../../results/model_res_1')


# 1. Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.add_special_tokens({"pad_token": "<|file_tune_right_pad_id|>"})
print("Check token:", tokenizer.tokenize('<|file_tune_right_pad_id|>'), tokenizer.pad_token, tokenizer.pad_token_id) # Check if tokens applied correctly
model.resize_token_embeddings(len(tokenizer))


print('#'*100)


# 2. Load and tokenize data then make dataloader
df = pd.read_json(data_dir)
df['input'] = df['input'].astype(str)   # Ensure that they are strings  (sanity check)
df['output'] = df['output'].astype(str) # Ensure that they are strings  (sanity check)
input_texts = df['input'].tolist()      # Convert to list
output_texts = df['output'].tolist()    # Convert to list
data = {'input': input_texts, 'output': output_texts}
dataset = Dataset.from_dict(data)

# Tokenize
max_length = 0
for input_text, output_text in zip(input_texts, output_texts):
    input_length = len(tokenizer.encode(input_text))
    output_length = len(tokenizer.encode(output_text))
    if input_length > max_length:
        max_length = input_length
    if output_length > max_length:
        max_length = output_length
def tokenize_function(examples):
    return tokenizer(
        examples['input'],
        text_target=examples['output'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create Dataloader
train_dataloader = DataLoader(tokenized_dataset, batch_size=train_args['batch_size'], shuffle=True)
val_dataloader = DataLoader(tokenized_dataset, batch_size=train_args['batch_size'], shuffle=True)
for i, batch in enumerate(train_dataloader):
    print(f"input_ids shape: {batch['input_ids'].shape}, attention_mask shape: {batch['attention_mask'].shape}, output_ids shape: {batch['labels'].shape}")


print('#'*100)


# 3. Define optimizer, scheduler, scaler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_args['learning_rate'],
    weight_decay=train_args['weight_decay'],
    betas=(0.9, 0.999)
)
num_training_steps = len(train_dataloader) * train_args['num_epochs']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=train_args['warmup_steps'],
    num_training_steps=num_training_steps
)

scaler = torch.amp.GradScaler()

writer = SummaryWriter(log_dir=log_dir)
model_things = {
    'model_name': model_name,
    'optimizer': optimizer.__class__.__name__,
    'num_training_steps': num_training_steps,
    'scheduler': scheduler.__class__.__name__,
    'train_args': train_args,
    'evaluation_args': evaluation_args,
    'train_dataloader_len': len(train_dataloader),
    'val_dataloader_len': len(val_dataloader)
}
with open(os.path.join(log_dir, 'model_metadata.json'), 'w') as f:
    json.dump(model_things, f, indent=4)


# 4. Training loop:
def main():
    start_time = datetime.now()
    train(model, tokenizer, optimizer, scaler, scheduler, writer, train_dataloader, val_dataloader, check_points_path, model_save_dir)
    print('-'*100)
    print('Model trained, evaluated and saved successfully!')
    print('-'*100)
    end_time = datetime.now()
    print(f"Training took: {end_time - start_time}")
    
if __name__ == "__main__":
    main()
