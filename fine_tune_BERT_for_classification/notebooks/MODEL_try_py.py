# Imports
import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np




# Device configuration
# torch.cuda.empty_cache()
# torch.cuda.reset_max_memory_cached()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# # Train set
# train_pos_folder = './data/aclImdb/train/pos'
# train_neg_folder = './data/aclImdb/train/neg'

# train_pos_sentences = [open(os.path.join(train_pos_folder, f)).read().strip() for f in os.listdir(train_pos_folder)]
# train_neg_sentences = [open(os.path.join(train_neg_folder, f)).read().strip() for f in os.listdir(train_neg_folder)]

# train_df = pd.DataFrame({
#     'text': train_pos_sentences + train_neg_sentences,
#     'label': [1] * len(train_pos_sentences) + [0] * len(train_neg_sentences)  # 1 for positive, 0 for negative
# })

# # Test set
# test_pos_folder = './data/aclImdb/test/pos'
# test_neg_folder = './data/aclImdb/test/neg'

# test_pos_sentences = [open(os.path.join(test_pos_folder, f)).read().strip() for f in os.listdir(test_pos_folder)]
# test_neg_sentences = [open(os.path.join(test_neg_folder, f)).read().strip() for f in os.listdir(test_neg_folder)]

# test_df = pd.DataFrame({
#     'text': test_pos_sentences + test_neg_sentences,
#     'label': [1] * len(test_pos_sentences) + [0] * len(test_neg_sentences)  # 1 for positive, 0 for negative
# })










train_pos_folder = './data/aclImdb/train/pos'
train_neg_folder = './data/aclImdb/train/neg'

# Ensure the correct encoding when reading files
train_pos_sentences = [open(os.path.join(train_pos_folder, f), encoding='utf-8').read().strip() for f in os.listdir(train_pos_folder)]
train_neg_sentences = [open(os.path.join(train_neg_folder, f), encoding='utf-8').read().strip() for f in os.listdir(train_neg_folder)]

train_df = pd.DataFrame({
    'text': train_pos_sentences + train_neg_sentences,
    'label': [1] * len(train_pos_sentences) + [0] * len(train_neg_sentences)  # 1 for positive, 0 for negative
})



test_pos_folder = './data/aclImdb/test/pos'
test_neg_folder = './data/aclImdb/test/neg'

# Ensure the correct encoding when reading files
test_pos_sentences = [open(os.path.join(test_pos_folder, f), encoding='utf-8').read().strip() for f in os.listdir(test_pos_folder)]
test_neg_sentences = [open(os.path.join(test_neg_folder, f), encoding='utf-8').read().strip() for f in os.listdir(test_neg_folder)]

test_df = pd.DataFrame({
    'text': test_pos_sentences + test_neg_sentences,
    'label': [1] * len(test_pos_sentences) + [0] * len(test_neg_sentences)  # 1 for positive, 0 for negative
})









# Check for NaN values in columns
print('train NANs:', train_df['text'].isna().sum(), train_df['label'].isna().sum())
print('test NANs:', test_df['text'].isna().sum(), test_df['label'].isna().sum())

# Check labels
train_unique_values = train_df['label'].unique()
test_unique_values = test_df['label'].unique()
print('Check labels:', train_unique_values, test_unique_values)

# Check max length
train_max_words = train_df['text'].apply(lambda x: len(x.split())).max()
test_max_words = test_df['text'].apply(lambda x: len(x.split())).max()
print('max_words:', train_max_words, test_max_words)



# Dataset
class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')    # as pytorch tensors

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor([label], dtype=torch.float)
        }
    


# Make torch DataLoader
batch_size = 32

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create data loaders
train_dataset = IMDBDataset(train_df, tokenizer, max_len=512)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = IMDBDataset(test_df, tokenizer, max_len=512)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



# Check tensors
for i, (train_batch, test_batch) in enumerate(zip(train_loader, test_loader)):
    if i == 30:
        break
    train_input_ids = train_batch['input_ids']
    train_attention_mask = train_batch['attention_mask']
    train_labels = train_batch['labels']
    
    test_input_ids = test_batch['input_ids']
    test_attention_mask = test_batch['attention_mask']
    test_labels = test_batch['labels']

    print(i, train_input_ids.shape, train_attention_mask.shape, train_labels.shape)
    print(i, test_input_ids.shape, test_attention_mask.shape, test_labels.shape)



# Define hyperparmaters
num_epochs = 3
learning_rate = 1e-5





# Define the BertForSentenceClassification model
class BertForSentenceClassification(nn.Module):
    def __init__(self, bert_model):
        super(BertForSentenceClassification, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs
    


# Apply and check the model

# Load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create an instance of the custom model
model = BertForSentenceClassification(bert_model)
model.to(device)



# Basic check
input_ids = torch.randint(0, 100, (32, 512)).to(device)  # random input IDs
attention_mask = torch.ones((32, 512)).to(device)  # random attention mask
output = model(input_ids, attention_mask)
print(output.shape)



# Define the optimizer and loss function
criterion = nn.BCELoss()

# # Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)



i = 0
losses = []
for epoch in range(1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        print(f"batch: {i}", end="\r")
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        i += 1
        if i == 5:
            break

    # Append the current loss to the list
    losses.append(loss.item())

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')