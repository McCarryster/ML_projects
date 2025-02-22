import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Initialize the model, optimizer, and loss function
device = 'cpu'  # Set device to CPU
model = SimpleNet().to(device)  # Move model to CPU
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001,
    betas=(0.9, 0.999)
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=5,
    num_training_steps=5
)
criterion = nn.CrossEntropyLoss()

# Create some dummy data
input_data = torch.randn(32, 10).to(device)  # Batch size of 32
target_data = torch.randint(0, 2, (32,)).to(device)  # Binary target

# Training loop with autocast on CPU
for epoch in range(5):  # Run for a few epochs
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):  # Enable autocasting for mixed precision on CPU
        output = model(input_data)  # Forward pass
        loss = criterion(output, target_data)  # Compute loss
    
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

