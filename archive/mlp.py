#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#%%
# Configuration parameters
HIDDEN_DIM = 64  # You can change this
A = 1.0  # Parameter 'a' in log(a+H)/(b+T)
B = 1.0  # Parameter 'b' in log(a+H)/(b+T)
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32

#%%
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

#%%
# Dataset class for generating training data
class LogRatioDataset(Dataset):
    def __init__(self, num_samples=10000, a=1.0, b=1.0, max_val=100):
        self.num_samples = num_samples
        self.a = a
        self.b = b
        
        # Generate random integer inputs H and T
        self.H = torch.randint(0, max_val, (num_samples,)).float()
        self.T = torch.randint(0, max_val, (num_samples,)).float()
        
        # Stack H and T as input features
        self.X = torch.stack([self.H, self.T], dim=1)
        
        # Compute target: log((a+H)/(b+T))
        self.y = torch.log((self.a + self.H) / (self.b + self.T)).unsqueeze(1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%%
# Create dataset and dataloader
dataset = LogRatioDataset(num_samples=10000, a=A, b=B)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create test dataset
test_dataset = LogRatioDataset(num_samples=1000, a=A, b=B)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(dataset)}")
print(f"Test samples: {len(test_dataset)}")

#%%
# Initialize model, loss function, and optimizer
model = MLP(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

#%%
# Training loop
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_train_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / len(dataloader)
    train_losses.append(avg_train_loss)
    
    # Evaluation
    model.eval()
    epoch_test_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            epoch_test_loss += loss.item()
    
    avg_test_loss = epoch_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

#%%
# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

#%%
# Test the model with some specific examples
model.eval()
test_cases = [
    [1, 1],
    [5, 3],
    [10, 7],
    [20, 15],
    [50, 30]
]

print(f"\nTesting with a={A}, b={B}")
print("H\tT\tPredicted\tActual\t\tError")
print("-" * 50)

with torch.no_grad():
    for h, t in test_cases:
        input_tensor = torch.tensor([[h, t]], dtype=torch.float32)
        predicted = model(input_tensor).item()
        actual = np.log((A + h) / (B + t))
        error = abs(predicted - actual)
        print(f"{h}\t{t}\t{predicted:.6f}\t{actual:.6f}\t{error:.6f}")

#%%


