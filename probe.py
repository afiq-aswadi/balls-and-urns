#%%
import transformer_lens
from samplers import generate_data, generate_data_with_p
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from model import deactivate_position
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import time
from utils import get_log_resids_from_sequential_ones, get_theoretical_log, calculate_optimal_loss
import itertools
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def prepend_bos(datasets, bos_token_value=2):
    """
    Prepend BOS token to each batch in the datasets.
    
    Args:
        datasets: List of torch tensors representing batches
        bos_token_value: Value of the BOS token (default: 2)
    
    Returns:
        List of modified tensors with BOS tokens prepended
    """
    modified_datasets = []
    for dataset in datasets:
        batch_size = dataset.shape[0]
        bos_tokens = torch.full((batch_size, 1), bos_token_value, dtype=torch.long)
        modified_dataset = torch.cat([bos_tokens, dataset], dim=1)
        modified_datasets.append(modified_dataset)
    return modified_datasets

#%%


def generate_all_binary_sequences_with_fixed_num_ones(n: int, num_ones: int, max_n_sequences: int = None, prepend_bos: bool = False, last_obs: int = None) -> torch.Tensor:
    """
    Generate all possible binary sequences of length n with exactly num_ones ones.
    If max_n_sequences is specified, only generate up to that many sequences.
    
    Args:
        n: Length of the sequence
        num_ones: Number of ones in each sequence
        max_n_sequences: Maximum number of sequences to generate (optional)
        prepend_bos: Whether to prepend BOS token (default: False)
        last_obs: If 0 or 1, fix the last token to this value (optional)
        
    Returns:
        torch.Tensor: Tensor of shape (num_permutations, n) or (num_permutations, n+1) if prepend_bos=True
    """
    if last_obs is not None and last_obs in [0, 1]:
        # If last token is fixed, we need to place ones in the first n-1 positions
        if last_obs == 1:
            # Last position is 1, so we need (num_ones - 1) ones in first (n-1) positions
            remaining_ones = num_ones - 1
        else:
            # Last position is 0, so we need num_ones ones in first (n-1) positions
            remaining_ones = num_ones
        
        # Check if it's possible to satisfy the constraint
        if remaining_ones < 0 or remaining_ones > (n - 1):
            return torch.empty((0, n), dtype=torch.long)
        
        # Generate combinations for the first n-1 positions
        positions_iter = itertools.combinations(range(n - 1), remaining_ones)
        if max_n_sequences is not None:
            positions_iter = itertools.islice(positions_iter, max_n_sequences)
        positions_list = list(positions_iter)
        num_permutations = len(positions_list)
        
        # Initialize the output tensor
        sequences = torch.zeros((num_permutations, n), dtype=torch.long)
        
        # Set the last position to the fixed value
        sequences[:, -1] = last_obs
        
        # Fill in the tensor with 1s at the appropriate positions in first n-1 positions
        for i, positions in enumerate(positions_list):
            for pos in positions:
                sequences[i, pos] = 1
    else:
        # Original behavior when last_obs is not specified or not 0/1
        positions_iter = itertools.combinations(range(n), num_ones)
        if max_n_sequences is not None:
            positions_iter = itertools.islice(positions_iter, max_n_sequences)
        positions_list = list(positions_iter)
        num_permutations = len(positions_list)
        
        # Initialize the output tensor
        sequences = torch.zeros((num_permutations, n), dtype=torch.long)
        
        # Fill in the tensor with 1s at the appropriate positions
        for i, positions in enumerate(positions_list):
            for pos in positions:
                sequences[i, pos] = 1
    
    # Prepend BOS token if requested
    if prepend_bos:
        bos_tokens = torch.full((num_permutations, 1), 2, dtype=torch.long)
        sequences = torch.cat([bos_tokens, sequences], dim=1)
    
    return sequences


#%%

coinformer_wit_bos_config = transformer_lens.HookedTransformerConfig(
    d_model=64, #embedding dimension
    d_head=64,
    n_layers=1,
    n_ctx=100,
    d_vocab=3,
    act_fn="relu",
    default_prepend_bos=False,  # Disable automatic BOS prepending since we do it manually
    normalization_type=None,
)

model = transformer_lens.HookedTransformer(coinformer_wit_bos_config).to(DEVICE)


#%%
def train_coinformer_model(model, 
                           num_epochs: int = 3,
                           learning_rate: float = 0.001,
                           batch_size: int = 64,
                           seq_length: int = 20,
                           num_batches: int = 100,
                           alpha_param: float = 1.0,
                           beta_param: float = 1.0,
                           bernoulli: bool = False,
                           bernoulli_p: float = 0.5,
                           flip_batch: bool = False,
                           pos_embed: bool = True,
                           add_zeros_and_ones: int = 0,
                           scale: float = 1.0,
                           bias: float = 0.0,
                           importance_sampling: bool = False,
                           importance_sampling_alpha: float = 1.0,
                           importance_sampling_beta: float = 1.0):
    """Train the Coinformer model."""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    # Deactivate positional embedding if specified
    if not pos_embed:
        model = deactivate_position(model)
    else:
        # Ensure positional embedding gradients are enabled when pos_embed=True
        for name, param in model.named_parameters():
            if "pos_embed" in name or "W_pos" in name:
                param.requires_grad = True
    

    for epoch in range(num_epochs):
        # Generate new data for each epoch
        datasets, priors = generate_data(batch_size=batch_size, seq_length=seq_length, 
                                   num_batches=num_batches, alpha=alpha_param, beta=beta_param, bernoulli=bernoulli, bernoulli_p=bernoulli_p, flip_batch=flip_batch
                                   , scale=scale, bias=bias)

        # Prepend BOS token to each batch
        datasets = prepend_bos(datasets)

        # Add the specified number of batches of all 0s and all 1s
        if add_zeros_and_ones > 0:
            zeros_ones_batches = []
            for _ in range(add_zeros_and_ones):
                zeros = torch.zeros((batch_size, seq_length), dtype=torch.long)
                ones = torch.ones((batch_size, seq_length), dtype=torch.long)
                zeros_ones_batches.extend([zeros, ones])
                priors.extend([0.0, 1.0])
            
            # Prepend BOS tokens to the newly added batches and extend datasets
            zeros_ones_with_bos = prepend_bos(zeros_ones_batches)
            datasets.extend(zeros_ones_with_bos)

        epoch_loss = 0
        
        # Plot histogram of priors for this epoch
        # plt.figure(figsize=(8, 4))
        # plt.hist(priors, bins=10, alpha=0.7, color='blue', edgecolor='black')
        # plt.title(f'Distribution of Prior Probabilities (Epoch {epoch+1})')
        # plt.xlabel('Probability Value')
        # plt.ylabel('Frequency')
        # plt.grid(True, alpha=0.3)
        # plt.show()
        if importance_sampling:
            optimal_loss = calculate_optimal_loss(importance_sampling_alpha, importance_sampling_beta)
        else:
            optimal_loss = calculate_optimal_loss(alpha_param, beta_param)

        for data_batch, prior in tqdm(zip(datasets, priors), desc=f"Epoch {epoch+1}/{num_epochs}"):
            data_batch = data_batch.to(DEVICE)
            # For each sequence, use all tokens except the last one as input
            inputs = data_batch[:, :-1]
            # Use all tokens except the first one as targets
            targets = data_batch[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape logits and targets for loss calculation
            logits_view = logits.view(-1, model.cfg.d_vocab)
            targets_view = targets.reshape(-1)
            
            # Calculate loss
            loss = criterion(logits_view, targets_view)
            if importance_sampling:
                weights = beta(importance_sampling_alpha, importance_sampling_beta).pdf(prior)
                loss = loss * weights

            epoch_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = epoch_loss / len(datasets)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Theoretical Loss Lower Bound: {optimal_loss:.4f}")

        # # After each epoch, analyze model's behavior on sequences with a single 1
        # model.eval()  # Set model to evaluation mode
        # with torch.no_grad():  # Disable gradient computation
        #     print(f"\\nAnalyzing model behavior after epoch {epoch+1}:")
        #     analyze_single_one_sequences(model, seq_length=20)
        # model.train()  # Set model back to training mode

    return losses

#%%

losses = train_coinformer_model(model,
                                num_epochs=10,
                                learning_rate=0.001,
                                batch_size=64,
                                seq_length=99,
                                num_batches=1000,
                                alpha_param=1.0,
                                beta_param=1.0,
                                pos_embed=True
                                )


#%%

batch_size = 100  # Number of rows
seq_length = 100  # Total length (including the first column of 2s)

# Initialize the tensor with zeros
tensor = torch.zeros((batch_size, seq_length), dtype=torch.long)

# Set the first column to 2 (BOS token)
tensor[:, 0] = 2

# Fill a lower‐triangular band of 1’s (including the diagonal) starting at column 1
for i in range(1, min(batch_size, seq_length)):
    tensor[i, 1 : i+1] = 1


logits, cache = model.run_with_cache(tensor)

resids = cache["resid_mid", -1]

resids.shape

mask2d = torch.triu(torch.ones((batch_size, seq_length), dtype=torch.bool), diagonal=0)
mask2d

relevant_resids = resids[mask2d]

N_obs = torch.zeros_like(tensor)

for i in range(seq_length):
    N_obs[:, i] = i

N_obs

relevant_N = N_obs[mask2d]

H_obs = tensor.clone()

H_obs[:,0] = 0

for i in range(1, batch_size):
    H_obs[i, : ] = torch.cumsum(tensor[i, :], dim=0) -2

H_obs

relevant_H = H_obs[mask2d]

print(relevant_resids.shape)
print(relevant_N.shape)
print(relevant_H.shape)

#%%
import torch
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=64):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Prepare data
# Move tensors to same device and convert to float
N = relevant_N.float().to(DEVICE)
H = relevant_H.float().to(DEVICE)
targets = relevant_resids.to(DEVICE)

# Create input tensor
X = torch.stack([N, H], dim=1)  # Shape: [num_samples, 2]

# Split into train and test sets
total_size = X.size(0)
test_size = int(0.2 * total_size)  # 20% for test set
train_size = total_size - test_size

indices = torch.randperm(total_size)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create train and test sets
X_train = X[train_indices]
targets_train = targets[train_indices]
X_test = X[test_indices]
targets_test = targets[test_indices]

# Create and train the model
mlp = MLP(input_dim=2, hidden_dim=128, output_dim=64).to(DEVICE)
optimizer = optim.Adam(mlp.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 30
batch_size = 256

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    mlp.train()
    epoch_train_loss = 0
    num_batches = (X_train.size(0) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_train.size(0))
        
        batch_X = X_train[start_idx:end_idx]
        batch_targets = targets_train[start_idx:end_idx]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = mlp(batch_X)
        
        # Calculate loss
        loss = criterion(outputs, batch_targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * (end_idx - start_idx)
    
    avg_train_loss = epoch_train_loss / X_train.size(0)
    train_losses.append(avg_train_loss)
    
    # Testing
    mlp.eval()
    with torch.no_grad():
        test_outputs = mlp(X_test)
        test_loss = criterion(test_outputs, targets_test)
        test_losses.append(test_loss.item())
    
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss.item():.6f}")

# Plot the training and test losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Test the model on held-out test set
with torch.no_grad():
    # Sample a few points from test set to compare predictions with actual values
    sample_indices = torch.randint(0, X_test.size(0), (5,))
    sample_X = X_test[sample_indices]
    sample_targets = targets_test[sample_indices]
    
    predictions = mlp(sample_X)
    
    for i in range(5):
        mse = criterion(predictions[i], sample_targets[i])
        
        # Calculate cosine similarity
        pred_np = predictions[i].cpu().numpy().reshape(1, -1)
        target_np = sample_targets[i].cpu().numpy().reshape(1, -1)
        cosine_sim = sklearn_cosine_similarity(pred_np, target_np)[0, 0]
        
        print(f"Input: N={sample_X[i][0].item():.1f}, H={sample_X[i][1].item():.1f}")
        print(f"Target: {sample_targets[i][:5]}")
        print(f"Prediction: {predictions[i][:5]}")
        print(f"MSE: {mse.item():.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        print()
# %%
