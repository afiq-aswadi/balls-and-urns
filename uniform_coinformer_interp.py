#%%
from model import coinformer_model_attn_only_config
from utils import calculate_posterior_mean, count_ones_and_zeros
from plot_utils import visualize_attention_patterns
from samplers import generate_data_with_p

from transformer_lens import HookedTransformer
import torch
import itertools
import numpy as np

#%%

model = HookedTransformer(coinformer_model_attn_only_config)
# Load the weights from the PT file
weights_path = "saved_models/uniform_coinformer_alpha1.0_beta1.0.pt"
model.load_state_dict(torch.load(weights_path))

# Verify the model loaded correctly
print("Model weights loaded successfully.")

#%%
visualize_attention_patterns(
    theta=0.5,
    model=model,
    seq_length=20,
)

#%%

num_ones = 5
n = 20

def generate_all_binary_sequences_with_fixed_num_ones(n: int, num_ones: int) -> torch.Tensor:
    """
    Generate all possible binary sequences of length n with exactly num_ones ones.
    
    Args:
        n: Length of the sequence
        num_ones: Number of ones in each sequence
        
    Returns:
        torch.Tensor: Tensor of shape (num_permutations, n) containing all permutations
    """
    # Generate all combinations of positions for the ones
    positions_list = list(itertools.combinations(range(n), num_ones))
    num_permutations = len(positions_list)
    
    # Initialize the output tensor
    sequences = torch.zeros((num_permutations, n), dtype=torch.long)
    
    # Fill in the tensor with 1s at the appropriate positions
    for i, positions in enumerate(positions_list):
        for pos in positions:
            sequences[i, pos] = 1
    
    return sequences

def generate_sequential_ones(n: int) -> torch.Tensor:
    """
    Generate a sequence of ones followed by zeros.
    
    Args:
        n: Length of the sequence
        
    Returns:
        torch.Tensor: Tensor of shape (1, n) containing the sequence
    """
    return torch.tril(torch.ones((n, n), dtype=torch.long).unsqueeze(0)


test_seq = generate_all_binary_sequences_with_fixed_num_ones(n, num_ones)

#%%
# after this the posterior should be: 11/21....... so the logit should be log of that plus some constant.
# will the constant be the same for all sequences? 
# logit for 1 is log(11)+c for some c... same for logit for 2.

def theoretical_logit(num_ones, n, alpha, beta):
    theo_0 = torch.log(torch.tensor(n - num_ones + beta)).unsqueeze(0)
    theo_1 = torch.log(torch.tensor(num_ones + alpha)).unsqueeze(0)
    return torch.cat([theo_0, theo_1], dim=0)

theoretical_logits = theoretical_logit(num_ones, n, 1, 1)


#%%
logits, cache = model.run_with_cache(test_seq)

#%%
last_logits = logits[:, -1, :]

#%%
import matplotlib.pyplot as plt

# Clear any previous plots
plt.close('all')

# Create a figure with subplots
plt.figure(figsize=(12, 6))

# Convert to numpy for easier manipulation
logits_np = last_logits.cpu().detach().numpy()
theo_np = theoretical_logits.cpu().detach().numpy()

# Create sequence indices for x-axis
seq_indices = np.arange(logits_np.shape[0])

# Plot the logits for class 0 and class 1
plt.subplot(1, 2, 1)
plt.bar(seq_indices - 0.2, logits_np[:, 0], width=0.4, label='Model Logit (0)')
plt.bar(seq_indices + 0.2, logits_np[:, 1], width=0.4, label='Model Logit (1)')
plt.xlabel('Sequence Index')
plt.ylabel('Logit Value')
plt.title('Model Logits for Each Sequence')
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate and plot the differences between model and theoretical logits
plt.subplot(1, 2, 2)

# Compute differences
diff0 = logits_np[:, 0] - theo_np[0]
diff1 = logits_np[:, 1] - theo_np[1]

# Calculate mean differences
mean_diff0 = np.mean(diff0)
mean_diff1 = np.mean(diff1)

plt.scatter(seq_indices, diff0, marker='o', label=f'Logit(0) - Theo(0), mean={mean_diff0:.4f}')
plt.scatter(seq_indices, diff1, marker='x', label=f'Logit(1) - Theo(1), mean={mean_diff1:.4f}')
plt.axhline(y=mean_diff0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=mean_diff1, color='g', linestyle='--', alpha=0.5)

plt.xlabel('Sequence Index')
plt.ylabel('Logit Difference')
plt.title('Differences Between Model and Theoretical Logits')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Calculate statistics for diff0 and diff1
print("Statistics for diff0 (Logit(0) - Theoretical(0)):")
print(f"  Mean: {np.mean(diff0):.6f}")
print(f"  Median: {np.median(diff0):.6f}")
print(f"  Standard Deviation: {np.std(diff0):.6f}")
print(f"  Min: {np.min(diff0):.6f}")
print(f"  Max: {np.max(diff0):.6f}")
print(f"  Range: {np.max(diff0) - np.min(diff0):.6f}")

print("\nStatistics for diff1 (Logit(1) - Theoretical(1)):")
print(f"  Mean: {np.mean(diff1):.6f}")
print(f"  Median: {np.median(diff1):.6f}")
print(f"  Standard Deviation: {np.std(diff1):.6f}")
print(f"  Min: {np.min(diff1):.6f}")
print(f"  Max: {np.max(diff1):.6f}")
print(f"  Range: {np.max(diff1) - np.min(diff1):.6f}")
# %%
# idea: reverse engineer the way model takes the logarithm.
# to get the logit we take the dot product of the residual and unembedding
# since this is a single layer attn only transformer, the attention head performs all the computation.
# so technically it should really be the softmaxing that computes the logarithm 
# the operation is just... your embedding + pos embedding then put that into attention. the output of that dotted with the unembed gives the logit

# Extract embedding weights
embedding_weights = model.embed.W_E.detach().cpu()
print("Embedding weights shape:", embedding_weights.shape)

# Extract positional embedding weights
positional_embedding_weights = model.pos_embed.W_pos.detach().cpu()
print("Positional embedding weights shape:", positional_embedding_weights.shape)

# Extract unembedding weights
unembedding_weights = model.unembed.W_U.detach().cpu()
print("Unembedding weights shape:", unembedding_weights.shape)

# Get the relevant residual streams at the last position for each sequence
residual_stream = cache["resid_post", 0][:, -1, :].detach().cpu()
print("Residual stream shape (last position):", residual_stream.shape)

#%%
test = cache["resid_pre", 0][:, -1, :].detach().cpu()
# %%
# List all keys available in the cache
print("Available keys in cache:")
for key in cache.keys():
    print(key)
# %%
cache["pattern",0].shape
# Let's inspect the attention patterns (softmax scores) at the last position for each sequence.
# These should reflect how the model aggregates information to compute the logit.
#%%

# Count ones and zeros for each sequence
num_ones_per_seq = test_seq.sum(dim=1).numpy()
num_zeros_per_seq = n - num_ones_per_seq

# Project residual stream onto a random direction or the unembedding direction for class 1
proj_direction = unembedding_weights[:, 1]  # shape: [d_model]
projections = (residual_stream @ proj_direction).numpy()

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.scatter(num_ones_per_seq, projections, label="Projection vs #ones")
plt.xlabel("Number of ones in sequence")
plt.ylabel("Projection of residual stream onto unembed(1)")
plt.title("Does the residual stream encode the number of ones?")
plt.grid(True)
plt.legend()
plt.show()
# %%

# Get the attention pattern at the last position for each sequence
attn_pattern = cache["pattern", 0][:, -1, :]  # shape: [num_sequences, seq_length]
attn_pattern.shape


#%%
# For each sequence, sum the attention paid to positions with a 1 and with a 0
attn_to_ones = (attn_pattern * test_seq).sum(dim=1)
attn_to_zeros = (attn_pattern * (1 - test_seq)).sum(dim=1)

# Take the log of these sums
log_attn_to_ones = torch.log(attn_to_ones + 1e-8)
log_attn_to_zeros = torch.log(attn_to_zeros + 1e-8)

# Compare to the model's logits for class 1 and class 0
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(log_attn_to_ones.numpy(), last_logits[:, 1].detach().numpy(), label="Class 1")
plt.xlabel("log(sum attention to ones)")
plt.ylabel("Model logit for class 1")
plt.title("Logit vs log(attn to ones)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(log_attn_to_zeros.numpy(), last_logits[:, 0].detach().numpy(), label="Class 0")
plt.xlabel("log(sum attention to zeros)")
plt.ylabel("Model logit for class 0")
plt.title("Logit vs log(attn to zeros)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
# %%
