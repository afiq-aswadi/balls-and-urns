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

def generate_tokens(model, input_tokens, max_new_tokens=10, do_sample=True, temperature=1.0, num_samples=1):
    """
    Generate tokens from the model without requiring a tokenizer.
    
    Args:
        model: The transformer model
        input_tokens: Starting tokens as a tensor of shape [batch_size, seq_len]
        max_new_tokens: Maximum number of new tokens to generate
        do_sample: Whether to sample from the distribution (True) or use greedy decoding (False)
        temperature: Temperature for sampling (higher = more random)
        num_samples: Number of samples to generate for each input sequence
    
    Returns:
        Generated sequence including the input tokens, shape [batch_size * num_samples, seq_len + max_new_tokens]
    """
    model.eval()
    
    with torch.no_grad():
        # Repeat input tokens for each sample
        current_tokens = input_tokens.repeat_interleave(num_samples, dim=0)
        
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = model(current_tokens)
            
            # Get the logits for the last position
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Sample from the probability distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding - take the most likely token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the new token to the sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if we hit the context limit
            if current_tokens.shape[1] >= model.cfg.n_ctx:
                break
    
    return current_tokens

# Test the generation function with multiple samples
generated_sequences = generate_tokens(model, torch.tensor([[2]]).to(DEVICE), max_new_tokens=10, do_sample=True, num_samples=5)
print(f"Generated sequences shape: {generated_sequences.shape}")
for i, seq in enumerate(generated_sequences):
    print(f"Sample {i+1}: {seq[1:].tolist()}")  # Exclude BOS token

#%%


def predictive_resampling(model: transformer_lens.HookedTransformer,
    prefix: torch.Tensor = None,
    num_samples: int = 1000,
    M: int = 50,
    device: str = 'cpu',
    max_seq_length: int = 50
):
    """
    Perform predictive resampling using the Transformer model.
    
    Args:
        model: The Transformer model to use for resampling.
        prefix: Optional prefix tensor to condition the model on. If None, starts with BOS token (2).
        num_samples: Number of samples to generate.
        M: Number of samples per batch.
        device: Device to run the model on ('cpu' or 'cuda').
        max_seq_length: Maximum length of generated sequences.
    
    Returns:
        A tensor of probabilities for token 1, shape (num_samples,).
    """
    model.to(device).eval()
    
    # Set default prefix to BOS token if none provided
    if prefix is None:
        prefix = torch.tensor([[2]], device=device)
    else:
        prefix = prefix.to(device)
    
    # Generate samples
    all_probs = []
    num_batches = (num_samples + M - 1) // M  # Ceiling division
    
    for i in range(num_batches):
        current_batch_size = min(M, num_samples - i * M)
        
        # Create batch by repeating the prefix
        batch_prefix = prefix.repeat(current_batch_size, 1)
        
        # Generate sequences for this batch
        batch_sequences = generate_tokens(
            model, 
            batch_prefix, 
            max_new_tokens=max_seq_length - prefix.shape[1], 
            do_sample=True
        )
        
        # Compute logits and extract probability of token 1
        with torch.no_grad():
            logits = model(batch_sequences)
            probs = torch.softmax(logits, dim=-1)
            # Extract probability of token 1 from the last position
            prob_token_1 = probs[:, -1, 1]  # Shape: (batch_size,)
            all_probs.append(prob_token_1.cpu())
    
    return torch.cat(all_probs, dim=0)[:num_samples]


def compare_priors(
    A,
    B,
    model: transformer_lens.HookedTransformer,
    device: str = 'cpu',
    n_samples: int = 1000,
    zero_position: bool = False,
):
    
    """
    Based on the PUQ framework by Fortini and Petrone et al.
    """

    print(f"→ Starting compare_priors on device='{device}'.")

    model.to(device).eval()

    # Build a uniform grid on [0,1] and compute true prior density of P(A)
    grid = np.linspace(0, 1, 300)
    true_prior = beta(A, B).pdf(grid)

    plt.figure(figsize=(8, 5))

    # Plot true prior
    plt.plot(
        grid,
        true_prior,
        label='True Prior',
        lw=2
    )

    # Generate prior samples via Transformer predictive resampling (prefix=None)
    t1 = time.time()
    prior_samples = predictive_resampling(
        model,
        prefix=None,
        num_samples=n_samples,
        M=50,
        device=device
    ).cpu().numpy()
    print(f"  Transformer predictive resampling for prior in {time.time() - t1:.3f} sec.")

    # Overlay histogram of resampled prior P(A)
    # Use fewer bins for better visual comparison with the PDF
    n_bins = 50
    plt.hist(
        prior_samples,
        bins=n_bins,
        density=True,
        alpha=0.6,
        range=(0, 1),  # Ensure histogram covers the same range as the PDF
        label='Transformer Predictive Resampling'
    )

    # Use simple title and xlabel
    plt.title('Prior distribution comparison')
    plt.xlabel('P(A)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


#%%

compare_priors(
    A=1.0,
    B=1.0,
    model=model,
    device=DEVICE,
    n_samples = 10000
)
# %%
sample = generate_all_binary_sequences_with_fixed_num_ones(
    n=10,
    num_ones=2,
    prepend_bos=True,
    last_obs=1
)
#%%
with torch.inference_mode():
    logits, cache = model.run_with_cache(sample)

resids = cache["resid_mid", -1][:, -1, :]

#%%
print(f"Residuals shape: {resids.shape}")

#%%
print("Cache keys:", list(cache.keys()))


# %%
# Compute cosine similarity matrix
cosine_sim_matrix = sklearn_cosine_similarity(resids.cpu().numpy())

# Plot the cosine similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(cosine_sim_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity Matrix of Residual Vectors')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Index')
plt.tight_layout()
plt.show()

#%%
cosine_sim_matrix

#%%
# Iterate through sequences with different numbers of ones for length 10, last_obs=1
length = 10
mean_residuals = []

for n_ones in range(1, 11):
    # Generate sequences with n_ones ones, length 10, last observation = 1
    sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=length,
        num_ones=n_ones,
        prepend_bos=True,
        last_obs=1
    )
    
    if sequences.numel() > 0:  # Check if sequences were generated
        with torch.inference_mode():
            logits, cache = model.run_with_cache(sequences)
        
        # Get residuals from the last layer, last position
        resids = cache["resid_mid", -1][:, -1, :]

        # Compute mean residual across all sequences for this n_ones
        mean_resid = resids.mean(dim=0)
        mean_residuals.append(mean_resid)
        
        print(f"n_ones={n_ones}: {len(sequences)} sequences, mean residual shape: {mean_resid.shape}")
    else:
        print(f"n_ones={n_ones}: No valid sequences generated")

# Stack all mean residuals into a single tensor
if mean_residuals:
    mean_residuals_tensor = torch.stack(mean_residuals)
    print(f"\nFinal mean residuals tensor shape: {mean_residuals_tensor.shape}")
else:
    print("No mean residuals computed")

# %%
incremental_residuals = []
for i in range(1, len(mean_residuals_tensor)):
    increment = mean_residuals_tensor[i] - mean_residuals_tensor[i - 1]
    incremental_residuals.append(increment)

incremental_residuals = torch.stack(incremental_residuals)
print(f"Incremental residuals shape: {incremental_residuals.shape}")
# %%
# Compute cosine similarity matrix
cosine_sim_matrix_inc = sklearn_cosine_similarity(incremental_residuals.cpu().numpy())

# Plot the cosine similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(cosine_sim_matrix_inc, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity Matrix of Residual Vectors')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Index')
plt.tight_layout()
plt.show()

incremental_residuals

#%%
# Calculate L2 norms of incremental residuals
l2_norms_inc = torch.norm(incremental_residuals, dim=1)
plt.figure(figsize=(12, 6))
plt.plot(range(2, len(l2_norms_inc) + 2), l2_norms_inc.cpu().numpy(), 'b-o', linewidth=2, markersize=6)
plt.title('L2 Norm of Incremental Residuals (Length 10, last_obs=1)')
plt.xlabel('Number of Ones')
plt.ylabel('L2 Norm')
plt.grid(True, alpha=0.3)

# %%
# Iterate through sequence lengths from 3 to 10 with fixed num_ones=3 and last_obs=1
num_ones = 3
mean_residuals_by_length = []

for seq_length in range(3, 11):
    # Generate sequences with 3 ones, varying length, last observation = 1
    sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=seq_length,
        num_ones=num_ones,
        prepend_bos=True,
        last_obs=1
    )
    
    if sequences.numel() > 0:  # Check if sequences were generated
        with torch.inference_mode():
            logits, cache = model.run_with_cache(sequences)
        
        # Get residuals from the last layer, last position
        resids = cache["resid_mid", -1][:, -1, :]
        
        # Compute mean residual across all sequences for this length
        mean_resid = resids.mean(dim=0)
        mean_residuals_by_length.append(mean_resid)
        
        print(f"length={seq_length}: {len(sequences)} sequences, mean residual shape: {mean_resid.shape}")
    else:
        print(f"length={seq_length}: No valid sequences generated")

# Stack all mean residuals into a single tensor
if mean_residuals_by_length:
    mean_residuals_by_length_tensor = torch.stack(mean_residuals_by_length)
    print(f"\nFinal mean residuals tensor shape: {mean_residuals_by_length_tensor.shape}")
    
    # Compute cosine similarity matrix
    cosine_sim_matrix_length = sklearn_cosine_similarity(mean_residuals_by_length_tensor.cpu().numpy())
    
    # Plot the cosine similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cosine_sim_matrix_length, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Matrix of Mean Residual Vectors (num_ones=3, varying length)')
    plt.xlabel('Sequence Length Index (3-10)')
    plt.ylabel('Sequence Length Index (3-10)')
    
    # Add custom tick labels
    length_labels = list(range(3, 11))
    plt.xticks(range(len(length_labels)), length_labels)
    plt.yticks(range(len(length_labels)), length_labels)
    
    plt.tight_layout()
    plt.show()
    
    print("\nCosine similarity matrix:")
    print(cosine_sim_matrix_length)
else:
    print("No mean residuals computed")
# %%

# Iterate through sequences with different numbers of ones for length 10, last_obs=1
length = 9
mean_residuals = []

for n_ones in range(1, length + 1):
    # Generate sequences with n_ones ones, length 10, last observation = 1
    sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=length,
        num_ones=n_ones,
        prepend_bos=True,
        last_obs=0
    )
    
    if sequences.numel() > 0:  # Check if sequences were generated
        with torch.inference_mode():
            logits, cache = model.run_with_cache(sequences)
        
        # Get residuals from the last layer, last position
        resids = cache["resid_mid", -1][:, -1, :]
        
        # Compute mean residual across all sequences for this n_ones
        mean_resid = resids.mean(dim=0)
        mean_residuals.append(mean_resid)
        
        print(f"n_ones={n_ones}: {len(sequences)} sequences, mean residual shape: {mean_resid.shape}")
    else:
        print(f"n_ones={n_ones}: No valid sequences generated")

# Stack all mean residuals into a single tensor
if mean_residuals:
    mean_residuals_tensor = torch.stack(mean_residuals)
    print(f"\nFinal mean residuals tensor shape: {mean_residuals_tensor.shape}")
else:
    print("No mean residuals computed")

# %%
incremental_residuals_len_nine = []
for i in range(1, len(mean_residuals_tensor)):
    increment = mean_residuals_tensor[i] - mean_residuals_tensor[i - 1]
    incremental_residuals_len_nine.append(increment)

incremental_residuals_len_nine = torch.stack(incremental_residuals_len_nine )
print(f"Incremental residuals shape: {incremental_residuals.shape}")
# %%
# Compute cosine similarity matrix
cosine_sim_matrix_inc = sklearn_cosine_similarity(incremental_residuals_len_nine.cpu().numpy())

# Plot the cosine similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(cosine_sim_matrix_inc, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity Matrix of Residual Vectors')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Index')
plt.tight_layout()
plt.show()


# %%
stacked_resids = torch.cat([incremental_residuals, incremental_residuals_len_nine], dim=0)
stacked_resids.shape

# %%
cosine_sim_matrix_inc = sklearn_cosine_similarity(stacked_resids.cpu().numpy())

# Plot the cosine similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(cosine_sim_matrix_inc, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity Matrix of Residual Vectors')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Index')
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.plot(stacked_resids[:, 0].cpu().numpy())
plt.title('First Column of Stacked Residuals')
plt.xlabel('Sequence Index')
plt.ylabel('Residual Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
seq1 = torch.tensor([2, 1, 0, 0, 1, 0, 1,0,0,1])
seq2 = torch.tensor([2, 1, 0, 0, 1, 0, 1,1,1,1])
# Visualize attention for seq1 and seq2
sequences = torch.stack([seq1, seq2]).to(DEVICE)

with torch.inference_mode():
    logits, cache = model.run_with_cache(sequences)

# Get attention patterns
attention_patterns = cache["pattern", 0]  # Shape: [batch, n_heads, seq_len, seq_len]

# Create visualization for both sequences
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for seq_idx in range(2):
    # Get attention pattern for this sequence (assuming single head)
    attn_pattern = attention_patterns[seq_idx, 0].cpu().numpy()  # Shape: [seq_len, seq_len]
    
    im = axes[seq_idx].imshow(attn_pattern, cmap='Blues', vmin=0, vmax=1)
    axes[seq_idx].set_title(f'Attention Pattern - Sequence {seq_idx + 1}: {sequences[seq_idx].tolist()}')
    axes[seq_idx].set_xlabel('Key Position')
    axes[seq_idx].set_ylabel('Query Position')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[seq_idx])
    
    # Add grid
    axes[seq_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print attention weights for the last position (most relevant for prediction)
print("Attention weights for last position:")
for seq_idx in range(2):
    last_pos_attn = attention_patterns[seq_idx, 0, -1].cpu().numpy()
    print(f"Sequence {seq_idx + 1}: {last_pos_attn}")
# %%
torch.abs(cache["pattern",0][1,:,-1,:] - cache["pattern",0][0,:,-1,:])
# %%
print("All cache keys:")
for key in sorted(cache.keys()):
    print(f"  {key}")
# %%
cache["v",-1].shape
# %%
# Get value weights and positional embeddings
v_weights = model.W_V[0]  # Shape: [d_model, d_head]
pos_embeddings = model.W_pos  # Shape: [n_ctx, d_model]

# Take matrix product: pos_embeddings @ v_weights
pos_v_product = pos_embeddings @ v_weights  # Shape: [n_ctx, d_head]

print(f"Value weights shape: {v_weights.shape}")
print(f"Positional embeddings shape: {pos_embeddings.shape}")
print(f"Matrix product shape: {pos_v_product.shape}")
# %%
pos_v_product.squeeze()

# %%