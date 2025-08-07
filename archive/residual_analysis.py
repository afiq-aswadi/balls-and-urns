#%%
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import plot_utils as pu
from train import train_coinformer_model
from samplers import generate_all_binary_sequences_with_fixed_num_ones, generate_sequential_ones
from utils import get_log_resids_from_sequential_ones, flip_batch, get_residual_cosine_similarity


def plot_residual_cosine_similarity_sorted_by_first(resids, seq_length, num_ones):
    """
    Plot cosine similarity matrix with columns sorted by similarity to the original first sequence.
    This function handles tensor manipulation and sorting logic, then uses pu.plot_residual_cosine_similarity.
    
    Args:
        resids: Residual vectors tensor
        seq_length: Length of sequences
        num_ones: Number of ones in sequences
    """
    # First get the cosine similarity matrix using the existing plotting function
    print("Original cosine similarity matrix:")
    cos_sim_matrix = get_residual_cosine_similarity(resids, print_stats=True)

    # Get the sorting order based on the first column (similarity with the first sequence)
    first_col_similarities = cos_sim_matrix[:, 0]
    sorted_indices_by_first_col = np.argsort(-first_col_similarities)
    
    # Reorder the columns of the original matrix based on these sorted indices
    cos_sim_matrix_cols_sorted_by_first = cos_sim_matrix[:, sorted_indices_by_first_col]
    
    print("\nPlotting cosine similarity matrix with columns sorted by similarity to the original first sequence")
    # Create a new figure for the reordered plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cos_sim_matrix_cols_sorted_by_first, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Cosine Similarity (Columns sorted by similarity to original first sequence)")
    plt.xlabel("Sequence Index (Reordered)")
    plt.ylabel("Sequence Index (Original Order)")
    plt.tight_layout()
    plt.show()
    
    return cos_sim_matrix, sorted_indices_by_first_col


def plot_original_vs_flipped_histogram(cosine_sim_flipped_vs_original, seq_length, num_ones):
    """
    Plot histogram of cosine similarities between original and flipped residuals.
    
    Args:
        cosine_sim_flipped_vs_original: Tensor of cosine similarities between corresponding residuals
        seq_length: Length of sequences
        num_ones: Number of ones in sequences
    """
    # Plot a histogram of these cosine similarities
    plt.figure(figsize=(8, 6))
    plt.hist(cosine_sim_flipped_vs_original.cpu().numpy(), bins=20, edgecolor='black')
    plt.title(f"Distribution of Cosine Similarities (Original vs Flipped Residuals)\nSeq Length: {seq_length}, Num Ones: {num_ones}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_flipped_residual_similarity(flipped_resids, seq_length, num_ones):
    """
    Plot cosine similarity matrix for flipped residuals using existing plot utility.
    This function handles tensor inputs and calls pu.plot_residual_cosine_similarity.
    
    Args:
        flipped_resids: Flipped residual vectors tensor
        seq_length: Length of sequences
        num_ones: Number of ones in sequences
    """
    print("Cosine similarity matrix for flipped residuals:")
    return pu.plot_residual_cosine_similarity(seq_length, num_ones, flipped_resids, print_stats=True)


def plot_combined_original_and_flipped_similarity(original_resids, flipped_resids, seq_length, num_ones):
    """
    Plot combined cosine similarity matrix for original and flipped residuals.
    This function handles tensor manipulation and uses pu.plot_residual_cosine_similarity for plotting.
    
    Args:
        original_resids: Original residual vectors tensor
        flipped_resids: Flipped residual vectors tensor
        seq_length: Length of sequences
        num_ones: Number of ones in sequences
    """
    # Combine original and flipped residuals
    combined_resids = torch.cat((original_resids, flipped_resids), dim=0)
    
    # Calculate cosine similarity matrix using existing plot utility
    # We'll temporarily disable stats printing since we want custom formatting
    combined_cos_sim_matrix = pu.plot_residual_cosine_similarity(
        seq_length=seq_length, 
        num_ones=num_ones, 
        resids=combined_resids, 
        print_stats=False
    )
    
    # Add custom annotations to show the separation between original and flipped
    num_original_seqs = original_resids.shape[0]
    print(f"\nCombined similarity matrix: {num_original_seqs} original + {num_original_seqs} flipped sequences")
    print("Matrix sections:")
    print("  Top-left: Original vs Original")
    print("  Top-right: Original vs Flipped") 
    print("  Bottom-left: Flipped vs Original")
    print("  Bottom-right: Flipped vs Flipped")
    
    return combined_cos_sim_matrix


def print_cross_similarity_stats(cross_similarity_block):
    """
    Print statistics for the cross-similarity block (original vs. flipped).
    
    Args:
        cross_similarity_block: Cross-similarity matrix block
    """
    print("\nCosine Similarity Stats (Original Residuals vs. Flipped Residuals):")
    if cross_similarity_block.size > 0:
        print(f"  Mean: {cross_similarity_block.mean():.4f}")
        print(f"  Std: {cross_similarity_block.std():.4f}")
        print(f"  Max: {cross_similarity_block.max():.4f}")
        print(f"  Min: {cross_similarity_block.min():.4f}")

        # Specifically, the diagonal of this block represents original_i vs flipped_i
        if cross_similarity_block.shape[0] == cross_similarity_block.shape[1]:
            diag_cross_similarity = np.diag(cross_similarity_block)
            print("\nCosine Similarity Stats (Original_i vs. Flipped_i Residuals):")
            print(f"  Mean: {diag_cross_similarity.mean():.4f}")
            print(f"  Std: {diag_cross_similarity.std():.4f}")
            print(f"  Max: {diag_cross_similarity.max():.4f}")
            print(f"  Min: {diag_cross_similarity.min():.4f}")
    else:
        print("  Not enough elements for statistics.")


def plot_cross_similarity_heatmap(cross_similarity_block, seq_length, num_ones):
    """
    Plot heatmap of the cross-similarity block (original vs. flipped).
    
    Args:
        cross_similarity_block: Cross-similarity matrix block
        seq_length: Length of sequences
        num_ones: Number of ones in sequences
    """
    plt.figure(figsize=(9, 7))
    plt.imshow(cross_similarity_block, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    title_str_cross = (f"Cosine Similarity: Original vs. Flipped Residuals\n"
                       f"Seq Length: {seq_length}, Num Ones: {num_ones}")
    plt.title(title_str_cross)
    plt.xlabel("Flipped Sequence Index")
    plt.ylabel("Original Sequence Index")
    plt.tight_layout()
    plt.show()

def split_sequence_by_index_value(sequences, index, value):
    """
    Split sequences into two groups based on the value at a specific index.
    
    Args:
        sequences: Tensor of sequences
        index: Index to check for the value
        value: Value to check for at the specified index
    
    Returns:
        Tuple of two tensors: one with sequences where sequences[:, index] == value,
        and another where sequences[:, index] != value.
    """
    mask = (sequences[:, index] == value)
    return sequences[mask], sequences[~mask]


#%%
attn_only = False
n_layers = 2
alpha = 1.0
beta = 1.0
load_model = True

#%%
uniform_transformer_config = HookedTransformerConfig(
    d_model=64,  # embedding dimension
    d_head=64,
    n_layers=2,
    n_ctx=100,
    d_vocab=2,
    act_fn="relu",
    default_prepend_bos=False,
    normalization_type=None,
    attn_only=False,
)

uniform_transformer = HookedTransformer(uniform_transformer_config)

if load_model:
    # Load the model's state dictionary
    save_dir = "saved_models"
    save_path = os.path.join(
        save_dir,
        f"uniform_coinformer_dmodel{uniform_transformer_config.d_model}_dhead{uniform_transformer_config.d_head}_layers{n_layers}_attnonly{attn_only}_alpha{alpha}_beta{beta}.pt"
    )
    uniform_transformer.load_state_dict(torch.load(save_path))
    print(f"Model loaded from {save_path}")
else:
    losses= train_coinformer_model(
        model=uniform_transformer,
        num_epochs=5,
        learning_rate=0.001,
        batch_size=64,
        seq_length=100,
        num_batches=1000,
        alpha_param=alpha,
        beta_param=beta,
        bernoulli=False,  # Set to False for uniform distribution
        bernoulli_p=None,  # No need for this parameter
        pos_embed=True,  # Activate positional embedding
        flip_batch=True,
        scale=1.0,
        bias=0.0,
        importance_sampling=False,
    )

#%%
seq_length = 10
num_ones = 2

all_binary_sequences = generate_all_binary_sequences_with_fixed_num_ones(
    n = seq_length,
    num_ones = num_ones,
    max_n_sequences= 500
)

all_binary_sequences_end_one, all_binary_sequences_end_zero = split_sequence_by_index_value(all_binary_sequences, -1, 1)

with torch.inference_mode():
    logits, cache = uniform_transformer.run_with_cache(all_binary_sequences)
    logits_end_one, cache_end_one = uniform_transformer.run_with_cache(all_binary_sequences_end_one)
    logits_end_zero, cache_end_zero = uniform_transformer.run_with_cache(all_binary_sequences_end_zero)

resids = cache["resid_post", -1][:, -1, :]
resids_end_one = cache_end_one["resid_post", -1][:, -1, :]
resids_end_zero = cache_end_zero["resid_post", -1][:, -1, :]


cos_sim_matrix = pu.plot_residual_cosine_similarity(seq_length, num_ones, resids)
cos_sim_matrix_end_one = pu.plot_residual_cosine_similarity(seq_length, num_ones, resids_end_one)
cos_sim_matrix_end_zero = pu.plot_residual_cosine_similarity(seq_length, num_ones, resids_end_zero)




#%%

# Get the sorting order based on the first column (similarity with the first sequence)
# We want to sort in descending order, so we negate the first column for argsort

# If you want to sort by the similarity of each sequence to a *specific* sequence (e.g., the first one in the *original* order)
# and then reorder the columns based on that:
cos_sim_matrix, sorted_indices_by_first_col = plot_residual_cosine_similarity_sorted_by_first(resids, seq_length, num_ones)

# To make the plot more interpretable, you might also want to reorder the rows
# so that the sequence that was originally first (and used for sorting) is still at the top.
# Or, more consistently, reorder rows by the same logic as columns.
cos_sim_matrix_reordered_symmetrically = cos_sim_matrix[sorted_indices_by_first_col, :]
cos_sim_matrix_reordered_symmetrically = cos_sim_matrix_reordered_symmetrically[:, sorted_indices_by_first_col]


#%%
flipped_binary_sequences = flip_batch(all_binary_sequences)
with torch.inference_mode():
    flipped_logits, flipped_cache = uniform_transformer.run_with_cache(flipped_binary_sequences)
flipped_resids = flipped_cache["resid_post", -1][:, -1, :]

# Plot cosine similarity for flipped residuals
cosine_sim_mat_flipped = plot_flipped_residual_similarity(flipped_resids, seq_length, num_ones)

# Calculate cosine similarity between corresponding residuals
cosine_sim_flipped_vs_original = torch.nn.functional.cosine_similarity(resids, flipped_resids, dim=1)

plot_original_vs_flipped_histogram(cosine_sim_flipped_vs_original, seq_length, num_ones)

#%%
combined_cos_sim_matrix = plot_combined_original_and_flipped_similarity(resids, flipped_resids, seq_length, num_ones)


#%%
# Print stats for the cross-similarity block (original vs. flipped)
# This is the top-right block of the combined_cos_sim_matrix
num_original_seqs = resids.shape[0]
cross_similarity_block = combined_cos_sim_matrix[:num_original_seqs, num_original_seqs:]
print_cross_similarity_stats(cross_similarity_block)

#%%
# Plot just the cross-similarity block (original vs. flipped)
plot_cross_similarity_heatmap(cross_similarity_block, seq_length, num_ones)

#%%
prompt_1 = [1] + [0] * 29
prompt_2 = [0] * 29 + [1]

prompt = torch.tensor([prompt_1, prompt_2], dtype=torch.long)

with torch.inference_mode():
    logits, cache = uniform_transformer.run_with_cache(prompt)

resids = cache["resid_post", -1][:, -1, :]

sim = cosine_similarity(resids[0].unsqueeze(0).cpu().numpy(), resids[1].unsqueeze(0).cpu().numpy())
# %%
sim
# %%
seq_length = 10
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()

for i, num_ones in enumerate(range(1,seq_length)):
    all_binary_sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=seq_length,
        num_ones=num_ones,
    )

    with torch.inference_mode():
        logits, cache = uniform_transformer.run_with_cache(all_binary_sequences)

    resids = cache["resid_post", -1][:, -1, :]
    
    # Get cosine similarity matrix without plotting individually
    cos_sim_matrix = get_residual_cosine_similarity(resids, print_stats=False)
    
    # Plot on the subplot
    im = axes[i].imshow(cos_sim_matrix, cmap='viridis', aspect='auto')
    axes[i].set_title(f'num_ones={num_ones}')
    axes[i].set_xlabel('Sequence Index')
    axes[i].set_ylabel('Sequence Index')

# Add a colorbar for the entire figure
fig.colorbar(im, ax=axes, label='Cosine Similarity', shrink=0.8)
plt.tight_layout()
plt.show()

# %%
seq_length = 10
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()

for i, num_ones in enumerate(range(1,seq_length)):
    all_binary_sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=seq_length,
        num_ones=num_ones,
    )

    all_binary_sequences_end_one, _ = split_sequence_by_index_value(all_binary_sequences, -1, 1)

    with torch.inference_mode():
        logits, cache = uniform_transformer.run_with_cache(all_binary_sequences_end_one)

    resids = cache["resid_post", -1][:, -1, :]
    
    # Get cosine similarity matrix without plotting individually
    cos_sim_matrix = get_residual_cosine_similarity(resids, print_stats=False)
    
    # Plot on the subplot
    im = axes[i].imshow(cos_sim_matrix, cmap='viridis', aspect='auto')
    axes[i].set_title(f'num_ones={num_ones}')
    axes[i].set_xlabel('Sequence Index')
    axes[i].set_ylabel('Sequence Index')

# Add a colorbar for the entire figure
fig.colorbar(im, ax=axes, label='Cosine Similarity', shrink=0.8)
plt.tight_layout()
plt.show()


#%%
seq_length = 10
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()

for i, num_ones in enumerate(range(1,seq_length)):
    all_binary_sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=seq_length,
        num_ones=num_ones,
    )

    _, all_binary_sequences_end_zero = split_sequence_by_index_value(all_binary_sequences, -1, 1)

    with torch.inference_mode():
        logits, cache = uniform_transformer.run_with_cache(all_binary_sequences_end_zero)

    resids = cache["resid_post", -1][:, -1, :]
    
    # Get cosine similarity matrix without plotting individually
    cos_sim_matrix = get_residual_cosine_similarity(resids, print_stats=False)
    
    # Plot on the subplot
    im = axes[i].imshow(cos_sim_matrix, cmap='viridis', aspect='auto')
    axes[i].set_title(f'num_ones={num_ones}')
    axes[i].set_xlabel('Sequence Index')
    axes[i].set_ylabel('Sequence Index')

# Add a colorbar for the entire figure
fig.colorbar(im, ax=axes, label='Cosine Similarity', shrink=0.8)
plt.tight_layout()
plt.show()
# %%
