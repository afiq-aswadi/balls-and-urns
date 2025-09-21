#%%
"""
Experiment 3: Permutation Invariance Analysis

This experiment tests whether the model's residual representations are permutation
invariant for sequences with the same number of 1s, as expected for Bayesian updating.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from core.config import PERMUTATION_CONFIG
from core.training import train_coinformer_model, save_model_with_config
from core.samplers import generate_all_binary_sequences_with_fixed_num_ones
from core.plotting import plot_residual_cosine_similarity
from core.utils import flip_batch, get_residual_cosine_similarity

#%% Configuration
config = PERMUTATION_CONFIG
seq_length = config.seq_length
print(f"Model config: d_model={config.model_config.d_model}")
print(f"Sequence length: {seq_length}")
print(f"Use BOS token: {config.model_config.use_bos_token}")

#%% Train model
print("Training coinformer model...")
model, losses = train_coinformer_model(config)
save_path = save_model_with_config(model, config, "permutation_invariance")
print(f"Model saved to: {save_path}")

#%% Generate test sequences with fixed number of ones
num_ones_values = [5, 10, 15, 20]  # Different numbers of ones to test
max_sequences = 1000  # Limit to avoid memory issues

all_results = {}

for num_ones in num_ones_values:
    if num_ones >= seq_length:
        continue
        
    print(f"\n=== Testing sequences with {num_ones} ones ===")
    
    # Generate all binary sequences with exactly num_ones ones
    sequences = generate_all_binary_sequences_with_fixed_num_ones(
        n=seq_length,
        num_ones=num_ones,
        max_n_sequences=max_sequences,
        use_bos_token=config.model_config.use_bos_token
    )
    
    print(f"Generated {sequences.shape[0]} sequences")
    
    # Get model outputs and residuals
    with torch.inference_mode():
        logits, cache = model.run_with_cache(sequences)
    
    # Extract residuals from the last position of the final layer
    residuals = cache["resid_post", -1][:, -1, :]  # [num_sequences, d_model]
    
    # Calculate cosine similarity matrix
    cos_sim_matrix = plot_residual_cosine_similarity(
        seq_length, num_ones, residuals, print_stats=True
    )
    
    # Store results
    all_results[num_ones] = {
        'sequences': sequences,
        'residuals': residuals,
        'cos_sim_matrix': cos_sim_matrix,
        'mean_similarity': np.mean(cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]),
        'std_similarity': np.std(cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]),
    }

#%% Test original vs flipped sequences
print("\n=== Testing Original vs Flipped Sequences ===")
num_ones = 10  # Choose a representative case
if num_ones in all_results:
    original_sequences = all_results[num_ones]['sequences']
    original_residuals = all_results[num_ones]['residuals']
    
    # Create flipped sequences
    flipped_sequences = flip_batch(original_sequences)
    
    # Get residuals for flipped sequences
    with torch.inference_mode():
        flipped_logits, flipped_cache = model.run_with_cache(flipped_sequences)
    
    flipped_residuals = flipped_cache["resid_post", -1][:, -1, :]
    
    # Calculate cosine similarity between corresponding original and flipped residuals
    pairwise_similarity = torch.nn.functional.cosine_similarity(
        original_residuals, flipped_residuals, dim=1
    )
    
    # Plot histogram of similarities
    plt.figure(figsize=(10, 6))
    plt.hist(pairwise_similarity.cpu().numpy(), bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Cosine Similarity (Original vs Flipped)')
    plt.ylabel('Frequency') 
    plt.title(f'Distribution of Similarities: Original vs Flipped Sequences ({num_ones} ones)')
    plt.axvline(x=pairwise_similarity.mean().item(), color='r', linestyle='--', 
                label=f'Mean: {pairwise_similarity.mean().item():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Mean similarity (original vs flipped): {pairwise_similarity.mean().item():.4f}")
    print(f"Std similarity (original vs flipped): {pairwise_similarity.std().item():.4f}")

#%% Combined analysis: Original, Flipped, and Cross similarities
if num_ones in all_results:
    # Combine original and flipped residuals
    combined_residuals = torch.cat([original_residuals, flipped_residuals], dim=0)
    combined_residuals_np = combined_residuals.cpu().numpy()
    
    # Calculate full cosine similarity matrix
    combined_cos_sim_matrix = cosine_similarity(combined_residuals_np)
    
    # Plot combined similarity matrix
    num_original = original_residuals.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_cos_sim_matrix, cmap='viridis', aspect='auto', vmin=-1.0, vmax=1.0)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Combined Similarity Matrix: Original, Flipped, and Cross\n'
             f'Seq Length: {seq_length}, Num Ones: {num_ones}')
    plt.xlabel('Sequence Index (0 to N-1: Original, N to 2N-1: Flipped)')
    plt.ylabel('Sequence Index (0 to N-1: Original, N to 2N-1: Flipped)')
    
    # Add dividing lines
    line_pos = num_original - 0.5
    plt.axvline(x=line_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.axhline(y=line_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze cross-similarity block (original vs flipped)
    cross_similarity_block = combined_cos_sim_matrix[:num_original, num_original:]
    
    print(f"\nCross-similarity statistics (original vs flipped):")
    print(f"  Mean: {cross_similarity_block.mean():.4f}")
    print(f"  Std: {cross_similarity_block.std():.4f}")
    print(f"  Min: {cross_similarity_block.min():.4f}")
    print(f"  Max: {cross_similarity_block.max():.4f}")
