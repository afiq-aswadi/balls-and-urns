#%%
import torch
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
import matplotlib.pyplot as plt

from train import train_coinformer_model
from model import coinformer_model_config, coinformer_model_attn_only_config
import plot_utils as pu
from utils import calculate_posterior_mean, get_log_loss, get_kl_divergence, get_log_resids_from_sequential_ones, get_theoretical_log, flip_batch
from samplers import generate_data_with_p, generate_data_with_p_list, generate_sequential_ones, generate_all_binary_sequences_with_fixed_num_ones
import os
from sklearn.metrics.pairwise import cosine_similarity

alpha = 1.0
beta = 1.0
n_layers = 2
attn_only = False
load_model = False

#%%

uniform_transformer_config = HookedTransformerConfig(
    d_model=128,  # embedding dimension
    d_head=128,
    n_layers=n_layers,
    n_ctx=100,
    d_vocab=2,
    act_fn="relu",
    default_prepend_bos=False,
    normalization_type=None,
    attn_only=attn_only,
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
        flip_batch=False,
        scale=1.0,
        bias=0.0,
        importance_sampling=False,
    )

#%%
thetas = np.linspace(0, 0.9, 10)
test_data, priors = generate_data_with_p_list(thetas,
    batch_size=64,
    seq_length=100,
    num_batches=1,
    flip_batch=False
)

#%%

pu.plot_probability_diff_surface(
    theta_values=thetas,
    model=uniform_transformer,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data_list=test_data
)

pu.plot_probability_diff(
    theta=0.5,
    model=uniform_transformer,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    norm='abs',
    data=test_data[4]
)

# %%
pu.plot_kl_divergence(
    theta=0.5,
    model=uniform_transformer,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data=test_data[4]
)

pu.plot_kl_divergence_surface(
    theta_values=thetas,
    model=uniform_transformer,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data_list=test_data
)

# #%%
# trans_log_loss, bayes_log_loss = get_log_loss(
#     model=uniform_transformer,
#     seq_length=100,
#     batch_size=32,
#     alpha0=alpha,
#     beta0=beta,
#     theta=0.5,
#     test_data=test_data[5],
# )

# print(f"Transformer log loss: {trans_log_loss}")
# print(f"Bayesian log loss: {bayes_log_loss}")
# %%
pu.visualize_attention_patterns(
    theta=0.5,
    model=uniform_transformer,
    seq_length=20
)

#%%
seq_length = 100

ith_logits, ith_residuals = get_log_resids_from_sequential_ones(uniform_transformer, seq_length)
theoretical_log = get_theoretical_log(seq_length)    
log_odds = ith_logits[:,1] - ith_logits[:,0]

pu.plot_log_odds_vs_theoretical_log(log_odds, theoretical_log)

#%%
seq_length = 30
num_ones = 20

all_binary_sequences = generate_all_binary_sequences_with_fixed_num_ones(
    n = seq_length,
    num_ones = num_ones,
    max_n_sequences= 1000
)

with torch.inference_mode():
    logits, cache = uniform_transformer.run_with_cache(all_binary_sequences)

resids = cache["resid_post", -1][:, -1, :]

cos_sim_matrix = pu.plot_residual_cosine_similarity(seq_length, num_ones, resids)



#%%
# Get the sorting order based on the first column (similarity with the first sequence)
# We want to sort in descending order, so we negate the first column for argsort

# If you want to sort by the similarity of each sequence to a *specific* sequence (e.g., the first one in the *original* order)
# and then reorder the columns based on that:
first_col_similarities = cos_sim_matrix[:, 0]
# Get indices that would sort the first column in descending order
sorted_indices_by_first_col = np.argsort(-first_col_similarities)

# Reorder the columns of the original matrix based on these sorted indices
cos_sim_matrix_cols_sorted_by_first = cos_sim_matrix[:, sorted_indices_by_first_col]

# To make the plot more interpretable, you might also want to reorder the rows
# so that the sequence that was originally first (and used for sorting) is still at the top.
# Or, more consistently, reorder rows by the same logic as columns.
cos_sim_matrix_reordered_symmetrically = cos_sim_matrix[sorted_indices_by_first_col, :]
cos_sim_matrix_reordered_symmetrically = cos_sim_matrix_reordered_symmetrically[:, sorted_indices_by_first_col]


print("Plotting cosine similarity matrix with columns sorted by similarity to the original first sequence")
# Create a new figure for the reordered plot
plt.figure(figsize=(8, 6))
plt.imshow(cos_sim_matrix_cols_sorted_by_first, cmap='viridis', aspect='auto')
plt.colorbar(label='Cosine Similarity')
plt.title("Cosine Similarity (Columns sorted by similarity to original first sequence)")
plt.xlabel("Sequence Index (Reordered)")
plt.ylabel("Sequence Index (Original Order)")
plt.tight_layout()
plt.show()




# %%
flipped_binary_sequences = flip_batch(all_binary_sequences)
with torch.inference_mode():
    flipped_logits, flipped_cache = uniform_transformer.run_with_cache(flipped_binary_sequences)
flipped_resids = flipped_cache["resid_post", -1][:, -1, :]


cosine_sim_mat_flipped = pu.plot_residual_cosine_similarity(seq_length, num_ones, flipped_resids, print_stats=False)


# Calculate cosine similarity between corresponding residuals
cosine_sim_flipped_vs_original = torch.nn.functional.cosine_similarity(resids, flipped_resids, dim=1)

# Plot a histogram of these cosine similarities
plt.figure(figsize=(8, 6))
plt.hist(cosine_sim_flipped_vs_original.cpu().numpy(), bins=20, edgecolor='black')
plt.title(f"Distribution of Cosine Similarities (Original vs Flipped Residuals)\nSeq Length: {seq_length}, Num Ones: {num_ones}")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%%
# Combine original and flipped residuals
combined_resids = torch.cat((resids, flipped_resids), dim=0)

# Calculate cosine similarity for the combined set
# Ensure data is on CPU and is a NumPy array for sklearn's cosine_similarity
if isinstance(combined_resids, torch.Tensor):
    combined_resids_np = combined_resids.cpu().numpy()
else:
    combined_resids_np = combined_resids # Assuming it's already a NumPy array if not a Tensor

combined_cos_sim_matrix = cosine_similarity(combined_resids_np)

# Plotting
num_original_seqs = resids.shape[0]
# num_total_seqs = combined_resids.shape[0] # This is 2 * num_original_seqs

plt.figure(figsize=(10, 8))
plt.imshow(combined_cos_sim_matrix, cmap='viridis', aspect='auto', vmin=-1.0, vmax=1.0)
plt.colorbar(label='Cosine Similarity')
title_str = (f"Cosine Similarity: Original, Flipped, and Cross Residuals\n"
             f"Seq Length: {seq_length}, Num Ones: {num_ones}\n"
             f"{num_original_seqs} original sequences, {num_original_seqs} flipped sequences")
plt.title(title_str)
plt.xlabel("Sequence Index (0 to N-1: Original, N to 2N-1: Flipped)")
plt.ylabel("Sequence Index (0 to N-1: Original, N to 2N-1: Flipped)")

# Add lines to separate original and flipped sections
# Line position is after the last original sequence, before the first flipped one
line_pos = num_original_seqs - 0.5
plt.axvline(x=line_pos, color='red', linestyle='--', linewidth=2)
plt.axhline(y=line_pos, color='red', linestyle='--', linewidth=2)

# Optional: more descriptive ticks if needed, though default might be okay
# ticks = [0, num_original_seqs - 1, num_original_seqs, 2 * num_original_seqs - 1]
# tick_labels = ['0 (Orig)', f'{num_original_seqs-1} (Orig)', f'{num_original_seqs} (Flip)', f'{2*num_original_seqs-1} (Flip)']
# plt.xticks(ticks, tick_labels, rotation=45, ha="right")
# plt.yticks(ticks, tick_labels)

plt.tight_layout()
plt.show()

# Print stats for the cross-similarity block (original vs. flipped)
# This is the top-right block of the combined_cos_sim_matrix
cross_similarity_block = combined_cos_sim_matrix[:num_original_seqs, num_original_seqs:]
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


#%%
# Plot just the cross-similarity block (original vs. flipped)
plt.figure(figsize=(9, 7))
plt.imshow(cross_similarity_block, cmap='coolwarm', aspect='auto', vmin=-1.0, vmax=1.0)
plt.colorbar(label='Cosine Similarity')
title_str_cross = (f"Cosine Similarity: Original vs. Flipped Residuals\n"
                   f"Seq Length: {seq_length}, Num Ones: {num_ones}")
plt.title(title_str_cross)
plt.xlabel("Flipped Sequence Index")
plt.ylabel("Original Sequence Index")
plt.tight_layout()
plt.show()

# %%
# Create directory for saved models if it doesn't exist
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Save the model's state dictionary
config = uniform_transformer.cfg
save_path = os.path.join(
    save_dir,
    f"uniform_coinformer_dmodel{config.d_model}_dhead{config.d_head}_layers{config.n_layers}_attnonly{config.attn_only}_alpha{alpha}_beta{beta}.pt"
)
torch.save(uniform_transformer.state_dict(), save_path)

print(f"Model saved to {save_path}")
# %%
