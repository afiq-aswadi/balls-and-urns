#%%
from model import coinformer_model_attn_only_config, coinformer_model_config    
from plot_utils import visualize_attention_patterns
from samplers import generate_data_with_p

from transformer_lens import HookedTransformer
import torch
import itertools
import numpy as np
import einops
from torch.nn.functional import softmax, log_softmax
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

#%%

model = HookedTransformer(coinformer_model_attn_only_config)

# Load the weights from the PT file
weights_path = "saved_models/uniform_coinformer_alpha1.0_beta1.0.pt"
model.load_state_dict(torch.load(weights_path))

# Verify the model loaded correctly
print("Model weights loaded successfully.")
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
    return torch.tril(torch.ones((n, n), dtype=torch.long)).unsqueeze(0)


test_seq = generate_all_binary_sequences_with_fixed_num_ones(n, num_ones)


#%%
# after this the posterior should be: 11/21....... so the logit should be log of that plus some constant.
# will the constant be the same for all sequences? 
# logit for 1 is log(11)+c for some c... same for logit for 2.
def theoretical_logit(n, alpha, beta):
    """
    Returns a tensor of shape (n+1, 2), where each row corresponds to a different number of ones (from 0 to n).
    The first column is the logit for class 0, the second column is the logit for class 1.
    """
    num_ones = torch.arange(0, n + 1)
    theo_0 = torch.log((n - num_ones + beta).float()).unsqueeze(1)
    theo_1 = torch.log((num_ones + alpha).float()).unsqueeze(1)
    return torch.cat([theo_0, theo_1], dim=1)

theoretical_logits = theoretical_logit(10, 1, 1)

#%%
logits, cache = model.run_with_cache(test_seq)

#%%
last_logits = logits[:, -1, :]

#%%# %%
# idea: reverse engineer the way model takes the logarithm.
# to get the logit we take the dot product of the residual and unembedding
# since this is a single layer attn only transformer, the attention head performs all the computation.
# so technically it should really be the softmaxing that computes the logarithm 
# the operation is just... your embedding + pos embedding then put that into attention. the output of that dotted with the unembed gives the logit
# %%
# List all keys available in the cache
print("Available keys in cache:")
for key in cache.keys():
    print(key)

# %%
embed = model.embed.W_E
W_U_diff = model.unembed.W_U[:,0] - model.unembed.W_U[:,1]
resid_pre_unembeds = einops.einsum(embed, W_U_diff, "i j, j ->i")
print(resid_pre_unembeds)
# %%
input_seq = generate_sequential_ones(10)
print(input_seq.shape)

#%%
logits, cache = model.run_with_cache(input_seq[0])
#%%
print("Available keys in cache:")
for key in cache.keys():
    print(key)
# %%
resid_n = cache["resid_post",0][:,-1,:]
print(resid_n.shape)
# %%
log_odds = einops.einsum(resid_n, W_U_diff, "i j, j ->i")
print(log_odds)

# in theory, log odds = log((1+ H)/ (N- H + 1))

#%%
logits_last = logits[:, -1, :]
logits_last[:,1] - logits_last[:,0]
# %%
## log odds = logit(1) - logit(0) = log(1+H) - log(N-H+1)
normalized_logits = log_softmax(logits_last, dim=-1) 
print(normalized_logits.shape)

# %%
normalized_logits[:,1] - normalized_logits[:,0]
# %%
theoretical_logit(1,10,1,1)[1] - theoretical_logit(1,10,1,1)[0]

# %%
theoretical_log_odds = theoretical_logits[:,1] - theoretical_logits[:,0]
print(theoretical_log_odds)

# %%
pos_embed = model.pos_embed.W_pos
pos_embed_norms = torch.norm(pos_embed, dim=-1)
max_norm = pos_embed_norms.max()
print(f"Max value of the norm of the positional embedding: {max_norm.item()}")
# %%
embed_norms = torch.norm(embed, dim=-1)
print(f"Norms of the token embeddings: {embed_norms}")

print(f"Norms of the positional embeddings: {pos_embed_norms}")
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(pos_embed_norms.detach().cpu().numpy(), marker='o')
plt.title("Norms of Positional Embeddings")
plt.xlabel("Position")
plt.ylabel("Norm")
plt.grid(True)
plt.show()
# %%
# Compute cosine similarity between each positional embedding and the two token embeddings

# pos_embed: (n_positions, d_model)
# embed: (2, d_model) since there are only two embeddings (for 0 and 1)
cos_sims = cosine_similarity(
    pos_embed.unsqueeze(1),  # (n_positions, 1, d_model)
    embed.unsqueeze(0),      # (1, 2, d_model)
    dim=-1
)  # (n_positions, 2)

plt.figure(figsize=(10, 6))
im = plt.imshow(cos_sims.detach().cpu().numpy(), aspect='auto', cmap='viridis')
plt.colorbar(im, label='Cosine Similarity')
plt.xlabel('Token (0 or 1)')
plt.ylabel('Position Index')
plt.title('Cosine Similarity: Positional Embeddings vs Token Embeddings')
plt.xticks([0, 1], ['0', '1'])
plt.show()
# %%
seq_of_five_ones = generate_all_binary_sequences_with_fixed_num_ones(10, 5)

logits_five_ones, cache_five_ones = model.run_with_cache(seq_of_five_ones)

# Compute the residuals at the last position for all sequences with five ones
resid_last = cache_five_ones["resid_post", 1][:, -1, :]  # shape: (num_sequences, d_model)

# Compute pairwise cosine similarities between all residual vectors

resid_last_np = resid_last.detach().cpu().numpy()
pairwise_cos_sim = sklearn_cosine_similarity(resid_last_np)

plt.figure(figsize=(8, 6))
plt.imshow(pairwise_cos_sim, aspect='auto', cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.title('Pairwise Cosine Similarity of Last Residuals (5 ones)')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Index')
plt.show()

# Print mean and min/max cosine similarity to show similarity
mean_cos_sim = np.mean(pairwise_cos_sim[np.triu_indices_from(pairwise_cos_sim, k=1)])
min_cos_sim = np.min(pairwise_cos_sim[np.triu_indices_from(pairwise_cos_sim, k=1)])
max_cos_sim = np.max(pairwise_cos_sim[np.triu_indices_from(pairwise_cos_sim, k=1)])
print(f"Mean pairwise cosine similarity: {mean_cos_sim:.4f}")
print(f"Min pairwise cosine similarity: {min_cos_sim:.4f}")
print(f"Max pairwise cosine similarity: {max_cos_sim:.4f}")
# %%
