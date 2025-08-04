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
from utils import get_log_resids_from_sequential_ones, flip_batch, get_residual_cosine_similarity, get_log_resids_from_sequential_zeros

#%%
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

def theoretical_log_odds(sequence_length, num_ones, alpha=1.0, beta=1.0):
    """
    Calculate the theoretical log odds for a sequence of a given length with a fixed number of ones.
    
    Args:
        sequence_length: Length of the binary sequence
        num_ones: Number of ones in the sequence
        alpha: Parameter for the beta distribution (prior for ones)
        beta: Parameter for the beta distribution (prior for zeros)
    
    Returns:
        Theoretical log odds value.
    """
    numerator = (num_ones + alpha)
    denominator = (sequence_length - num_ones + beta)
    return numerator, denominator, np.log(numerator / denominator)

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
        flip_batch=True,
        scale=1.0,
        bias=0.0,
        importance_sampling=False,
    )

#%%
ith_logits_one, ith_resids_one = get_log_resids_from_sequential_ones(uniform_transformer, 100) #### index off by two: zeroth logit represents log(2)
ith_logits_zero, ith_resids_zero = get_log_resids_from_sequential_zeros(uniform_transformer, 100)

#%%
sequence_length = 20
num_ones = 5
numerator, denominator, theoretical_log_odds_value = theoretical_log_odds(sequence_length, num_ones, alpha, beta)
sequences = generate_all_binary_sequences_with_fixed_num_ones(sequence_length, num_ones, max_n_sequences=1000)
sequences_end_ones, sequences_end_zeros = split_sequence_by_index_value(sequences, sequence_length - 1, 1)


logits, cache = uniform_transformer.run_with_cache(sequences_end_ones)
ones_resids = cache["resid_post",-1][:, -1, :]
#%%

pu.plot_residual_cosine_similarity(
    seq_length=sequence_length,
    num_ones=num_ones,
    resids=ones_resids,
)

mean_resid = torch.mean(ones_resids, dim=0)



# %%
def get_resid_vector_from_ones_and_zeros_resids(ones_resids, zeros_resids, numerator, denominator):
    """
    Calculate the residual vector for a sequence of ones and zeros.
    
    Args:
        ones_resids: Residuals for sequences ending with ones
        zeros_resids: Residuals for sequences ending with zeros
        numerator: Numerator for the theoretical log odds
        denominator: Denominator for the theoretical log odds
    
    Returns:
        Residual vector for the sequence.
    """
    return ones_resids[int(numerator + 2)].cpu() + zeros_resids[int(denominator + 2)].cpu()


def get_resid_vector_from_ones_resids(ones_resids, numerator, denominator):
    """
    Calculate the residual vector for sequences ending with ones.
    
    Args:
        ones_resids: Residuals for sequences ending with ones
        numerator: Numerator for the theoretical log odds
        denominator: Denominator for the theoretical log odds
    
    Returns:
        Residual vector for the sequence.
    """
    return ones_resids[int(numerator + 2)].cpu() - ones_resids[int(denominator + 2)].cpu()


ones_and_zeros_resids = get_resid_vector_from_ones_and_zeros_resids(
    ones_resids=ith_resids_one,
    zeros_resids=ith_resids_zero,
    numerator=numerator,
    denominator=denominator
)

ones_resids_only = get_resid_vector_from_ones_resids(
    ones_resids=ith_resids_one,
    numerator=numerator,
    denominator=denominator
)

#%%
print(cosine_similarity(mean_resid.unsqueeze(0).cpu(), ones_and_zeros_resids.unsqueeze(0).cpu()))
print(cosine_similarity(mean_resid.unsqueeze(0).cpu(), ones_resids_only.unsqueeze(0).cpu()))
# %%
torch.softmax(ith_logits_one.cpu(), dim = -1) + torch.softmax(ith_logits_zero.cpu(), dim = -1)
# %%
