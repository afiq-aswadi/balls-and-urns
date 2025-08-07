#%%
import torch
import numpy as np
from transformer_lens import HookedTransformer

from train import train_coinformer_model
from model import coinformer_model_config, coinformer_model_attn_only_config
import plot_utils as pu
from utils import calculate_posterior_mean, get_log_loss, get_kl_divergence
from samplers import generate_data_with_p, generate_data_with_p_list
import os

alpha = 1.0
beta = 8.0
#%%
# uniform_transformer = HookedTransformer(coinformer_model_config)
uniform_transformer_no_importance = HookedTransformer(coinformer_model_config)
uniform_transformer_importance = HookedTransformer(coinformer_model_config)

losses_no_importance = train_coinformer_model(
    model=uniform_transformer_no_importance,
    num_epochs=10,
    learning_rate=0.001,
    batch_size=64,
    seq_length=100,
    num_batches=10000,
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

losses_importance = train_coinformer_model(
    model=uniform_transformer_importance,
    num_epochs=10,
    learning_rate=0.001,
    batch_size=64,
    seq_length=100,
    num_batches=10000,
    alpha_param=1,
    beta_param=1,
    bernoulli=False,  # Set to False for uniform distribution
    bernoulli_p=None,  # No need for this parameter
    pos_embed=True,  # Activate positional embedding
    flip_batch=False,
    scale=1.0,
    bias=0.0,
    importance_sampling=True,
    importance_sampling_alpha=alpha,
    importance_sampling_beta=beta,
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
    model=uniform_transformer_no_importance,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data_list=test_data
)

pu.plot_probability_diff(
    theta=0.5,
    model=uniform_transformer_no_importance,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    norm='abs',
    data=test_data[4]
)

#%%

pu.plot_probability_diff_surface(
    theta_values=thetas,
    model=uniform_transformer_importance,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data_list=test_data
)

pu.plot_probability_diff(
    theta=0.5,
    model=uniform_transformer_importance,
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
    model=uniform_transformer_no_importance,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data=test_data[4]
)

pu.plot_kl_divergence_surface(
    theta_values=thetas,
    model=uniform_transformer_no_importance,
    seq_length=20,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data_list=test_data
)

#%%
pu.plot_kl_divergence(
    theta=0.5,
    model=uniform_transformer_importance,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data=test_data[4]
)

pu.plot_kl_divergence_surface(
    theta_values=thetas,
    model=uniform_transformer_importance,
    seq_length=20,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    data_list=test_data
)



#%%
trans_log_loss, bayes_log_loss = get_log_loss(
    model=uniform_transformer_no_importance,
    seq_length=100,
    batch_size=32,
    alpha0=alpha,
    beta0=beta,
    theta=0.5,
    test_data=test_data[5],
)

print(f"Transformer log loss: {trans_log_loss}")
print(f"Bayesian log loss: {bayes_log_loss}")
# %%
pu.visualize_attention_patterns(
    theta=0.5,
    model=uniform_transformer,
    seq_length=20
)
# %%
# Create directory for saved models if it doesn't exist
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Save the model's state dictionary
save_path = os.path.join(save_dir, f"uniform_coinformer_alpha{alpha}_beta{beta}.pt")
torch.save(uniform_transformer.state_dict(), save_path)

print(f"Model saved to {save_path}")


# %%
