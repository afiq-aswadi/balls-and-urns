#%%
"""
Experiment 1: Basic Bayesian Updating

This experiment verifies that the coinformer model learns to perform Bayesian updating
by comparing its predictions against the theoretical Bayesian posterior.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from core.config import BAYESIAN_UPDATING_CONFIG
from core.training import train_coinformer_model, save_model_with_config
from core.samplers import generate_data_with_p_list
from core.plotting import (
    plot_probability_diff_surface, plot_probability_diff,
    plot_kl_divergence, plot_kl_divergence_surface,
    visualize_attention_patterns
)
from core.utils import get_log_loss

#%% Configuration
config = BAYESIAN_UPDATING_CONFIG
print(f"Model config: d_model={config.model_config.d_model}, d_head={config.model_config.d_head}")
print(f"Prior: α={config.alpha}, β={config.beta}")
print(f"Training: {config.num_epochs} epochs, {config.num_batches} batches")

#%% Train model
print("Training coinformer model...")
model, losses = train_coinformer_model(config)

#%% Generate test data
print("Generating test data...")
test_data, priors = generate_data_with_p_list(
    config.theta_values,
    batch_size=config.batch_size,
    seq_length=config.seq_length,
    num_batches=1,
    flip_batch=config.flip_batch,
    use_bos_token=config.model_config.use_bos_token
)

#%% Evaluate probability differences
print("Evaluating model predictions vs Bayesian posterior...")

plot_probability_diff_surface(
    theta_values=config.theta_values,
    model=model,
    seq_length=config.seq_length,
    batch_size=32,
    alpha0=config.alpha,
    beta0=config.beta,
    data_list=test_data
)

plot_probability_diff(
    theta=0.5,
    model=model,
    seq_length=config.seq_length,
    batch_size=32,
    alpha0=config.alpha,
    beta0=config.beta,
    norm='abs',
    data=test_data[4]
)

#%% KL divergence analysis
print("Analyzing KL divergence...")

plot_kl_divergence(
    theta=0.5,
    model=model,
    seq_length=config.seq_length,
    batch_size=32,
    alpha0=config.alpha,
    beta0=config.beta,
    data=test_data[4]
)

plot_kl_divergence_surface(
    theta_values=config.theta_values,
    model=model,
    seq_length=config.seq_length,
    batch_size=32,
    alpha0=config.alpha,
    beta0=config.beta,
    data_list=test_data
)

#%% Log loss comparison
print("Comparing log losses...")
trans_log_loss, bayes_log_loss = get_log_loss(
    model=model,
    seq_length=config.seq_length,
    batch_size=32,
    alpha0=config.alpha,
    beta0=config.beta,
    theta=0.5,
    test_data=test_data[5],
)

print(f"Transformer log loss: {trans_log_loss:.4f}")
print(f"Bayesian log loss: {bayes_log_loss:.4f}")
print(f"Ratio (should approach 1.0): {trans_log_loss/bayes_log_loss:.4f}")

#%% Visualize attention patterns
print("Visualizing attention patterns...")
visualize_attention_patterns(
    theta=0.5,
    model=model,
    seq_length=20
)

#%% Save model
save_path = save_model_with_config(model, config, "bayesian_updating")
print(f"Model saved to: {save_path}")

#%% Summary
print(f"""
=== Experiment 1 Summary ===
Model successfully trained for {config.num_epochs} epochs.
Final training loss: {losses[-1]:.4f}

Key Results:
- Transformer vs Bayesian log loss ratio: {trans_log_loss/bayes_log_loss:.4f}
- Model learns Bayesian updating: {'✓' if abs(trans_log_loss/bayes_log_loss - 1.0) < 0.1 else '✗'}

The model should show:
1. Probability predictions converging to Bayesian posterior means
2. KL divergence decreasing over sequence length
3. Log loss ratio approaching 1.0
""")