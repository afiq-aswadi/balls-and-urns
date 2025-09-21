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
import matplotlib.pyplot as plt
from core.config import BAYESIAN_UPDATING_CONFIG
from core.training import train_coinformer_model, save_model_with_config
from core.samplers import generate_data_with_p_list, generate_sequential_ones
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

#%%
data = generate_sequential_ones(10, use_bos_token=False)

#%% Symmetry check on constant sequences
print("Checking model symmetry on all-zeros and all-ones sequences...")

model.eval()
device = next(model.parameters()).device

all_ones_seq = torch.ones((1, config.seq_length), dtype=torch.long, device=device)
all_zeros_seq = torch.zeros_like(all_ones_seq)

with torch.inference_mode():
    logits_all_ones = model(all_ones_seq)
    logits_all_zeros = model(all_zeros_seq)

probs_all_ones = torch.softmax(logits_all_ones, dim=-1)[0, 1:, 1].cpu().numpy()
probs_all_zeros = torch.softmax(logits_all_zeros, dim=-1)[0, 1:, 1].cpu().numpy()
positions = np.arange(1, config.seq_length)

plt.figure(figsize=(8, 4))
plt.plot(positions, probs_all_ones, label='All ones sequence', linewidth=2)
plt.plot(positions, probs_all_zeros, label='All zeros sequence', linewidth=2, linestyle='--')
plt.xlabel('Sequence Position')
plt.ylabel('P(next token = 1)')
plt.title('Model predictions on constant sequences')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


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
# %%
