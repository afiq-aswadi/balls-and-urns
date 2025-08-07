#%%
"""
Experiment 4: Importance Sampling Training

This experiment compares standard training against importance sampling training,
where we train on a different distribution than our target evaluation distribution.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.config import IMPORTANCE_SAMPLING_CONFIG, BAYESIAN_UPDATING_CONFIG
from core.training import train_coinformer_model, save_model_with_config
from core.samplers import generate_data_with_p_list
from core.plotting import (
    plot_probability_diff_surface, plot_probability_diff,
    plot_kl_divergence, plot_kl_divergence_surface
)
from utils import get_log_loss

#%% Configuration
standard_config = BAYESIAN_UPDATING_CONFIG
importance_config = IMPORTANCE_SAMPLING_CONFIG

print("=== Importance Sampling Experiment ===")
print("Standard training:")
print(f"  Prior: α={standard_config.alpha}, β={standard_config.beta}")
print(f"  Training batches: {standard_config.num_batches}")

print("Importance sampling training:")
print(f"  Proposal: α={importance_config.alpha}, β={importance_config.beta}")
print(f"  Target: α={importance_config.importance_sampling_alpha}, β={importance_config.importance_sampling_beta}")
print(f"  Training batches: {importance_config.num_batches}")

#%% Train both models
print("\nTraining standard model...")
standard_model, standard_losses = train_coinformer_model(standard_config)
standard_save_path = save_model_with_config(standard_model, standard_config, "standard_training")

print("\nTraining importance sampling model...")
importance_model, importance_losses = train_coinformer_model(importance_config)
importance_save_path = save_model_with_config(importance_model, importance_config, "importance_sampling")

print(f"\nModels saved:")
print(f"  Standard: {standard_save_path}")
print(f"  Importance: {importance_save_path}")

#%% Compare training curves
plt.figure(figsize=(10, 6))
plt.plot(standard_losses, label=f'Standard (final: {standard_losses[-1]:.4f})', marker='o')
plt.plot(importance_losses, label=f'Importance Sampling (final: {importance_losses[-1]:.4f})', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison: Standard vs Importance Sampling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%% Generate test data for evaluation (use target distribution parameters)
print("\nGenerating test data for evaluation...")
target_alpha = importance_config.importance_sampling_alpha
target_beta = importance_config.importance_sampling_beta

# Use target distribution parameters for fair comparison
eval_theta_values = np.linspace(0.1, 0.9, 10)
test_data, priors = generate_data_with_p_list(
    eval_theta_values,
    batch_size=64,
    seq_length=importance_config.seq_length,
    num_batches=1,
    flip_batch=False,
    use_bos_token=importance_config.model_config.use_bos_token
)

#%% Evaluate both models on the target distribution
print("Evaluating models on target distribution...")

models = {
    'Standard': standard_model,
    'Importance Sampling': importance_model
}

evaluation_results = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name} model...")
    
    # Calculate log loss on target distribution
    trans_log_loss, bayes_log_loss = get_log_loss(
        model=model,
        seq_length=importance_config.seq_length,
        batch_size=32,
        alpha0=target_alpha,
        beta0=target_beta,
        theta=0.5,
        test_data=test_data[4],
    )
    
    evaluation_results[model_name] = {
        'trans_log_loss': trans_log_loss,
        'bayes_log_loss': bayes_log_loss,
        'log_loss_ratio': trans_log_loss / bayes_log_loss,
        'final_training_loss': importance_losses[-1] if model_name == 'Importance Sampling' else standard_losses[-1]
    }
    
    print(f"  Transformer log loss: {trans_log_loss:.4f}")
    print(f"  Bayesian log loss: {bayes_log_loss:.4f}")
    print(f"  Ratio: {trans_log_loss / bayes_log_loss:.4f}")

#%% Compare probability differences on target distribution
print("\nComparing probability differences...")

for model_name, model in models.items():
    print(f"Plotting probability differences for {model_name}...")
    
    plot_probability_diff_surface(
        theta_values=eval_theta_values,
        model=model,
        seq_length=importance_config.seq_length,
        batch_size=32,
        alpha0=target_alpha,
        beta0=target_beta,
        data_list=test_data
    )
    
    plot_probability_diff(
        theta=0.5,
        model=model,
        seq_length=importance_config.seq_length,
        batch_size=32,
        alpha0=target_alpha,
        beta0=target_beta,
        norm='abs',
        data=test_data[4]
    )

#%% Compare KL divergences
print("\nComparing KL divergences...")

kl_results = {}

for model_name, model in models.items():
    print(f"Calculating KL divergence for {model_name}...")
    
    plot_kl_divergence(
        theta=0.5,
        model=model,
        seq_length=importance_config.seq_length,
        batch_size=32,
        alpha0=target_alpha,
        beta0=target_beta,
        data=test_data[4]
    )
    
    # Store KL divergence for comparison (we'd need to modify the function to return values)
    # For now, just plot the surfaces
    plot_kl_divergence_surface(
        theta_values=eval_theta_values,
        model=model,
        seq_length=importance_config.seq_length,
        batch_size=32,
        alpha0=target_alpha,
        beta0=target_beta,
        data_list=test_data
    )

#%% Detailed performance comparison
print(f"\n=== Performance Comparison ===")
print(f"Target distribution: α={target_alpha}, β={target_beta}")
print()

comparison_metrics = ['trans_log_loss', 'log_loss_ratio', 'final_training_loss']
for metric in comparison_metrics:
    print(f"{metric}:")
    for model_name, results in evaluation_results.items():
        print(f"  {model_name:20s}: {results[metric]:.4f}")
    print()

# Determine which method is better
standard_ratio = evaluation_results['Standard']['log_loss_ratio']
importance_ratio = evaluation_results['Importance Sampling']['log_loss_ratio']

improvement = (standard_ratio - importance_ratio) / standard_ratio * 100
if improvement > 0:
    print(f"Importance sampling improves performance by {improvement:.1f}%")
else:
    print(f"Standard training performs {-improvement:.1f}% better than importance sampling")

#%% Distribution analysis
print(f"\n=== Distribution Analysis ===")
print("Proposal distribution (what importance sampling trains on):")
print(f"  α={importance_config.alpha}, β={importance_config.beta}")
print(f"  Mean: {importance_config.alpha / (importance_config.alpha + importance_config.beta):.3f}")

print("Target distribution (what we evaluate on):")
print(f"  α={target_alpha}, β={target_beta}")
print(f"  Mean: {target_alpha / (target_alpha + target_beta):.3f}")

# Plot the two distributions for visualization
from scipy.stats import beta as beta_dist

x = np.linspace(0, 1, 1000)
proposal_pdf = beta_dist.pdf(x, importance_config.alpha, importance_config.beta)
target_pdf = beta_dist.pdf(x, target_alpha, target_beta)

plt.figure(figsize=(10, 6))
plt.plot(x, proposal_pdf, label=f'Proposal: β({importance_config.alpha}, {importance_config.beta})', linewidth=2)
plt.plot(x, target_pdf, label=f'Target: β({target_alpha}, {target_beta})', linewidth=2)
plt.xlabel('Probability θ')
plt.ylabel('Density')
plt.title('Proposal vs Target Distributions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%% Summary
print(f"""
=== Importance Sampling Experiment Summary ===

Training Setup:
- Standard model trained on β({standard_config.alpha}, {standard_config.beta})
- Importance sampling model trained on β({importance_config.alpha}, {importance_config.beta}) 
  but targeting β({target_alpha}, {target_beta})

Evaluation Results (on target distribution):
- Standard model log loss ratio: {standard_ratio:.4f}
- Importance sampling log loss ratio: {importance_ratio:.4f}
- Performance change: {improvement:+.1f}%

Key Insights:
- Importance sampling {'helps' if improvement > 5 else 'has minimal effect on' if abs(improvement) < 5 else 'hurts'} performance on the target distribution
- Training losses: Standard={standard_losses[-1]:.4f}, Importance={importance_losses[-1]:.4f}
- Both models should converge to Bayesian updating, but importance sampling allows 
  training on easier/more common cases while still learning the target behavior

This technique is valuable when:
1. Target distribution is hard to sample from
2. We want to focus learning on specific regions
3. Training data is limited or expensive
""")