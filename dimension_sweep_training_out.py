"""
Dimension Sweep Training Script

This script systematically trains coinformer models across different architectural
dimensions to understand the relationship between model capacity and performance
on Bayesian updating tasks.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from itertools import product
from tqdm import tqdm

from core.config import ExperimentConfig
from core.models import ModelConfig
from core.training import train_coinformer_model, save_model_with_config
from core.samplers import generate_data_with_p_list
from core.utils import get_log_loss


def create_dimension_configs(
    d_model_values=[32, 64, 128, 256],
    d_head_values=[16, 32, 64, 128],
    d_mlp_values=None,
    base_config=None
):
    """
    Create a list of experiment configurations for dimension sweep.
    
    Args:
        d_model_values: List of d_model values to test
        d_head_values: List of d_head values to test  
        d_mlp_values: List of d_mlp values to test. If None, defaults to [2*d_model, 4*d_model] for each d_model
        base_config: Base configuration to modify
        
    Returns:
        List of (config_name, ExperimentConfig) tuples
    """
    if base_config is None:
        base_config = ExperimentConfig(
            alpha=1.0,
            beta=1.0,
            num_epochs=5,
            num_batches=1000,
            seq_length=50,
            learning_rate=0.001,
            batch_size=64
        )
    
    configs = []
    
    for d_model, d_head in product(d_model_values, d_head_values):
        # Skip invalid configurations (d_head should typically be <= d_model)
        if d_head > d_model:
            continue
        
        # Set d_mlp values for this d_model if not provided
        if d_mlp_values is None:
            current_d_mlp_values = [2 * d_model, 4 * d_model]
        else:
            current_d_mlp_values = d_mlp_values
            
        for d_mlp in current_d_mlp_values:
            # Create model config with fixed n_layers=1
            model_config = ModelConfig(
                d_model=d_model,
                d_head=d_head,
                d_mlp=d_mlp,
                n_layers=1  # Fixed to 1 layer
            )
            
            # Create experiment config
            config = ExperimentConfig(
                model_config=model_config,
                alpha=base_config.alpha,
                beta=base_config.beta,
                num_epochs=base_config.num_epochs,
                num_batches=base_config.num_batches,
                seq_length=base_config.seq_length,
                learning_rate=base_config.learning_rate,
                batch_size=base_config.batch_size
            )
            
            config_name = f"d{d_model}_h{d_head}_mlp{d_mlp}"
            configs.append((config_name, config))
    
    return configs


def run_dimension_sweep(
    configs,
    results_dir="dimension_sweep_results",
    evaluate_models=True,
    save_models=True
):
    """
    Run training for all dimension configurations.
    
    Args:
        configs: List of (config_name, ExperimentConfig) tuples
        results_dir: Directory to save results
        evaluate_models: Whether to evaluate trained models
        save_models: Whether to save trained models
        
    Returns:
        Dictionary with training results
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_results_dir = f"{results_dir}_{timestamp}"
    os.makedirs(full_results_dir, exist_ok=True)
    
    # Initialize results tracking
    results = {
        'configs': {},
        'training_losses': {},
        'models': {},
        'evaluations': {},
        'metadata': {
            'timestamp': timestamp,
            'total_configs': len(configs),
            'results_dir': full_results_dir
        }
    }
    
    print(f"=== Dimension Sweep Training ===")
    print(f"Training {len(configs)} configurations...")
    print(f"Results will be saved to: {full_results_dir}")
    
    # Train each configuration
    for i, (config_name, config) in enumerate(tqdm(configs, desc="Training models")):
        print(f"\n[{i+1}/{len(configs)}] Training {config_name}...")
        print(f"  d_model={config.model_config.d_model}, d_head={config.model_config.d_head}, n_layers={config.model_config.n_layers}")
        
        try:
            # Train model
            model, losses = train_coinformer_model(config, verbose=False)
            
            # Store results
            results['configs'][config_name] = config
            results['training_losses'][config_name] = losses
            results['models'][config_name] = model
            
            print(f"  Final training loss: {losses[-1]:.4f}")
            
            # Save model if requested
            if save_models:
                model_dir = os.path.join(full_results_dir, "models")
                save_path = save_model_with_config(model, config, f"sweep_{config_name}", model_dir)
                print(f"  Saved model to: {save_path}")
            
        except Exception as e:
            print(f"  ERROR training {config_name}: {e}")
            results['training_losses'][config_name] = None
            results['models'][config_name] = None
            continue
    
    # Evaluate models if requested
    if evaluate_models:
        print("\n=== Evaluating Models ===")
        evaluate_dimension_sweep(results, full_results_dir)
    
    # Save results summary
    save_results_summary(results, full_results_dir)
    
    return results


def evaluate_dimension_sweep(results, results_dir):
    """Evaluate all trained models and compute performance metrics."""
    # Generate test data
    test_config = ExperimentConfig()  # Use default config for test data
    test_data, priors = generate_data_with_p_list(
        test_config.theta_values,
        batch_size=test_config.batch_size,
        seq_length=test_config.seq_length,
        num_batches=1,
        flip_batch=test_config.flip_batch,
        use_bos_token=test_config.model_config.use_bos_token
    )
    
    for config_name, model in results['models'].items():
        if model is None:
            continue
            
        try:
            print(f"Evaluating {config_name}...")
            config = results['configs'][config_name]
            
            # Calculate log loss
            trans_log_loss, bayes_log_loss = get_log_loss(
                model=model,
                seq_length=test_config.seq_length,
                batch_size=32,
                alpha0=test_config.alpha,
                beta0=test_config.beta,
                theta=0.5,
                test_data=test_data[5],
            )
            
            # Calculate model parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            # Store evaluation results
            results['evaluations'][config_name] = {
                'd_model': config.model_config.d_model,
                'd_head': config.model_config.d_head,
                'd_mlp': config.model_config.d_mlp,
                'n_layers': config.model_config.n_layers,
                'param_count': param_count,
                'final_training_loss': results['training_losses'][config_name][-1],
                'trans_log_loss': trans_log_loss,
                'bayes_log_loss': bayes_log_loss,
                'log_loss_ratio': trans_log_loss / bayes_log_loss,
            }
            
        except Exception as e:
            print(f"  ERROR evaluating {config_name}: {e}")
            continue


def save_results_summary(results, results_dir):
    """Save results summary to files."""
    # Create summary DataFrame
    eval_data = []
    for config_name, eval_result in results['evaluations'].items():
        eval_data.append({
            'config_name': config_name,
            **eval_result
        })
    
    if eval_data:
        df = pd.DataFrame(eval_data)
        
        # Save CSV
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved evaluation results to: {csv_path}")
        
        # Create summary plots
        create_summary_plots(df, results_dir)
    
    # Save training losses
    losses_path = os.path.join(results_dir, "training_losses.json")
    with open(losses_path, 'w') as f:
        # Convert to serializable format
        serializable_losses = {k: v for k, v in results['training_losses'].items() if v is not None}
        json.dump(serializable_losses, f, indent=2)
    
    # Save metadata
    metadata_path = os.path.join(results_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(results['metadata'], f, indent=2)


def create_summary_plots(df, results_dir):
    """Create summary plots of the dimension sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance vs Parameters
    ax = axes[0, 0]
    scatter = ax.scatter(df['param_count'], df['log_loss_ratio'], 
                        c=df['d_model'], s=60, alpha=0.7, cmap='viridis')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Log Loss Ratio (Trans/Bayes)')
    ax.set_title('Performance vs Model Size')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Bayesian')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='d_model')
    
    # 2. d_model vs Performance  
    ax = axes[0, 1]
    for d_mlp in sorted(df['d_mlp'].unique()):
        subset = df[df['d_mlp'] == d_mlp]
        ax.plot(subset['d_model'], subset['log_loss_ratio'], 
               marker='o', label=f'd_mlp={d_mlp}', alpha=0.7)
    ax.set_xlabel('d_model')
    ax.set_ylabel('Log Loss Ratio')
    ax.set_title('d_model vs Performance by d_mlp')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. d_mlp vs Performance
    ax = axes[1, 0]
    for d_model in sorted(df['d_model'].unique()):
        subset = df[df['d_model'] == d_model]
        ax.plot(subset['d_mlp'], subset['log_loss_ratio'], 
               marker='s', label=f'd_model={d_model}', alpha=0.7)
    ax.set_xlabel('d_mlp')
    ax.set_ylabel('Log Loss Ratio')
    ax.set_title('d_mlp vs Performance by d_model')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Training Loss Distribution
    ax = axes[1, 1]
    ax.hist(df['final_training_loss'], bins=15, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Training Loss')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Training Losses')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dimension_sweep_summary.png"), dpi=300, bbox_inches='tight')
    plt.show()


def print_best_configurations(results, top_k=5):
    """Print the best performing configurations."""
    if not results['evaluations']:
        print("No evaluation results available.")
        return
    
    # Sort by log loss ratio (lower is better)
    sorted_configs = sorted(
        results['evaluations'].items(),
        key=lambda x: x[1]['log_loss_ratio']
    )
    
    print(f"\n=== Top {top_k} Performing Configurations ===")
    for i, (config_name, eval_result) in enumerate(sorted_configs[:top_k]):
        print(f"{i+1}. {config_name}")
        print(f"   d_model={eval_result['d_model']}, d_head={eval_result['d_head']}, d_mlp={eval_result['d_mlp']}")
        print(f"   Parameters: {eval_result['param_count']:,}")
        print(f"   Log Loss Ratio: {eval_result['log_loss_ratio']:.4f}")
        print(f"   Training Loss: {eval_result['final_training_loss']:.4f}")
        print()


if __name__ == "__main__":
    # Define dimension ranges to explore
    d_model_values = [32, 64, 128, 256]
    d_head_values = [16, 32, 64]  # Keeping d_head <= d_model
    d_mlp_values = None  # Will default to [2*d_model, 4*d_model] for each d_model
    
    # Create configurations
    configs = create_dimension_configs(
        d_model_values=d_model_values,
        d_head_values=d_head_values,
        d_mlp_values=d_mlp_values
    )
    
    print(f"Created {len(configs)} dimension configurations to test")
    
    # Run the sweep
    results = run_dimension_sweep(
        configs,
        results_dir="dimension_sweep_results",
        evaluate_models=True,
        save_models=True
    )
    
    # Print best configurations
    print_best_configurations(results, top_k=5)
    
    print("\nDimension sweep complete!")