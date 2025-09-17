#%%
"""
Experiment 5: Fixed Positional Embedding Analysis

This experiment trains models with fixed positional embeddings where the positional
embeddings increment by 1 in the last dimension of d_model. We focus on d_model=4 
first to understand how this affects probability updating behavior.

The positional embeddings are non-trainable, and we compare performance with 
the baseline models from dimension sweep training.
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
from core.models import ModelConfig, create_coinformer_model, set_fixed_positional_embedding
from core.training import train_coinformer_model, save_model_with_config
from core.samplers import generate_data_with_p_list, generate_data
from core.utils import get_log_loss

#%%

def create_fixed_pos_embed_configs(
    d_model_values=[4],
    d_head_values=[2, 4],
    d_mlp_values=[16, 32],
    base_config=None
):
    """
    Create experiment configurations for fixed positional embedding experiments.
    All models use exactly 1 attention head and 1 layer, with fixed positional embeddings.
    
    Args:
        d_model_values: List of d_model values to test (default: [4] to start small)
        d_head_values: List of d_head values to test  
        d_mlp_values: List of d_mlp values to test
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
            seq_length=99,
            learning_rate=0.001,
            batch_size=64
        )
    
    configs = []
    
    for d_model, d_head in product(d_model_values, d_head_values):
        # Skip invalid configurations (d_head should typically be <= d_model)
        if d_head > d_model:
            continue
            
        for d_mlp in d_mlp_values:
            # Create model config with fixed n_layers=1 and n_heads=1
            model_config = ModelConfig(
                d_model=d_model,
                d_head=d_head,
                n_heads=1,  # Fixed to 1 attention head
                d_mlp=d_mlp,
                n_layers=1,  # Fixed to 1 layer
                use_bos_token=True
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
            
            config_name = f"fixed_pos_d{d_model}_h{d_head}_mlp{d_mlp}"
            configs.append((config_name, config))
    
    return configs


def train_model_with_fixed_pos_embed(config, training_data=None, verbose=False, bos_gradient_on = False):
    """
    Train a model with fixed positional embeddings.
    
    Args:
        config: ExperimentConfig
        training_data: Optional pre-generated training data
        verbose: Whether to print verbose output
        
    Returns:
        (model, losses): Trained model and training losses
    """
    # Create model
    model = create_coinformer_model(config.model_config)
    
    # Set fixed positional embeddings
    model = set_fixed_positional_embedding(model)

    if bos_gradient_on:
        model.pos_embed.W_pos.requires_grad = True
    if verbose:
        print(f"Model created with d_model={config.model_config.d_model}")
        print(f"Fixed positional embeddings set in last dimension")
        print(f"Positional embeddings shape: {model.pos_embed.W_pos.shape}")
        print(f"Positional embeddings require_grad: {model.pos_embed.W_pos.requires_grad}")
        print(f"Sample positional embeddings (first 5 positions):")
        for i in range(min(5, model.pos_embed.W_pos.shape[0])):
            print(f"  Position {i}: {model.pos_embed.W_pos[i].tolist()}")
    
    # Train model using the existing training function
    # Note: We need to modify the training function to handle pre-created models
    # For now, we'll create the model within the training function and then replace it
    
    # Generate training data if not provided
    if training_data is None:
        training_datasets, training_priors = generate_data(
            batch_size=config.batch_size, 
            seq_length=config.seq_length,
            num_batches=config.num_batches, 
            alpha=config.alpha, 
            beta=config.beta, 
            bernoulli=config.bernoulli, 
            bernoulli_p=config.bernoulli_p, 
            flip_batch=config.flip_batch,
            scale=config.scale, 
            bias=config.bias,
            use_bos_token=config.model_config.use_bos_token
        )
        training_data = (training_datasets, training_priors)
    
    # Manual training loop since we need to use our pre-configured model
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    training_datasets, training_priors = training_data
    
    model.train()
    for epoch in range(config.num_epochs):
        epoch_losses = []
        
        for batch_idx in range(len(training_datasets)):
            inputs = training_datasets[batch_idx].to(device)
            targets = inputs[:, 1:].contiguous()  # Shift for next token prediction
            inputs = inputs[:, :-1].contiguous()
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(inputs)
            
            # Compute loss
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return model, losses


def run_fixed_pos_embed_sweep(
    configs,
    results_dir="fixed_pos_embed_results",
    evaluate_models=True,
    save_models=True,
    seed=42,
    bos_gradient_on = False
):
    """
    Run training for all fixed positional embedding configurations.
    
    Args:
        configs: List of (config_name, ExperimentConfig) tuples
        results_dir: Directory to save results
        evaluate_models: Whether to evaluate trained models
        save_models: Whether to save trained models
        seed: Random seed for reproducible results
        
    Returns:
        Dictionary with training results
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_results_dir = f"{results_dir}_{timestamp}"
    os.makedirs(full_results_dir, exist_ok=True)
    
    # Generate fixed training data using the first config's parameters
    reference_config = configs[0][1]
    print(f"Generating fixed training data using reference config parameters...")
    print(f"  alpha={reference_config.alpha}, beta={reference_config.beta}")
    print(f"  batch_size={reference_config.batch_size}, seq_length={reference_config.seq_length}")
    print(f"  num_batches={reference_config.num_batches}")
    
    training_datasets, training_priors = generate_data(
        batch_size=reference_config.batch_size, 
        seq_length=reference_config.seq_length,
        num_batches=reference_config.num_batches, 
        alpha=reference_config.alpha, 
        beta=reference_config.beta, 
        bernoulli=reference_config.bernoulli, 
        bernoulli_p=reference_config.bernoulli_p, 
        flip_batch=reference_config.flip_batch,
        scale=reference_config.scale, 
        bias=reference_config.bias,
        use_bos_token=reference_config.model_config.use_bos_token
    )
    
    # Initialize results tracking
    results = {
        'configs': {},
        'training_losses': {},
        'models': {},
        'evaluations': {},
        'training_data': (training_datasets, training_priors),
        'metadata': {
            'timestamp': timestamp,
            'total_configs': len(configs),
            'results_dir': full_results_dir,
            'seed': seed,
            'fixed_training_data': True,
            'experiment_type': 'fixed_positional_embedding'
        }
    }
    
    print(f"=== Fixed Positional Embedding Training ===")
    print(f"Training {len(configs)} configurations with FIXED positional embeddings...")
    print(f"Random seed: {seed}")
    print(f"Results will be saved to: {full_results_dir}")
    
    # Train each configuration
    for i, (config_name, config) in enumerate(tqdm(configs, desc="Training models")):
        print(f"\n[{i+1}/{len(configs)}] Training {config_name}...")
        print(f"  d_model={config.model_config.d_model}, d_head={config.model_config.d_head}, n_heads={config.model_config.n_heads}, d_mlp={config.model_config.d_mlp}")
        
        try:
            # Train model with fixed positional embeddings
            model, losses = train_model_with_fixed_pos_embed(
                config, 
                training_data=(training_datasets, training_priors),
                verbose=True,
                bos_gradient_on=bos_gradient_on
            )
            
            # Store results
            results['configs'][config_name] = config
            results['training_losses'][config_name] = losses
            results['models'][config_name] = model
            
            print(f"  Final training loss: {losses[-1]:.4f}")
            
            # Save model if requested
            if save_models:
                model_dir = os.path.join(full_results_dir, "models")
                save_path = save_model_with_config(model, config, f"fixed_pos_{config_name}", model_dir)
                print(f"  Saved model to: {save_path}")
            
        except Exception as e:
            print(f"  ERROR training {config_name}: {e}")
            results['training_losses'][config_name] = None
            results['models'][config_name] = None
            continue
    
    # Evaluate models if requested
    if evaluate_models:
        print("\n=== Evaluating Models ===")
        evaluate_fixed_pos_embed_models(results, full_results_dir)
    
    # Save results summary
    save_fixed_pos_embed_results(results, full_results_dir)
    
    return results


def evaluate_fixed_pos_embed_models(results, results_dir):
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
            trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Store evaluation results
            trans_log_loss_scalar = trans_log_loss.cpu().item() if hasattr(trans_log_loss, 'cpu') else float(trans_log_loss)
            bayes_log_loss_scalar = bayes_log_loss.cpu().item() if hasattr(bayes_log_loss, 'cpu') else float(bayes_log_loss)
            final_training_loss_scalar = results['training_losses'][config_name][-1]
            if hasattr(final_training_loss_scalar, 'cpu'):
                final_training_loss_scalar = final_training_loss_scalar.cpu().item()
            
            results['evaluations'][config_name] = {
                'd_model': config.model_config.d_model,
                'd_head': config.model_config.d_head,
                'n_heads': config.model_config.n_heads,
                'd_mlp': config.model_config.d_mlp,
                'n_layers': config.model_config.n_layers,
                'param_count': param_count,
                'trainable_param_count': trainable_param_count,
                'final_training_loss': final_training_loss_scalar,
                'trans_log_loss': trans_log_loss_scalar,
                'bayes_log_loss': bayes_log_loss_scalar,
                'log_loss_ratio': trans_log_loss_scalar / bayes_log_loss_scalar,
                'experiment_type': 'fixed_pos_embed'
            }
            
        except Exception as e:
            print(f"  ERROR evaluating {config_name}: {e}")
            continue


def save_fixed_pos_embed_results(results, results_dir):
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
        create_fixed_pos_embed_plots(df, results_dir)
    
    # Save training losses
    losses_path = os.path.join(results_dir, "training_losses.json")
    with open(losses_path, 'w') as f:
        serializable_losses = {k: v for k, v in results['training_losses'].items() if v is not None}
        json.dump(serializable_losses, f, indent=2)
    
    # Save metadata
    metadata_path = os.path.join(results_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(results['metadata'], f, indent=2)


def create_fixed_pos_embed_plots(df, results_dir):
    """Create summary plots for fixed positional embedding results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance vs Parameters
    ax = axes[0, 0]
    scatter = ax.scatter(df['trainable_param_count'], df['log_loss_ratio'], 
                        c=df['d_model'], s=60, alpha=0.7, cmap='viridis')
    ax.set_xlabel('Trainable Parameters')
    ax.set_ylabel('Log Loss Ratio (Trans/Bayes)')
    ax.set_title('Performance vs Trainable Model Size\n(Fixed Pos Embeddings)')
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
    ax.set_title('d_model vs Performance by d_mlp\n(Fixed Pos Embeddings)')
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
    ax.set_title('d_mlp vs Performance by d_model\n(Fixed Pos Embeddings)')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Training Loss Distribution
    ax = axes[1, 1]
    ax.hist(df['final_training_loss'], bins=10, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Training Loss')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Training Losses\n(Fixed Pos Embeddings)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fixed_pos_embed_summary.png"), dpi=300, bbox_inches='tight')
    plt.show()


def print_best_fixed_pos_embed_configs(results, top_k=5):
    """Print the best performing configurations."""
    if not results['evaluations']:
        print("No evaluation results available.")
        return
    
    # Sort by log loss ratio (lower is better)
    sorted_configs = sorted(
        results['evaluations'].items(),
        key=lambda x: x[1]['log_loss_ratio']
    )
    
    print(f"\n=== Top {top_k} Performing Fixed Pos Embed Configurations ===")
    for i, (config_name, eval_result) in enumerate(sorted_configs[:top_k]):
        print(f"{i+1}. {config_name}")
        print(f"   d_model={eval_result['d_model']}, d_head={eval_result['d_head']}, n_heads={eval_result['n_heads']}, d_mlp={eval_result['d_mlp']}")
        print(f"   Total Parameters: {eval_result['param_count']:,}")
        print(f"   Trainable Parameters: {eval_result['trainable_param_count']:,}")
        print(f"   Log Loss Ratio: {eval_result['log_loss_ratio']:.4f}")
        print(f"   Training Loss: {eval_result['final_training_loss']:.4f}")
        print()


#%%

if __name__ == "__main__":
    # Define dimension ranges to explore for d_model=4
    d_model_values = [4,16,32]
    d_head_values = [2] 
    d_mlp_values = [64,128]
    
    # Create configurations
    configs = create_fixed_pos_embed_configs(
        d_model_values=d_model_values,
        d_head_values=d_head_values,
        d_mlp_values=d_mlp_values
    )
    
    print(f"Created {len(configs)} fixed positional embedding configurations to test")
    for config_name, config in configs:
        print(f"  {config_name}: d_model={config.model_config.d_model}, d_head={config.model_config.d_head}, d_mlp={config.model_config.d_mlp}")
    
    # Run the sweep
    results = run_fixed_pos_embed_sweep(
        configs,
        results_dir="fixed_pos_embed_results",
        evaluate_models=True,
        save_models=True,
        seed=42,
        bos_gradient_on=True  # Fixed seed for reproducible results
    )
    
    # Print best configurations
    print_best_fixed_pos_embed_configs(results, top_k=len(configs))
    
    print("\nFixed positional embedding sweep complete!")
# %%
