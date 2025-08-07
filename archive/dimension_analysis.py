#%%
"""
Dimension Analysis Experiment for Coinformer Models

This script implements a comprehensive experiment to test whether low embedding dimensions
and attention head dimensions are sufficient for computing sufficient statistics in 
Bayesian updating tasks.

Key Functions:
- run_dimension_experiment(): Run full experiment across all dimension combinations
- run_quick_comparison_experiment(): Quick test with 4 key model configurations
- plot_experiment_results(): Create comprehensive visualization of results
- plot_next_token_probabilities_by_dimension(): Plot next token probabilities side by side grouped by dimension
- plot_individual_model_analysis(): Detailed analysis of individual models
- evaluate_model_bayesian_performance(): Assess Bayesian updating performance

Usage:
1. Run the script directly: python dimension_analysis.py
2. Choose between quick comparison or full experiment
3. Results will be plotted and saved to CSV

The experiment tests:
- d_model: [64, 16, 4, 2] - embedding dimensions
- d_head: [8, 4, 2] - attention head dimensions  
- d_mlp: [256, 128, 64] - MLP hidden dimensions

All models use n_heads=1 to isolate the effect of head dimension.
"""

import transformer_lens
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils import calculate_optimal_loss
from samplers import generate_data
from plot_utils import plot_probability_diff, plot_kl_divergence, plot_incremental_log_odds_cosine
from model import deactivate_position

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def prepend_bos(datasets, bos_token_value=2):
    """
    Prepend BOS token to each batch in the datasets.
    
    Args:
        datasets: List of torch tensors representing batches
        bos_token_value: Value of the BOS token (default: 2)
    
    Returns:
        List of modified tensors with BOS tokens prepended
    """
    modified_datasets = []
    for dataset in datasets:
        batch_size = dataset.shape[0]
        bos_tokens = torch.full((batch_size, 1), bos_token_value, dtype=torch.long)
        modified_dataset = torch.cat([bos_tokens, dataset], dim=1)
        modified_datasets.append(modified_dataset)
    return modified_datasets


#%%

def generate_all_binary_sequences_with_fixed_num_ones(n: int, num_ones: int, max_n_sequences: int = None, prepend_bos: bool = False, last_obs: int = None) -> torch.Tensor:
    """
    Generate all possible binary sequences of length n with exactly num_ones ones.
    If max_n_sequences is specified, only generate up to that many sequences.
    
    Args:
        n: Length of the sequence
        num_ones: Number of ones in each sequence
        max_n_sequences: Maximum number of sequences to generate (optional)
        prepend_bos: Whether to prepend BOS token (default: False)
        last_obs: If 0 or 1, fix the last token to this value (optional)
        
    Returns:
        torch.Tensor: Tensor of shape (num_permutations, n) or (num_permutations, n+1) if prepend_bos=True
    """
    if last_obs is not None and last_obs in [0, 1]:
        # If last token is fixed, we need to place ones in the first n-1 positions
        if last_obs == 1:
            # Last position is 1, so we need (num_ones - 1) ones in first (n-1) positions
            remaining_ones = num_ones - 1
        else:
            # Last position is 0, so we need num_ones ones in first (n-1) positions
            remaining_ones = num_ones
        
        # Check if it's possible to satisfy the constraint
        if remaining_ones < 0 or remaining_ones > (n - 1):
            return torch.empty((0, n), dtype=torch.long)
        
        # Generate combinations for the first n-1 positions
        positions_iter = itertools.combinations(range(n - 1), remaining_ones)
        if max_n_sequences is not None:
            positions_iter = itertools.islice(positions_iter, max_n_sequences)
        positions_list = list(positions_iter)
        num_permutations = len(positions_list)
        
        # Initialize the output tensor
        sequences = torch.zeros((num_permutations, n), dtype=torch.long)
        
        # Set the last position to the fixed value
        sequences[:, -1] = last_obs
        
        # Fill in the tensor with 1s at the appropriate positions in first n-1 positions
        for i, positions in enumerate(positions_list):
            for pos in positions:
                sequences[i, pos] = 1
    else:
        # Original behavior when last_obs is not specified or not 0/1
        positions_iter = itertools.combinations(range(n), num_ones)
        if max_n_sequences is not None:
            positions_iter = itertools.islice(positions_iter, max_n_sequences)
        positions_list = list(positions_iter)
        num_permutations = len(positions_list)
        
        # Initialize the output tensor
        sequences = torch.zeros((num_permutations, n), dtype=torch.long)
        
        # Fill in the tensor with 1s at the appropriate positions
        for i, positions in enumerate(positions_list):
            for pos in positions:
                sequences[i, pos] = 1
    
    # Prepend BOS token if requested
    if prepend_bos:
        bos_tokens = torch.full((num_permutations, 1), 2, dtype=torch.long)
        sequences = torch.cat([bos_tokens, sequences], dim=1)
    
    return sequences


#%%

def create_model_config(d_model, d_head, d_mlp, n_heads=1, n_layers=1, n_ctx=100, d_vocab=3):
    """
    Create a transformer configuration with specified dimensions.
    
    Args:
        d_model: Embedding dimension
        d_head: Attention head dimension
        d_mlp: MLP hidden dimension
        n_heads: Number of attention heads (default: 1)
        n_layers: Number of layers (default: 1)
        n_ctx: Context length (default: 100)
        d_vocab: Vocabulary size (default: 3)
        
    Returns:
        transformer_lens.HookedTransformerConfig: Model configuration
    """
    return transformer_lens.HookedTransformerConfig(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        act_fn="relu",
        default_prepend_bos=False,
        normalization_type=None,
    )

def generate_consistent_training_data(batch_size=64, seq_length=20, num_batches=100, 
                                    alpha_param=1.0, beta_param=1.0, seed=42):
    """
    Generate consistent training data for all experiments.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        num_batches: Number of batches
        alpha_param: Alpha parameter for beta distribution
        beta_param: Beta parameter for beta distribution
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (datasets, priors) - List of datasets and corresponding priors
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate data
    datasets, priors = generate_data(
        batch_size=batch_size, 
        seq_length=seq_length, 
        num_batches=num_batches, 
        alpha=alpha_param, 
        beta=beta_param, 
        bernoulli=False, 
        bernoulli_p=0.5, 
        flip_batch=False
    )
    
    # Prepend BOS token to each batch
    datasets = prepend_bos(datasets)
    
    return datasets, priors

def train_coinformer_model_with_data(model, datasets, priors,
                                   num_epochs: int = 3,
                                   learning_rate: float = 0.001,
                                   importance_sampling: bool = False,
                                   importance_sampling_alpha: float = 1.0,
                                   importance_sampling_beta: float = 1.0):
    """Train the Coinformer model with pre-generated data."""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    # Calculate optimal loss
    if importance_sampling:
        optimal_loss = calculate_optimal_loss(importance_sampling_alpha, importance_sampling_beta)
    else:
        optimal_loss = calculate_optimal_loss(1.0, 1.0)  # Default alpha=1, beta=1

    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for data_batch, prior in tqdm(zip(datasets, priors), desc=f"Epoch {epoch+1}/{num_epochs}"):
            data_batch = data_batch.to(DEVICE)
            # For each sequence, use all tokens except the last one as input
            inputs = data_batch[:, :-1]
            # Use all tokens except the first one as targets
            targets = data_batch[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape logits and targets for loss calculation
            logits_view = logits.view(-1, model.cfg.d_vocab)
            targets_view = targets.reshape(-1)
            
            # Calculate loss
            loss = criterion(logits_view, targets_view)
            if importance_sampling:
                weights = beta(importance_sampling_alpha, importance_sampling_beta).pdf(prior)
                loss = loss * weights

            epoch_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = epoch_loss / len(datasets)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Theoretical Loss Lower Bound: {optimal_loss:.4f}")

    return losses

def evaluate_model_bayesian_performance(model, test_theta=0.5, seq_length=20, batch_size=32):
    """
    Evaluate how well the model performs Bayesian updating.
    
    Args:
        model: Trained model
        test_theta: Test probability for evaluation
        seq_length: Sequence length for evaluation
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    # Generate test data
    test_data, _ = generate_data(
        batch_size=batch_size, 
        seq_length=seq_length, 
        num_batches=1, 
        alpha=1.0, 
        beta=1.0, 
        bernoulli=True, 
        bernoulli_p=test_theta
    )
    test_data = prepend_bos(test_data)[0].to(DEVICE)
    
    # test_data now has shape (batch_size, seq_length+1) due to BOS token
    actual_seq_length = test_data.shape[1] - 1  # Remove BOS token for calculations
    
    with torch.no_grad():
        # Get model predictions
        logits = model(test_data[:, :-1])
        probs = torch.softmax(logits, dim=-1)[..., 1]  # P(X_t=1)
        
        # Calculate Bayesian posterior means
        # Remove BOS token (first column) for calculations
        test_float = test_data[:, 1:].float()  # Shape: (batch_size, seq_length)
        num_ones_cumsum = torch.cumsum(test_float, dim=1)
        num_elements = torch.arange(1, actual_seq_length + 1, device=test_data.device, dtype=torch.float).unsqueeze(0).expand(batch_size, -1)
        alpha_n = 1.0 + num_ones_cumsum
        beta_n = 1.0 + (num_elements - num_ones_cumsum)
        bayesian_posteriors = alpha_n / (alpha_n + beta_n)
        
        # Calculate errors
        # probs has shape (batch_size, seq_length) where probs[:, t] predicts x_{t+1}
        # bayesian_posteriors has shape (batch_size, seq_length) where bayesian_posteriors[:, t] is posterior after seeing x_1, ..., x_{t+1}
        # We want to compare probs[:, t] (predicting x_{t+1}) with bayesian_posteriors[:, t-1] (posterior after seeing x_1, ..., x_t)
        # So we compare probs[:, 1:] with bayesian_posteriors[:, :-1]
        errors = torch.abs(probs[:, 1:] - bayesian_posteriors[:, :-1])  # Correct alignment
        mean_error = errors.mean().item()
        max_error = errors.max().item()
        
        # Calculate KL divergence
        kl_divs = []
        # Compare probs[:, t] (predicting x_{t+1}) with bayesian_posteriors[:, t-1] (posterior after seeing x_1, ..., x_t)
        # But we need to ensure we don't go out of bounds
        for t in range(1, actual_seq_length):  # Changed from actual_seq_length + 1 to actual_seq_length
            p_model = probs[:, t]
            p_bayes = bayesian_posteriors[:, t-1]
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_model = torch.clamp(p_model, epsilon, 1 - epsilon)
            p_bayes = torch.clamp(p_bayes, epsilon, 1 - epsilon)
            kl = p_model * torch.log(p_model / p_bayes) + (1 - p_model) * torch.log((1 - p_model) / (1 - p_bayes))
            kl_divs.append(kl.mean().item())
        
        avg_kl = np.mean(kl_divs)
    
    model.train()
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'avg_kl_divergence': avg_kl,
        'probs': probs.cpu().numpy(),
        'bayesian_posteriors': bayesian_posteriors.cpu().numpy()
    }

#%%
def run_dimension_experiment(d_model_values=[64, 16, 4, 2], 
                           d_head_values=[8, 4, 2], 
                           d_mlp_values=[256, 128, 64],
                           num_epochs=5,
                           learning_rate=0.001,
                           batch_size=64,
                           seq_length=20,
                           num_batches=100,
                           test_theta=0.5):
    """
    Run the dimension analysis experiment.
    
    Args:
        d_model_values: List of d_model values to test
        d_head_values: List of d_head values to test
        d_mlp_values: List of d_mlp values to test
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        seq_length: Sequence length
        num_batches: Number of batches
        test_theta: Test probability for evaluation
        
    Returns:
        dict: Dictionary containing all experiment results
    """
    print("Generating consistent training data...")
    datasets, priors = generate_consistent_training_data(
        batch_size=batch_size,
        seq_length=seq_length,
        num_batches=num_batches
    )
    
    results = {
        'configs': [],
        'losses': [],
        'bayesian_performance': [],
        'model_params': []
    }
    
    # Create models directory if it doesn't exist
    import os
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    total_experiments = len(d_model_values) * len(d_head_values) * len(d_mlp_values)
    experiment_count = 0
    
    for d_model in d_model_values:
        for d_head in d_head_values:
            for d_mlp in d_mlp_values:
                experiment_count += 1
                print(f"\n{'='*60}")
                print(f"Experiment {experiment_count}/{total_experiments}")
                print(f"Config: d_model={d_model}, d_head={d_head}, d_mlp={d_mlp}")
                print(f"{'='*60}")
                
                try:
                    # Create model configuration
                    config = create_model_config(d_model, d_head, d_mlp)
                    
                    # Create and train model
                    model = transformer_lens.HookedTransformer(config).to(DEVICE)
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    
                    # Train model
                    losses = train_coinformer_model_with_data(
                        model, datasets, priors,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate
                    )
                    
                    # Evaluate Bayesian performance
                    bayesian_metrics = evaluate_model_bayesian_performance(
                        model, test_theta, seq_length, batch_size
                    )
                    
                    # Store results
                    config_dict = {
                        'd_model': d_model,
                        'd_head': d_head,
                        'd_mlp': d_mlp,
                        'total_params': total_params
                    }
                    
                    results['configs'].append(config_dict)
                    results['losses'].append(losses)
                    results['bayesian_performance'].append(bayesian_metrics)
                    results['model_params'].append(total_params)
                    
                    print(f"Final Loss: {losses[-1]:.4f}")
                    print(f"Mean Error: {bayesian_metrics['mean_error']:.4f}")
                    print(f"Avg KL Divergence: {bayesian_metrics['avg_kl_divergence']:.4f}")
                    print(f"Total Parameters: {total_params:,}")
                    
                    # Save model in models directory
                    model_filename = os.path.join(models_dir, f"model_d{d_model}_h{d_head}_mlp{d_mlp}.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'losses': losses,
                        'bayesian_metrics': bayesian_metrics,
                        'total_params': total_params
                    }, model_filename) 
                    print(f"Model saved to {model_filename}")
                    
                    # Free up memory
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    print(f"❌ ERROR in experiment {experiment_count}: {e}")
                    print(f"Failed config: d_model={d_model}, d_head={d_head}, d_mlp={d_mlp}")
                    import traceback
                    traceback.print_exc()
                    
                    # Still increment experiment count so we can see which ones failed
                    print(f"Continuing to next experiment...")
                    continue
    
    # Print final summary
    successful_experiments = len(results['configs'])
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments planned: {total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Failed experiments: {total_experiments - successful_experiments}")
    
    if successful_experiments < total_experiments:
        print(f"⚠️  {total_experiments - successful_experiments} experiments failed!")
        print("Check the error messages above for details.")
    else:
        print("✅ All experiments completed successfully!")
    
    return results

#%%
def plot_experiment_results(results):
    """
    Create comprehensive plots for the dimension analysis experiment.
    
    Args:
        results: Dictionary containing experiment results
    
    Returns:
        pd.DataFrame: Results dataframe for further analysis
    """
    # Validate input data
    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")
    
    required_keys = ['configs', 'losses', 'bayesian_performance']
    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required key in results: {key}")
    
    # Create DataFrame for easier plotting
    df_data = []
    for i, config in enumerate(results['configs']):
        try:
            # Safely extract data with error checking
            final_loss = results['losses'][i][-1] if results['losses'][i] else float('inf')
            mean_error = results['bayesian_performance'][i].get('mean_error', float('inf'))
            avg_kl_div = results['bayesian_performance'][i].get('avg_kl_divergence', float('inf'))
            
            df_data.append({
                'd_model': config['d_model'],
                'd_head': config['d_head'],
                'd_mlp': config['d_mlp'],
                'total_params': config['total_params'],
                'final_loss': final_loss,
                'mean_error': mean_error,
                'avg_kl_divergence': avg_kl_div
            })
        except (IndexError, KeyError, TypeError) as e:
            print(f"Warning: Error processing config {i}: {e}")
            continue
    
    if not df_data:
        raise ValueError("No valid data found to create plots")
    
    df = pd.DataFrame(df_data)
    
    # Create separate plots for each d_model value
    d_model_values = sorted(df['d_model'].unique())
    
    for d_model in d_model_values:
        df_model = df[df['d_model'] == d_model]
        
        plt.figure(figsize=(15, 12))  # Adjusted size for 3x2 grid
        plt.suptitle(f'Results for d_model = {d_model}', fontsize=16)
        
        # 1. Loss curves for this d_model
        plt.subplot(3, 2, 1)  # Changed to 3x2 grid
        for i, config in enumerate(results['configs']):
            if config['d_model'] == d_model:
                label = f"d_head={config['d_head']}, d_mlp={config['d_mlp']}"
                plt.plot(results['losses'][i], label=label, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Final loss vs parameters
        plt.subplot(3, 2, 2)
        plt.scatter(df_model['total_params'], df_model['final_loss'], alpha=0.7)
        plt.xlabel('Total Parameters')
        plt.ylabel('Final Loss')
        plt.title('Final Loss vs Model Size')
        plt.grid(True, alpha=0.3)
        
        # 3. Mean error vs parameters
        plt.subplot(3, 2, 3)
        plt.scatter(df_model['total_params'], df_model['mean_error'], alpha=0.7)
        plt.xlabel('Total Parameters')
        plt.ylabel('Mean Error')
        plt.title('Bayesian Error vs Model Size')
        plt.grid(True, alpha=0.3)
        
        # 4. KL divergence vs parameters
        plt.subplot(3, 2, 4)
        plt.scatter(df_model['total_params'], df_model['avg_kl_divergence'], alpha=0.7)
        plt.xlabel('Total Parameters')
        plt.ylabel('Avg KL Divergence')
        plt.title('KL Divergence vs Model Size')
        plt.grid(True, alpha=0.3)
        
        # 5. Performance by d_head
        plt.subplot(3, 2, 5)
        d_head_groups = df_model.groupby('d_head')
        for d_head, group in d_head_groups:
            plt.scatter(group['total_params'], group['mean_error'], 
                       label=f'd_head={d_head}', alpha=0.7)
        plt.xlabel('Total Parameters')
        plt.ylabel('Mean Error')
        plt.title('Error by d_head')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Performance by d_mlp
        plt.subplot(3, 2, 6)
        d_mlp_groups = df_model.groupby('d_mlp')
        for d_mlp, group in d_mlp_groups:
            plt.scatter(group['total_params'], group['mean_error'], 
                       label=f'd_mlp={d_mlp}', alpha=0.7)
        plt.xlabel('Total Parameters')
        plt.ylabel('Mean Error')
        plt.title('Error by d_mlp')
        plt.legend()
        plt.grid(True, alpha=0.3)

        
        plt.tight_layout()
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for d_model in d_model_values:
        df_model = df[df['d_model'] == d_model]
        print(f"\nResults for d_model = {d_model}:")
        
        print(f"\nBest performing configuration:")
        if len(df_model) > 0:
            best_config = df_model.loc[df_model['mean_error'].idxmin()]
            print(f"  d_head: {best_config['d_head']}")
            print(f"  d_mlp: {best_config['d_mlp']}")
            print(f"  total_params: {best_config['total_params']:,}")
            print(f"  mean_error: {best_config['mean_error']:.4f}")
            print(f"  final_loss: {best_config['final_loss']:.4f}")
        else:
            print("  No configurations found for this d_model")
        
        print(f"\nSmallest model with good performance (error < 0.1):")
        good_models = df_model[df_model['mean_error'] < 0.1]
        if len(good_models) > 0:
            smallest_good = good_models.loc[good_models['total_params'].idxmin()]
            print(f"  d_head: {smallest_good['d_head']}")
            print(f"  d_mlp: {smallest_good['d_mlp']}")
            print(f"  total_params: {smallest_good['total_params']:,}")
            print(f"  mean_error: {smallest_good['mean_error']:.4f}")
        else:
            print("  No models achieved error < 0.1")
    
    return df



def plot_next_token_probabilities_by_dimension(results, group_by='d_model', seq_length=99):
    """
    Plot next token probabilities for a sequence of 1s, organized by specified dimension values side by side.
    
    Args:
        results: Dictionary containing experiment results
        group_by: Dimension to group by ('d_model', 'd_head', or 'd_mlp')
        seq_length: Length of the sequence of 1s to test (default: 99)
    
    Examples:
        # Group by embedding dimension
        plot_next_token_probabilities_by_dimension(results, group_by='d_model')
        
        # Group by attention head dimension
        plot_next_token_probabilities_by_dimension(results, group_by='d_head')
        
        # Group by MLP dimension
        plot_next_token_probabilities_by_dimension(results, group_by='d_mlp')
    """
    if not isinstance(results, dict) or 'configs' not in results:
        raise ValueError("Results must be a dictionary containing 'configs' key")
    
    if group_by not in ['d_model', 'd_head', 'd_mlp']:
        raise ValueError("group_by must be one of: 'd_model', 'd_head', 'd_mlp'")
    
    # Get unique values for the specified dimension and sort them
    dimension_values = sorted(set(config[group_by] for config in results['configs']))
    
    if not dimension_values:
        print(f"No {group_by} values found in results")
        return
    
    # Create figure with subplots side by side
    fig, axes = plt.subplots(1, len(dimension_values), figsize=(5 * len(dimension_values), 6))
    
    # If only one value, make axes a list for consistency
    if len(dimension_values) == 1:
        axes = [axes]
    
    fig.suptitle(f'Next Token Probabilities for Sequence of 1s (grouped by {group_by})', fontsize=16)
    
    # Create input sequence: BOS token followed by sequence of 1s
    input_seq = torch.tensor([[2] + [1] * seq_length]).to(DEVICE)  # [2] is BOS token
    
    for idx, dim_value in enumerate(dimension_values):
        ax = axes[idx]
        
        # Plot for each configuration with this dimension value
        for i, config in enumerate(results['configs']):
            if config[group_by] == dim_value:
                try:
                    model_path = f"trained_models/model_d{config['d_model']}_h{config['d_head']}_mlp{config['d_mlp']}.pt"
                    
                    # Check if model file exists
                    if not os.path.exists(model_path):
                        print(f"Warning: Model file not found: {model_path}")
                        continue
                    
                    # Load the saved model data
                    # Note: weights_only=False needed for PyTorch 2.6+ to load custom objects like HookedTransformerConfig
                    saved_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
                    
                    # Use the actual transformer config from the saved file
                    actual_config = saved_data['config']
                    model = transformer_lens.HookedTransformer(actual_config).to(DEVICE)
                    model.load_state_dict(saved_data['model_state_dict'])
                    model.eval()
                    
                    with torch.no_grad():
                        logits = model(input_seq)
                        probs = torch.softmax(logits[0], dim=-1)  # Get probabilities for each position
                        prob_of_1 = probs[:, 1].cpu().numpy()  # Probability of token 1
                    
                    # Create label showing the other two dimensions
                    if group_by == 'd_model':
                        label = f"d_head={config['d_head']}, d_mlp={config['d_mlp']}"
                    elif group_by == 'd_head':
                        label = f"d_model={config['d_model']}, d_mlp={config['d_mlp']}"
                    else:  # group_by == 'd_mlp'
                        label = f"d_model={config['d_model']}, d_head={config['d_head']}"
                    
                    ax.plot(range(len(prob_of_1)), prob_of_1, label=label, alpha=0.7)
                    
                    # Clean up model from memory
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    print(f"Error loading model for d_model={config['d_model']}, d_head={config['d_head']}, d_mlp={config['d_mlp']}: {e}")
                    continue
        
        ax.set_xlabel('Position')
        ax.set_ylabel('P(next token = 1)')
        ax.set_title(f'{group_by} = {dim_value}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_models(results_df):
    """
    Compare and summarize different model configurations.
    
    Args:
        results_df: DataFrame containing experiment results with columns:
                   d_model, d_head, d_mlp, total_params, mean_error, final_loss
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY") 
    print("="*60)

    # Find overall best model
    best_model = results_df.loc[results_df['mean_error'].idxmin()]
    print("\nBest Overall Model:")
    print(f"  d_model: {best_model['d_model']}")
    print(f"  d_head: {best_model['d_head']}")
    print(f"  d_mlp: {best_model['d_mlp']}")
    print(f"  Parameters: {best_model['total_params']:,}")
    print(f"  Mean Error: {best_model['mean_error']:.4f}")
    print(f"  Final Loss: {best_model['final_loss']:.4f}")

    # Find most parameter-efficient model
    good_models = results_df[results_df['mean_error'] < 0.1]
    if len(good_models) > 0:
        efficient_model = good_models.loc[good_models['total_params'].idxmin()]
        print("\nMost Parameter-Efficient Model (error < 0.1):")
        print(f"  d_model: {efficient_model['d_model']}")
        print(f"  d_head: {efficient_model['d_head']}")
        print(f"  d_mlp: {efficient_model['d_mlp']}")
        print(f"  Parameters: {efficient_model['total_params']:,}")
        print(f"  Mean Error: {efficient_model['mean_error']:.4f}")
    
    # Analyze dimension impact
    print("\nDimension Analysis:")
    
    for dim in ['d_model', 'd_head', 'd_mlp']:
        dim_groups = results_df.groupby(dim)['mean_error']
        print(f"\n{dim} impact:")
        print("  Size  |  Avg Error  |  Min Error")
        print("  " + "-"*30)
        for size, errors in dim_groups:
            print(f"  {size:<5} |  {errors.mean():.4f}    |  {errors.min():.4f}")

    # Parameter efficiency analysis
    param_bins = pd.qcut(results_df['total_params'], q=4)
    param_analysis = results_df.groupby(param_bins)['mean_error'].agg(['mean', 'min'])
    
    print("\nParameter Efficiency Analysis:")
    print("  Param Range  |  Avg Error  |  Best Error")
    print("  " + "-"*45)
    for bin_range, stats in param_analysis.iterrows():
        range_str = f"{bin_range.left:,.0f}-{bin_range.right:,.0f}"
        print(f"  {range_str:<12} |  {stats['mean']:.4f}    |  {stats['min']:.4f}")


def plot_individual_model_analysis(model, config_name, test_theta=0.5, seq_length=20, batch_size=32):
    """
    Create detailed analysis plots for a single model.
    
    Args:
        model: Trained model to analyze
        config_name: Name/description of the model configuration
        test_theta: Test probability for evaluation
        seq_length: Sequence length for evaluation
        batch_size: Batch size for evaluation
    """
    print(f"\nAnalyzing model: {config_name}")
    
    # Evaluate model performance
    metrics = evaluate_model_bayesian_performance(model, test_theta, seq_length, batch_size)
    
    # Create comprehensive plots
    plt.figure(figsize=(15, 10))
    
    # 1. Model vs Bayesian predictions over sequence positions
    plt.subplot(2, 3, 1)
    positions = range(1, seq_length)
    mean_model_probs = np.mean(metrics['probs'][:, 1:], axis=0)  # Model predictions for x_2, ..., x_seq_length
    mean_bayes_probs = np.mean(metrics['bayesian_posteriors'][:, :-1], axis=0)  # Posterior after x_1, ..., x_{seq_length-1}
    
    plt.plot(positions, mean_model_probs, 'b-', label='Model Predictions', linewidth=2)
    plt.plot(positions, mean_bayes_probs, 'r--', label='Bayesian Posteriors', linewidth=2)
    plt.xlabel('Sequence Position')
    plt.ylabel('Probability of 1')
    plt.title(f'Model vs Bayesian Predictions (θ={test_theta})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Error over sequence positions
    plt.subplot(2, 3, 2)
    errors = np.abs(mean_model_probs - mean_bayes_probs)
    plt.plot(positions, errors, 'g-', linewidth=2)
    plt.xlabel('Sequence Position')
    plt.ylabel('Absolute Error')
    plt.title(f'Prediction Error Over Positions')
    plt.grid(True, alpha=0.3)
    
    # 3. Error distribution
    plt.subplot(2, 3, 3)
    all_errors = np.abs(metrics['probs'][:, 1:] - metrics['bayesian_posteriors'][:, :-1]).flatten()
    plt.hist(all_errors, bins=30, alpha=0.7, color='orange')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (Mean: {metrics["mean_error"]:.4f})')
    plt.grid(True, alpha=0.3)
    
    # 4. Scatter plot of model vs Bayesian predictions
    plt.subplot(2, 3, 4)
    model_flat = metrics['probs'][:, 1:].flatten()  # Skip BOS token
    bayes_flat = metrics['bayesian_posteriors'].flatten()  # Already correct shape
    plt.scatter(bayes_flat, model_flat, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
    plt.xlabel('Bayesian Posterior')
    plt.ylabel('Model Prediction')
    plt.title('Model vs Bayesian Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Performance metrics
    plt.subplot(2, 3, 5)
    metrics_text = f"""
    Mean Error: {metrics['mean_error']:.4f}
    Max Error: {metrics['max_error']:.4f}
    Avg KL Divergence: {metrics['avg_kl_divergence']:.4f}
    """
    plt.text(0.1, 0.5, metrics_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.axis('off')
    plt.title('Performance Metrics')
    
    # 6. Model configuration info
    plt.subplot(2, 3, 6)
    config_text = f"""
    {config_name}
    
    d_model: {model.cfg.d_model}
    d_head: {model.cfg.d_head}
    d_mlp: {model.cfg.d_mlp}
    n_heads: {model.cfg.n_heads}
    n_layers: {model.cfg.n_layers}
    """
    plt.text(0.1, 0.5, config_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    plt.axis('off')
    plt.title('Model Configuration')
    
    plt.tight_layout()
    plt.suptitle(f'Detailed Analysis: {config_name}', fontsize=16, y=1.02)
    plt.show()
    
    return metrics

def run_quick_comparison_experiment():
    """
    Run a quick comparison experiment with a few key configurations.
    This is useful for getting initial insights without running the full experiment.
    """
    print("Running quick comparison experiment...")
    
    # Test a few key configurations
    configs = [
        {'d_model': 64, 'd_head': 8, 'd_mlp': 256, 'name': 'Large Model'},
        {'d_model': 16, 'd_head': 4, 'd_mlp': 128, 'name': 'Medium Model'},
        {'d_model': 4, 'd_head': 2, 'd_mlp': 64, 'name': 'Small Model'},
        {'d_model': 2, 'd_head': 2, 'd_mlp': 32, 'name': 'Tiny Model'},
    ]
    
    # Generate consistent training data
    datasets, priors = generate_consistent_training_data(
        batch_size=32, seq_length=20, num_batches=50
    )
    
    results = []
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"Config: d_model={config['d_model']}, d_head={config['d_head']}, d_mlp={config['d_mlp']}")
        print(f"{'='*50}")
        
        # Create and train model
        model_config = create_model_config(
            config['d_model'], config['d_head'], config['d_mlp']
        )
        model = transformer_lens.HookedTransformer(model_config).to(DEVICE)
        
        # Train model
        losses = train_coinformer_model_with_data(
            model, datasets, priors, num_epochs=3, learning_rate=0.001
        )
        
        # Evaluate performance
        metrics = evaluate_model_bayesian_performance(model, test_theta=0.5)
        
        # Store results
        results.append({
            'name': config['name'],
            'config': config,
            'losses': losses,
            'metrics': metrics,
            'total_params': sum(p.numel() for p in model.parameters())
        })
        
        # Create detailed analysis plots
        plot_individual_model_analysis(model, config['name'])
    
    # Print summary
    print("\n" + "="*60)
    print("QUICK COMPARISON SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Parameters: {result['total_params']:,}")
        print(f"  Final Loss: {result['losses'][-1]:.4f}")
        print(f"  Mean Error: {result['metrics']['mean_error']:.4f}")
        print(f"  Avg KL Divergence: {result['metrics']['avg_kl_divergence']:.4f}")
    
    return results

#%%

results = run_dimension_experiment(
    d_model_values=[16, 4, 2],  # Test embedding dimensions
    d_head_values=[8, 4, 2],        # Test attention head dimensions  
    d_mlp_values=[256, 64],    # Test MLP dimensions
    num_epochs=10,                   # Number of training epochs
    batch_size=64,                  # Batch size
    seq_length=100,                  # Sequence length
    num_batches=100,                # Number of batches
    test_theta=0.5                  # Test probability
)

#%%
df = plot_experiment_results(results)
compare_models(df)

# Plot next token probabilities side by side grouped by different dimensions
plot_next_token_probabilities_by_dimension(results, group_by='d_model')
# You can also group by other dimensions:
plot_next_token_probabilities_by_dimension(results, group_by='d_head')
plot_next_token_probabilities_by_dimension(results, group_by='d_mlp')

df.to_csv('dimension_analysis_results.csv', index=False)
print("Results saved to dimension_analysis_results.csv")


# %%
