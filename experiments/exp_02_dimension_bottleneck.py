#%%
"""
Experiment 2: Dimension Bottleneck Analysis

This experiment loads models trained from dimension_sweep_training and analyzes
how different architectural choices affect probability updating across sequences.
Creates interactive plots showing probability evolution for different theta values.

Configuration options:
- USE_ALL_MODELS: Set to True to analyze all models in the directory, False to select best performing
- NUM_SELECTED_MODELS: Number of models to select when USE_ALL_MODELS is False
- SELECTION_CRITERION: Metric to use for model selection (default: 'log_loss_ratio_clean')
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.config import ExperimentConfig, ModelConfig
from core.training import load_model_from_config
from core.samplers import generate_data_with_p
from core.utils import get_autoregressive_predictions

#%% Configuration
# Data source configuration
LOAD_FROM_CSV = False  # Set to True to load existing CSV data instead of recomputing
SKIP_MODEL_LOADING = False  # Set to True to skip model loading and just use CSV data for plots

# USAGE MODES:
# 1. Full computation: LOAD_FROM_CSV=False, SKIP_MODEL_LOADING=False (compute everything)
# 2. Plot-only mode: LOAD_FROM_CSV=True, SKIP_MODEL_LOADING=True (fast plotting from CSV)
# 3. Data-only mode: LOAD_FROM_CSV=False, SKIP_MODEL_LOADING=True (compute data, skip models)

# Experiment configuration options (only used if LOAD_FROM_CSV is False)
USE_ALL_MODELS = True  # Set to True to use all models, False to select best performing
# NUM_SELECTED_MODELS = 6  # Number of models to select if USE_ALL_MODELS is False
# SELECTION_CRITERION = 'log_loss_ratio_clean'  # Metric to use for model selection

# Load the dimension sweep results
SWEEP_RESULTS_DIR = "/Users/afiqabdillah/balls-and-urns/experiments/dimension_sweep_results_20250807_072621"

if LOAD_FROM_CSV:
    print("=== Loading Data from CSV Files ===")
    # Load pre-computed data from CSV files
    predictions_df = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_predictions.csv"))
    theoretical_df = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_theoretical.csv"))
    sequences_df = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_sequences.csv"))
    summary_df = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_summary.csv"))
    
    # Parse the nested arrays from string format back to lists
    import ast
    predictions_df['predictions'] = predictions_df['predictions'].apply(ast.literal_eval)
    theoretical_df['theoretical_predictions_avg'] = theoretical_df['theoretical_predictions_avg'].apply(ast.literal_eval)
    if 'theoretical_predictions_all' in theoretical_df.columns:
        theoretical_df['theoretical_predictions_all'] = theoretical_df['theoretical_predictions_all'].apply(ast.literal_eval)
    sequences_df['sequences'] = sequences_df['sequences'].apply(ast.literal_eval)
    
    # Extract theta values and model information from loaded data
    theta_values = sorted(predictions_df['theta'].unique())
    selected_models = predictions_df[['model_name', 'd_model', 'd_head', 'n_heads', 'd_mlp', 'n_layers', 'log_loss_ratio']].drop_duplicates()
    selected_models = selected_models.rename(columns={'model_name': 'config_name', 'log_loss_ratio': 'log_loss_ratio_clean'})
    
    # Reconstruct probability_results and all_sequences from DataFrames
    probability_results = {}
    all_sequences = {}
    
    for _, row in predictions_df.iterrows():
        theta = row['theta']
        model_name = row['model_name']
        predictions = row['predictions']
        
        if theta not in probability_results:
            probability_results[theta] = {}
        probability_results[theta][model_name] = np.array(predictions)
    
    for _, row in sequences_df.iterrows():
        theta = row['theta']
        sequences_list = row['sequences']
        all_sequences[theta] = [torch.tensor(seq_dict['sequence']) for seq_dict in sequences_list]
    
    print(f"Loaded data for {len(theta_values)} theta values and {len(selected_models)} models")
    
else:
    # Original data loading and processing
    evaluation_results = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, "evaluation_results.csv"))

    # Clean up the tensor strings in the CSV to extract numerical values
    def extract_tensor_value(tensor_str):
        """Extract numerical value from tensor string format."""
        if "tensor(" in str(tensor_str):
            # Extract number between tensor( and ,
            import re
            match = re.search(r'tensor\(([\d\.]+)', str(tensor_str))
            if match:
                return float(match.group(1))
        return float(tensor_str)

    evaluation_results['log_loss_ratio_clean'] = evaluation_results['log_loss_ratio'].apply(extract_tensor_value)

print("=== Dimension Bottleneck Analysis with Sweep Models ===")

if not LOAD_FROM_CSV:
    print(f"Found {len(evaluation_results)} trained models")
    print(f"Results directory: {SWEEP_RESULTS_DIR}")

    # Select models based on configuration
    if USE_ALL_MODELS:
        selected_models = evaluation_results
        print(f"\nUsing ALL {len(selected_models)} models for analysis")
    else:
        selected_models = evaluation_results.nsmallest(NUM_SELECTED_MODELS, SELECTION_CRITERION)
        print(f"\nSelected {len(selected_models)} best performing models (by {SELECTION_CRITERION}):")
else:
    print(f"Loaded data from CSV files")
    print(f"Results directory: {SWEEP_RESULTS_DIR}")

print(f"\nAnalyzing {len(selected_models)} models:")
for _, row in selected_models.iterrows():
    config_col = 'config_name' if 'config_name' in row else 'model_name'
    print(f"  {row[config_col]}: d_model={row['d_model']}, d_head={row['d_head']}, d_mlp={row['d_mlp']}, ratio={row['log_loss_ratio_clean']:.4f}")

#%% Load trained models from dimension sweep (only if needed)
if not LOAD_FROM_CSV and not SKIP_MODEL_LOADING:
    models = {}
    model_configs = {}

    def create_config_from_row(row):
        """Create ExperimentConfig from evaluation results row."""
        model_config = ModelConfig(
            d_model=int(row['d_model']),
            d_head=int(row['d_head']),
            n_heads=int(row['n_heads']),
            d_mlp=int(row['d_mlp']),
            n_layers=int(row['n_layers']),
            use_bos_token=True  # Models from dimension sweep were trained with BOS tokens
        )
        
        return ExperimentConfig(
            model_config=model_config,
            alpha=1.0,  # Default values from dimension sweep training
            beta=1.0,
            num_epochs=5,
            num_batches=1000,
            seq_length=99,
            learning_rate=0.001,
            batch_size=64
        )

    print("\n=== Loading Models ===")
    for _, row in selected_models.iterrows():
        config_name = row['config_name'] if 'config_name' in row else row['model_name']
        print(f"Loading {config_name}...")
        
        # Create configuration
        config = create_config_from_row(row)
        model_configs[config_name] = config
        
        # Construct model file path
        model_filename = f"sweep_{config_name}_dmodel{row['d_model']}_dhead{row['d_head']}_layers{row['n_layers']}_alpha{config.alpha}_beta{config.beta}_bos.pt"
        model_path = os.path.join(SWEEP_RESULTS_DIR, "models", model_filename)
        
        if os.path.exists(model_path):
            try:
                model = load_model_from_config(config, model_path)
                models[config_name] = model
                print(f"  Successfully loaded: {config_name}")
            except Exception as e:
                print(f"  Error loading {config_name}: {e}")
        else:
            print(f"  Model file not found: {model_filename}")

    print(f"\nSuccessfully loaded {len(models)} models")
else:
    # Create empty models dict for CSV mode or when skipping model loading
    models = {}
    print("\nSkipping model loading (using CSV data or SKIP_MODEL_LOADING=True)")

#%% Generate test sequences for different theta values (only if needed)
num_sequences = 5
if not LOAD_FROM_CSV:
    def generate_sequences_for_theta(theta, num_sequences=10, seq_length=50, use_bos_token=True):
        """Generate multiple sequences with a given theta value."""
        data, _ = generate_data_with_p(
            p=theta,
            batch_size=num_sequences,
            seq_length=seq_length,
            num_batches=1,
            flip_batch=False,
            use_bos_token=use_bos_token
        )
        # Extract all sequences from the single batch
        return [data[0][i] for i in range(num_sequences)]

    # Generate test sequences for theta values from 0.0 to 1.0
    # Include edge cases theta=0 and theta=1 where behavior is deterministic
    theta_values = np.concatenate([[0.0], np.arange(0.1, 1.0, 0.1), [1.0]])
    # IMPORTANT: Models were trained with seq_length=99 + BOS token = 100 total tokens
    # So we need to use seq_length=99 for testing to match training configuration
    seq_length = 99

    print(f"\n=== Generating Test Sequences ===")
    print(f"Theta values: {theta_values}")
    print(f"Sequences per theta: {num_sequences}")
    print(f"Sequence length: {seq_length}")

    all_sequences = {}
    for theta in theta_values:
        print(f"Generating sequences for theta={theta:.1f}...")
        # For edge cases (theta=0 or theta=1), only generate one sequence since behavior is deterministic
        sequences_needed = 1 if theta in [0.0, 1.0] else num_sequences
        # Models from dimension sweep were trained with BOS tokens
        all_sequences[theta] = generate_sequences_for_theta(theta, sequences_needed, seq_length, use_bos_token=True)
else:
    print(f"\n=== Using Pre-loaded Sequences ===")
    print(f"Theta values: {theta_values}")
    num_sequences = len(all_sequences[theta_values[0]])  # Get num_sequences from loaded data
    seq_length = all_sequences[theta_values[0]][0].shape[0] - 1  # Subtract 1 for BOS token
    print(f"Sequences per theta: {num_sequences}")
    print(f"Sequence length: {seq_length}")

#%% Compute probability updates for each model and sequence (only if needed)
if not LOAD_FROM_CSV:
    def compute_probability_updates(model, sequence):
        """Compute probability updates for a single sequence."""
        predictions = get_autoregressive_predictions(model, sequence)
        return predictions

    print(f"\n=== Computing Probability Updates ===")
    # Store results: {theta: {model_name: [avg_probs_over_sequences]}}
    probability_results = {}

    for theta in theta_values:
        print(f"Processing theta={theta:.1f}...")
        probability_results[theta] = {}
        
        for model_name, model in models.items():
            print(f"  Model: {model_name}")
            
            # Collect predictions for all sequences with this theta
            all_predictions = []
            for seq_idx, sequence in enumerate(all_sequences[theta]):
                try:
                    predictions = compute_probability_updates(model, sequence)
                    all_predictions.append(predictions)
                except Exception as e:
                    print(f"    Error with sequence {seq_idx}: {e}")
                    continue
            
            if all_predictions:
                # Average predictions across sequences
                # Convert to numpy array and compute mean
                all_predictions = np.array(all_predictions)  # Shape: (num_sequences, seq_length-1)
                avg_predictions = np.mean(all_predictions, axis=0)
                probability_results[theta][model_name] = avg_predictions
            else:
                print(f"    No valid predictions for {model_name} with theta={theta:.1f}")
else:
    print(f"\n=== Using Pre-loaded Probability Results ===")
    print(f"Loaded predictions for {len(probability_results)} theta values")

#%% Save Results and Sequences to DataFrames (only if computed fresh)
if not LOAD_FROM_CSV:
    print(f"\n=== Saving Results and Sequences ===")

    # Create efficient nested DataFrame for probability predictions
    print("Creating nested probability predictions DataFrame...")
    prediction_records = []
    for theta in theta_values:
        for model_name in models.keys():
            if model_name in probability_results[theta]:
                predictions = probability_results[theta][model_name]
                # Get model configuration info once per model
                config_col = 'config_name' if 'config_name' in selected_models.columns else 'model_name'
                row_data = selected_models[selected_models[config_col] == model_name].iloc[0]
                prediction_records.append({
                    'theta': theta,
                    'model_name': model_name,
                    'predictions': predictions.tolist(),  # Store entire prediction array
                    'd_model': int(row_data['d_model']),
                    'd_head': int(row_data['d_head']),
                    'n_heads': int(row_data['n_heads']),
                    'd_mlp': int(row_data['d_mlp']),
                    'n_layers': int(row_data['n_layers']),
                    'log_loss_ratio': row_data['log_loss_ratio_clean']
                })

    predictions_df = pd.DataFrame(prediction_records)

    # Create nested DataFrame for theoretical Bayesian predictions  
    print("Creating nested theoretical Bayesian predictions DataFrame...")
    theoretical_records = []
    for theta in theta_values:
        # Compute theoretical predictions for this theta
        all_theoretical_probs = []
        sequence_theoretical_probs_list = []
    
        for seq_idx, sequence in enumerate(all_sequences[theta]):
            sequence_theoretical_probs = []
            # Start with Beta(1,1) prior: alpha=1, beta=1
            alpha, beta = 1, 1
        
            # Account for BOS token at position 0 (token ID 2)
            # Start from position 1 (first actual data token) and predict from position 2 onwards
            for pos in range(2, len(sequence)):
                # Update posterior based on observed token at position pos-1
                if sequence[pos-1] == 1:
                    alpha += 1
                else:
                    beta += 1
            
                # Posterior mean for Beta(alpha, beta) is alpha / (alpha + beta)
                posterior_mean = alpha / (alpha + beta)
                sequence_theoretical_probs.append(posterior_mean)
        
            all_theoretical_probs.append(sequence_theoretical_probs)
            sequence_theoretical_probs_list.append(sequence_theoretical_probs)
    
        # Store both individual sequence predictions and average
        if all_theoretical_probs:
            theoretical_probs_avg = np.mean(all_theoretical_probs, axis=0)
            theoretical_records.append({
                'theta': theta,
                'num_sequences': len(all_sequences[theta]),
                'theoretical_predictions_avg': theoretical_probs_avg.tolist(),
                'theoretical_predictions_all': sequence_theoretical_probs_list,  # List of lists for multiple sequences
            })

    theoretical_df = pd.DataFrame(theoretical_records)

    # Create nested DataFrame for sequences (one row per theta)
    print("Creating nested sequences DataFrame...")
    sequence_records = []
    for theta in theta_values:
        sequences_list = []
        for seq_idx, sequence in enumerate(all_sequences[theta]):
            sequence_list = sequence.tolist()
            sequences_list.append({
                'sequence_idx': seq_idx,
                'sequence': sequence_list,
                'sequence_length': len(sequence_list),
                'num_ones': sum(1 for token in sequence_list[1:] if token == 1),  # Exclude BOS token
                'num_zeros': sum(1 for token in sequence_list[1:] if token == 0),  # Exclude BOS token
                'empirical_probability': sum(1 for token in sequence_list[1:] if token == 1) / (len(sequence_list) - 1)  # Exclude BOS
            })
    
        sequence_records.append({
            'theta': theta,
            'num_sequences': len(all_sequences[theta]),
            'sequences': sequences_list  # List of sequence dictionaries
        })

    sequences_df = pd.DataFrame(sequence_records)

    # Save all DataFrames to the sweep results directory
    print(f"Saving DataFrames to {SWEEP_RESULTS_DIR}...")

    predictions_df.to_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_predictions.csv"), index=False)
    theoretical_df.to_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_theoretical.csv"), index=False)
    sequences_df.to_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_sequences.csv"), index=False)

    print(f"Saved {len(prediction_records)} prediction records to dimension_bottleneck_predictions.csv")
    print(f"Saved {len(theoretical_records)} theoretical records to dimension_bottleneck_theoretical.csv")
    print(f"Saved {len(sequence_records)} sequence records to dimension_bottleneck_sequences.csv")

    # Save summary statistics
    print("Creating summary statistics...")
    summary_stats = []
    for model_name in models.keys():
        model_data = predictions_df[predictions_df['model_name'] == model_name]
        if not model_data.empty:
            # Get model configuration
            row_data = selected_models[selected_models['config_name'] == model_name].iloc[0]
        
            # Calculate statistics across all theta values and positions
            # Extract all predictions from nested structure
            all_predictions = []
            for _, row in model_data.iterrows():
                all_predictions.extend(row['predictions'])
        
            summary_stats.append({
                'model_name': model_name,
                'd_model': int(row_data['d_model']),
                'd_head': int(row_data['d_head']),
                'n_heads': int(row_data['n_heads']),
                'd_mlp': int(row_data['d_mlp']),
                'n_layers': int(row_data['n_layers']),
                'log_loss_ratio': row_data['log_loss_ratio_clean'],
                'mean_predicted_prob': np.mean(all_predictions),
                'std_predicted_prob': np.std(all_predictions),
                'min_predicted_prob': np.min(all_predictions),
                'max_predicted_prob': np.max(all_predictions),
                'num_predictions': len(all_predictions),
                'num_theta_values': len(model_data)
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(SWEEP_RESULTS_DIR, "dimension_bottleneck_summary.csv"), index=False)
    print(f"Saved summary statistics to dimension_bottleneck_summary.csv")

    # Save helper functions for working with nested data
    helper_code = '''
    # Helper functions for working with nested DataFrame structure

    import pandas as pd
    import numpy as np

    def expand_predictions_df(df):
        """
        Expand nested predictions DataFrame to long format.
    
        Args:
            df: DataFrame with 'predictions' column containing lists
        
        Returns:
            DataFrame in long format with one row per (theta, model, position)
        """
        expanded_records = []
        for _, row in df.iterrows():
            for position, prob in enumerate(row['predictions'], 1):
                expanded_records.append({
                    'theta': row['theta'],
                    'model_name': row['model_name'],
                    'position': position,
                    'predicted_probability': prob,
                    'd_model': row['d_model'],
                    'd_head': row['d_head'],
                    'n_heads': row['n_heads'],
                    'd_mlp': row['d_mlp'],
                    'n_layers': row['n_layers'],
                    'log_loss_ratio': row['log_loss_ratio']
                })
        return pd.DataFrame(expanded_records)

    def expand_theoretical_df(df):
        """
        Expand nested theoretical DataFrame to long format.
    
        Args:
            df: DataFrame with 'theoretical_predictions_avg' column containing lists
        
        Returns:
            DataFrame in long format with one row per (theta, position)
        """
        expanded_records = []
        for _, row in df.iterrows():
            for position, prob in enumerate(row['theoretical_predictions_avg'], 1):
                expanded_records.append({
                    'theta': row['theta'],
                    'position': position,
                    'theoretical_probability': prob
                })
        return pd.DataFrame(expanded_records)

    def expand_sequences_df(df):
        """
        Expand nested sequences DataFrame to long format.
    
        Args:
            df: DataFrame with 'sequences' column containing lists of dicts
        
        Returns:
            DataFrame in long format with one row per sequence
        """
        expanded_records = []
        for _, row in df.iterrows():
            for seq_dict in row['sequences']:
                expanded_records.append({
                    'theta': row['theta'],
                    'sequence_idx': seq_dict['sequence_idx'],
                    'sequence_length': seq_dict['sequence_length'],
                    'sequence': seq_dict['sequence'],
                    'num_ones': seq_dict['num_ones'],
                    'num_zeros': seq_dict['num_zeros'],
                    'empirical_probability': seq_dict['empirical_probability']
                })
        return pd.DataFrame(expanded_records)

    # Example usage:
    # predictions_df = pd.read_csv('dimension_bottleneck_predictions.csv')
    # predictions_long = expand_predictions_df(predictions_df)
    '''

    with open(os.path.join(SWEEP_RESULTS_DIR, "data_helpers.py"), 'w') as f:
        f.write(helper_code)
    print("Saved helper functions to data_helpers.py")
else:
    print(f"\n=== Using Pre-loaded DataFrames ===")
    print("DataFrames already loaded from CSV files")

#%% Create Interactive Plots with Plotly
print(f"\n=== Creating Interactive Plots ===")

# Create individual plots for each theta value
def create_interactive_plot_for_theta(theta, save_html=True):
    """Create an interactive plot for a specific theta value."""
    fig = go.Figure()
    
    # Get model names from probability_results instead of models dict
    model_names = list(probability_results[theta].keys()) if theta in probability_results else []
    
    # Sort models by architectural parameters (d_model, d_head, d_mlp)
    def sort_key(model_name):
        # Extract d_model, d_head, d_mlp from model name like "d8_h4_mlp128"
        parts = model_name.split('_')
        d_model = int(parts[0][1:])  # Remove 'd' prefix
        d_head = int(parts[1][1:])   # Remove 'h' prefix  
        d_mlp = int(parts[2][3:])    # Remove 'mlp' prefix
        return (d_model, d_head, d_mlp)
    
    model_names = sorted(model_names, key=sort_key)
    
    # Color palette for models
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    if len(model_names) > len(colors):
        colors = colors * (len(model_names) // len(colors) + 1)
    
    # Add traces for each model
    for j, model_name in enumerate(model_names):
        if model_name in probability_results[theta]:
            predictions = probability_results[theta][model_name]
            positions = np.arange(1, len(predictions) + 1)
            
            # Get model info for legend
            config_col = 'config_name' if 'config_name' in selected_models.columns else 'model_name'
            row = selected_models[selected_models[config_col] == model_name].iloc[0]
            legend_name = f"{model_name}<br>d_model={row['d_model']}, d_head={row['d_head']}, d_mlp={row['d_mlp']}"
            
            fig.add_trace(go.Scatter(
                x=positions,
                y=predictions,
                mode='lines+markers',
                name=legend_name,
                line=dict(width=2, color=colors[j % len(colors)]),
                marker=dict(size=4),
                hovertemplate=f'<b>{model_name}</b><br>' +
                             'Position: %{x}<br>' +
                             'Probability: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
    
    # Add theoretical Bayesian updating line
    # For Bayesian updating with Beta(1,1) prior
    # Compute theoretical updates for the same sequences used for model predictions
    all_theoretical_probs = []
    for sequence in all_sequences[theta]:
        sequence_theoretical_probs = []
        # Start with Beta(1,1) prior: alpha=1, beta=1
        alpha, beta = 1, 1
        
        # Account for BOS token at position 0 (token ID 2)
        # Start from position 1 (first actual data token) and predict from position 2 onwards
        for pos in range(2, len(sequence)):
            # Update posterior based on observed token at position pos-1
            # Note: sequence[pos-1] is the previous actual data token (not BOS)
            if sequence[pos-1] == 1:
                alpha += 1
            else:
                beta += 1
            
            # Posterior mean for Beta(alpha, beta) is alpha / (alpha + beta)
            posterior_mean = alpha / (alpha + beta)
            sequence_theoretical_probs.append(posterior_mean)
        
        all_theoretical_probs.append(sequence_theoretical_probs)
    
    # Average theoretical probabilities across sequences (same as done for model predictions)
    theoretical_probs = np.mean(all_theoretical_probs, axis=0)
    positions = np.arange(1, len(theoretical_probs) + 1)
    
    fig.add_trace(go.Scatter(
        x=positions,
        y=theoretical_probs,
        mode='lines',
        name='Theoretical Bayesian',
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate='Theoretical<br>Position: %{x}<br>Probability: %{y:.4f}<extra></extra>'
    ))
    
    # Determine actual number of sequences used for this theta
    actual_sequences = len(all_sequences[theta])
    sequences_text = f"{actual_sequences} sequence{'s' if actual_sequences > 1 else ''}"
    if theta in [0.0, 1.0]:
        sequences_text += " (deterministic case)"
    
    fig.update_layout(
        title=f'Probability Updates for θ = {theta:.1f}<br>Average over {sequences_text}',
        xaxis_title='Position in Sequence',
        yaxis_title='Predicted Probability of Next Token = 1',
        hovermode='closest',
        width=900,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    # Add horizontal line at true theta
    fig.add_hline(y=theta, line_dash="dot", line_color="black", 
                  annotation_text=f"True θ = {theta:.1f}")
    
    if save_html:
        filename = f"probability_updates_theta_{theta:.1f}.html"
        fig.write_html(filename)
        print(f"Saved plot to {filename}")
    
    fig.show()
    return fig

# Create plots for all theta values
interactive_figs = {}
for theta in theta_values:
    print(f"Creating plot for theta={theta:.1f}")
    interactive_figs[theta] = create_interactive_plot_for_theta(theta, save_html=True)

#%% Create Master Interactive Plot with Theta Slider
print(f"\n=== Creating Master Interactive Plot ===")

def create_master_interactive_plot():
    """Create a master plot with slider to toggle between theta values."""
    
    # Calculate optimal grid layout for theta values
    n_theta = len(theta_values)
    if n_theta <= 3:
        rows, cols = 1, n_theta
    elif n_theta <= 6:
        rows, cols = 2, 3
    elif n_theta <= 9:
        rows, cols = 3, 3
    else:
        rows = int(np.ceil(np.sqrt(n_theta)))
        cols = int(np.ceil(n_theta / rows))
    
    # Create subplots - one for each theta
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'θ = {theta:.1f}' for theta in theta_values],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Get all unique model names from probability_results
    all_model_names = set()
    for theta in theta_values:
        if theta in probability_results:
            all_model_names.update(probability_results[theta].keys())
    
    # Sort models by architectural parameters (d_model, d_head, d_mlp)
    def sort_key(model_name):
        # Extract d_model, d_head, d_mlp from model name like "d8_h4_mlp128"
        parts = model_name.split('_')
        d_model = int(parts[0][1:])  # Remove 'd' prefix
        d_head = int(parts[1][1:])   # Remove 'h' prefix  
        d_mlp = int(parts[2][3:])    # Remove 'mlp' prefix
        return (d_model, d_head, d_mlp)
    
    all_model_names = sorted(list(all_model_names), key=sort_key)
    
    # Color palette for models - extend with additional color palettes if needed
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    # Ensure we have enough colors for all models
    if len(all_model_names) > len(colors):
        colors = colors * (len(all_model_names) // len(colors) + 1)
    
    for i, theta in enumerate(theta_values):
        row = i // cols + 1
        col = i % cols + 1
        
        for j, model_name in enumerate(all_model_names):
            if model_name in probability_results[theta]:
                predictions = probability_results[theta][model_name]
                positions = np.arange(1, len(predictions) + 1)
                
                # Get model info
                config_col = 'config_name' if 'config_name' in selected_models.columns else 'model_name'
                row_data = selected_models[selected_models[config_col] == model_name].iloc[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=positions,
                        y=predictions,
                        mode='lines+markers',
                        name=f"{model_name} (d{row_data['d_model']}_h{row_data['d_head']})",
                        legendgroup=model_name,  # Group traces by model name
                        line=dict(width=2, color=colors[j % len(colors)]),
                        marker=dict(size=3),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=f'<b>{model_name}</b><br>' +
                                     'Position: %{x}<br>' +
                                     'Probability: %{y:.4f}<br>' +
                                     '<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        # Add theoretical Bayesian updating line for each subplot
        # Compute theoretical updates for the same sequences used for model predictions
        all_theoretical_probs = []
        for sequence in all_sequences[theta]:
            sequence_theoretical_probs = []
            # Start with Beta(1,1) prior: alpha=1, beta=1
            alpha, beta = 1, 1
            
            # Account for BOS token at position 0 (token ID 2)
            # Start from position 1 (first actual data token) and predict from position 2 onwards
            for pos in range(2, len(sequence)):
                # Update posterior based on observed token at position pos-1
                # Note: sequence[pos-1] is the previous actual data token (not BOS)
                if sequence[pos-1] == 1:
                    alpha += 1
                else:
                    beta += 1
                
                # Posterior mean for Beta(alpha, beta) is alpha / (alpha + beta)
                posterior_mean = alpha / (alpha + beta)
                sequence_theoretical_probs.append(posterior_mean)
            
            all_theoretical_probs.append(sequence_theoretical_probs)
        
        # Average theoretical probabilities across sequences (same as done for model predictions)
        if all_theoretical_probs:
            theoretical_probs = np.mean(all_theoretical_probs, axis=0)
            positions = np.arange(1, len(theoretical_probs) + 1)
            
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=theoretical_probs,
                    mode='lines',
                    name='Theoretical Bayesian',
                    legendgroup='theoretical',  # Group all theoretical lines together
                    line=dict(color='red', width=3, dash='dash'),
                    showlegend=(i == 0),  # Only show legend for first subplot
                    hovertemplate='Theoretical<br>Position: %{x}<br>Probability: %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Add horizontal line at true theta for each subplot
        fig.add_hline(y=theta, line_dash="dot", line_color="black", 
                      row=row, col=col, opacity=0.7)
    
    fig.update_layout(
        title='Probability Updates Across Different θ Values<br>Models from Dimension Sweep Training',
        height=900,
        width=1200,
        hovermode='closest'
    )
    
    # Update all subplot axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Only update if subplot exists (handles cases where grid has empty cells)
            subplot_index = (i - 1) * cols + (j - 1)
            if subplot_index < len(theta_values):
                fig.update_xaxes(title_text='Position in Sequence', row=i, col=j)
                fig.update_yaxes(title_text='Prob(next=1)', row=i, col=j)
    
    fig.write_html("master_probability_updates.html")
    print("Saved master plot to master_probability_updates.html")
    fig.show()
    return fig

master_fig = create_master_interactive_plot()

#%% Summary Analysis
print(f"""
=== Dimension Bottleneck Analysis Summary ===

Loaded Models from Dimension Sweep:
""")
for _, row in selected_models.iterrows():
    print(f"{row['config_name']:15s}: d_model={row['d_model']}, d_head={row['d_head']}, d_mlp={row['d_mlp']}, ratio={row['log_loss_ratio_clean']:.4f}")

# Calculate total sequences accounting for edge cases
total_sequences = sum(len(all_sequences[theta]) for theta in theta_values)
print(f"""
Generated {total_sequences} test sequences across {len(theta_values)} theta values
Theta values: {[f"{t:.1f}" for t in theta_values]}
- Regular cases (θ=0.1-0.9): {num_sequences} sequences each
- Edge cases (θ=0.0, 1.0): 1 sequence each (deterministic)

Key Findings:
- Interactive plots show how different architectures handle probability updating
- Models with lower log-loss ratios should track theoretical Bayesian updating better
- Edge cases (θ=0,1) provide insight into deterministic behavior
- Theoretical Bayesian lines (red dashed) provide ground truth for comparison
- HTML files saved for interactive exploration

Files generated:
- probability_updates_theta_X.X.html (individual plots for each theta with theoretical lines)
- master_probability_updates.html (overview of all theta values with theoretical lines)
- dimension_bottleneck_predictions.csv (NESTED: model predictions arrays by theta/model)
- dimension_bottleneck_theoretical.csv (NESTED: theoretical predictions by theta)
- dimension_bottleneck_sequences.csv (NESTED: all sequences grouped by theta)
- dimension_bottleneck_summary.csv (summary statistics by model)
- data_helpers.py (functions to expand nested DataFrames to long format)

Data Structure:
- Predictions DF: {len(predictions_df)} rows (one per model-theta combination)
- Theoretical DF: {len(theoretical_df)} rows (one per theta)
- Sequences DF: {len(sequences_df)} rows (one per theta)
- Summary DF: {len(summary_df)} rows (one per model)

This nested structure dramatically reduces file size and redundancy while maintaining
all information. Use data_helpers.py functions to expand to long format when needed.
""")
# %%
