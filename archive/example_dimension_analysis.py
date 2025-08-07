#!/usr/bin/env python3
"""
Example script demonstrating the dimension analysis experiment.

This script shows how to use the dimension analysis functionality to test
whether low embedding dimensions and attention head dimensions are sufficient
for computing sufficient statistics in Bayesian updating tasks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dimension_analysis import (
    run_dimension_experiment,
    run_quick_comparison_experiment,
    plot_experiment_results,
    create_model_config,
    generate_consistent_training_data,
    train_coinformer_model_with_data,
    evaluate_model_bayesian_performance,
    plot_individual_model_analysis
)

def example_1_quick_comparison():
    """
    Example 1: Run a quick comparison experiment with 4 different model sizes.
    This is useful for getting initial insights quickly.
    """
    print("="*60)
    print("EXAMPLE 1: Quick Comparison Experiment")
    print("="*60)
    
    print("Running quick comparison experiment...")
    results = run_quick_comparison_experiment()
    
    print("\nResults Summary:")
    for result in results:
        print(f"  {result['name']}:")
        print(f"    Parameters: {result['total_params']:,}")
        print(f"    Final Loss: {result['losses'][-1]:.4f}")
        print(f"    Mean Error: {result['metrics']['mean_error']:.4f}")
        print(f"    Avg KL Divergence: {result['metrics']['avg_kl_divergence']:.4f}")
    
    return results

def example_2_single_model_analysis():
    """
    Example 2: Analyze a single model in detail.
    This shows how to create and analyze a specific model configuration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Single Model Analysis")
    print("="*60)
    
    # Create a specific model configuration
    config = create_model_config(d_model=16, d_head=4, d_mlp=128)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 3)
    )
    model.cfg = config  # Mock the config for demonstration
    
    print(f"Created model with config:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_head: {config.d_head}")
    print(f"  d_mlp: {config.d_mlp}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate training data
    datasets, priors = generate_consistent_training_data(
        batch_size=16, seq_length=15, num_batches=20, seed=42
    )
    
    print(f"Generated {len(datasets)} training batches")
    print(f"Data shape: {datasets[0].shape}")
    
    return model, datasets, priors

def example_3_custom_experiment():
    """
    Example 3: Run a custom experiment with specific dimension values.
    This shows how to customize the experiment parameters.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Experiment")
    print("="*60)
    
    # Define custom dimension values
    d_model_values = [32, 16, 8]  # Test embedding dimensions
    d_head_values = [8, 4]        # Test attention head dimensions
    d_mlp_values = [128, 64]      # Test MLP dimensions
    
    print("Running custom experiment with:")
    print(f"  d_model values: {d_model_values}")
    print(f"  d_head values: {d_head_values}")
    print(f"  d_mlp values: {d_mlp_values}")
    
    # Run experiment with smaller parameters for faster execution
    results = run_dimension_experiment(
        d_model_values=d_model_values,
        d_head_values=d_head_values,
        d_mlp_values=d_mlp_values,
        num_epochs=2,           # Fewer epochs for faster execution
        batch_size=16,          # Smaller batch size
        seq_length=15,          # Shorter sequences
        num_batches=20,         # Fewer batches
        test_theta=0.5          # Test probability
    )
    
    # Plot results
    df = plot_experiment_results(results)
    
    return results, df

def example_4_parameter_analysis():
    """
    Example 4: Analyze the relationship between model parameters and performance.
    This shows how to extract insights from the experiment results.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Parameter Analysis")
    print("="*60)
    
    # Run a small experiment first
    results = run_dimension_experiment(
        d_model_values=[32, 16, 8],
        d_head_values=[8, 4],
        d_mlp_values=[128, 64],
        num_epochs=2,
        batch_size=16,
        seq_length=15,
        num_batches=20,
        test_theta=0.5
    )
    
    # Extract data for analysis
    configs = results['configs']
    performances = results['bayesian_performance']
    
    print("Parameter-Performance Analysis:")
    print("-" * 40)
    
    # Analyze by d_model
    d_model_groups = {}
    for i, config in enumerate(configs):
        d_model = config['d_model']
        if d_model not in d_model_groups:
            d_model_groups[d_model] = []
        d_model_groups[d_model].append(performances[i]['mean_error'])
    
    print("Performance by d_model:")
    for d_model, errors in d_model_groups.items():
        avg_error = np.mean(errors)
        print(f"  d_model={d_model}: avg_error={avg_error:.4f}")
    
    # Analyze by d_head
    d_head_groups = {}
    for i, config in enumerate(configs):
        d_head = config['d_head']
        if d_head not in d_head_groups:
            d_head_groups[d_head] = []
        d_head_groups[d_head].append(performances[i]['mean_error'])
    
    print("\nPerformance by d_head:")
    for d_head, errors in d_head_groups.items():
        avg_error = np.mean(errors)
        print(f"  d_head={d_head}: avg_error={avg_error:.4f}")
    
    return results

def main():
    """
    Main function that runs all examples.
    """
    print("Dimension Analysis Experiment Examples")
    print("="*60)
    print("This script demonstrates how to use the dimension analysis functionality.")
    print("Choose which example to run:")
    print("1. Quick comparison experiment")
    print("2. Single model analysis")
    print("3. Custom experiment")
    print("4. Parameter analysis")
    print("5. Run all examples")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        example_1_quick_comparison()
    elif choice == "2":
        example_2_single_model_analysis()
    elif choice == "3":
        example_3_custom_experiment()
    elif choice == "4":
        example_4_parameter_analysis()
    elif choice == "5":
        print("Running all examples...")
        example_1_quick_comparison()
        example_2_single_model_analysis()
        example_3_custom_experiment()
        example_4_parameter_analysis()
    else:
        print("Invalid choice. Running quick comparison...")
        example_1_quick_comparison()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("Check the generated plots and results for insights.")

if __name__ == "__main__":
    main() 