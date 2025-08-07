# Coinformer Experiments

This repository contains experiments on transformer models learning Bayesian updating for coin flip sequences.

## Project Structure

```
├── core/                          # Core modules (clean, reusable)
│   ├── models.py                  # Model architectures and configs
│   ├── config.py                  # Experiment configurations  
│   ├── training.py                # Training utilities
│   ├── samplers.py                # Data generation
│   └── plotting.py                # Visualization utilities
├── experiments/                   # Main experiment scripts
│   ├── exp_01_bayesian_updating.py    # Basic Bayesian updating verification
│   ├── exp_02_dimension_bottleneck.py # Architecture analysis (d_model, d_head)
│   ├── exp_03_permutation_invariance.py # Residual permutation invariance
│   └── exp_04_importance_sampling.py   # Training efficiency with importance sampling
├── utils.py                       # General utilities
└── saved_models/                  # Saved model checkpoints
```

## Key Features

### Model Architecture Options
- **Vocabulary size**: 2 (tokens: 0,1) or 3 (tokens: BOS,0,1) 
- **Positional embeddings**: Configurable on/off
- **Architecture variants**: Attention-only, with MLP, different layer counts
- **Dimension analysis**: Configurable d_model and d_head for bottleneck studies

### Experiment Configurations
Each experiment has clean, dataclass-based configurations for:
- Model architecture parameters
- Training hyperparameters  
- Prior distribution parameters (α, β)
- Importance sampling settings
- BOS token and positional embedding options

## Running Experiments

Each experiment script is self-contained with `#%%` cell structure for interactive execution:

```python
# Example: Run basic Bayesian updating experiment
python experiments/exp_01_bayesian_updating.py

# Or run interactively in your IDE with cell execution
```

### Experiment 1: Bayesian Updating
Tests whether the model learns proper Bayesian posterior updating by comparing predictions against theoretical values.

### Experiment 2: Dimension Bottleneck  
Analyzes how different architectural choices (d_model, d_head) affect Bayesian updating performance.

### Experiment 3: Permutation Invariance
Tests whether residual representations are permutation invariant for sequences with the same number of 1s.

### Experiment 4: Importance Sampling
Compares standard training vs importance sampling where training and evaluation distributions differ.

## Configuration Examples

```python
from core.config import ExperimentConfig
from core.models import ModelConfig

# Basic configuration
config = ExperimentConfig(
    model_config=ModelConfig(d_model=64, d_head=32),
    alpha=1.0, beta=1.0,
    num_epochs=5, num_batches=1000
)

# BOS token configuration  
bos_config = ExperimentConfig(
    model_config=ModelConfig(use_bos_token=True),
    alpha=1.0, beta=1.0
)

# Importance sampling
importance_config = ExperimentConfig(
    importance_sampling=True,
    alpha=1.0, beta=1.0,  # Proposal distribution
    importance_sampling_alpha=1.0,  # Target distribution 
    importance_sampling_beta=8.0
)
```

## Model Saving/Loading

Models are automatically saved with descriptive filenames including architecture and training parameters:

```
bayesian_updating_dmodel64_dhead32_layers2_alpha1.0_beta1.0.pt
importance_sampling_dmodel64_dhead32_layers2_alpha1.0_beta1.0_importance.pt
```

## Legacy Files

The original experiment files remain for reference but the new structure provides:
- ✅ No duplicate code
- ✅ Clean separation of concerns  
- ✅ Consistent #%% structure
- ✅ BOS token support
- ✅ Configurable architectures
- ✅ ML experiment best practices