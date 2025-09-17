"""
Experiment configurations for coinformer research.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from .models import ModelConfig, DEFAULT_CONFIG, ATTENTION_ONLY_CONFIG, BOS_TOKEN_CONFIG


@dataclass
class ExperimentConfig:
    """Configuration for coinformer experiments."""
    # Model configuration
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())
    
    # Prior parameters
    alpha: float = 1.0
    beta: float = 1.0
    
    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    seq_length: int = 100  # Base sequence length (BOS token adds +1 if use_bos_token=True)
    num_batches: int = 1000
    
    # Data parameters
    bernoulli: bool = False
    bernoulli_p: Optional[float] = None
    flip_batch: bool = False
    scale: float = 1.0
    bias: float = 0.0
    
    # Importance sampling
    importance_sampling: bool = False
    importance_sampling_alpha: Optional[float] = None
    importance_sampling_beta: Optional[float] = None
    
    # Evaluation parameters
    theta_values: np.ndarray = None
    
    def __post_init__(self):
        if self.theta_values is None:
            self.theta_values = np.linspace(0, 0.9, 10)
        if self.importance_sampling_alpha is None:
            self.importance_sampling_alpha = self.alpha
        if self.importance_sampling_beta is None:
            self.importance_sampling_beta = self.beta


# Experiment 1: Basic Bayesian Updating
BAYESIAN_UPDATING_CONFIG = ExperimentConfig(
    model_config=DEFAULT_CONFIG,
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
    seq_length=50,
)

# Experiment 2: Dimension Bottleneck Configs
SMALL_MODEL_CONFIG = ExperimentConfig(
    model_config=ModelConfig(d_model=32, d_head=16),
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
)

LARGE_MODEL_CONFIG = ExperimentConfig(
    model_config=ModelConfig(d_model=128, d_head=64),
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
)

BOTTLENECK_HEAD_CONFIG = ExperimentConfig(
    model_config=ModelConfig(d_model=64, d_head=16),
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
)

# Experiment 3: Permutation Invariance
PERMUTATION_CONFIG = ExperimentConfig(
    model_config=DEFAULT_CONFIG,
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
    seq_length=30,  # Shorter sequences for permutation analysis
)

# Experiment 4: Importance Sampling
IMPORTANCE_SAMPLING_CONFIG = ExperimentConfig(
    model_config=DEFAULT_CONFIG,
    alpha=1.0,  # Proposal distribution
    beta=1.0,
    importance_sampling=True,
    importance_sampling_alpha=1.0,  # Target distribution
    importance_sampling_beta=8.0,
    num_epochs=10,
    num_batches=2000,
)

# BOS Token Experiments
BOS_BAYESIAN_CONFIG = ExperimentConfig(
    model_config=BOS_TOKEN_CONFIG,
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
)

# No Positional Embedding Experiments
NO_POS_CONFIG = ExperimentConfig(
    model_config=ModelConfig(pos_embed_config=None),
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
)

# Attention-only experiments
ATTN_ONLY_CONFIG = ExperimentConfig(
    model_config=ATTENTION_ONLY_CONFIG,
    alpha=1.0,
    beta=1.0,
    num_epochs=5,
    num_batches=1000,
)