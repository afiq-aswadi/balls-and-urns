"""
Experiment configuration and common parameters for coinformer experiments.
"""
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@dataclass
class ExperimentConfig:
    """Configuration for coinformer experiments."""
    # Model parameters
    d_model: int = 64
    d_head: int = 32
    n_layers: int = 2
    n_ctx: int = 100
    d_vocab: int = 2
    act_fn: str = "relu"
    attn_only: bool = False
    
    # Prior parameters
    alpha: float = 1.0
    beta: float = 1.0
    
    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    seq_length: int = 100
    num_batches: int = 10000
    
    # Data parameters
    bernoulli: bool = False
    bernoulli_p: Optional[float] = None
    pos_embed: bool = True
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

# Common experiment configurations
DEFAULT_CONFIG = ExperimentConfig()

IMPORTANCE_SAMPLING_CONFIG = ExperimentConfig(
    importance_sampling=True,
    alpha=1.0,
    beta=1.0,
    importance_sampling_alpha=1.0,
    importance_sampling_beta=8.0
)

ATTENTION_ONLY_CONFIG = ExperimentConfig(
    d_model=64,
    d_head=64,
    n_layers=1,
    attn_only=True
)