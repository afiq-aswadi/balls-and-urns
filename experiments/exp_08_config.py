"""
Configuration for exp_08: Multi-seed small transformer experiment
"""
import os
import torch
from datetime import datetime
from core.config import ExperimentConfig, ModelConfig
from core.models import PosEmbedConfig, PosEmbedType

# Model configuration
MODEL_CONFIG = ModelConfig(
    d_model=2,
    d_head=1,
    n_heads=1,
    d_mlp=16,
    n_layers=1,
    n_ctx = 20,
    use_bos_token=False,
    pos_embed_config=PosEmbedConfig(PosEmbedType.LOG, trainable=False) #set what positional embedding we want here
)

# Experiment configuration
EXP_CONFIG = ExperimentConfig(
    model_config=MODEL_CONFIG,
    alpha=1.0,
    beta=1.0,
    num_epochs=10,
    learning_rate=1e-3,
    batch_size=64,
    seq_length=21,
    num_batches=5000,
)

# Multi-seed setup
SEEDS = [100, 200, 300]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Directory configuration
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# EXP_RESULTS_DIR = f"/Users/afiqabdillah/balls-and-urns/results/exp_08_small_transformer/{TIMESTAMP}"
EXP_RESULTS_DIR = f"/Users/afiqabdillah/balls-and-urns/results/exp_08_small_transformer/20250904_122246"

def get_seed_checkpoint_dir(seed: int) -> str:
    """Get the checkpoint directory for a specific seed."""
    return os.path.join(EXP_RESULTS_DIR, "models", f"seed_{seed}")

def get_seed_results_dir(seed: int) -> str:
    """Get the results directory for a specific seed."""
    return os.path.join(EXP_RESULTS_DIR, "results", f"seed_{seed}")

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(EXP_RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXP_RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(EXP_RESULTS_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(EXP_RESULTS_DIR, "aggregated"), exist_ok=True)
    
    for seed in SEEDS:
        os.makedirs(get_seed_checkpoint_dir(seed), exist_ok=True)
        os.makedirs(get_seed_results_dir(seed), exist_ok=True)

def set_global_seed(seed: int) -> None:
    """Set global seed for reproducibility."""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # PyTorch deterministic algorithms where supported
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def build_title_suffix(exp_cfg: ExperimentConfig) -> str:
    """Build a descriptive title suffix for plots."""
    cfg = exp_cfg.model_config
    bos = "on" if cfg.use_bos_token else "off"
    return (
        f"d_model={cfg.d_model}, d_mlp={cfg.d_mlp}, layers={cfg.n_layers}, "
        f"BOS={bos}, seq_len={exp_cfg.seq_length}, alpha={exp_cfg.alpha}, beta={exp_cfg.beta}"
    )