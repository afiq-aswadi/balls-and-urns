"""
We train a small transformer across multiple seedsand see how embeddings/residuals develop across training. 
We denote H = 1 and T = 0
"""
#%%
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
from core.training import calculate_optimal_loss
from core.models import create_coinformer_model
from core.samplers import generate_data, generate_sequential_ones, add_bos_token
from core.utils import get_autoregressive_predictions


import glob
import re
from datetime import datetime

#%%

model_config = ModelConfig(
    d_model=2,
    d_head=1,
    n_heads=1,
    d_mlp=32,
    n_layers=1,
    use_bos_token=True)

#%% Training setup and loop (single small transformer, save checkpoint each epoch)
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Define an experiment configuration (single small model)
exp_config = ExperimentConfig(
    model_config=model_config,
    alpha=1.0,
    beta=1.0,
    num_epochs=10,
    learning_rate=1e-3,
    batch_size=64,
    # IMPORTANT: models trained elsewhere with BOS used seq_length=99 so total tokens = 100
    seq_length=10,
    num_batches=10000,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Global seed for reproducibility (override with env EXP07_SEED)
# SEED = int(os.getenv("EXP07_SEED", "1337"))
SEED = 100

# Create experiment results directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_RESULTS_DIR = f"/Users/afiqabdillah/balls-and-urns/results/exp_07_small_transformer/{TIMESTAMP}"
os.makedirs(EXP_RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {EXP_RESULTS_DIR}")


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def build_title_suffix(exp_cfg: ExperimentConfig) -> str:
    cfg = exp_cfg.model_config
    bos = "on" if cfg.use_bos_token else "off"
    return (
        f"d_model={cfg.d_model}, d_mlp={cfg.d_mlp}, layers={cfg.n_layers}, "
        f"BOS={bos}, seq_len={exp_cfg.seq_length}, alpha={exp_cfg.alpha}, beta={exp_cfg.beta}"
    )



def train_single_model_and_checkpoint(config: ExperimentConfig, checkpoint_dir: str, seed: int) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set seed before any model init or sampling
    set_global_seed(seed)

    # Create model
    model = create_coinformer_model(config.model_config).to(DEVICE)

    # Optimizer / loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Theoretical lower bound for reference
    theo_lower_bound = calculate_optimal_loss(config.alpha, config.beta)

    # Save initialization checkpoint (epoch 0)
    cfg0 = config.model_config
    ckpt_name0 = (
        f"small_transformer_epoch_0_"
        f"dmodel{cfg0.d_model}_dhead{cfg0.d_head}_layers{cfg0.n_layers}_"
        f"alpha{config.alpha}_beta{config.beta}_seed{seed}{'_bos' if cfg0.use_bos_token else ''}.pt"
    )
    ckpt_path0 = os.path.join(checkpoint_dir, ckpt_name0)
    if not os.path.exists(ckpt_path0):
        torch.save({
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_epoch_loss": None,
            "config": {
                "alpha": config.alpha,
                "beta": config.beta,
                "seed": seed,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "seq_length": config.seq_length,
                "num_batches": config.num_batches,
                "model_config": {
                    "d_model": cfg0.d_model,
                    "d_head": cfg0.d_head,
                    "n_heads": cfg0.n_heads,
                    "d_mlp": cfg0.d_mlp,
                    "n_layers": cfg0.n_layers,
                    "use_bos_token": cfg0.use_bos_token,
                    "use_pos_embed": cfg0.use_pos_embed,
                    "attn_only": cfg0.attn_only,
                },
            },
        }, ckpt_path0)
        print(f"Saved initialization checkpoint: {ckpt_path0}")

    for epoch in range(config.num_epochs):
        # Fresh data each epoch - SAMPLE p from Beta(alpha, beta)
        datasets, priors = generate_data(
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            num_batches=config.num_batches,
            alpha=config.alpha,
            beta=config.beta,
            bernoulli=False,
            flip_batch=False,
            use_bos_token=config.model_config.use_bos_token,
        )

        epoch_loss = 0.0
        progress = tqdm(zip(datasets, priors), total=len(datasets), desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for data_batch, _ in progress:
            data_batch = data_batch.to(DEVICE)

            # Inputs are all tokens except the last; targets are all except the first
            inputs = data_batch[:, :-1]
            targets = data_batch[:, 1:]

            logits = model(inputs)
            loss = criterion(logits.view(-1, model.cfg.d_vocab), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(datasets)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, theoretical_lower_bound={theo_lower_bound:.4f}")

        # Save checkpoint for this epoch
        cfg = config.model_config
        ckpt_name = (
            f"small_transformer_epoch_{epoch+1}_"
            f"dmodel{cfg.d_model}_dhead{cfg.d_head}_layers{cfg.n_layers}_"
            f"alpha{config.alpha}_beta{config.beta}_seed{seed}{'_bos' if cfg.use_bos_token else ''}.pt"
        )
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_epoch_loss": avg_loss,
            "config": {
                "alpha": config.alpha,
                "beta": config.beta,
                "seed": seed,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "seq_length": config.seq_length,
                "num_batches": config.num_batches,
                "model_config": {
                    "d_model": cfg.d_model,
                    "d_head": cfg.d_head,
                    "n_heads": cfg.n_heads,
                    "d_mlp": cfg.d_mlp,
                    "n_layers": cfg.n_layers,
                    "use_bos_token": cfg.use_bos_token,
                    "use_pos_embed": cfg.use_pos_embed,
                    "attn_only": cfg.attn_only,
                },
            },
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

#%%
def build_probe_batch(seq_length: int, use_bos: bool) -> torch.Tensor:
    # Lower triangular ones (ensure last token is 0 by excluding the all-ones row)
    lower_full = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))
    lower = lower_full[:-1, :]  # rows 0..seq_length-2 have last token = 0
    # Upper triangular ones (last token is always 1)
    upper = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.long))
    # All-zeros sequence (tails only)
    zeros_row = torch.zeros((1, seq_length), dtype=torch.long)
    # Order: zeros, lower (last=0), upper (last=1)
    batch = torch.cat([zeros_row, lower, upper], dim=0)
    if use_bos:
        batch = add_bos_token(batch)
    return batch

#%%

if __name__ == "__main__":
    checkpoint_dir = os.path.join(EXP_RESULTS_DIR, "models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # do_train = os.getenv("EXP07_SKIP_TRAIN", "0") != "1"
    do_train = True
    if do_train:
        # Ensure seed is set in driver as well
        set_global_seed(SEED)
        train_single_model_and_checkpoint(exp_config, checkpoint_dir, SEED)

    # After training: build epoch-wise dataframe from checkpoints and create plots



# %%
