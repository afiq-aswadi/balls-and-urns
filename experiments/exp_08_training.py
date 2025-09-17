"""
Training script for exp_08: Multi-seed small transformer experiment
Trains models across multiple seeds with checkpointing at each epoch.
"""
#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from core.config import ExperimentConfig
from core.training import calculate_optimal_loss
from core.models import create_coinformer_model
from core.samplers import generate_data

from exp_08_config import (
    EXP_CONFIG, SEEDS, DEVICE, EXP_RESULTS_DIR,
    get_seed_checkpoint_dir, ensure_directories, set_global_seed
)


def train_single_model_and_checkpoint(config: ExperimentConfig, checkpoint_dir: str, seed: int) -> None:
    """Train a single model with checkpointing at each epoch."""
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
                    "pos_embed_config": cfg0.pos_embed_config,
                    "attn_only": cfg0.attn_only,
                },
            },
        }, ckpt_path0)
        print(f"Saved initialization checkpoint: {ckpt_path0}")
    
    # Training loop
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
        progress = tqdm(zip(datasets, priors), total=len(datasets), 
                       desc=f"Seed {seed} - Epoch {epoch+1}/{config.num_epochs}")
        
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
        print(f"Seed {seed} - Epoch {epoch+1}: avg_loss={avg_loss:.4f}, theoretical_lower_bound={theo_lower_bound:.4f}")
        
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
                    "pos_embed_config": cfg.pos_embed_config,
                    "attn_only": cfg.attn_only,
                },
            },
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

#%%
def main():
    """Main training loop across all seeds."""
    ensure_directories()
    print(f"Results will be saved to: {EXP_RESULTS_DIR}")
    
    print(f"Training models for seeds: {SEEDS}")
    print(f"Configuration: {EXP_CONFIG}")
    print(f"Positional embedding config: {EXP_CONFIG.model_config.pos_embed_config}")
    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"Training seed {seed}")
        print(f"{'='*50}")
        
        checkpoint_dir = get_seed_checkpoint_dir(seed)
        train_single_model_and_checkpoint(EXP_CONFIG, checkpoint_dir, seed)
        
        print(f"Completed training for seed {seed}")
    
    print(f"\n{'='*50}")
    print("All seeds completed!")
    print(f"Results saved to: {EXP_RESULTS_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
# %%
