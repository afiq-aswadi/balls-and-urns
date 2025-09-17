"""
Data collection script for exp_08: Multi-seed small transformer experiment
Processes checkpoints from all seeds and creates aggregated datasets.
"""

#%%

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import glob
import re
from typing import List, Tuple

from core.models import create_coinformer_model
from core.config import ModelConfig
from core.samplers import add_bos_token

from exp_08_config import (
    EXP_CONFIG, SEEDS, DEVICE,
    get_seed_checkpoint_dir, get_seed_results_dir, set_global_seed, EXP_RESULTS_DIR
)


def build_probe_batch(seq_length: int, use_bos: bool) -> torch.Tensor:
    """Build probe batch for analysis.""" #TODO: seems add bos token messes up with seq_length?
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


def load_checkpoints_sorted(ckpt_dir: str) -> List[str]:
    """Load and sort checkpoints by epoch number."""
    pattern = os.path.join(ckpt_dir, "small_transformer_epoch_*.pt")
    files = glob.glob(pattern)
    
    def epoch_num(fp):
        m = re.search(r"epoch_(\d+)_", os.path.basename(fp))
        return int(m.group(1)) if m else -1
    
    return sorted(files, key=epoch_num)


def filter_checkpoints_by_config(ckpt_paths: List[str], exp_cfg, seed: int) -> List[str]:
    """Keep only checkpoints whose saved config matches the current experiment config and seed."""
    matched = []
    for p in ckpt_paths:
        try:
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            cfg = ck.get("config", {})
            mc = cfg.get("model_config", {})
   
            if (
                cfg.get("alpha") == exp_cfg.alpha and
                cfg.get("beta") == exp_cfg.beta and
                cfg.get("seed") == seed and
                cfg.get("seq_length") == exp_cfg.seq_length and
                mc.get("d_model") == exp_cfg.model_config.d_model and
                mc.get("d_head") == exp_cfg.model_config.d_head and
                mc.get("n_layers") == exp_cfg.model_config.n_layers and
                mc.get("use_bos_token") == exp_cfg.model_config.use_bos_token and
                # Skip pos_embed_config check - old checkpoints use different format
                mc.get("attn_only") == exp_cfg.model_config.attn_only and
                mc.get("d_mlp") == exp_cfg.model_config.d_mlp
            ):
                matched.append(p)
                print(f"    -> MATCHED")
            else:
                print(f"    -> NO MATCH")
        except Exception as e:
            print(f"  Error loading {os.path.basename(p)}: {e}")
            continue
    return matched


def compute_token_probabilities(residuals: torch.Tensor, model) -> np.ndarray:
    """Compute token probabilities from residuals using the unembedding layer."""
    with torch.no_grad():
        logits = model.unembed(residuals)  # [B, T, d_vocab]
        probs = torch.softmax(logits, dim=-1)  # [B, T, d_vocab]
        # Return probability of token 1 (heads)
        return probs[:, :, 1].detach().cpu().numpy()


def build_improved_epoch_dataframe(checkpoints: List[str], model_cfg: ModelConfig, seed: int) -> pd.DataFrame:
    """Build improved epoch dataframe with combined pre/post MLP data and probability distributions."""
    records = []
    
    # Probe inputs
    seq_len = EXP_CONFIG.seq_length
    if model_cfg.use_bos_token:
        batch_len = seq_len - 1
    else:
        batch_len = seq_len - 1
    batch = build_probe_batch(batch_len,  use_bos=model_cfg.use_bos_token).to(DEVICE) #note: -1 because of bos
    B, T = batch.shape
    
    # Pre-compute N and H
    # N should be 1-indexed position (so first real token is N=1)
    pos_idx = torch.arange(T, device=batch.device).unsqueeze(0).expand(B, -1) + 1  # Now 1-indexed
    
    # H should include the current token in the count
    tokens_for_count = batch.clone()
    if model_cfg.use_bos_token:
        tokens_for_count[:, 0] = 0  # BOS doesn't count toward H
    H_inclusive = torch.cumsum(tokens_for_count, dim=1)  # This now includes current token
    
    # Per-sequence source labels: 'zeros' (0), 'lower' (1), 'upper' (2)
    zeros_count = 1
    lower_count = seq_len - 1
    upper_count = seq_len
    source_seq = torch.empty((B,), dtype=torch.long, device=batch.device)
    source_seq[:zeros_count] = 0
    source_seq[zeros_count:zeros_count+lower_count] = 1
    source_seq[zeros_count+lower_count:] = 2
    source_mat = source_seq.unsqueeze(1).expand(-1, T)
    
    # Last token per sequence (exclude BOS): simply the last column of batch
    last_token_seq = batch[:, -1].clone()
    last_token_mat = last_token_seq.unsqueeze(1).expand(-1, T)
    
    # Token observed at each (B, T) position (includes BOS at col 0 if present)
    token_at_pos = batch.clone()
    
    # We only keep positions t >= 1 when BOS is used (BOS is at t=0, N=1)
    valid_mask = torch.ones_like(pos_idx, dtype=torch.bool)
    if model_cfg.use_bos_token:
        valid_mask[:, 0] = False  # Skip BOS position (t=0, but N would be 1)
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        epoch = ckpt.get("epoch", None)
        
        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        with torch.no_grad():
            _, cache = model.run_with_cache(batch)
        
        pre_attn = cache["resid_pre", 0] # [B, T, d_model]
        # Pre-MLP (after attention, before MLP) for layer 0
        pre_mlp = cache["resid_mid", 0]  # [B, T, d_model]
        # Post-MLP for layer 0
        post_mlp = cache["resid_post", 0]  # [B, T, d_model]
        
        # Compute probabilities for heatmap
        pre_mlp_probs = compute_token_probabilities(pre_mlp, model)  # [B, T]
        post_mlp_probs = compute_token_probabilities(post_mlp, model)  # [B, T]
        
        # Flatten and collect - now combining pre_mlp and post_mlp in single rows
        valid_indices = torch.where(valid_mask)
        for i, (b_idx, t_idx) in enumerate(zip(valid_indices[0], valid_indices[1])):
            b_idx, t_idx = int(b_idx), int(t_idx)
            
            # Get coordinates for this position
            pre_attn_x, pre_attn_y = float(pre_attn[b_idx, t_idx, 0]), float(pre_attn[b_idx, t_idx, 1])
            pre_x, pre_y = float(pre_mlp[b_idx, t_idx, 0]), float(pre_mlp[b_idx, t_idx, 1])
            post_x, post_y = float(post_mlp[b_idx, t_idx, 0]), float(post_mlp[b_idx, t_idx, 1])
            
            # Get probability distributions
            pre_prob_token1 = float(pre_mlp_probs[b_idx, t_idx])
            post_prob_token1 = float(post_mlp_probs[b_idx, t_idx])


            records.append({
                "seed": seed,
                "epoch": epoch,
                "N": int(pos_idx[b_idx, t_idx]),
                "H": int(H_inclusive[b_idx, t_idx]),
                "source": {0: "zeros", 1: "lower", 2: "upper"}[int(source_mat[b_idx, t_idx])],
                # "final_token": int(last_token_mat[b_idx, t_idx]) if t_idx != b_idx - 1 else 4
                "final_token": int(token_at_pos[b_idx, t_idx]),
                "token_at_N": int(token_at_pos[b_idx, t_idx]),
                # Pre-attention residuals
                "pre_attn_x": pre_attn_x,
                "pre_attn_y": pre_attn_y,
                # Pre-MLP residuals
                "pre_mlp_x": pre_x,
                "pre_mlp_y": pre_y,
                "pre_mlp_prob_token1": pre_prob_token1,
                # Post-MLP residuals
                "post_mlp_x": post_x,
                "post_mlp_y": post_y,
                "post_mlp_prob_token1": post_prob_token1,
            })
    
    return pd.DataFrame.from_records(records)


def build_embed_unembed_trajectories(checkpoints: List[str], model_cfg: ModelConfig, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect token embedding and unembedding vectors across epochs."""
    emb_records = []
    unemb_records = []
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        epoch = ckpt.get("epoch", None)
        
        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        with torch.no_grad():
            W_E = model.embed.W_E.detach().cpu().numpy()  # [d_vocab, d_model]
            W_U = model.unembed.W_U.detach().cpu().numpy()  # [d_model, d_vocab]
        
        d_vocab = model.cfg.d_vocab
        # Plot tokens 0,1 and BOS (id=2) if present
        tokens_of_interest = [0, 1] + ([2] if d_vocab > 2 else [])
        
        for tok in tokens_of_interest:
            if tok < d_vocab and model.cfg.d_model >= 2:
                x_e, y_e = float(W_E[tok, 0]), float(W_E[tok, 1])
                emb_records.append({
                    "seed": seed,
                    "epoch": epoch,
                    "token_id": tok,
                    "token_name": {0: "0", 1: "1", 2: "BOS"}.get(tok, str(tok)),
                    "x": x_e,
                    "y": y_e,
                })
                x_u, y_u = float(W_U[0, tok]), float(W_U[1, tok])
                unemb_records.append({
                    "seed": seed,
                    "epoch": epoch,
                    "token_id": tok,
                    "token_name": {0: "0", 1: "1", 2: "BOS"}.get(tok, str(tok)),
                    "x": x_u,
                    "y": y_u,
                })
    
    return pd.DataFrame.from_records(emb_records), pd.DataFrame.from_records(unemb_records)


def build_unembed_bias_diffs(checkpoints: List[str], model_cfg: ModelConfig, seed: int) -> pd.DataFrame:
    """Collect unembed bias differences (b_U[1] - b_U[0]) across epochs."""
    records = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        epoch = ckpt.get("epoch", None)

        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            b_U = model.unembed.b_U.detach().cpu().numpy()  # [d_vocab]
            bias_diff = float(b_U[1] - b_U[0]) if b_U.shape[0] >= 2 else 0.0

        records.append({
            "seed": seed,
            "epoch": epoch,
            "bias_diff_1_minus_0": bias_diff,
        })

    return pd.DataFrame.from_records(records)


def build_positional_embed_trajectories(checkpoints: List[str], model_cfg: ModelConfig, seed: int, max_positions: int) -> pd.DataFrame:
    """Collect positional embedding vectors across epochs for positions [0, max_positions)."""
    records = []
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        epoch = ckpt.get("epoch", None)
        
        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        with torch.no_grad():
            W_pos = model.pos_embed.W_pos.detach().cpu().numpy()  # [n_ctx, d_model]
        
        n_ctx = W_pos.shape[0]
        d_model = W_pos.shape[1]
        limit = min(max_positions, n_ctx)
        
        if d_model < 2:
            continue
            
        for pos in range(limit):
            x, y = float(W_pos[pos, 0]), float(W_pos[pos, 1])
            records.append({
                "seed": seed,
                "epoch": epoch,
                "position": pos,
                "x": x,
                "y": y,
            })
    
    return pd.DataFrame.from_records(records)


def process_seed(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process all data for a single seed."""
    print(f"Processing seed {seed}...")
    
    checkpoint_dir = get_seed_checkpoint_dir(seed)
    all_ckpts = load_checkpoints_sorted(checkpoint_dir)
    ckpts = filter_checkpoints_by_config(all_ckpts, EXP_CONFIG, seed)
    
    if len(ckpts) != len(all_ckpts):
        print(f"  Filtered checkpoints: using {len(ckpts)} of {len(all_ckpts)} that match current config/seed.")
    
    if not ckpts:
        print(f"  No valid checkpoints found for seed {seed}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Build improved epoch dataframe
    print(f"  Building epoch dataframe...")
    df_epoch = build_improved_epoch_dataframe(ckpts, EXP_CONFIG.model_config, seed)
    
    # Build embedding/unembedding trajectories
    print(f"  Building token trajectories...")
    emb_df, unemb_df = build_embed_unembed_trajectories(ckpts, EXP_CONFIG.model_config, seed)
    
    # Build positional embedding trajectories
    print(f"  Building positional embedding trajectories...")
    max_positions = EXP_CONFIG.seq_length + (1 if EXP_CONFIG.model_config.use_bos_token else 0)
    pos_df = build_positional_embed_trajectories(ckpts, EXP_CONFIG.model_config, seed, max_positions)
    
    # Build unembed bias differences
    print(f"  Building unembed bias diffs...")
    bias_df = build_unembed_bias_diffs(ckpts, EXP_CONFIG.model_config, seed)
    
    # Save individual seed results
    results_dir = get_seed_results_dir(seed)
    df_epoch.to_csv(os.path.join(results_dir, "epoch_data.csv"), index=False)
    emb_df.to_csv(os.path.join(results_dir, "embed_trajectories.csv"), index=False)
    unemb_df.to_csv(os.path.join(results_dir, "unembed_trajectories.csv"), index=False)
    pos_df.to_csv(os.path.join(results_dir, "pos_embed_trajectories.csv"), index=False)
    bias_df.to_csv(os.path.join(results_dir, "unembed_bias_diffs.csv"), index=False)
    
    print(f"  Completed seed {seed}")
    return df_epoch, emb_df, unemb_df, pos_df, bias_df

#%%
def main():
    """Main data collection process across all seeds."""
    print(f"Starting data collection for exp_08")
    print(f"Processing seeds: {SEEDS}")
    print(f"Results directory: {EXP_RESULTS_DIR}")
    
    # Process each seed
    all_epoch_dfs = []
    all_emb_dfs = []
    all_unemb_dfs = []
    all_pos_dfs = []
    all_bias_dfs = []
    
    for seed in SEEDS:
        df_epoch, emb_df, unemb_df, pos_df, bias_df = process_seed(seed)
        if not df_epoch.empty:
            all_epoch_dfs.append(df_epoch)
            all_emb_dfs.append(emb_df)
            all_unemb_dfs.append(unemb_df)
            all_pos_dfs.append(pos_df)
            all_bias_dfs.append(bias_df)
    
    # Aggregate all data
    print("\nAggregating data across all seeds...")
    aggregated_dir = os.path.join(EXP_RESULTS_DIR, "aggregated")
    
    if all_epoch_dfs:
        aggregated_epoch_df = pd.concat(all_epoch_dfs, ignore_index=True)
        aggregated_epoch_df.to_csv(os.path.join(aggregated_dir, "aggregated_epoch_data.csv"), index=False)
        print(f"Saved aggregated epoch data: {len(aggregated_epoch_df)} rows")
    
    if all_emb_dfs:
        aggregated_emb_df = pd.concat(all_emb_dfs, ignore_index=True)
        aggregated_emb_df.to_csv(os.path.join(aggregated_dir, "aggregated_embed_trajectories.csv"), index=False)
        print(f"Saved aggregated embedding trajectories: {len(aggregated_emb_df)} rows")
    
    if all_unemb_dfs:
        aggregated_unemb_df = pd.concat(all_unemb_dfs, ignore_index=True)
        aggregated_unemb_df.to_csv(os.path.join(aggregated_dir, "aggregated_unembed_trajectories.csv"), index=False)
        print(f"Saved aggregated unembedding trajectories: {len(aggregated_unemb_df)} rows")
    
    if all_pos_dfs:
        aggregated_pos_df = pd.concat(all_pos_dfs, ignore_index=True)
        aggregated_pos_df.to_csv(os.path.join(aggregated_dir, "aggregated_pos_embed_trajectories.csv"), index=False)
        print(f"Saved aggregated positional embedding trajectories: {len(aggregated_pos_df)} rows")
    
    if all_bias_dfs:
        aggregated_bias_df = pd.concat(all_bias_dfs, ignore_index=True)
        aggregated_bias_df.to_csv(os.path.join(aggregated_dir, "aggregated_unembed_bias_diffs.csv"), index=False)
        print(f"Saved aggregated unembed bias diffs: {len(aggregated_bias_df)} rows")
    
    print(f"\nData collection completed!")
    print(f"Results saved to: {EXP_RESULTS_DIR}")

#%%
if __name__ == "__main__":
    main()
# %%
