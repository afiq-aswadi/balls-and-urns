"""
We train a small transformer and see how embeddings/residuals develop across training. 
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
from core.models import create_coinformer_model, PosEmbedConfig, PosEmbedType
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
    n_ctx = 5,
    pos_embed_config=None,
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
    seq_length=5,
    num_batches=100,
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
                    "pos_embed_config": cfg0.pos_embed_config,
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
                    "pos_embed_config": cfg.pos_embed_config,
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


    def load_checkpoints_sorted(ckpt_dir: str):
        pattern = os.path.join(ckpt_dir, "small_transformer_epoch_*.pt")
        files = glob.glob(pattern)
        def epoch_num(fp):
            m = re.search(r"epoch_(\d+)_", os.path.basename(fp))
            return int(m.group(1)) if m else -1
        return sorted(files, key=epoch_num)

    def filter_checkpoints_by_config(ckpt_paths: list, exp_cfg: ExperimentConfig, seed: int) -> list:
        """Keep only checkpoints whose saved config matches the current experiment config and seed."""
        matched = []
        for p in ckpt_paths:
            try:
                ck = torch.load(p, map_location=DEVICE)
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
                    mc.get("attn_only") == exp_cfg.model_config.attn_only and
                    mc.get("d_mlp") == exp_cfg.model_config.d_mlp
                ):
                    matched.append(p)
            except Exception:
                # Skip unreadable/legacy checkpoints
                continue
        # Keep chronological order
        return matched

    def build_epoch_dataframe(checkpoints: list, model_cfg: ModelConfig) -> pd.DataFrame:
        records = []
        # Probe inputs
        seq_len = exp_config.seq_length
        batch = build_probe_batch(seq_len -1 , use_bos=model_cfg.use_bos_token).to(DEVICE)
        B, T = batch.shape
        # Pre-compute N and H
        pos_idx = torch.arange(T, device=batch.device).unsqueeze(0).expand(B, -1)
        # Exclude BOS for H count
        tokens_for_count = batch.clone()
        if model_cfg.use_bos_token:
            tokens_for_count[:, 0] = 0
        H_inclusive = torch.cumsum(tokens_for_count, dim=1)
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
        # Broadcast last token to all positions for convenient flattening
        last_token_mat = last_token_seq.unsqueeze(1).expand(-1, T)
        # Token observed at each (B, T) position (includes BOS at col 0 if present)
        token_at_pos = batch.clone()

        # We only keep positions t >= 1 when BOS is used
        valid_mask = torch.ones_like(pos_idx, dtype=torch.bool)
        if model_cfg.use_bos_token:
            valid_mask[:, 0] = False

        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            epoch = ckpt.get("epoch", None)

            model = create_coinformer_model(model_cfg).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"]) 
            model.eval()

            with torch.no_grad():
                _, cache = model.run_with_cache(batch)

            # Pre-MLP (after attention, before MLP) for layer 0
            pre_mlp = cache["resid_mid", 0]  # [B, T, d_model]
            # Post-MLP for layer 0
            post_mlp = cache["resid_post", 0]  # [B, T, d_model]

            # Flatten and collect
            for name, tensor in [("pre_mlp", pre_mlp), ("post_mlp", post_mlp)]:
                V = tensor[valid_mask].detach().cpu().numpy().reshape(-1, model.cfg.d_model)
                Ns = pos_idx[valid_mask].detach().cpu().numpy()
                Hs = H_inclusive[valid_mask].detach().cpu().numpy()
                Ss = source_mat[valid_mask].detach().cpu().numpy()
                Ls = last_token_mat[valid_mask].detach().cpu().numpy()
                Ts = token_at_pos[valid_mask].detach().cpu().numpy()
                for (x, y), n, h, s, l, tcur in zip(V, Ns, Hs, Ss, Ls, Ts):
                    records.append({
                        "epoch": epoch,
                        "name": name,
                        "N": int(n),
                        "H": int(h),
                        "source": {0: "zeros", 1: "lower", 2: "upper"}[int(s)],
                        "final_token": int(l),
                        "token_at_N": int(tcur),
                        "x": float(x),
                        "y": float(y),
                    })

        return pd.DataFrame.from_records(records)

    def plot_epoch_views(df: pd.DataFrame, save_html_prefix: str = None, group_size: int = 10, title_suffix: str = ""):
        figs = {}
        for name in ["pre_mlp", "post_mlp"]:
            sub = df[df["name"] == name].copy()
            for last in [0, 1]:
                sub_last = sub[sub["final_token"] == last]
                if sub_last.empty:
                    continue
                epochs = sorted(sub_last["epoch"].unique())

                # Shared axis ranges across epochs for this subset
                xmin = float(sub_last["x"].min())
                xmax = float(sub_last["x"].max())
                ymin = float(sub_last["y"].min())
                ymax = float(sub_last["y"].max())
                xr = xmax - xmin; yr = ymax - ymin
                xmin -= 0.05 * xr if xr > 0 else 0.1
                xmax += 0.05 * xr if xr > 0 else 0.1
                ymin -= 0.05 * yr if yr > 0 else 0.1
                ymax += 0.05 * yr if yr > 0 else 0.1

                # Define N position groups
                max_N = int(sub_last["N"].max()) if not sub_last.empty else 0
                num_groups = (max_N + group_size) // group_size
                group_labels = []
                group_ranges = []
                for g in range(num_groups):
                    start = g * group_size
                    end = min((g + 1) * group_size - 1, max_N)
                    group_labels.append(f"N {start}-{end}")
                    group_ranges.append((start, end))

                # Precompute per-epoch, per-group arrays
                epoch_group_x = [[[] for _ in range(num_groups)] for _ in epochs]
                epoch_group_y = [[[] for _ in range(num_groups)] for _ in epochs]
                epoch_group_text = [[[] for _ in range(num_groups)] for _ in epochs]
                epoch_group_custom = [[[] for _ in range(num_groups)] for _ in epochs]
                for ei, e in enumerate(epochs):
                    dfe = sub_last[sub_last["epoch"] == e]
                    for gi, (start, end) in enumerate(group_ranges):
                        sub_g = dfe[(dfe["N"] >= start) & (dfe["N"] <= end)]
                        epoch_group_x[ei][gi] = sub_g["x"].tolist()
                        epoch_group_y[ei][gi] = sub_g["y"].tolist()
                        epoch_group_text[ei][gi] = sub_g.get("source", pd.Series([None]*len(sub_g))).tolist()
                        # custom: N, H, token_at_N, final_token
                        epoch_group_custom[ei][gi] = np.stack([
                            sub_g["N"].values if len(sub_g) else np.array([]),
                            sub_g["H"].values if len(sub_g) else np.array([]),
                            sub_g["token_at_N"].values if len(sub_g) else np.array([]),
                            sub_g["final_token"].values if len(sub_g) else np.array([]),
                        ], axis=1).tolist() if len(sub_g) else []

                # Create one trace per group; slider updates x/y/text/customdata
                fig = go.Figure()
                colors = px.colors.qualitative.Dark24
                init_ei = len(epochs) - 1
                for gi, label in enumerate(group_labels):
                    color = colors[gi % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=epoch_group_x[init_ei][gi],
                        y=epoch_group_y[init_ei][gi],
                        mode="markers",
                        text=epoch_group_text[init_ei][gi],
                        customdata=epoch_group_custom[init_ei][gi],
                        marker=dict(size=5, color=color),
                        name=label,
                        hovertemplate="group=%{name}<br>source=%{text}<br>N=%{customdata[0]}<br>H=%{customdata[1]}<br>obs@N=%{customdata[2]}<br>seq_last=%{customdata[3]}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                        showlegend=True,
                    ))

                # Slider steps: update data for all group traces
                steps = []
                for ei, e in enumerate(epochs):
                    step = dict(
                        method="update",
                        args=[
                            {
                                "x": [epoch_group_x[ei][gi] for gi in range(num_groups)],
                                "y": [epoch_group_y[ei][gi] for gi in range(num_groups)],
                                "text": [epoch_group_text[ei][gi] for gi in range(num_groups)],
                                "customdata": [epoch_group_custom[ei][gi] for gi in range(num_groups)],
                            },
                            {"title": f"{name} (last={last}) - {'init' if e == 0 else f'epoch {e}'} — {title_suffix}"},
                        ],
                    )
                    steps.append(step)

                sliders = [dict(active=len(epochs)-1, pad={"t": 30}, steps=steps)]
                fig.update_layout(
                    title=f"{name} (seq_last={last}) - {'init' if epochs[-1] == 0 else f'epoch {epochs[-1]}' } — {title_suffix}",
                    xaxis_title="dim 1",
                    yaxis_title="dim 2",
                    showlegend=True,
                    sliders=sliders,
                    width=800,
                    height=600,
                )
                fig.update_xaxes(range=[xmin, xmax])
                fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)

                if save_html_prefix:
                    out = f"{save_html_prefix}_{name}_last{last}.html"
                    fig.write_html(out)
                    print(f"Saved {out}")
                figs[f"{name}_last{last}"] = fig
        return figs

    def build_embed_unembed_trajectories(checkpoints: list, model_cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collect token embedding and unembedding vectors across epochs.

        Returns two dataframes with columns: epoch, token_id, token_name, x, y
        """
        emb_records = []
        unemb_records = []
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
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
                        "epoch": epoch,
                        "token_id": tok,
                        "token_name": {0: "0", 1: "1", 2: "BOS"}.get(tok, str(tok)),
                        "x": x_e,
                        "y": y_e,
                    })
                    x_u, y_u = float(W_U[0, tok]), float(W_U[1, tok])
                    unemb_records.append({
                        "epoch": epoch,
                        "token_id": tok,
                        "token_name": {0: "0", 1: "1", 2: "BOS"}.get(tok, str(tok)),
                        "x": x_u,
                        "y": y_u,
                    })

        return pd.DataFrame.from_records(emb_records), pd.DataFrame.from_records(unemb_records)

    def plot_token_trajectories(df: pd.DataFrame, title: str) -> go.Figure:
        """Plot trajectories for tokens (lines across epochs) in 2D."""
        if df.empty:
            return go.Figure()
        # Shared bounds
        xmin = float(df["x"].min()); xmax = float(df["x"].max())
        ymin = float(df["y"].min()); ymax = float(df["y"].max())
        xr = xmax - xmin; yr = ymax - ymin
        xmin -= 0.05 * xr if xr > 0 else 0.1
        xmax += 0.05 * xr if xr > 0 else 0.1
        ymin -= 0.05 * yr if yr > 0 else 0.1
        ymax += 0.05 * yr if yr > 0 else 0.1

        fig = go.Figure()
        # Use deterministic colors per token
        token_ids = sorted(df["token_id"].unique())
        color_map = px.colors.qualitative.Set2
        for i, tok in enumerate(token_ids):
            sub = df[df["token_id"] == tok].sort_values("epoch")
            color = color_map[i % len(color_map)]
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"], mode="lines+markers+text",
                line=dict(color=color), marker=dict(color=color, size=7),
                text=[str(e) for e in sub["epoch"].values],
                textposition="top center",
                name=f"token {sub['token_name'].iloc[0]}",
                hovertemplate="token=%{name}<br>epoch=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            ))

        fig.update_layout(
            title=title,
            xaxis_title="dim 1",
            yaxis_title="dim 2",
            showlegend=True,
            width=800,
            height=600,
        )
        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)
        return fig

    def build_positional_embed_trajectories(checkpoints: list, model_cfg: ModelConfig, max_positions: int) -> pd.DataFrame:
        """Collect positional embedding vectors across epochs for positions [0, max_positions)."""
        records = []
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
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
                    "epoch": epoch,
                    "position": pos,
                    "x": x,
                    "y": y,
                })
        return pd.DataFrame.from_records(records)

    def plot_position_trajectories(df: pd.DataFrame, title: str) -> go.Figure:
        """Plot trajectories for positions (lines across epochs) in 2D."""
        if df.empty:
            return go.Figure()
        xmin = float(df["x"].min()); xmax = float(df["x"].max())
        ymin = float(df["y"].min()); ymax = float(df["y"].max())
        xr = xmax - xmin; yr = ymax - ymin
        xmin -= 0.05 * xr if xr > 0 else 0.1
        xmax += 0.05 * xr if xr > 0 else 0.1
        ymin -= 0.05 * yr if yr > 0 else 0.1
        ymax += 0.05 * yr if yr > 0 else 0.1

        fig = go.Figure()
        color_map = px.colors.qualitative.Plotly
        pos_ids = sorted(df["position"].unique())
        for i, pos in enumerate(pos_ids):
            sub = df[df["position"] == pos].sort_values("epoch")
            color = color_map[i % len(color_map)]
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"], mode="lines+markers+text",
                line=dict(color=color), marker=dict(color=color, size=6),
                text=[str(e) for e in sub["epoch"].values],
                textposition="top center",
                name=f"pos {pos}",
                hovertemplate="pos=%{name}<br>epoch=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            ))

        fig.update_layout(
            title=title,
            xaxis_title="dim 1",
            yaxis_title="dim 2",
            width=900,
            height=700,
        )
        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)
        return fig

    def plot_positional_epoch_slider(df: pd.DataFrame, title: str, group_size: int = 10) -> go.Figure:
        """Plot positional embeddings with an epoch slider and group toggles via legend (first ten, next ten, ...)."""
        if df.empty:
            return go.Figure()
        epochs = sorted(df["epoch"].unique())
        xmin = float(df["x"].min()); xmax = float(df["x"].max())
        ymin = float(df["y"].min()); ymax = float(df["y"].max())
        xr = xmax - xmin; yr = ymax - ymin
        xmin -= 0.05 * xr if xr > 0 else 0.1
        xmax += 0.05 * xr if xr > 0 else 0.1
        ymin -= 0.05 * yr if yr > 0 else 0.1
        ymax += 0.05 * yr if yr > 0 else 0.1

        # Define position groups
        max_pos = int(df["position"].max())
        num_groups = (max_pos + group_size) // group_size
        group_labels = []
        group_ranges = []
        for g in range(num_groups):
            start = g * group_size
            end = min((g + 1) * group_size - 1, max_pos)
            group_labels.append(f"pos {start}-{end}")
            group_ranges.append((start, end))

        # Precompute per-epoch, per-group arrays
        epoch_group_x = [[[] for _ in range(num_groups)] for _ in epochs]
        epoch_group_y = [[[] for _ in range(num_groups)] for _ in epochs]
        epoch_group_text = [[[] for _ in range(num_groups)] for _ in epochs]
        for ei, e in enumerate(epochs):
            dfe = df[df["epoch"] == e]
            for gi, (start, end) in enumerate(group_ranges):
                sub = dfe[(dfe["position"] >= start) & (dfe["position"] <= end)].sort_values("position")
                epoch_group_x[ei][gi] = sub["x"].tolist()
                epoch_group_y[ei][gi] = sub["y"].tolist()
                epoch_group_text[ei][gi] = [str(p) for p in sub["position"].tolist()]

        # Create one trace per group; slider will update their x/y/text across epochs
        fig = go.Figure()
        colors = px.colors.qualitative.Dark24
        init_ei = len(epochs) - 1
        for gi, label in enumerate(group_labels):
            color = colors[gi % len(colors)]
            fig.add_trace(go.Scatter(
                x=epoch_group_x[init_ei][gi],
                y=epoch_group_y[init_ei][gi],
                mode="markers+text",
                text=epoch_group_text[init_ei][gi],
                textposition="top center",
                marker=dict(size=7, color=color),
                name=label,
                hovertemplate="%{name}<br>pos=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                showlegend=True,
            ))

        # Slider steps: update x/y/text for all group traces
        steps = []
        for ei, e in enumerate(epochs):
            step = dict(
                method="update",
                args=[
                    {
                        "x": [epoch_group_x[ei][gi] for gi in range(num_groups)],
                        "y": [epoch_group_y[ei][gi] for gi in range(num_groups)],
                        "text": [epoch_group_text[ei][gi] for gi in range(num_groups)],
                    },
                    {"title": f"{title} - {'init' if e == 0 else f'epoch {e}'}"},
                ],
            )
            steps.append(step)

        sliders = [dict(active=len(epochs) - 1, pad={"t": 30}, steps=steps)]
        fig.update_layout(
            title=f"{title} - {'init' if epochs and epochs[-1] == 0 else f'epoch {epochs[-1]}' }",
            xaxis_title="dim 1",
            yaxis_title="dim 2",
            showlegend=True,
            sliders=sliders,
            width=900,
            height=700,
        )
        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)
        return fig

    all_ckpts = load_checkpoints_sorted(checkpoint_dir)
    ckpts = filter_checkpoints_by_config(all_ckpts, exp_config, SEED)
    if len(ckpts) != len(all_ckpts):
        print(f"Filtered checkpoints: using {len(ckpts)} of {len(all_ckpts)} that match current config/seed.")

    title_suffix = build_title_suffix(exp_config)
    emb_out = None
    unemb_out = None

    if ckpts:
        df_epoch = build_epoch_dataframe(ckpts, exp_config.model_config)
        df_path = os.path.join(EXP_RESULTS_DIR, "epochwise_data.csv")
        df_epoch.to_csv(df_path, index=False)
        print(f"Saved epoch dataframe to {df_path}")
        plot_epoch_views(
            df_epoch,
            save_html_prefix=os.path.join(EXP_RESULTS_DIR, "residual_analysis"),
            title_suffix=title_suffix,
        )

        # Build and plot embedding/unembedding trajectories in single plots
        emb_df, unemb_df = build_embed_unembed_trajectories(ckpts, exp_config.model_config)
        # Save compact token-level trajectories CSV
        try:
            traj_df = pd.concat([
                emb_df.assign(kind="embed"),
                unemb_df.assign(kind="unembed")
            ], ignore_index=True)
            traj_csv = os.path.join(EXP_RESULTS_DIR, "token_trajectories.csv")
            traj_df.to_csv(traj_csv, index=False)
            print(f"Saved token-level trajectories to {traj_csv}")
        except Exception as e:
            print(f"Warning: failed to save token-level trajectories CSV: {e}")
        emb_fig = plot_token_trajectories(emb_df, title=f"Embedding trajectories across epochs — {title_suffix}")
        unemb_fig = plot_token_trajectories(unemb_df, title=f"Unembedding trajectories across epochs — {title_suffix}")
        emb_out = os.path.join(EXP_RESULTS_DIR, "embed_trajectories.html")
        unemb_out = os.path.join(EXP_RESULTS_DIR, "unembed_trajectories.html")
        emb_fig.write_html(emb_out)
        unemb_fig.write_html(unemb_out)
        print(f"Saved {emb_out}")
        print(f"Saved {unemb_out}")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        # cuDNN determinism
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    else:
        print("No checkpoints matched; skipping residual and token trajectory exports.")
    # PyTorch deterministic algorithms where supported
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # Positional embeddings trajectories (limit to used positions)
    max_positions = exp_config.seq_length + (1 if exp_config.model_config.use_bos_token else 0)
    pos_df = build_positional_embed_trajectories(ckpts, exp_config.model_config, max_positions=max_positions)
    pos_fig_slider = plot_positional_epoch_slider(pos_df, title=f"Positional embeddings — {title_suffix}", group_size=10)
    pos_out = os.path.join(EXP_RESULTS_DIR, "pos_embed_trajectories.html")
    pos_fig_slider.write_html(pos_out)
    print(f"Saved {pos_out}")

# %%
