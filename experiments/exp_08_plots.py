import os
import glob
import re
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.models import create_coinformer_model
from core.config import ExperimentConfig, ModelConfig



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
                    hovertemplate="group=%{name}<br>source=%{text}<br>N=%{customdata[0]}<br>H=%{customdata[1]}<br>token@N=%{customdata[2]}<br>final=%{customdata[3]}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
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
                title=f"{name} (last={last}) - {'init' if epochs[-1] == 0 else f'epoch {epochs[-1]}' } — {title_suffix}",
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
if ckpts:
    df_epoch = build_epoch_dataframe(ckpts, exp_config.model_config)
    df_path = os.path.join(EXP_RESULTS_DIR, "epochwise_data.csv")
    df_epoch.to_csv(df_path, index=False)
    print(f"Saved epoch dataframe to {df_path}")
    title_suffix = build_title_suffix(exp_config)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
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
print(f"Saved {emb_out}")
print(f"Saved {unemb_out}")
# Positional embeddings trajectories (limit to used positions)
max_positions = exp_config.seq_length + (1 if exp_config.model_config.use_bos_token else 0)
pos_df = build_positional_embed_trajectories(ckpts, exp_config.model_config, max_positions=max_positions)
pos_fig_slider = plot_positional_epoch_slider(pos_df, title=f"Positional embeddings — {title_suffix}", group_size=10)
pos_out = os.path.join(EXP_RESULTS_DIR, "pos_embed_trajectories.html")
pos_fig_slider.write_html(pos_out)
print(f"Saved {pos_out}")