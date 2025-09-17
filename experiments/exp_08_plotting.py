"""
Plotting script for exp_08: Multi-seed small transformer experiment
Creates visualizations with heatmap-based residual analysis and joint-seed plots.
"""
#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from exp_08_config import (
    EXP_CONFIG, SEEDS, build_title_suffix, EXP_RESULTS_DIR
)



def get_position_group_color(group_index: int, total_groups: int) -> str:
    """Generate consistent gradient colors for position groups using viridis colormap."""
    import matplotlib.pyplot as plt
    viridis = plt.cm.get_cmap('viridis')
    
    # Map group index to color using viridis gradient
    group_color_fraction = group_index / max(1, total_groups - 1)  # 0 to 1
    group_color = viridis(group_color_fraction)
    # Convert to RGB string
    return f'rgb({int(group_color[0]*255)}, {int(group_color[1]*255)}, {int(group_color[2]*255)})'


def plot_residual_heatmap(df: pd.DataFrame, residual_type: str, seed: int, 
                         save_html_prefix: str = None, group_size: int = 10, 
                         title_suffix: str = "") -> dict:
    """
    Plot residual analysis with heatmap coloring based on token 1 probability.
    
    Args:
        df: DataFrame with residual data
        residual_type: 'pre_mlp' or 'post_mlp' or 'pre_attn'
        seed: Seed number
        save_html_prefix: Prefix for saving HTML files
        group_size: Size of position groups
        title_suffix: Additional title information
    """
    figs = {}
    
    # Column names for this residual type
    x_col = f"{residual_type}_x"
    y_col = f"{residual_type}_y"
    # Always use post_mlp probability for heatmap (final probability)
    prob_col = "post_mlp_prob_token1"
    
    for final_token in [0, 1]:
        sub = df[df["final_token"] == final_token].copy()
        if sub.empty:
            continue
            
        epochs = sorted(sub["epoch"].unique())
        
        # Shared axis ranges across epochs for this subset
        xmin = float(sub[x_col].min())
        xmax = float(sub[x_col].max())
        ymin = float(sub[y_col].min())
        ymax = float(sub[y_col].max())
        xr = xmax - xmin; yr = ymax - ymin
        xmin -= 0.05 * xr if xr > 0 else 0.1
        xmax += 0.05 * xr if xr > 0 else 0.1
        ymin -= 0.05 * yr if yr > 0 else 0.1
        ymax += 0.05 * yr if yr > 0 else 0.1
        
        # Define N position groups
        max_N = int(sub["N"].max()) if not sub.empty else 0
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
        epoch_group_colors = [[[] for _ in range(num_groups)] for _ in epochs]
        epoch_group_custom = [[[] for _ in range(num_groups)] for _ in epochs]
        
        for ei, e in enumerate(epochs):
            dfe = sub[sub["epoch"] == e]
            for gi, (start, end) in enumerate(group_ranges):
                sub_g = dfe[(dfe["N"] >= start) & (dfe["N"] <= end)]
                epoch_group_x[ei][gi] = sub_g[x_col].tolist()
                epoch_group_y[ei][gi] = sub_g[y_col].tolist()
                epoch_group_text[ei][gi] = sub_g.get("source", pd.Series([None]*len(sub_g))).tolist()
                # Use probability of token 1 as color intensity
                epoch_group_colors[ei][gi] = sub_g[prob_col].tolist()
                # Custom data: N, H, token_at_N, final_token, probability
                epoch_group_custom[ei][gi] = np.stack([
                    sub_g["N"].values if len(sub_g) else np.array([]),
                    sub_g["H"].values if len(sub_g) else np.array([]),
                    sub_g["token_at_N"].values if len(sub_g) else np.array([]),
                    sub_g["final_token"].values if len(sub_g) else np.array([]),
                    sub_g[prob_col].values if len(sub_g) else np.array([]),
                ], axis=1).tolist() if len(sub_g) else []
        
        # Create figure with heatmap coloring
        fig = go.Figure()
        init_ei = len(epochs) - 1
        
        # Define a consistent colorscale across all groups
        colorscale = "Viridis"  # Good for probability heatmaps
        
        for gi, label in enumerate(group_labels):
            # Use scatter with color mapping instead of fixed colors
            fig.add_trace(go.Scatter(
                x=epoch_group_x[init_ei][gi],
                y=epoch_group_y[init_ei][gi],
                mode="markers",
                text=epoch_group_text[init_ei][gi],
                customdata=epoch_group_custom[init_ei][gi],
                marker=dict(
                    size=6,
                    color=epoch_group_colors[init_ei][gi],
                    colorscale=colorscale,
                    cmin=0,
                    cmax=1,
                    showscale=(gi == 0),  # Only show colorbar for first trace
                    colorbar=dict(
                        title="P(token=1)", 
                        x=1.02,  # Move colorbar to the right to avoid legend overlap
                        len=0.7   # Make colorbar shorter
                    ) if gi == 0 else None
                ),
                name=label,
                hovertemplate="group=%{name}<br>source=%{text}<br>N=%{customdata[0]}<br>H=%{customdata[1]}<br>token@N=%{customdata[2]}<br>final=%{customdata[3]}<br>P(token=1)=%{customdata[4]:.3f}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
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
                        "marker.color": [epoch_group_colors[ei][gi] for gi in range(num_groups)],
                        "customdata": [epoch_group_custom[ei][gi] for gi in range(num_groups)],
                    },
                    {"title": f"Seed {seed} - {residual_type} (final={final_token}) - {'init' if e == 0 else f'epoch {e}'} — {title_suffix}"},
                ],
            )
            steps.append(step)
        
        sliders = [dict(active=len(epochs)-1, pad={"t": 30}, steps=steps)]
        fig.update_layout(
            title=f"Seed {seed} - {residual_type} (final={final_token}) - {'init' if epochs[-1] == 0 else f'epoch {epochs[-1]}' } — {title_suffix}",
            xaxis_title="dim 1",
            yaxis_title="dim 2",
            showlegend=True,
            sliders=sliders,
            width=1000,  # Wider to accommodate moved colorbar
            height=700,
            legend=dict(
                x=0.01,    # Move legend to left to avoid colorbar
                y=0.99
            )
        )
        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)
        
        # Add origin lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
        
        if save_html_prefix:
            out = f"{save_html_prefix}_seed{seed}_{residual_type}_final{final_token}.html"
            fig.write_html(out)
            print(f"Saved {out}")
        
        figs[f"seed{seed}_{residual_type}_final{final_token}"] = fig
    
    return figs


def plot_positional_trajectories(df: pd.DataFrame, title: str, seed: int = None, 
                                save_html_prefix: str = None, title_suffix: str = "") -> go.Figure:
    """Plot positional embeddings as scatter plot with epoch slider similar to residual plots."""
    if df.empty:
        return go.Figure()
    
    # Filter by seed if specified
    if seed is not None:
        df = df[df["seed"] == seed]
        if df.empty:
            return go.Figure()
    
    epochs = sorted(df["epoch"].unique())
    pos_ids = sorted(df["position"].unique())
    
    # Group positions by tens (0-9, 10-19, etc)
    max_pos = max(pos_ids) if pos_ids else 0
    group_size = 10
    num_groups = (max_pos + group_size) // group_size
    group_labels = []
    group_ranges = []
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size - 1, max_pos)
        # Display labels as 1-based inclusive ranges
        group_labels.append(f"pos {start+1}-{end+1}")
        group_ranges.append((start, end))
    
    # Shared axis ranges across epochs
    xmin = float(df["x"].min())
    xmax = float(df["x"].max())
    ymin = float(df["y"].min())
    ymax = float(df["y"].max())
    xr = xmax - xmin; yr = ymax - ymin
    xmin -= 0.05 * xr if xr > 0 else 0.1
    xmax += 0.05 * xr if xr > 0 else 0.1
    ymin -= 0.05 * yr if yr > 0 else 0.1
    ymax += 0.05 * yr if yr > 0 else 0.1
    
    # Precompute per-epoch, per-group arrays
    epoch_group_x = [[[] for _ in range(num_groups)] for _ in epochs]
    epoch_group_y = [[[] for _ in range(num_groups)] for _ in epochs]
    epoch_group_text = [[[] for _ in range(num_groups)] for _ in epochs]
    epoch_group_colors = [[[] for _ in range(num_groups)] for _ in epochs]
    
    for ei, e in enumerate(epochs):
        dfe = df[df["epoch"] == e]
        for gi, (start, end) in enumerate(group_ranges):
            sub_g = dfe[(dfe["position"] >= start) & (dfe["position"] <= end)]
            epoch_group_x[ei][gi] = sub_g["x"].tolist()
            epoch_group_y[ei][gi] = sub_g["y"].tolist()
            epoch_group_text[ei][gi] = sub_g["position"].tolist()  # Show actual position in hover
            # Use position as color gradient within each group
            epoch_group_colors[ei][gi] = sub_g["position"].tolist()
    
    # Create figure with scatter points using gradient colors
    fig = go.Figure()
    init_ei = len(epochs) - 1
    
    for gi, label in enumerate(group_labels):
        # Use consistent gradient color for this group
        group_color = get_position_group_color(gi, len(group_labels))
        
        fig.add_trace(go.Scatter(
            x=epoch_group_x[init_ei][gi],
            y=epoch_group_y[init_ei][gi],
            mode="markers",
            text=epoch_group_text[init_ei][gi],
            marker=dict(size=8, color=group_color),
            name=label,
            hovertemplate="group=%{name}<br>pos=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
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
                },
                {"title": f"{title} - {'init' if e == 0 else f'epoch {e}'} — {title_suffix}"},
            ],
        )
        steps.append(step)
    
    sliders = [dict(active=len(epochs)-1, pad={"t": 30}, steps=steps)]
    fig.update_layout(
        title=f"{title} - {'init' if epochs[-1] == 0 else f'epoch {epochs[-1]}'} — {title_suffix}",
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        showlegend=True,
        sliders=sliders,
        width=800,
        height=700,
    )
    fig.update_xaxes(range=[xmin, xmax])
    fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)
    
    # Add origin lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
    
    if save_html_prefix:
        out = f"{save_html_prefix}_seed{seed}_positional.html"
        fig.write_html(out)
        print(f"Saved {out}")
    
    return fig


def plot_token_trajectories(df: pd.DataFrame, title: str, seed: int = None) -> go.Figure:
    """Plot trajectories for tokens (lines across epochs) in 2D."""
    if df.empty:
        return go.Figure()
    
    # Filter by seed if specified
    if seed is not None:
        df = df[df["seed"] == seed]
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
    
    # Add origin lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
    
    return fig


def _compute_unembed_diff_by_epoch(seed_unemb_df: pd.DataFrame) -> dict:
    """Return dict[epoch] -> (dx, dy) where (dx, dy) = W_U[:,1] - W_U[:,0] for that epoch."""
    diff_by_epoch = {}
    if seed_unemb_df.empty:
        return diff_by_epoch
    epochs = sorted(seed_unemb_df["epoch"].unique())
    for e in epochs:
        sub = seed_unemb_df[seed_unemb_df["epoch"] == e]
        tok0 = sub[sub["token_id"] == 0]
        tok1 = sub[sub["token_id"] == 1]
        if not tok0.empty and not tok1.empty:
            # Each should be one row per epoch per token
            x0, y0 = float(tok0.iloc[0]["x"]), float(tok0.iloc[0]["y"])
            x1, y1 = float(tok1.iloc[0]["x"]), float(tok1.iloc[0]["y"])
            diff_by_epoch[e] = (x1 - x0, y1 - y0)
    return diff_by_epoch


def _compute_unembed_bias_diff_by_epoch(seed_bias_df: pd.DataFrame) -> dict:
    """Return dict[epoch] -> bias_diff = b_U[1] - b_U[0] for that epoch."""
    bias_by_epoch = {}
    if seed_bias_df is None or seed_bias_df.empty:
        return bias_by_epoch
    epochs = sorted(seed_bias_df["epoch"].unique())
    for e in epochs:
        sub = seed_bias_df[seed_bias_df["epoch"] == e]
        if not sub.empty and "bias_diff_1_minus_0" in sub.columns:
            bias_by_epoch[e] = float(sub.iloc[0]["bias_diff_1_minus_0"])
    return bias_by_epoch


def plot_post_mlp_projection(seed_epoch_df: pd.DataFrame, seed_unemb_df: pd.DataFrame, seed: int,
                             save_path: str = None, title_suffix: str = "",
                             seed_bias_df: pd.DataFrame = None) -> go.Figure:
    """3D scatter: x=N, y=H, z=<post_mlp_resid, W_U[:,1]-W_U[:,0]> + (b1-b0); split by final_token."""
    fig = go.Figure()
    if seed_epoch_df.empty or seed_unemb_df.empty:
        return fig
    final_epoch = int(seed_epoch_df["epoch"].max())
    final_df = seed_epoch_df[seed_epoch_df["epoch"] == final_epoch]
    if final_df.empty:
        return fig
    diff_by_epoch = _compute_unembed_diff_by_epoch(seed_unemb_df)
    if final_epoch not in diff_by_epoch:
        return fig
    dx, dy = diff_by_epoch[final_epoch]
    bias_by_epoch = _compute_unembed_bias_diff_by_epoch(seed_bias_df) if seed_bias_df is not None else {}
    bias_term = bias_by_epoch.get(final_epoch, 0.0)

    for final_token in [0, 1]:
        token_df = final_df[final_df["final_token"] == final_token]
        if token_df.empty:
            continue
        proj = token_df["post_mlp_x"].values * dx + token_df["post_mlp_y"].values * dy + bias_term
        colorscale = "Blues" if final_token == 0 else "Reds"
        fig.add_trace(
            go.Scatter3d(
                x=token_df["N"],
                y=token_df["H"],
                z=proj,
                mode="markers",
                marker=dict(
                    size=3,
                    color=token_df["post_mlp_prob_token1"],
                    colorscale=colorscale,
                    cmin=0,
                    cmax=1,
                    showscale=(final_token == 1),
                    colorbar=dict(
                        title="P(token=1) - Reds",
                        len=0.8
                    ) if final_token == 1 else None
                ),
                name=f"final_token={final_token}",
                hovertemplate="N=%{x}<br>H=%{y}<br>proj=%{z:.3f}<br>P1=%{customdata[2]:.3f}<extra></extra>",
                customdata=np.column_stack([
                    token_df["N"], token_df["H"], token_df["post_mlp_prob_token1"]
                ]),
            )
        )

    # Add theoretical log-odds points across all final tokens
    theo_z = np.log((1.0 + final_df["H"].values) /(1.0 + final_df["N"].values - final_df["H"].values))
    fig.add_trace(
        go.Scatter3d(
            x=final_df["N"],
            y=final_df["H"],
            z=theo_z,
            mode="markers",
            marker=dict(size=2, color="black"),
            name="theoretical log-odds",
            hovertemplate="N=%{x}<br>H=%{y}<br>theory=%{z:.3f}<extra></extra>",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"Seed {seed} - 3D: N,H vs post-MLP projection — {title_suffix}",
        scene=dict(
            xaxis_title="N",
            yaxis_title="H",
            zaxis_title="projection"
        ),
        showlegend=True,
        width=900,
        height=700,
    )

    if save_path:
        fig.write_html(save_path)
    return fig


def create_joint_seed_plot(epoch_df: pd.DataFrame, emb_df: pd.DataFrame, 
                          unemb_df: pd.DataFrame, bias_df: pd.DataFrame, pos_df: pd.DataFrame, seed: int, 
                          save_path: str = None, title_suffix: str = "") -> go.Figure:
    """
    Create a joint plot combining embedding trajectories and residual analysis for a single seed.
    """
    # Filter data by seed

    epochs = sorted(epoch_df["epoch"].unique())

    seed_epoch_df = epoch_df[epoch_df["seed"] == seed]
    seed_emb_df = emb_df[emb_df["seed"] == seed]
    seed_unemb_df = unemb_df[unemb_df["seed"] == seed]
    seed_pos_df = pos_df[pos_df["seed"] == seed] if not pos_df.empty else pd.DataFrame()
    seed_bias_df = bias_df[bias_df["seed"] == seed] if (bias_df is not None and not bias_df.empty) else pd.DataFrame()
    
    if seed_epoch_df.empty:
        return go.Figure()
    
    # Create subplot layout: 3x3 grid (third row adds projection plot)
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            f"Embedding Trajectories - Seed {seed}",
            f"Unembedding Trajectories - Seed {seed}",
            f"Positional Embedding Trajectories - Seed {seed}",
            f"Pre-Attention Residuals (final epoch) - Seed {seed}",
            f"Pre-MLP Residuals (final epoch) - Seed {seed}",
            f"Post-MLP Residuals (final epoch) - Seed {seed}",
            f"",
            f"Post-MLP dot(W_U[1]-W_U[0]) - Seed {seed}",
            f""
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
    )
    
    # 1. Embedding trajectories (top-left)
    if not seed_emb_df.empty:
        token_ids = sorted(seed_emb_df["token_id"].unique())
        color_map = px.colors.qualitative.Set2
        for i, tok in enumerate(token_ids):
            sub = seed_emb_df[seed_emb_df["token_id"] == tok].sort_values("epoch")
            color = color_map[i % len(color_map)]
            fig.add_trace(
                go.Scatter(
                    x=sub["x"], y=sub["y"], mode="lines+markers+text",
                    line=dict(color=color), marker=dict(color=color, size=5),
                    text=[str(e) for e in sub["epoch"].values],  # Add epoch markers
                    textposition="top center",
                    name=f"emb_token_{sub['token_name'].iloc[0]}",
                    showlegend=False,  # Hide legend to reduce clutter
                    hovertemplate="token=%{name}<br>epoch=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=1, col=1
            )
    
    # 2. Unembedding trajectories (top-right)
    if not seed_unemb_df.empty:
        token_ids = sorted(seed_unemb_df["token_id"].unique())
        for i, tok in enumerate(token_ids):
            sub = seed_unemb_df[seed_unemb_df["token_id"] == tok].sort_values("epoch")
            color = color_map[i % len(color_map)]
            fig.add_trace(
                go.Scatter(
                    x=sub["x"], y=sub["y"], mode="lines+markers+text",
                    line=dict(color=color), marker=dict(color=color, size=5),
                    text=[str(e) for e in sub["epoch"].values],  # Add epoch markers
                    textposition="top center",
                    name=f"unemb_token_{sub['token_name'].iloc[0]}",
                    showlegend=False,  # Hide legend to reduce clutter
                    hovertemplate="token=%{name}<br>epoch=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=1, col=2
            )
    
    # 3. Positional embedding trajectories (top-right) - with sliders like residuals
    pos_epoch_data = {}
    seed_pos_df = pos_df[pos_df["seed"] == seed] if not pos_df.empty else pd.DataFrame()
    if not seed_pos_df.empty:
        pos_ids = sorted(seed_pos_df["position"].unique())
        
        # Group positions by tens
        max_pos = max(pos_ids) if pos_ids else 0
        group_size = 10
        num_pos_groups = (max_pos + group_size) // group_size
        pos_group_ranges = []
        pos_group_labels = []
        for g in range(num_pos_groups):
            start = g * group_size
            end = min((g + 1) * group_size - 1, max_pos)
            # 1-based labels for readability
            pos_group_labels.append(f"pos {start+1}-{end+1}")
            pos_group_ranges.append((start, end))
        
        # Prepare positional data for each epoch and group
        for epoch in epochs:
            epoch_pos_data = seed_pos_df[seed_pos_df["epoch"] == epoch]
            if not epoch_pos_data.empty:
                pos_epoch_data[epoch] = {}
                for gi, (start, end) in enumerate(pos_group_ranges):
                    group_data = epoch_pos_data[(epoch_pos_data["position"] >= start) & (epoch_pos_data["position"] <= end)]
                    if not group_data.empty:
                        pos_epoch_data[epoch][gi] = {
                            'x': group_data["x"].tolist(),
                            'y': group_data["y"].tolist(),
                            'colors': group_data["position"].tolist(),
                            'text': group_data["position"].tolist()
                        }
                    else:
                        pos_epoch_data[epoch][gi] = {'x': [], 'y': [], 'colors': [], 'text': []}
        
        # Add initial traces for positional embeddings (final epoch)
        final_epoch = max(epochs)
        if final_epoch in pos_epoch_data:
            for gi, label in enumerate(pos_group_labels):
                if gi in pos_epoch_data[final_epoch]:
                    data = pos_epoch_data[final_epoch][gi]
                    # Use consistent gradient color for this group
                    group_color = get_position_group_color(gi, len(pos_group_labels))
                    fig.add_trace(
                        go.Scatter(
                            x=data['x'],
                            y=data['y'],
                            mode="markers",
                            text=data['text'],
                            marker=dict(size=6, color=group_color),
                            name=label,
                            showlegend=True,
                            hovertemplate="group=%{name}<br>pos=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                        ),
                        row=1, col=3
                    )

    final_epoch = seed_epoch_df["epoch"].max()
    final_epoch_df = seed_epoch_df[seed_epoch_df["epoch"] == final_epoch]
    # 4. Pre-attention residuals for final epoch (bottom-left) - separate by final_token
    if not final_epoch_df.empty:
        for final_token in [0, 1]:
            token_df = final_epoch_df[final_epoch_df["final_token"] == final_token]
            if not token_df.empty:
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter(
                        x=token_df["pre_attn_x"],
                        y=token_df["pre_attn_y"],
                        mode="markers",
                        marker=dict(size=6, color=token_df["post_mlp_prob_token1"]),
                        name=f"Pre-Attention final_token={final_token} ({'Blues' if final_token == 0 else 'Reds'})",
                        showlegend=True,
                        hovertemplate=f"final_token={final_token}<br>N=%{{customdata[0]}}<br>H=%{{customdata[1]}}<br>P(token=1)=%{{customdata[2]:.3f}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                        customdata=np.column_stack([
                            token_df["N"],
                            token_df["H"],
                            token_df["post_mlp_prob_token1"]
                        ])
                    ),
                    row=2, col=1
                )

    # 4. Pre-MLP residuals for final epoch (bottom-center) - separate by final_token
    if not final_epoch_df.empty:
        # Split by final_token for different color gradients
        for final_token in [0, 1]:
            token_df = final_epoch_df[final_epoch_df["final_token"] == final_token]
            if not token_df.empty:
                # Use different colorscales for different final tokens
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter(
                        x=token_df["pre_mlp_x"],
                        y=token_df["pre_mlp_y"],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=token_df["post_mlp_prob_token1"],  # Use final probability
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=(final_token == 1),  # Only show colorbar for Reds (final_token=1)
                            colorbar=dict(
                                title="P(token=1) - Reds", 
                                x=0.3,  # Position between pre-mlp and post-mlp plots, avoid overlap
                                len=0.28,
                                yanchor="bottom",
                                y=0.12
                            ) if final_token == 1 else None
                        ),
                        name=f"Pre-MLP final_token={final_token} ({'Blues' if final_token == 0 else 'Reds'})",
                        showlegend=True,
                        hovertemplate=f"final_token={final_token}<br>N=%{{customdata[0]}}<br>H=%{{customdata[1]}}<br>P(token=1)=%{{customdata[2]:.3f}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                        customdata=np.column_stack([
                            token_df["N"],
                            token_df["H"],
                            token_df["post_mlp_prob_token1"]
                        ])
                    ),
                    row=2, col=2
                )
    
    # 5. Post-MLP residuals for final epoch (bottom-right) - separate by final_token
    if not final_epoch_df.empty:
        for final_token in [0, 1]:
            token_df = final_epoch_df[final_epoch_df["final_token"] == final_token]
            if not token_df.empty:
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter(
                        x=token_df["post_mlp_x"],
                        y=token_df["post_mlp_y"],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=token_df["post_mlp_prob_token1"],
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=(final_token == 0),  # Only show colorbar for Blues (final_token=0)
                            colorbar=dict(
                                title="P(token=1) - Blues", 
                                x=0.66,  # Position to the right of post-mlp plot
                                len=0.28,
                                yanchor="bottom",
                                y=0.12
                            ) if final_token == 0 else None
                        ),
                        name=f"Post-MLP final_token={final_token} ({'Blues' if final_token == 0 else 'Reds'})",
                        showlegend=True,
                        hovertemplate=f"final_token={final_token}<br>N=%{{customdata[0]}}<br>H=%{{customdata[1]}}<br>P(token=1)=%{{customdata[2]:.3f}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                        customdata=np.column_stack([
                            token_df["N"],
                            token_df["H"],
                            token_df["post_mlp_prob_token1"]
                        ])
                    ),
                    row=2, col=3
                )

    
    
    # 6. Post-MLP projection onto W_U[1]-W_U[0] + (b1-b0) (row 3, col 2) — 3D
    seed_unemb_df = unemb_df[unemb_df["seed"] == seed]
    if not final_epoch_df.empty and not seed_unemb_df.empty:
        diff_by_epoch = _compute_unembed_diff_by_epoch(seed_unemb_df)
        bias_by_epoch = _compute_unembed_bias_diff_by_epoch(seed_bias_df)
        if final_epoch in diff_by_epoch:
            dx, dy = diff_by_epoch[final_epoch]
            bias_term = bias_by_epoch.get(final_epoch, 0.0)
            for final_token in [0, 1]:
                token_df = final_epoch_df[final_epoch_df["final_token"] == final_token]
                if token_df.empty:
                    continue
                proj = token_df["post_mlp_x"].values * dx + token_df["post_mlp_y"].values * dy + bias_term
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter3d(
                        x=token_df["N"],
                        y=token_df["H"],
                        z=proj,
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=token_df["post_mlp_prob_token1"],
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=(final_token == 1),
                            colorbar=dict(
                                title="P(token=1) - Reds",
                                len=0.4
                            ) if final_token == 1 else None
                        ),
                        name=f"Projection final_token={final_token}",
                        showlegend=True,
                        hovertemplate=f"final_token={final_token}<br>N=%{{x}}<br>H=%{{y}}<br>proj=%{{z:.3f}}<br>P1=%{{customdata[2]:.3f}}<extra></extra>",
                        customdata=np.column_stack([
                            token_df["N"], token_df["H"], token_df["post_mlp_prob_token1"],
                        ])
                    ),
                    row=3, col=2
                )

            # Theoretical log-odds for final epoch (single trace)
            theo_z = np.log((1.0 + final_epoch_df["H"].values) / (1.0 + final_epoch_df["N"].values - final_epoch_df["H"].values))
            fig.add_trace(
                go.Scatter3d(
                    x=final_epoch_df["N"],
                    y=final_epoch_df["H"],
                    z=theo_z,
                    mode="markers",
                    marker=dict(size=2, color="black"),
                    name="theoretical log-odds",
                    showlegend=True,
                    hovertemplate="N=%{x}<br>H=%{y}<br>theory=%{z:.3f}<extra></extra>",
                ),
                row=3, col=2
            )

    # Update layout
    fig.update_layout(
        title=f"Joint Analysis - Seed {seed} — {title_suffix}",
        height=1400,
        width=1800,  # Even wider to accommodate colorbars
        showlegend=True
    )
    
    # Update axes to maintain aspect ratio and add origin lines
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.update_xaxes(title_text="dim 1", row=row, col=col)
            fig.update_yaxes(title_text="dim 2", scaleanchor=f"x{'' if row==1 and col==1 else (row-1)*3+col}", 
                           scaleratio=1, row=row, col=col)
            # Add origin lines to each subplot
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.3, row=row, col=col, exclude_empty_subplots=False)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.3, row=row, col=col, exclude_empty_subplots=False)
    # Row 3 scene formatting (3D) — use scene2 for (row=3,col=2)
    fig.update_layout(scene2=dict(xaxis_title="N", yaxis_title="H", zaxis_title="projection"))
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved joint plot: {save_path}")
    
    return fig


def create_joint_seed_plot_with_sliders(epoch_df: pd.DataFrame, emb_df: pd.DataFrame, 
                                        unemb_df: pd.DataFrame, bias_df: pd.DataFrame, pos_df: pd.DataFrame, seed: int, 
                                        save_path: str = None, title_suffix: str = "") -> go.Figure:
    """
    Create a joint plot with epoch sliders for residual analysis.
    """
    # Filter data by seed
    seed_epoch_df = epoch_df[epoch_df["seed"] == seed]
    seed_emb_df = emb_df[emb_df["seed"] == seed]
    seed_unemb_df = unemb_df[unemb_df["seed"] == seed]
    seed_bias_df = bias_df[bias_df["seed"] == seed] if (bias_df is not None and not bias_df.empty) else pd.DataFrame()
    
    if seed_epoch_df.empty:
        return go.Figure()
    
    # Create subplot layout: 3x3 grid (third row adds projection plot)
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            f"Embedding Trajectories - Seed {seed}",
            f"Unembedding Trajectories - Seed {seed}",
            f"Positional Embedding Trajectories - Seed {seed}",
            f"Pre-Attention Residuals - Seed {seed}",
            f"Pre-MLP Residuals - Seed {seed}",
            f"Post-MLP Residuals - Seed {seed}",
            f"",
            f"Post-MLP dot(W_U[1]-W_U[0]) - Seed {seed}",
            f""
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
    )
    
    epochs = sorted(seed_epoch_df["epoch"].unique())
    
    # 1. Embedding trajectories (top-left) - same as before
    if not seed_emb_df.empty:
        token_ids = sorted(seed_emb_df["token_id"].unique())
        color_map = px.colors.qualitative.Set2
        for i, tok in enumerate(token_ids):
            sub = seed_emb_df[seed_emb_df["token_id"] == tok].sort_values("epoch")
            color = color_map[i % len(color_map)]
            fig.add_trace(
                go.Scatter(
                    x=sub["x"], y=sub["y"], mode="lines+markers+text",
                    line=dict(color=color), marker=dict(color=color, size=4),
                    text=[str(e) for e in sub["epoch"].values],
                    textposition="top center",
                    name=f"emb_token_{sub['token_name'].iloc[0]}",
                    showlegend=False,
                    hovertemplate="token=%{name}<br>epoch=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=1, col=1
            )
    
    # 2. Unembedding trajectories (top-right) - same as before
    if not seed_unemb_df.empty:
        token_ids = sorted(seed_unemb_df["token_id"].unique())
        for i, tok in enumerate(token_ids):
            sub = seed_unemb_df[seed_unemb_df["token_id"] == tok].sort_values("epoch")
            color = color_map[i % len(color_map)]
            fig.add_trace(
                go.Scatter(
                    x=sub["x"], y=sub["y"], mode="lines+markers+text",
                    line=dict(color=color), marker=dict(color=color, size=4),
                    text=[str(e) for e in sub["epoch"].values],
                    textposition="top center",
                    name=f"unemb_token_{sub['token_name'].iloc[0]}",
                    showlegend=False,
                    hovertemplate="token=%{name}<br>epoch=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=1, col=2
            )
    
    # Add positional embedding data (if available) - with sliders like residuals
    pos_epoch_data = {}
    seed_pos_df = pos_df[pos_df["seed"] == seed] if not pos_df.empty else pd.DataFrame()
    if not seed_pos_df.empty:
        pos_ids = sorted(seed_pos_df["position"].unique())
        
        # Group positions by tens
        max_pos = max(pos_ids) if pos_ids else 0
        group_size = 10
        num_pos_groups = (max_pos + group_size) // group_size
        pos_group_ranges = []
        pos_group_labels = []
        for g in range(num_pos_groups):
            start = g * group_size
            end = min((g + 1) * group_size - 1, max_pos)
            pos_group_labels.append(f"pos {start+1}-{end+1}")
            pos_group_ranges.append((start, end))
        
        # Prepare positional data for each epoch and group
        for epoch in epochs:
            epoch_pos_data = seed_pos_df[seed_pos_df["epoch"] == epoch]
            if not epoch_pos_data.empty:
                pos_epoch_data[epoch] = {}
                for gi, (start, end) in enumerate(pos_group_ranges):
                    group_data = epoch_pos_data[(epoch_pos_data["position"] >= start) & (epoch_pos_data["position"] <= end)]
                    if not group_data.empty:
                        pos_epoch_data[epoch][gi] = {
                            'x': group_data["x"].tolist(),
                            'y': group_data["y"].tolist(),
                            'colors': group_data["position"].tolist(),
                            'text': group_data["position"].tolist()
                        }
                    else:
                        pos_epoch_data[epoch][gi] = {'x': [], 'y': [], 'colors': [], 'text': []}
        
        # Add initial traces for positional embeddings (final epoch)
        final_epoch = max(epochs)
        if final_epoch in pos_epoch_data:
            for gi, label in enumerate(pos_group_labels):
                if gi in pos_epoch_data[final_epoch]:
                    data = pos_epoch_data[final_epoch][gi]
                    # Use consistent gradient color for this group
                    group_color = get_position_group_color(gi, len(pos_group_labels))
                    fig.add_trace(
                        go.Scatter(
                            x=data['x'],
                            y=data['y'],
                            mode="markers",
                            text=data['text'],
                            marker=dict(size=6, color=group_color),
                            name=label,
                            showlegend=True,
                            hovertemplate="group=%{name}<br>pos=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                        ),
                        row=1, col=3
                    )
    
    # Prepare residual data with sliders - separate by final_token
    # We'll create data arrays for each epoch and final_token combination
    pre_attn_epoch_data = {}
    pre_mlp_epoch_data = {}
    post_mlp_epoch_data = {}
    
    for epoch in epochs:
        epoch_data = seed_epoch_df[seed_epoch_df["epoch"] == epoch]
        if not epoch_data.empty:
            pre_attn_epoch_data[epoch] = {}
            pre_mlp_epoch_data[epoch] = {}
            post_mlp_epoch_data[epoch] = {}
            
            for final_token in [0, 1]:
                token_data = epoch_data[epoch_data["final_token"] == final_token]
                if not token_data.empty:
                    pre_attn_epoch_data[epoch][final_token] = {
                        'x': token_data["pre_attn_x"].tolist(),
                        'y': token_data["pre_attn_y"].tolist(),
                        'colors': token_data["post_mlp_prob_token1"].tolist(),
                        'customdata': np.column_stack([
                            token_data["N"],
                            token_data["H"],
                            token_data["post_mlp_prob_token1"],
                            np.full(len(token_data), final_token)
                        ]).tolist()
                    }
                    pre_mlp_epoch_data[epoch][final_token] = {
                        'x': token_data["pre_mlp_x"].tolist(),
                        'y': token_data["pre_mlp_y"].tolist(),
                        'colors': token_data["post_mlp_prob_token1"].tolist(),  # Use final probability
                        'customdata': np.column_stack([
                            token_data["N"],
                            token_data["H"],
                            token_data["post_mlp_prob_token1"],
                            np.full(len(token_data), final_token)  # Add final_token to customdata
                        ]).tolist()
                    }
                    post_mlp_epoch_data[epoch][final_token] = {
                        'x': token_data["post_mlp_x"].tolist(),
                        'y': token_data["post_mlp_y"].tolist(),
                        'colors': token_data["post_mlp_prob_token1"].tolist(),
                        'customdata': np.column_stack([
                            token_data["N"],
                            token_data["H"],
                            token_data["post_mlp_prob_token1"],
                            np.full(len(token_data), final_token)
                        ]).tolist()
                    }
                else:
                    # Empty data for missing final_token
                    pre_attn_epoch_data[epoch][final_token] = {'x': [], 'y': [], 'colors': [], 'customdata': []}
                    pre_mlp_epoch_data[epoch][final_token] = {'x': [], 'y': [], 'colors': [], 'customdata': []}
                    post_mlp_epoch_data[epoch][final_token] = {'x': [], 'y': [], 'colors': [], 'customdata': []}

    # Prepare projection data per epoch using unembedding difference
    seed_unemb_df = unemb_df[unemb_df["seed"] == seed]
    unemb_diff_by_epoch = _compute_unembed_diff_by_epoch(seed_unemb_df)
    bias_diff_by_epoch = _compute_unembed_bias_diff_by_epoch(seed_bias_df)
    proj_epoch_data = {}
    theoretical_epoch_data = {}
    for epoch in epochs:
        epoch_data = seed_epoch_df[seed_epoch_df["epoch"] == epoch]
        if not epoch_data.empty and epoch in unemb_diff_by_epoch:
            dx, dy = unemb_diff_by_epoch[epoch]
            bias_term = bias_diff_by_epoch.get(epoch, 0.0)
            proj_epoch_data[epoch] = {}
            for final_token in [0, 1]:
                token_data = epoch_data[epoch_data["final_token"] == final_token]
                if not token_data.empty:
                    proj_vals = (token_data["post_mlp_x"].values * dx + token_data["post_mlp_y"].values * dy + bias_term).tolist()
                    proj_epoch_data[epoch][final_token] = {
                        'x': proj_vals,
                        'y': [0.0] * len(proj_vals),
                        'colors': token_data["post_mlp_prob_token1"].tolist(),
                        'customdata': np.column_stack([
                            token_data["N"], token_data["H"], token_data["post_mlp_prob_token1"],
                            np.full(len(token_data), final_token)
                        ]).tolist()
                    }
                else:
                    proj_epoch_data[epoch][final_token] = {'x': [], 'y': [], 'colors': [], 'customdata': []}
            # Also compute theoretical log-odds for this epoch (all points together)
            Ns = epoch_data["N"].values.tolist()
            Hs = epoch_data["H"].values.tolist()
            theo_z = np.log((1.0 + epoch_data["H"].values) / (1.0 + epoch_data["N"].values - epoch_data["H"].values)).tolist()
            theoretical_epoch_data[epoch] = {'x': Ns, 'y': Hs, 'z': theo_z}
    
    # Add initial traces for residuals (final epoch) - separate by final_token
    final_epoch = max(epochs)
    
    # 3. Pre-Attention residuals (bottom-left) - add traces for both final_token values
    if final_epoch in pre_attn_epoch_data:
        for final_token in [0, 1]:
            if final_token in pre_attn_epoch_data[final_epoch]:
                data = pre_attn_epoch_data[final_epoch][final_token]
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter(
                        x=data['x'],
                        y=data['y'],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=data['colors'],
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=False
                        ),
                        name=f"Pre-Attn final_token={final_token} ({'Blues' if final_token == 0 else 'Reds'})",
                        showlegend=True,
                        customdata=data['customdata'],
                        hovertemplate=f"final_token={final_token}<br>N=%{{customdata[0]}}<br>H=%{{customdata[1]}}<br>P(token=1)=%{{customdata[2]:.3f}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                    ),
                    row=2, col=1
                )

    # 4. Pre-MLP residuals (bottom-center) - add traces for both final_token values
    if final_epoch in pre_mlp_epoch_data:
        for final_token in [0, 1]:
            if final_token in pre_mlp_epoch_data[final_epoch]:
                data = pre_mlp_epoch_data[final_epoch][final_token]
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter(
                        x=data['x'],
                        y=data['y'],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=data['colors'],
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=(final_token == 1),  # Only show colorbar for Reds (final_token=1)
                            colorbar=dict(
                                title="P(token=1) - Reds", 
                                x=0.3,  # Position between pre-mlp and post-mlp plots, avoid overlap
                                len=0.28,
                                yanchor="bottom",
                                y=0.12
                            ) if final_token == 1 else None
                        ),
                        name=f"Pre-MLP final_token={final_token} ({'Blues' if final_token == 0 else 'Reds'})",
                        showlegend=True,
                        customdata=data['customdata'],
                        hovertemplate=f"final_token={final_token}<br>N=%{{customdata[0]}}<br>H=%{{customdata[1]}}<br>P(token=1)=%{{customdata[2]:.3f}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                    ),
                    row=2, col=2
                )
    
    # 5. Post-MLP residuals (bottom-right) - add traces for both final_token values
    if final_epoch in post_mlp_epoch_data:
        for final_token in [0, 1]:
            if final_token in post_mlp_epoch_data[final_epoch]:
                data = post_mlp_epoch_data[final_epoch][final_token]
                colorscale = "Blues" if final_token == 0 else "Reds"
                fig.add_trace(
                    go.Scatter(
                        x=data['x'],
                        y=data['y'],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=data['colors'],
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=(final_token == 0),  # Only show colorbar for Blues (final_token=0)
                            colorbar=dict(
                                title="P(token=1) - Blues", 
                                x=0.66,  # Position to the right of post-mlp plot
                                len=0.28,
                                yanchor="bottom",
                                y=0.12
                            ) if final_token == 0 else None
                        ),
                        name=f"Post-MLP final_token={final_token} ({'Blues' if final_token == 0 else 'Reds'})",
                        showlegend=True,
                        customdata=data['customdata'],
                        hovertemplate=f"final_token={final_token}<br>N=%{{customdata[0]}}<br>H=%{{customdata[1]}}<br>P(token=1)=%{{customdata[2]:.3f}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                    ),
                    row=2, col=3
                )
    
    # 6. Projection traces at final epoch (row 3, col 2) — 3D
    if final_epoch in proj_epoch_data:
        for final_token in [0, 1]:
            if final_token in proj_epoch_data[final_epoch]:
                data = proj_epoch_data[final_epoch][final_token]
                colorscale = "Blues" if final_token == 0 else "Reds"
                # Build 3D data from stored projection values (x=proj, we need N,H back from customdata)
                Ns = [row[0] for row in data['customdata']]
                Hs = [row[1] for row in data['customdata']]
                P1s = [row[2] for row in data['customdata']]
                fig.add_trace(
                    go.Scatter3d(
                        x=Ns,
                        y=Hs,
                        z=data['x'],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=P1s,
                            colorscale=colorscale,
                            cmin=0,
                            cmax=1,
                            showscale=(final_token == 1),
                            colorbar=dict(
                                title="P(token=1) - Reds",
                                len=0.4
                            ) if final_token == 1 else None
                        ),
                        name=f"Projection final_token={final_token}",
                        showlegend=True,
                        customdata=data['customdata'],
                        hovertemplate=f"final_token={final_token}<br>N=%{{x}}<br>H=%{{y}}<br>proj=%{{z:.3f}}<br>P1=%{{marker.color:.3f}}<extra></extra>",
                    ),
                    row=3, col=2
                )
        # Add initial theoretical log-odds trace
        if final_epoch in theoretical_epoch_data:
            tdata = theoretical_epoch_data[final_epoch]
            fig.add_trace(
                go.Scatter3d(
                    x=tdata['x'],
                    y=tdata['y'],
                    z=tdata['z'],
                    mode="markers",
                    marker=dict(size=2, color="black"),
                    name="theoretical log-odds",
                    showlegend=True,
                    hovertemplate="N=%{x}<br>H=%{y}<br>theory=%{z:.3f}<extra></extra>",
                ),
                row=3, col=2
            )
    
    # Count the number of static traces (embedding, unembedding) and dynamic traces (positional, residuals, projection)
    # that should be affected by sliders
    num_emb_traces = len(seed_emb_df["token_id"].unique()) if not seed_emb_df.empty else 0
    num_unemb_traces = len(seed_unemb_df["token_id"].unique()) if not seed_unemb_df.empty else 0
    # Positional traces are now grouped, so count groups instead of individual positions
    num_pos_traces = len(pos_group_ranges) if not seed_pos_df.empty else 0
    num_static_traces = num_emb_traces + num_unemb_traces  # Only embedding and unembedding are static
    
    # Create slider steps to update positional and residual traces; keep embedding/unembedding static
    steps = []
    for epoch in epochs:
        if epoch in pre_mlp_epoch_data and epoch in post_mlp_epoch_data:
            # Indices of traces that change with slider (positional groups + 6 residuals + 2 projections + 1 theory)
            dynamic_indices = list(range(num_static_traces, num_static_traces + num_pos_traces + 9))

            # Build arrays in the same order: all positional groups first, then residuals pre(0), pre(1), post(0), post(1), then projection (0),(1)
            x_vals = []
            y_vals = []
            z_vals = []
            color_vals = []
            custom_vals = []

            for gi in range(num_pos_traces):
                if epoch in pos_epoch_data and gi in pos_epoch_data[epoch]:
                    x_vals.append(pos_epoch_data[epoch][gi]['x'])
                    y_vals.append(pos_epoch_data[epoch][gi]['y'])
                else:
                    x_vals.append([])
                    y_vals.append([])
                # Use consistent gradient color for this group
                color_vals.append(get_position_group_color(gi, num_pos_traces))
                custom_vals.append(None)
                z_vals.append(None)

            for final_token in [0, 1]:
                d = pre_attn_epoch_data.get(epoch, {}).get(final_token, {'x': [], 'y': [], 'colors': [], 'customdata': []})
                x_vals.append(d['x'])
                y_vals.append(d['y'])
                color_vals.append(d['colors'])
                custom_vals.append(d['customdata'])
                z_vals.append(None)
            for final_token in [0, 1]:
                d = pre_mlp_epoch_data[epoch].get(final_token, {'x': [], 'y': [], 'colors': [], 'customdata': []})
                x_vals.append(d['x'])
                y_vals.append(d['y'])
                color_vals.append(d['colors'])
                custom_vals.append(d['customdata'])
                z_vals.append(None)
            for final_token in [0, 1]:
                d = post_mlp_epoch_data[epoch].get(final_token, {'x': [], 'y': [], 'colors': [], 'customdata': []})
                x_vals.append(d['x'])
                y_vals.append(d['y'])
                color_vals.append(d['colors'])
                custom_vals.append(d['customdata'])
            # Projection traces
            for final_token in [0, 1]:
                d = proj_epoch_data.get(epoch, {}).get(final_token, {'x': [], 'y': [], 'colors': [], 'customdata': []})
                # For 3D projection traces: we stored N,H in customdata and proj in x
                Ns = [row[0] for row in d['customdata']] if d['customdata'] else []
                Hs = [row[1] for row in d['customdata']] if d['customdata'] else []
                projs = d['x']
                x_vals.append(Ns)
                y_vals.append(Hs)
                z_vals.append(projs)
                color_vals.append(d['colors'])
                custom_vals.append(d['customdata'])

            # Theoretical trace
            if epoch in theoretical_epoch_data:
                tdata = theoretical_epoch_data[epoch]
                x_vals.append(tdata['x'])
                y_vals.append(tdata['y'])
                z_vals.append(tdata['z'])
                color_vals.append('black')
                custom_vals.append(None)

            step = dict(
                method="restyle",
                args=[
                    {
                        'x': x_vals,
                        'y': y_vals,
                        'z': z_vals,
                        'marker.color': color_vals,
                        'customdata': custom_vals
                    },
                    dynamic_indices
                ],
                label=f"{'Init' if epoch == 0 else f'Epoch {epoch}'}"
            )
            steps.append(step)
    
    # Add slider
    if len(steps) > 1:
        sliders = [dict(
            active=len(steps)-1,  # Start with final epoch
            pad={"t": 30},
            steps=steps
        )]
    else:
        sliders = []
    
    # Update layout
    fig.update_layout(
        title=f"Joint Analysis with Sliders - Seed {seed} - {'init' if final_epoch == 0 else f'epoch {final_epoch}'} — {title_suffix}",
        height=1400,
        width=1800,  # Even wider to accommodate colorbars
        showlegend=True,  # Show legends incl. positional groups
        sliders=sliders
    )
    if sliders:
        fig.update_layout(sliders=[dict(
            active=sliders[0]['active'],
            pad=sliders[0]['pad'],
            steps=steps,
            currentvalue=dict(prefix="Epoch: ", visible=True)
        )])
    
    # Update axes to maintain aspect ratio and add origin lines
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.update_xaxes(title_text="dim 1", row=row, col=col)
            fig.update_yaxes(title_text="dim 2", scaleanchor=f"x{'' if row==1 and col==1 else (row-1)*3+col}", 
                           scaleratio=1, row=row, col=col)
            # Add origin lines to each subplot
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.3, row=row, col=col, exclude_empty_subplots=False)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.3, row=row, col=col, exclude_empty_subplots=False)
    # Row 3 scene formatting (3D) — use scene2 for (row=3,col=2)
    fig.update_layout(scene2=dict(xaxis_title="N", yaxis_title="H", zaxis_title="projection"))
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved joint plot with sliders: {save_path}")
    
    return fig


def main():
    """Main plotting process."""
    print(f"Starting plotting for exp_08")
    print(f"Reading data from: {EXP_RESULTS_DIR}")
    
    # Load aggregated data
    aggregated_dir = os.path.join(EXP_RESULTS_DIR, "aggregated")
    
    try:
        epoch_df = pd.read_csv(os.path.join(aggregated_dir, "aggregated_epoch_data.csv"))
        print(f"Loaded epoch data: {len(epoch_df)} rows")
    except FileNotFoundError:
        print("No aggregated epoch data found")
        epoch_df = pd.DataFrame()
    
    try:
        emb_df = pd.read_csv(os.path.join(aggregated_dir, "aggregated_embed_trajectories.csv"))
        print(f"Loaded embedding trajectories: {len(emb_df)} rows")
    except FileNotFoundError:
        print("No aggregated embedding data found")
        emb_df = pd.DataFrame()
    
    try:
        unemb_df = pd.read_csv(os.path.join(aggregated_dir, "aggregated_unembed_trajectories.csv"))
        print(f"Loaded unembedding trajectories: {len(unemb_df)} rows")
    except FileNotFoundError:
        print("No aggregated unembedding data found")
        unemb_df = pd.DataFrame()
    
    try:
        bias_df = pd.read_csv(os.path.join(aggregated_dir, "aggregated_unembed_bias_diffs.csv"))
        print(f"Loaded unembed bias diffs: {len(bias_df)} rows")
    except FileNotFoundError:
        print("No aggregated unembed bias diffs found")
        bias_df = pd.DataFrame()
    
    try:
        pos_df = pd.read_csv(os.path.join(aggregated_dir, "aggregated_pos_embed_trajectories.csv"))
        print(f"Loaded positional embedding trajectories: {len(pos_df)} rows")
    except FileNotFoundError:
        print("No aggregated positional embedding data found")
        pos_df = pd.DataFrame()
    
    # Create plots directory
    plots_dir = os.path.join(EXP_RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    title_suffix = build_title_suffix(EXP_CONFIG)
    
    # Generate per-seed plots
    print("\nGenerating per-seed plots...")
    for seed in SEEDS:
        print(f"Processing seed {seed}...")
        seed_plots_dir = os.path.join(plots_dir, f"seed_{seed}")
        os.makedirs(seed_plots_dir, exist_ok=True)
        
        # Filter data by seed
        seed_epoch_df = epoch_df[epoch_df["seed"] == seed] if not epoch_df.empty else pd.DataFrame()
        seed_emb_df = emb_df[emb_df["seed"] == seed] if not emb_df.empty else pd.DataFrame()
        seed_unemb_df = unemb_df[unemb_df["seed"] == seed] if not unemb_df.empty else pd.DataFrame()
        seed_pos_df = pos_df[pos_df["seed"] == seed] if not pos_df.empty else pd.DataFrame()
        
        # 1. Residual heatmap plots with probability coloring
        if not seed_epoch_df.empty:
            print(f"  Creating residual heatmap plots...")
            pre_mlp_figs = plot_residual_heatmap(
                seed_epoch_df, "pre_mlp", seed,
                save_html_prefix=os.path.join(seed_plots_dir, "residual_heatmap"),
                title_suffix=title_suffix
            )
            post_mlp_figs = plot_residual_heatmap(
                seed_epoch_df, "post_mlp", seed,
                save_html_prefix=os.path.join(seed_plots_dir, "residual_heatmap"),
                title_suffix=title_suffix
            )
        
        # 2. Token trajectory plots
        if not seed_emb_df.empty:
            print(f"  Creating embedding trajectory plot...")
            emb_fig = plot_token_trajectories(
                seed_emb_df, f"Seed {seed} - Embedding trajectories — {title_suffix}", seed
            )
            emb_path = os.path.join(seed_plots_dir, f"embedding_trajectories_seed{seed}.html")
            emb_fig.write_html(emb_path)
            print(f"  Saved {emb_path}")
        
        if not seed_unemb_df.empty:
            print(f"  Creating unembedding trajectory plot...")
            unemb_fig = plot_token_trajectories(
                seed_unemb_df, f"Seed {seed} - Unembedding trajectories — {title_suffix}", seed
            )
            unemb_path = os.path.join(seed_plots_dir, f"unembedding_trajectories_seed{seed}.html")
            unemb_fig.write_html(unemb_path)
            print(f"  Saved {unemb_path}")
        
        # 2b. Standalone post-MLP projection plot
        if not seed_epoch_df.empty and not seed_unemb_df.empty:
            print(f"  Creating post-MLP projection plot...")
            proj_path = os.path.join(seed_plots_dir, f"post_mlp_projection_seed{seed}.html")
            seed_bias_df = bias_df[bias_df["seed"] == seed] if not bias_df.empty else pd.DataFrame()
            _ = plot_post_mlp_projection(seed_epoch_df, seed_unemb_df, seed, save_path=proj_path, title_suffix=title_suffix, seed_bias_df=seed_bias_df)
            print(f"  Saved {proj_path}")
            
        # 3. Positional embedding trajectory plots
        if not seed_pos_df.empty:
            print(f"  Creating positional embedding trajectory plot...")
            pos_fig = plot_positional_trajectories(
                pos_df, f"Positional Embedding Trajectories - Seed {seed}", seed,
                save_html_prefix=os.path.join(seed_plots_dir, "positional_trajectories"),
                title_suffix=title_suffix
            )
        
        # 4. Joint seed plot
        if not seed_epoch_df.empty or not seed_emb_df.empty or not seed_unemb_df.empty or not seed_pos_df.empty:
            print(f"  Creating joint analysis plot...")
            joint_path = os.path.join(seed_plots_dir, f"joint_analysis_seed{seed}.html")
            joint_fig = create_joint_seed_plot(
                epoch_df, emb_df, unemb_df, bias_df, pos_df, seed,
                save_path=joint_path,
                title_suffix=title_suffix
            )
            
            # 5. Joint seed plot with sliders
            print(f"  Creating joint analysis plot with sliders...")
            joint_slider_path = os.path.join(seed_plots_dir, f"joint_analysis_with_sliders_seed{seed}.html")
            joint_slider_fig = create_joint_seed_plot_with_sliders(
                epoch_df, emb_df, unemb_df, bias_df, pos_df, seed,
                save_path=joint_slider_path,
                title_suffix=title_suffix
            )
    
    print(f"\nPlotting completed!")
    print(f"All plots saved to: {plots_dir}")

#%%
if __name__ == "__main__":
    main()
# %%
