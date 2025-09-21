"""
Loading in one of the tiny transformers and seeing what's going 
"""

#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_DIR = os.path.join(PROJECT_ROOT, "html")
os.makedirs(HTML_DIR, exist_ok=True)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from typing import List, Tuple, Dict, Any
import einops
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import create_coinformer_model
from core.config import ModelConfig
from core.samplers import add_bos_token

from exp_08_config import (
    EXP_CONFIG, SEEDS, DEVICE,
    get_seed_checkpoint_dir, get_seed_results_dir, set_global_seed, EXP_RESULTS_DIR
)

from exp_08_data_collection import build_probe_batch, load_checkpoints_sorted, filter_checkpoints_by_config, get_seed_checkpoint_dir, compute_token_probabilities
#%%
TARGET_SEEDS = list(SEEDS)
if len(TARGET_SEEDS) == 0:
    raise ValueError("No seeds provided in SEEDS configuration.")


model_config = EXP_CONFIG.model_config


model = create_coinformer_model(model_config).to(DEVICE)
model.eval()



# %%
# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Print parameters by layer/component
print("\nParameters by component:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters, shape {list(param.shape)}")

#%%

probe_batch = build_probe_batch(20, use_bos=False).to(DEVICE)

#%%   
normalise = True #if true, then h_on_n = (h + 1) / (n + 2)
def get_probe_n_and_h(probe_batch: torch.Tensor, normalise: bool = False) -> Tuple[torch.Tensor, torch.Tensor]: #todo: transfer to other script
    """
    Given a batch of binary sequences (1=head, 0=tails) possibly with a BOS or other
    non-{0,1} tokens, return two tensors (n, h) of the same shape as probe_batch:

      - n[..., t] = number of coin flips (0 or 1) seen up to and including position t
      - h[..., t] = number of heads (==1) seen up to and including position t

    Any token not equal to 0 or 1 (e.g., a BOS token) is ignored in both counts.
    Works for any batch shape; counts are computed along the last dimension.
    """
    # Identify valid flips and heads
    is_flip = (probe_batch == 0) | (probe_batch == 1)
    is_head = (probe_batch == 1)

    # Inclusive cumulative sums along the sequence dimension
    n = is_flip.to(probe_batch.dtype).cumsum(dim=-1)
    h = is_head.to(probe_batch.dtype).cumsum(dim=-1)
    if normalise:
        one = torch.tensor(1, device=probe_batch.device, dtype=probe_batch.dtype)
        two = torch.tensor(2, device=probe_batch.device, dtype=probe_batch.dtype)
        h_on_n = (h + one) / (n + two)
    else:
        h_on_n = h / n
    return n, h, h_on_n


n, h, h_on_n = get_probe_n_and_h(probe_batch, normalise=normalise)


# %%
def prepare_checkpoint_scatter(
    probe_batch: torch.Tensor,
    cache_z: torch.Tensor,
    h_on_n: torch.Tensor,
    n: torch.Tensor,
    h: torch.Tensor,
    label: str,
    label_short: str,
    seed: int,
) -> dict:
    """Build scatter plot data for a single checkpoint."""
    cache_z_cpu = cache_z.detach().cpu()
    h_on_n_cpu = h_on_n.detach().cpu()
    n_cpu = n.detach().cpu()
    h_cpu = h.detach().cpu()
    probe_batch_cpu = probe_batch.detach().cpu()

    mask = torch.isfinite(h_on_n_cpu) & (n_cpu > 0)
    if mask.sum().item() == 0:
        return {}

    seq_idx, pos_idx = torch.nonzero(mask, as_tuple=True)

    x_vals = h_on_n_cpu[seq_idx, pos_idx].numpy()
    y_vals = cache_z_cpu[seq_idx, pos_idx].numpy()
    n_vals = n_cpu[seq_idx, pos_idx].numpy()
    h_vals = h_cpu[seq_idx, pos_idx].numpy()
    token_vals = probe_batch_cpu[seq_idx, pos_idx].numpy()

    point_colors = np.where(
        token_vals == 1,
        '#d62728',  # heads (token == 1)
        np.where(token_vals == 0, '#1f77b4', '#7f7f7f')  # tails vs other tokens
    )

    customdata = np.column_stack([
        seq_idx.numpy(),
        pos_idx.numpy(),
        token_vals,
        n_vals,
        h_vals,
    ])

    return {
        "x": x_vals,
        "y": y_vals,
        "colors": point_colors,
        "customdata": customdata,
        "label": label,
        "label_short": label_short,
        "seed": seed,
    }


def plot_cache_z_vs_h_on_n(
    scatter_datasets: List[dict],
    html_dir: str,
    seed: int,
    filename: str = "cache_z_vs_h_on_n"
) -> None:
    """Scatter plot of cache z values vs empirical head frequency with checkpoint slider."""
    scatter_datasets = [data for data in scatter_datasets if data]
    if not scatter_datasets:
        return

    title_base = (
        f'Cache z vs (h+1)/(n+2) Scatter (Seed {seed})'
        if normalise
        else f'Cache z vs h/n Scatter (Seed {seed})'
    )
    ratio_label = '(h+1)/(n+2)' if normalise else 'h/n'

    fig = go.Figure()
    slider_steps = []

    # Fix a shared y-axis range so the slider does not rescale traces
    y_all = np.concatenate([
        np.asarray(data["y"]).ravel()
        for data in scatter_datasets
        if "y" in data and data["y"].size > 0
    ])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        if y_min == y_max:
            padding = 1.0 if y_min == 0 else abs(y_min) * 0.1
            y_min -= padding
            y_max += padding
        else:
            padding = 0.05 * (y_max - y_min)
            y_min -= padding
            y_max += padding
    else:
        y_min = y_max = None

    for idx, data in enumerate(scatter_datasets):
        visible = (idx == 0)
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode='markers',
            marker=dict(size=6, opacity=0.7, color=data["colors"]),
            customdata=data["customdata"],
            name=data["label"],
            visible=visible,
            hovertemplate=(
                f'checkpoint={data["label"]}'
                '<br>seq=%{customdata[0]}'
                '<br>pos=%{customdata[1]}'
                '<br>token=%{customdata[2]}'
                '<br>n=%{customdata[3]}'
                '<br>h=%{customdata[4]}'
                f'<br>{ratio_label}=%{{x:.3f}}'
                '<br>z=%{y}<extra></extra>'
            )
        ))

        visible_array = [False] * len(scatter_datasets)
        visible_array[idx] = True
        slider_steps.append(dict(
            method='update',
            label=data["label_short"],
            args=[
                {"visible": visible_array},
                {"title": {"text": f'{title_base}<br>{data["label"]}'}}
            ],
        ))

    fig.update_layout(
        title=dict(text=f'{title_base}<br>{scatter_datasets[0]["label"]}', x=0.5),
        xaxis_title='(h+1)/(n+2)' if normalise else 'h/n',
        yaxis_title='cache z',
        showlegend=False,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Checkpoint: "},
            pad={"t": 30},
            steps=slider_steps,
        )]
    )
    fig.update_xaxes(range=[0, 1])
    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    html_path = os.path.join(html_dir, f'{filename}_seed_{seed}_normalise_{normalise}.html')
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
# %%
plots_generated = False
per_seed_latest: Dict[int, Dict[str, Any]] = {}

for seed in TARGET_SEEDS:
    checkpoint_dir = get_seed_checkpoint_dir(seed)
    all_ckpts = load_checkpoints_sorted(checkpoint_dir)
    ckpts = filter_checkpoints_by_config(all_ckpts, EXP_CONFIG, seed)

    if len(ckpts) == 0:
        continue

    seed_scatter: List[dict] = []

    for idx, ckpt_path in enumerate(ckpts):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        _, cache = model.run_with_cache(probe_batch)
        cache_z_current = cache["z", -1].squeeze()

        h_on_n_aligned = h_on_n[:cache_z_current.shape[0], :cache_z_current.shape[1]]
        n_aligned = n[:cache_z_current.shape[0], :cache_z_current.shape[1]]
        h_aligned = h[:cache_z_current.shape[0], :cache_z_current.shape[1]]
        probe_batch_aligned = probe_batch[:cache_z_current.shape[0], :cache_z_current.shape[1]]

        ckpt_label = os.path.basename(ckpt_path)
        step_match = re.search(r'(\d+)', ckpt_label)
        ckpt_label_short = step_match.group(1) if step_match else f'idx{idx}'
        label = f'seed={seed} | {ckpt_label}'
        label_short = f'{seed}-{ckpt_label_short}'

        scatter_data = prepare_checkpoint_scatter(
            probe_batch=probe_batch_aligned,
            cache_z=cache_z_current,
            h_on_n=h_on_n_aligned,
            n=n_aligned,
            h=h_aligned,
            label=label,
            label_short=label_short,
            seed=seed,
        )
        if scatter_data:
            seed_scatter.append(scatter_data)
            per_seed_latest[seed] = {
                "cache_z": cache_z_current,
                "n": n_aligned,
                "h": h_aligned,
                "probe": probe_batch_aligned,
                "ckpt_path": ckpt_path,
            }

    if seed_scatter:
        plot_cache_z_vs_h_on_n(
            scatter_datasets=seed_scatter,
            html_dir=HTML_DIR,
            seed=seed,
        )
        plots_generated = True

if not plots_generated:
    raise ValueError("No checkpoints available for plotting cache activations across seeds.")

if not per_seed_latest:
    raise ValueError("No seed data available for downstream visualisations.")
# %% Plotly interactive heatmaps with n,h in hover and HTML export

# Probe batch interactive heatmap
batch_size, seq_len = probe_batch.shape
customdata = torch.stack([n, h], dim=-1).cpu().numpy()  # [B, T, 2]
fig_probe = go.Figure(data=go.Heatmap(
    z=probe_batch.cpu().numpy(),
    x=list(range(seq_len)),
    y=list(range(batch_size)),
    colorscale='RdBu',
    zmid=0,
    customdata=customdata,
    hovertemplate='seq=%{y}<br>pos=%{x}<br>n=%{customdata[0]}<br>h=%{customdata[1]}<br>value=%{z}<extra></extra>'
))
fig_probe.update_layout(
    title_text='Probe Batch Sequences',
    xaxis_title='Position',
    yaxis_title='Sequence'
)
fig_probe.write_html(os.path.join(HTML_DIR, 'probe_batch_all_seeds.html'), include_plotlyjs='cdn', full_html=True)

for seed, seed_data in per_seed_latest.items():
    cache_z_seed = seed_data["cache_z"]
    n_seed = seed_data["n"]
    h_seed = seed_data["h"]
    probe_seed = seed_data["probe"]

    z_arr = cache_z_seed.detach().cpu().numpy()
    b2, t2 = z_arr.shape
    nh_custom = torch.stack([n_seed, h_seed], dim=-1).cpu().numpy()

    fig_cache = go.Figure(data=go.Heatmap(
        z=z_arr,
        x=list(range(t2)),
        y=list(range(b2)),
        colorscale='RdBu',
        zmid=0,
        customdata=nh_custom,
        hovertemplate='seq=%{y}<br>pos=%{x}<br>n=%{customdata[0]}<br>h=%{customdata[1]}<br>z=%{z}<extra></extra>'
    ))
    fig_cache.update_layout(
        title_text=f'Cache z (attention function) Seed {seed}',
        xaxis_title='Position',
        yaxis_title='Sequence'
    )
    fig_cache.write_html(os.path.join(HTML_DIR, f'cache_z_seed_{seed}.html'), include_plotlyjs='cdn', full_html=True)

    fig_lines = go.Figure()
    for i in range(b2):
        fig_lines.add_trace(go.Scatter(
            y=z_arr[i],
            x=list(range(t2)),
            name=f'Sequence {i}',
            customdata=nh_custom[i],
            hovertemplate='pos=%{x}<br>z=%{y}<br>n=%{customdata[0]}<br>h=%{customdata[1]}<extra></extra>'
        ))

    fig_lines.update_layout(
        title_text=f'Cache z Values Per Sequence (Seed {seed})',
        xaxis_title='Position',
        yaxis_title='z value',
        showlegend=True
    )

    fig_lines.write_html(os.path.join(HTML_DIR, f'cache_z_lines_seed_{seed}.html'), include_plotlyjs='cdn', full_html=True)

    # Aggregate cache activations over (tails, heads) counts and split by final token
    tails_seed = n_seed - h_seed
    seq_len_seed = probe_seed.shape[1]
    grid_size = seq_len_seed + 1  # allow all possible counts from 0..seq_len
    grid_shape = (grid_size, grid_size)

    ht_sums = {token: np.zeros(grid_shape, dtype=float) for token in (0, 1)}
    ht_counts = {token: np.zeros(grid_shape, dtype=int) for token in (0, 1)}

    z_np = z_arr
    h_np = h_seed.detach().cpu().numpy()
    t_np = tails_seed.detach().cpu().numpy()
    n_np = n_seed.detach().cpu().numpy()
    token_np = probe_seed.detach().cpu().numpy()

    for seq_idx in range(b2):
        for pos_idx in range(t2):
            z_val = z_np[seq_idx, pos_idx]
            if not np.isfinite(z_val):
                continue
            token_val = int(token_np[seq_idx, pos_idx])
            if token_val not in (0, 1):
                continue
            if n_np[seq_idx, pos_idx] <= 0:
                continue
            h_val = int(h_np[seq_idx, pos_idx])
            t_val = int(t_np[seq_idx, pos_idx])
            if not (0 <= h_val <= seq_len_seed and 0 <= t_val <= seq_len_seed):
                continue
            ht_sums[token_val][t_val, h_val] += z_val
            ht_counts[token_val][t_val, h_val] += 1

    ht_avg = {token: np.full(grid_shape, np.nan, dtype=float) for token in (0, 1)}
    for token in (0, 1):
        mask = ht_counts[token] > 0
        if np.any(mask):
            ht_avg[token][mask] = ht_sums[token][mask] / ht_counts[token][mask]

    non_nan_lists = [arr[~np.isnan(arr)] for arr in ht_avg.values() if np.any(~np.isnan(arr))]
    if non_nan_lists:
        combined_values = np.concatenate(non_nan_lists)
        color_extent = float(np.max(np.abs(combined_values)))
        if color_extent == 0.0:
            color_extent = 1.0
    else:
        color_extent = 1.0

    fig_ht = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['Token = 0 (Tails)', 'Token = 1 (Heads)'],
        horizontal_spacing=0.08,
    )

    x_values = list(range(grid_size))
    y_values = list(range(grid_size))

    for col_idx, token in enumerate((0, 1), start=1):
        fig_ht.add_trace(
            go.Heatmap(
                z=ht_avg[token],
                x=x_values,
                y=y_values,
                coloraxis='coloraxis',
                customdata=ht_counts[token],
                hovertemplate='T=%{y}<br>H=%{x}<br>avg z=%{z:.3f}<br>count=%{customdata}<extra></extra>',
            ),
            row=1,
            col=col_idx,
        )

    fig_ht.update_yaxes(autorange='reversed', row=1, col=1)
    fig_ht.update_yaxes(autorange='reversed', row=1, col=2)
    fig_ht.update_xaxes(title_text='H count', row=1, col=1)
    fig_ht.update_xaxes(title_text='H count', row=1, col=2)
    fig_ht.update_yaxes(title_text='T count', row=1, col=1)
    fig_ht.update_yaxes(title_text='T count', row=1, col=2)

    fig_ht.update_layout(
        title_text=f'Cache z by H/T Counts (Seed {seed})',
        coloraxis=dict(
            colorscale='RdBu',
            cmid=0,
            cmin=-color_extent,
            cmax=color_extent,
            colorbar_title='cache z',
        ),
    )

    fig_ht.write_html(
        os.path.join(HTML_DIR, f'cache_z_heads_tails_seed_{seed}.html'),
        include_plotlyjs='cdn',
        full_html=True,
    )

    ckpt_path = seed_data["ckpt_path"]
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt = torch.zeros(probe_seed.shape[1], dtype=probe_seed.dtype, device=probe_seed.device)
    logits, _ = model.run_with_cache(prompt)

    probs = torch.softmax(logits, dim=-1)
    if probs.dim() == 3:
        prob_token_1_tensor = probs[0, :, 1]
    else:
        prob_token_1_tensor = probs[:, 1]
    prob_token_1 = prob_token_1_tensor.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(prob_token_1)), prob_token_1, 'bo-', linewidth=2, markersize=6, label='P(next token = 1)')
    plt.title(f'Probability of Next Token Being 1 (Seed {seed})')
    plt.xlabel('Token Position')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HTML_DIR, f'prob_next_token_1_seed_{seed}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
# %%
