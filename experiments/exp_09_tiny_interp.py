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
from typing import List, Tuple
import einops
import plotly.graph_objects as go

from core.models import create_coinformer_model
from core.config import ModelConfig
from core.samplers import add_bos_token

from exp_08_config import (
    EXP_CONFIG, SEEDS, DEVICE,
    get_seed_checkpoint_dir, get_seed_results_dir, set_global_seed, EXP_RESULTS_DIR
)

from exp_08_data_collection import build_probe_batch, load_checkpoints_sorted, filter_checkpoints_by_config, get_seed_checkpoint_dir, compute_token_probabilities
#%%
SEED = 300
checkpoint_dir = get_seed_checkpoint_dir(SEED)
all_ckpts = load_checkpoints_sorted(checkpoint_dir)
ckpts = filter_checkpoints_by_config(all_ckpts, EXP_CONFIG, SEED)
ckpt = torch.load(ckpts[-1],map_location=DEVICE,weights_only=False)


model_config = EXP_CONFIG.model_config


model = create_coinformer_model(model_config).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
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

probe_batch = build_probe_batch(20, use_bos=False)

#%%   

def get_probe_n_and_h(probe_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: #todo: transfer to other script
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
    h_on_n = h / n
    return n, h, h_on_n


n, h, h_on_n = get_probe_n_and_h(probe_batch)


# %%
logits, cache = model.run_with_cache(probe_batch)
cache_z = cache["z",-1]
cache_z = cache_z.squeeze()


def plot_cache_z_vs_h_on_n(
    cache_z: torch.Tensor,
    h_on_n: torch.Tensor,
    n: torch.Tensor,
    h: torch.Tensor,
    html_dir: str,
    seed: int,
    filename: str = "cache_z_vs_h_on_n"
) -> None:
    """Scatter plot of cache z values against empirical head frequency (h/n)."""
    cache_z_cpu = cache_z.detach().cpu()
    h_on_n_cpu = h_on_n.detach().cpu()
    n_cpu = n.detach().cpu()
    h_cpu = h.detach().cpu()

    mask = torch.isfinite(h_on_n_cpu) & (n_cpu > 0)
    if mask.sum().item() == 0:
        return

    seq_idx, pos_idx = torch.nonzero(mask, as_tuple=True)

    x_vals = h_on_n_cpu[seq_idx, pos_idx].numpy()
    y_vals = cache_z_cpu[seq_idx, pos_idx].numpy()
    n_vals = n_cpu[seq_idx, pos_idx].numpy()
    h_vals = h_cpu[seq_idx, pos_idx].numpy()

    customdata = np.column_stack([
        seq_idx.numpy(),
        pos_idx.numpy(),
        n_vals,
        h_vals,
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=dict(size=6, opacity=0.7),
        customdata=customdata,
        hovertemplate='seq=%{customdata[0]}<br>pos=%{customdata[1]}<br>n=%{customdata[2]}<br>h=%{customdata[3]}<br>h/n=%{x:.3f}<br>z=%{y}<extra></extra>'
    ))

    fig.update_layout(
        title_text=f'Cache z vs h/n Scatter (Seed {seed})',
        xaxis_title='h/n',
        yaxis_title='cache z',
        showlegend=False
    )
    fig.update_xaxes(range=[0, 1])

    html_path = os.path.join(html_dir, f'{filename}_seed_{seed}.html')
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
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
    title_text=f'Probe Batch Sequences (Seed {SEED})',
    xaxis_title='Position',
    yaxis_title='Sequence'
)
fig_probe.write_html(os.path.join(HTML_DIR, f'probe_batch_seed_{SEED}.html'), include_plotlyjs='cdn', full_html=True)

# Cache z interactive heatmap
z_arr = cache_z.cpu().detach().numpy()
b2, t2 = z_arr.shape
if (b2, t2) != (batch_size, seq_len):
    nh_custom = torch.stack([n[:b2, :t2], h[:b2, :t2]], dim=-1).cpu().numpy()
else:
    nh_custom = customdata

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
    title_text=f'Cache z (attention function) Seed {SEED}',
    xaxis_title='Position',
    yaxis_title='Sequence'
)
fig_cache.write_html(os.path.join(HTML_DIR, f'cache_z_seed_{SEED}.html'), include_plotlyjs='cdn', full_html=True)
# %%
# Line plots for each row in cache_z
fig_lines = go.Figure()

for i in range(cache_z.shape[0]):
    fig_lines.add_trace(go.Scatter(
        y=cache_z[i].cpu().numpy(),
        x=list(range(cache_z.shape[1])),
        name=f'Sequence {i}',
        customdata=nh_custom[i],
        hovertemplate='pos=%{x}<br>z=%{y}<br>n=%{customdata[0]}<br>h=%{customdata[1]}<extra></extra>'
    ))

fig_lines.update_layout(
    title_text=f'Cache z Values Per Sequence (Seed {SEED})',
    xaxis_title='Position',
    yaxis_title='z value',
    showlegend=True
)

fig_lines.write_html(os.path.join(HTML_DIR, f'cache_z_lines_seed_{SEED}.html'), include_plotlyjs='cdn', full_html=True)

# %%
h_on_n_aligned = h_on_n[:cache_z.shape[0], :cache_z.shape[1]]
n_aligned = n[:cache_z.shape[0], :cache_z.shape[1]]
h_aligned = h[:cache_z.shape[0], :cache_z.shape[1]]

plot_cache_z_vs_h_on_n(
    cache_z=cache_z,
    h_on_n=h_on_n_aligned,
    n=n_aligned,
    h=h_aligned,
    html_dir=HTML_DIR,
    seed=SEED,
)

# %%
model.embed.W_E[1] - model.embed.W_E[0]
# %%
model.W_O.squeeze()
# %%
print(model.W_Q.squeeze())
print(model.W_K.squeeze())
print(model.W_V.squeeze())
# %%
print(model.W_E)
# %%
