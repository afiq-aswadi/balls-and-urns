"""
Loading in one of the tiny transformers and seeing what's going 
"""

#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(PLOT_DIR, exist_ok=True)

import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Optional, Tuple

from core.models import create_coinformer_model

from exp_08_config import EXP_CONFIG, SEEDS, DEVICE, get_seed_checkpoint_dir

from exp_08_data_collection import build_probe_batch, load_checkpoints_sorted, filter_checkpoints_by_config
#%%
TARGET_SEEDS = list(SEEDS)
if len(TARGET_SEEDS) == 0:
    raise ValueError("No seeds provided in SEEDS configuration.")


model_config = EXP_CONFIG.model_config


model = create_coinformer_model(model_config).to(DEVICE)
model.eval()



# %%
probe_batch = build_probe_batch(20, use_bos=False).to(DEVICE)

#%%   
normalise = True  # if true, then h_on_n = (h + 1) / (n + 2)


def get_probe_n_and_h(probe_batch: torch.Tensor, normalise: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return slope, intercept, and R^2 for a least-squares line fit."""
    if x.size < 2:
        return None
    if np.allclose(x, x[0]):
        return None
    if np.allclose(y, y[0]):
        return None

    a = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(a, y, rcond=None)[0]

    y_pred = slope * x + intercept
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return None
    ss_res = np.sum((y - y_pred) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return slope, intercept, r_squared


def build_outlier_mask(
    h_vals: np.ndarray,
    n_vals: np.ndarray,
) -> np.ndarray:
    """Flag positions considered outliers for regression fitting."""
    return np.isclose(h_vals, 0.0) | np.isclose(n_vals - h_vals, 0.0)


def plot_last_checkpoint_scatter(
    seed: int,
    ckpt_label: str,
    cache_z: torch.Tensor,
    h_on_n: torch.Tensor,
    n: torch.Tensor,
    h: torch.Tensor,
    probe_batch_seed: torch.Tensor,
    normalise: bool,
    fig_dir: str,
) -> None:
    cache_z_cpu = cache_z.detach().cpu()
    h_on_n_cpu = h_on_n.detach().cpu()
    n_cpu = n.detach().cpu()
    h_cpu = h.detach().cpu()

    mask = torch.isfinite(h_on_n_cpu) & (n_cpu > 0)
    if mask.sum().item() == 0:
        print(f"[seed {seed}] No valid points to plot.")
        return

    seq_idx, pos_idx = torch.nonzero(mask, as_tuple=True)
    x_vals = h_on_n_cpu[seq_idx, pos_idx].numpy()
    y_vals = cache_z_cpu[seq_idx, pos_idx].numpy()
    token_vals = probe_batch_seed.detach().cpu()[seq_idx, pos_idx].numpy()
    h_vals = h_cpu[seq_idx, pos_idx].numpy()
    n_vals = n_cpu[seq_idx, pos_idx].numpy()

    valid_tokens = np.isin(token_vals, (0, 1))
    if not np.any(valid_tokens):
        print(f"[seed {seed}] No 0/1 tokens present for plotting.")
        return

    x_vals = x_vals[valid_tokens]
    y_vals = y_vals[valid_tokens]
    token_vals = token_vals[valid_tokens]
    h_vals = h_vals[valid_tokens]
    n_vals = n_vals[valid_tokens]

    ratio_label = '(h+1)/(n+2)' if normalise else 'h/n'
    token_colors = {0: '#1f77b4', 1: '#d62728'}
    info_lines = []

    fig, ax = plt.subplots(figsize=(8, 6))

    for token in (0, 1):
        token_mask = token_vals == token
        if not np.any(token_mask):
            continue

        x_token = x_vals[token_mask]
        y_token = y_vals[token_mask]
        h_token = h_vals[token_mask]
        n_token = n_vals[token_mask]

        ax.scatter(
            x_token,
            y_token,
            s=20,
            alpha=0.6,
            color=token_colors[token],
            label=f'token {token} points',
        )

        outlier_mask = build_outlier_mask(h_token, n_token)
        fit_all = fit_linear_regression(x_token, y_token)

        if fit_all is not None:
            slope_all, intercept_all, r2_all = fit_all
            x_line = np.linspace(x_token.min(), x_token.max(), 100)
            y_line = slope_all * x_line + intercept_all
            ax.plot(
                x_line,
                y_line,
                color=token_colors[token],
                linewidth=2,
                label=f'token {token} regression (all)',
            )
            info_lines.append(f'token {token} all: R²={r2_all:.4f}')
            info_lines.append(f'token {token} all slope: {slope_all:.4f}')
            print(f'[seed {seed} | token {token}] R² (all points): {r2_all:.4f}')
        else:
            print(f'[seed {seed} | token {token}] Not enough data for regression (all points).')

        keep_mask = ~outlier_mask
        if np.count_nonzero(keep_mask) >= 2:
            x_filtered = x_token[keep_mask]
            y_filtered = y_token[keep_mask]
            fit_filtered = fit_linear_regression(x_filtered, y_filtered)
            if fit_filtered is not None:
                slope_f, intercept_f, r2_f = fit_filtered
                x_line = np.linspace(x_filtered.min(), x_filtered.max(), 100)
                y_line = slope_f * x_line + intercept_f
                ax.plot(
                    x_line,
                    y_line,
                    color=token_colors[token],
                    linewidth=2,
                    linestyle='--',
                    label=f'token {token} regression (filtered)',
                )
                info_lines.append(f'token {token} filtered: R²={r2_f:.4f}')
                info_lines.append(f'token {token} filtered slope: {slope_f:.4f}')
                print(f'[seed {seed} | token {token}] R² (filtered): {r2_f:.4f}')
            else:
                print(f'[seed {seed} | token {token}] Not enough data for filtered regression.')
        else:
            print(f'[seed {seed} | token {token}] Not enough points after filtering outliers.')

    if not info_lines:
        info_lines.append('No regression fits available.')

    ax.text(
        0.02,
        0.98,
        '\n'.join(info_lines),
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
    )

    ax.set_xlabel(ratio_label)
    ax.set_ylabel('cache z')
    ax.set_title(f'seed {seed} | {ckpt_label}')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    safe_label = re.sub(r'[^0-9A-Za-z_-]+', '_', ckpt_label)
    filename = f'cache_z_vs_ratio_seed_{seed}_{safe_label}_normalise_{normalise}_linear_regression.png'
    fig_path = os.path.join(fig_dir, filename)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f'[seed {seed}] Saved figure to {fig_path}')
# %%
plots_generated = False

for seed in TARGET_SEEDS:
    checkpoint_dir = get_seed_checkpoint_dir(seed)
    all_ckpts = load_checkpoints_sorted(checkpoint_dir)
    ckpts = filter_checkpoints_by_config(all_ckpts, EXP_CONFIG, seed)

    if len(ckpts) == 0:
        print(f"[seed {seed}] No checkpoints found after filtering.")
        continue

    last_ckpt_path = ckpts[-1]
    ckpt = torch.load(last_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, cache = model.run_with_cache(probe_batch)
    cache_z_last = cache["z", -1].squeeze()

    h_on_n_aligned = h_on_n[:cache_z_last.shape[0], :cache_z_last.shape[1]]
    n_aligned = n[:cache_z_last.shape[0], :cache_z_last.shape[1]]
    h_aligned = h[:cache_z_last.shape[0], :cache_z_last.shape[1]]
    probe_batch_aligned = probe_batch[:cache_z_last.shape[0], :cache_z_last.shape[1]]

    ckpt_label = os.path.basename(last_ckpt_path)
    plot_last_checkpoint_scatter(
        seed=seed,
        ckpt_label=ckpt_label,
        cache_z=cache_z_last,
        h_on_n=h_on_n_aligned,
        n=n_aligned,
        h=h_aligned,
        probe_batch_seed=probe_batch_aligned,
        normalise=normalise,
        fig_dir=PLOT_DIR,
    )
    plots_generated = True

if not plots_generated:
    raise ValueError("No checkpoints available for plotting the latest cache activations.")

# %%
