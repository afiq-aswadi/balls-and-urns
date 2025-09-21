"""
Loading in one of the tiny transformers and seeing what's going 

If PosEmbedType is Linear, then positional embedding = (i / n_ctx) 
If PosEmbedType is Log, then positional embedding = log((i+1) / (n_ctx+1))

In the final dimension.
"""

#%%
import sys
import sympy
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_DIR = os.path.join(PROJECT_ROOT, "html")
os.makedirs(HTML_DIR, exist_ok=True)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import glob
import re
from typing import List, Tuple
import einops
import plotly.graph_objects as go

from core.models import create_coinformer_model
from core.config import ModelConfig
from core.samplers import add_bos_token, generate_all_binary_sequences_with_fixed_num_ones

from exp_08_config import (
    EXP_CONFIG, SEEDS, DEVICE,
    get_seed_checkpoint_dir, get_seed_results_dir, set_global_seed, EXP_RESULTS_DIR
)

from exp_08_data_collection import build_probe_batch, load_checkpoints_sorted, filter_checkpoints_by_config, get_seed_checkpoint_dir, compute_token_probabilities
#%%
SEED = 100
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
print(model.W_Q.squeeze())
print(model.b_Q.squeeze())
print(model.W_K.squeeze())
print(model.b_K.squeeze())
print(model.W_V.squeeze())
print(model.b_V.squeeze())
print(model.W_E.squeeze())
# %%
k = einops.einsum(model.W_E.squeeze(), model.W_Q.squeeze(), "th d, d -> th ") + model.b_Q.squeeze()
q = einops.einsum(model.W_E.squeeze(), model.W_K.squeeze(), "th d, d -> th ") + model.b_K.squeeze()
v = einops.einsum(model.W_E.squeeze(), model.W_V.squeeze(), "th d, d -> th ") + model.b_V.squeeze()
# %%
print(k)
print(q)
print(v)
# %%

last_obs = 0
batch = generate_all_binary_sequences_with_fixed_num_ones(20,5)
# Filter batch for sequences where the last value is zero
last_values = batch[:, -1]
zero_mask = last_values == last_obs
batch_filtered = batch[zero_mask]
print(f"Original batch size: {batch.shape[0]}")
print(f"Filtered batch size: {batch_filtered.shape[0]} (last value = {last_obs})")

# %%
logits, cache = model.run_with_cache(batch_filtered)
# %%
z = cache["z", -1].squeeze()
last_z = z[:,-1]
# %%
import matplotlib.pyplot as plt
import numpy as np

# Create histogram of last_z values
plt.figure(figsize=(10, 6))
plt.hist(last_z.cpu().detach().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.title(f'Histogram of last_z values, last value = {last_obs}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Print some statistics
print(f"Shape of last_z: {last_z.shape}, last value = {last_obs}")
print(f"Min value: {last_z.min().item():.4f}")
print(f"Max value: {last_z.max().item():.4f}")
print(f"Mean value: {last_z.mean().item():.4f}")
print(f"Std value: {last_z.std().item():.4f}")

# %%
# %%
sym_i = sympy.symbols("i", integer=True, nonnegative=True)
n_ctx = model.cfg.n_ctx
scale = EXP_CONFIG.model_config.pos_embed_config.scale
pos_term = scale * sympy.log((sym_i + 1) / (n_ctx + 1))

W_E_mat = model.W_E.squeeze().detach().cpu()
W_Q_vec = model.W_Q.squeeze().detach().cpu()
W_K_vec = model.W_K.squeeze().detach().cpu()
W_V_vec = model.W_V.squeeze().detach().cpu()
b_Q_vec = model.b_Q.squeeze().detach().cpu()
b_K_vec = model.b_K.squeeze().detach().cpu()
b_V_vec = model.b_V.squeeze().detach().cpu()

k_base = W_E_mat @ W_Q_vec + b_Q_vec
q_base = W_E_mat @ W_K_vec + b_K_vec
v_base = W_E_mat @ W_V_vec + b_V_vec

W_Q_last = sympy.nsimplify(float(W_Q_vec[-1]), rational=False)
W_K_last = sympy.nsimplify(float(W_K_vec[-1]), rational=False)
W_V_last = sympy.nsimplify(float(W_V_vec[-1]), rational=False)

labels = list(range(k_base.shape[0]))
if model.cfg.d_vocab == 2:
    labels = ["0", "1"]
elif model.cfg.d_vocab == 3:
    labels = ["BOS", "0", "1"]

token_exprs = []
for idx, label in enumerate(labels):
    k_const = sympy.nsimplify(float(k_base[idx].item()), rational=False)
    q_const = sympy.nsimplify(float(q_base[idx].item()), rational=False)
    v_const = sympy.nsimplify(float(v_base[idx].item()), rational=False)
    k_expr = sympy.simplify(k_const + W_Q_last * pos_term)
    q_expr = sympy.simplify(q_const + W_K_last * pos_term)
    v_expr = sympy.simplify(v_const + W_V_last * pos_term)
    kq_expr = sympy.expand(k_expr * q_expr)
    exp_kq_expr = sympy.simplify(sympy.exp(kq_expr))
    exp_kq_v_expr = sympy.simplify(exp_kq_expr * v_expr)
    token_exprs.append({
        "label": label,
        "k_expr": k_expr,
        "q_expr": q_expr,
        "v_expr": v_expr,
        "kq_expr": kq_expr,
        "exp_kq_expr": exp_kq_expr,
        "exp_kq_v_expr": exp_kq_v_expr,
    })
    print(f"seed {SEED}")
    print(f"token {label}")
    print("  k(i) =", k_expr)
    print("  q(i) =", q_expr)
    print("  v(i) =", v_expr)
    print("  kq(i) =", kq_expr)
    print("  exp(kq(i)) =", exp_kq_expr)
    print("  exp(kq(i)) * v(i) =", exp_kq_v_expr)
    print()

target_labels = {"0", "1"}
i_values = np.arange(n_ctx)
evaluated_tokens = []

for entry in token_exprs:
    if entry["label"] not in target_labels:
        continue
    k_func = sympy.lambdify(sym_i, entry["k_expr"], "numpy")
    q_func = sympy.lambdify(sym_i, entry["q_expr"], "numpy")
    v_func = sympy.lambdify(sym_i, entry["v_expr"], "numpy")
    k_vals = np.asarray(k_func(i_values), dtype=np.float64)
    q_vals = np.asarray(q_func(i_values), dtype=np.float64)
    v_vals = np.asarray(v_func(i_values), dtype=np.float64)
    kq_vals = k_vals * q_vals
    exp_kq_vals = np.exp(kq_vals)
    exp_kq_v_vals = exp_kq_vals * v_vals
    evaluated_tokens.append({
        "label": entry["label"],
        "k_vals": k_vals,
        "q_vals": q_vals,
        "v_vals": v_vals,
        "kq_vals": kq_vals,
        "exp_kq_vals": exp_kq_vals,
        "exp_kq_v_vals": exp_kq_v_vals,
    })

if evaluated_tokens:
    plot_defs = [
        ("k(i) vs i", "k_vals", "k(i)"),
        ("q(i) vs i", "q_vals", "q(i)"),
        ("v(i) vs i", "v_vals", "v(i)"),
        ("kq(i) vs i", "kq_vals", "k(i) * q(i)"),
        ("exp(kq(i)) vs i", "exp_kq_vals", "exp(kq(i))"),
        ("exp(kq(i)) * v(i) vs i", "exp_kq_v_vals", "exp(kq(i)) * v(i)"),
    ]

    usable_defs = []
    for title, key, ylabel in plot_defs:
        if any(key in token_data for token_data in evaluated_tokens):
            usable_defs.append((title, key, ylabel))

    if usable_defs:
        n_plots = len(usable_defs)
        n_cols = min(2, n_plots)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(9 * n_cols / 2, 4 * n_rows),
            sharex=False,
        )
        axes = np.atleast_1d(axes).flatten()
        for idx, (ax, (title, key, ylabel)) in enumerate(zip(axes, usable_defs)):
            has_series = False
            for token_data in evaluated_tokens:
                values = token_data.get(key)
                if values is None:
                    continue
                ax.plot(i_values, values, label=f"token {token_data['label']}")
                has_series = True
            if not has_series:
                ax.set_visible(False)
                continue
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if idx // n_cols == n_rows - 1:
                ax.set_xlabel("i")
            ax.legend()
            ax.grid(True, alpha=0.3)
        for ax in axes[n_plots:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show()

# %%
