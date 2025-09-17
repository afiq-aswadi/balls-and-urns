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
from core.training import load_model_from_config
from core.samplers import generate_data_with_p, generate_all_binary_sequences_with_fixed_num_ones
from core.utils import get_autoregressive_predictions
from transformer_lens import HookedTransformer

#%%
SWEEP_RESULTS_DIR = "/Users/afiqabdillah/balls-and-urns/experiments/dimension_sweep_results_20250807_131232"

model_config = ModelConfig(
    d_model=2,
    d_head=1,
    n_heads=1,
    d_mlp=32,
    n_layers=1,
    use_bos_token=True
)

# Load the trained model from sweep results
def load_sweep_model(sweep_dir, model_config):
    """Simple function to load a model from sweep results based on ModelConfig."""
    # Construct the expected filename
    d_model = model_config.d_model
    d_head = model_config.d_head
    d_mlp = model_config.d_mlp
    
    filename = f"sweep_d{d_model}_h{d_head}_mlp{d_mlp}_dmodel{d_model}_dhead{d_head}_layers1_alpha1.0_beta1.0_bos.pt"
    model_path = os.path.join(sweep_dir, "models", filename)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    # Create experiment config and load model
    exp_config = ExperimentConfig(model_config=model_config)
    model = load_model_from_config(exp_config, model_path)
    print(f"Loaded model from: {model_path}")
    return model

# Load the model
model = load_sweep_model(SWEEP_RESULTS_DIR, model_config)


    # %%
# Print model layers
print("=== Model Layer Structure ===")
print("\nLayer Structure and Parameters:")
for name, module in model.named_modules():
    if name == '':  # Skip the root module
        continue
    num_params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {type(module).__name__} ({num_params:,} parameters)")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal Parameters: {total_params:,}")

# %%
# Get positional embeddings
pos_emb = model.W_pos.detach()

# Calculate cosine similarities between all pairs of positional embeddings
n_positions = pos_emb.shape[0]
cos_sim = torch.zeros((n_positions, n_positions))

for i in range(n_positions):
    for j in range(n_positions):
        cos_sim[i,j] = torch.nn.functional.cosine_similarity(pos_emb[i], pos_emb[j], dim=0)

# Create heatmap using plotly
fig = go.Figure(data=go.Heatmap(
    z=cos_sim,
    x=[f'Pos {i}' for i in range(n_positions)],
    y=[f'Pos {i}' for i in range(n_positions)],
    colorscale='RdBu',
    zmid=0
))

fig.update_layout(
    title='Cosine Similarities Between Positional Embeddings',
    xaxis_title='Position',
    yaxis_title='Position',
    width=600,
    height=600
)

fig.show()

# Save the plot
fig.write_html("positional_embeddings_cosine_sim.html")
print("Saved positional embeddings cosine similarity plot to positional_embeddings_cosine_sim.html")

#%%
# Calculate norms of positional embeddings
pos_emb_norms = torch.norm(pos_emb, dim=1).cpu()

# Create line plot using plotly
fig = go.Figure(data=go.Scatter(
    x=list(range(n_positions)),
    y=pos_emb_norms,
    mode='lines+markers'
))

fig.update_layout(
    title='Norms of Positional Embeddings',
    xaxis_title='Position',
    yaxis_title='L2 Norm',
    width=800,
    height=400
)

fig.show()


# %%
theta = 0.5

# Generate a fixed sequence [2,0,1,0,1,0,1,0,1]
seq_length = 9
random_sequence = torch.tensor( [[0,1] * 4])

# Add BOS token if the model uses it
if model_config.use_bos_token:
    from core.samplers import add_bos_token
    random_sequence = add_bos_token(random_sequence)

print(f"Generated sequence with theta={theta}:")
print(f"Sequence: {random_sequence.squeeze().tolist()}")

# Count the observations (0s and 1s)
if model_config.use_bos_token:
    # Skip the BOS token (first token)
    obs_sequence = random_sequence.squeeze()[1:]
else:
    obs_sequence = random_sequence.squeeze()

count_0s = (obs_sequence == 0).sum().item()
count_1s = (obs_sequence == 1).sum().item()
print(f"Observations: {count_0s} zeros, {count_1s} ones")
print(f"Empirical probability: {count_1s / (count_0s + count_1s):.3f}")

#%%
# Extract attention patterns
# Run the model to get attention patterns
with torch.no_grad():
    logits, cache = model.run_with_cache(random_sequence)

# Get attention patterns from the first (and only) layer
attention_pattern = cache["pattern", 0]  # Shape: (batch, head, query_pos, key_pos)
attention_pattern = attention_pattern.squeeze(0).squeeze(0)  # Remove batch and head dimensions

print(f"Attention pattern shape: {attention_pattern.shape}")


# Create a heatmap of the attention pattern
fig = go.Figure(data=go.Heatmap(
    z=attention_pattern.cpu().numpy(),
    colorscale='Viridis',
    zmin=0,
    zmax=1
))

fig.update_layout(
    title='Attention Pattern',
    xaxis_title='Key Position',
    yaxis_title='Query Position',
    width=800,
    height=600
)

fig.show()

#%%

data = generate_all_binary_sequences_with_fixed_num_ones(10,5,use_bos_token=True)
# Filter sequences that end with 1
# data_ends_with_1 = data[data[:,-1] == 0]
# print(f"Original sequences: {len(data)}")
# print(f"Sequences ending with 1: {len(data_ends_with_1)}")
# data = data_ends_with_1


#%%

with torch.no_grad():
    logits, cache = model.run_with_cache(data)

resids = cache["resid_post",-1][:,-1,:]

#%%

# Get logits at final position for all sequences
final_logits = logits[:,-1,:]

# Calculate cosine similarity matrix between all logit vectors
logits_cosine_sim = torch.nn.functional.cosine_similarity(final_logits.unsqueeze(1), final_logits.unsqueeze(0), dim=2)

# Create heatmap of logits cosine similarities
fig = go.Figure(data=go.Heatmap(
    z=logits_cosine_sim.cpu().numpy(),
    colorscale='RdBu',
    zmin=-1,
    zmax=1
))

fig.update_layout(
    title='Cosine Similarity of Final Position Logits',
    xaxis_title='Sequence Index',
    yaxis_title='Sequence Index',
    width=800,
    height=800
)

fig.show()

#%%

# Calculate L2 norm of final logits
logit_norms = torch.norm(final_logits, dim=1)

# Create bar plot of logit norms
fig = go.Figure(data=go.Bar(
    y=logit_norms.cpu().numpy(),
))

fig.update_layout(
    title='L2 Norm of Final Position Logits',
    xaxis_title='Sequence Index', 
    yaxis_title='L2 Norm',
    width=800,
    height=600
)

fig.show()


#%%
# Apply softmax to get probabilities
probs = torch.nn.functional.softmax(final_logits, dim=-1)

# Get probability assigned to token 1
prob_of_one = probs[:, 1]

# Create bar plot of probabilities
fig = go.Figure(data=go.Bar(
    y=prob_of_one.cpu().numpy(),
))

fig.update_layout(
    title='Probability Assigned to Token 1',
    xaxis_title='Sequence Index',
    yaxis_title='Probability',
    width=800,
    height=600
)

fig.show()




# %%
# Calculate cosine similarity matrix between all residual vectors
cosine_sim = torch.nn.functional.cosine_similarity(resids.unsqueeze(1), resids.unsqueeze(0), dim=2)

# Create heatmap of cosine similarities
fig = go.Figure(data=go.Heatmap(
    z=cosine_sim.cpu().numpy(),
    colorscale='RdBu',
    zmin=-1,
    zmax=1
))

fig.update_layout(
    title='Cosine Similarity of Residual Vectors',
    xaxis_title='Sequence Index',
    yaxis_title='Sequence Index', 
    width=800,
    height=800
)

fig.show()

# %%
# Calculate L2 norm of each residual vector
resid_norms = torch.norm(resids, dim=1)

# Create bar plot of norms
fig = go.Figure(data=go.Bar(
    y=resid_norms.cpu().numpy(),
))

fig.update_layout(
    title='L2 Norm of Residual Vectors',
    xaxis_title='Sequence Index',
    yaxis_title='L2 Norm',
    width=800,
    height=600
)

fig.show()

# %%
### some kind of relationship between pos embed and attn output

attn_output = cache["attn_out",-1][-1] #just get for one sequence
pos_embed = model.W_pos.detach()[:data.detach().shape[1],:]

# %%
# Get the weight matrices
W_Q = model.W_Q.detach().cpu().numpy()
W_K = model.W_K.detach().cpu().numpy()
W_V = model.W_V.detach().cpu().numpy()
W_O = model.W_O.detach().cpu().numpy()
print("W_Q", W_Q)
print("W_K", W_K)
print("W_V", W_V)
print("W_O", W_O)
# %%
