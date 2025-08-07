#%%
# Train a coinformer on bernoulli prior -- and attempt to reverse engineer the model
import torch
from train import train_coinformer_model
from model import model_config
from transformer_lens import HookedTransformer
from utils import get_autoregressive_predictions
from torch.nn.functional import cosine_similarity

#%%
bernoulli_transformer =  HookedTransformer(model_config)
# Set all query and key matrices to ones
for layer in range(bernoulli_transformer.cfg.n_layers):
    # Set query matrices to ones
    bernoulli_transformer.blocks[layer].attn.W_Q.data = torch.ones_like(
        bernoulli_transformer.blocks[layer].attn.W_Q.data
    ) * 0.5
    # Set key matrices to ones
    bernoulli_transformer.blocks[layer].attn.W_K.data = torch.ones_like(
        bernoulli_transformer.blocks[layer].attn.W_K.data
    )* 0.5
bernoulli_transformer.W_pos.data = torch.zeros_like(
    bernoulli_transformer.W_pos.data
)  # Set positional embedding to zero

losses = train_coinformer_model(
    model=bernoulli_transformer,
    num_epochs=3,
    learning_rate=0.001,
    batch_size=64,
    seq_length=100,
    num_batches=100,
    alpha=1.0,
    beta=1.0,
    bernoulli=True,
    bernoulli_p=0.5,
    pos_embed=True,  # Deactivate positional embedding 
    flip_batch=True,
)

#%%
print(bernoulli_transformer.W_Q)
print(bernoulli_transformer.W_K)

#%%
zero_prompt = torch.tensor([0]*10)
one_prompt = torch.tensor([1]*10)

#%%
with torch.no_grad():
    zero_logits, zero_cache = bernoulli_transformer.run_with_cache(zero_prompt)
    one_logits, one_cache = bernoulli_transformer.run_with_cache(one_prompt)

    # Get the last residual vector after layer norm but before unembedding
    zero_last_resid = zero_cache["ln_final.hook_normalized"]
    one_last_resid = one_cache["ln_final.hook_normalized"]

print(zero_last_resid.shape)

#%%
cos_sim = cosine_similarity(zero_last_resid, one_last_resid, dim=-1)
print("Cosine Similarity:", cos_sim)

#%%
prompt = torch.zeros(10,10, dtype=torch.int64) #note: we need int64 to be able to index the embedding
for i in range(10):
    prompt[i,i] = 1

eye_logits, eye_cache = bernoulli_transformer.run_with_cache(prompt)
# %%
zero_cache["hook_embed"].shape
one_cache["hook_embed"].shape  # Added shape to one_cache["hook_embed"]
# %%
