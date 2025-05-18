#%%
from model import coinformer_model_attn_only_config
from utils import calculate_posterior_mean, count_ones_and_zeros
from plot_utils import visualize_attention_patterns
from samplers import generate_data_with_p


from transformer_lens import HookedTransformer
import torch

#%%

model = HookedTransformer(coinformer_model_attn_only_config)
# Load the weights from the PT file
weights_path = "saved_models/uniform_coinformer_alpha1_beta1.pt"
model.load_state_dict(torch.load(weights_path))

# Verify the model loaded correctly
print("Model weights loaded successfully.")

#%%
visualize_attention_patterns(
    theta=0.5,
    model=model,
    seq_length=20,
)

#%%
test_data, prior = generate_data_with_p(
    p=0.5,
    batch_size=64,
    seq_length=20,
    num_batches=1,
)

#%%
logits, cache = model.run_with_cache(test_data[0])
