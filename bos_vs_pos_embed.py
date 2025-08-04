#%%
#todo: import transformer, import samplers, train with bos
import transformer_lens
from samplers import generate_data, generate_data_with_p
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from model import deactivate_position
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from utils import get_log_resids_from_sequential_ones, get_theoretical_log, calculate_optimal_loss

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def prepend_bos(datasets, bos_token_value=2):
    """
    Prepend BOS token to each batch in the datasets.
    
    Args:
        datasets: List of torch tensors representing batches
        bos_token_value: Value of the BOS token (default: 2)
    
    Returns:
        List of modified tensors with BOS tokens prepended
    """
    modified_datasets = []
    for dataset in datasets:
        batch_size = dataset.shape[0]
        bos_tokens = torch.full((batch_size, 1), bos_token_value, dtype=torch.long)
        modified_dataset = torch.cat([bos_tokens, dataset], dim=1)
        modified_datasets.append(modified_dataset)
    return modified_datasets

#%%
coinformer_wit_bos_config = transformer_lens.HookedTransformerConfig(
    d_model=64, #embedding dimension
    d_head=64,
    n_layers=1,
    n_ctx=100,
    d_vocab=3,
    act_fn="relu",
    default_prepend_bos=False,  # Disable automatic BOS prepending since we do it manually
    normalization_type=None,
)

model = transformer_lens.HookedTransformer(coinformer_wit_bos_config).to(DEVICE)

def train_coinformer_model(model, 
                           num_epochs: int = 3,
                           learning_rate: float = 0.001,
                           batch_size: int = 64,
                           seq_length: int = 20,
                           num_batches: int = 100,
                           alpha_param: float = 1.0,
                           beta_param: float = 1.0,
                           bernoulli: bool = False,
                           bernoulli_p: float = 0.5,
                           flip_batch: bool = False,
                           pos_embed: bool = True,
                           add_zeros_and_ones: int = 0,
                           scale: float = 1.0,
                           bias: float = 0.0,
                           importance_sampling: bool = False,
                           importance_sampling_alpha: float = 1.0,
                           importance_sampling_beta: float = 1.0):
    """Train the Coinformer model."""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    # Deactivate positional embedding if specified
    if not pos_embed:
        model = deactivate_position(model)
    

    for epoch in range(num_epochs):
        # Generate new data for each epoch
        datasets, priors = generate_data(batch_size=batch_size, seq_length=seq_length, 
                                   num_batches=num_batches, alpha=alpha_param, beta=beta_param, bernoulli=bernoulli, bernoulli_p=bernoulli_p, flip_batch=flip_batch
                                   , scale=scale, bias=bias)

        # Prepend BOS token to each batch
        datasets = prepend_bos(datasets)

        # Add the specified number of batches of all 0s and all 1s
        if add_zeros_and_ones > 0:
            zeros_ones_batches = []
            for _ in range(add_zeros_and_ones):
                zeros = torch.zeros((batch_size, seq_length), dtype=torch.long)
                ones = torch.ones((batch_size, seq_length), dtype=torch.long)
                zeros_ones_batches.extend([zeros, ones])
                priors.extend([0.0, 1.0])
            
            # Prepend BOS tokens to the newly added batches and extend datasets
            zeros_ones_with_bos = prepend_bos(zeros_ones_batches)
            datasets.extend(zeros_ones_with_bos)

        epoch_loss = 0
        
        # Plot histogram of priors for this epoch
        # plt.figure(figsize=(8, 4))
        # plt.hist(priors, bins=10, alpha=0.7, color='blue', edgecolor='black')
        # plt.title(f'Distribution of Prior Probabilities (Epoch {epoch+1})')
        # plt.xlabel('Probability Value')
        # plt.ylabel('Frequency')
        # plt.grid(True, alpha=0.3)
        # plt.show()
        if importance_sampling:
            optimal_loss = calculate_optimal_loss(importance_sampling_alpha, importance_sampling_beta)
        else:
            optimal_loss = calculate_optimal_loss(alpha_param, beta_param)

        for data_batch, prior in tqdm(zip(datasets, priors), desc=f"Epoch {epoch+1}/{num_epochs}"):
            data_batch = data_batch.to(DEVICE)
            # For each sequence, use all tokens except the last one as input
            inputs = data_batch[:, :-1]
            # Use all tokens except the first one as targets
            targets = data_batch[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape logits and targets for loss calculation
            logits_view = logits.view(-1, model.cfg.d_vocab)
            targets_view = targets.reshape(-1)
            
            # Calculate loss
            loss = criterion(logits_view, targets_view)
            if importance_sampling:
                weights = beta(importance_sampling_alpha, importance_sampling_beta).pdf(prior)
                loss = loss * weights

            epoch_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = epoch_loss / len(datasets)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Theoretical Loss Lower Bound: {optimal_loss:.4f}")

        # # After each epoch, analyze model's behavior on sequences with a single 1
        # model.eval()  # Set model to evaluation mode
        # with torch.no_grad():  # Disable gradient computation
        #     print(f"\\nAnalyzing model behavior after epoch {epoch+1}:")
        #     analyze_single_one_sequences(model, seq_length=20)
        # model.train()  # Set model back to training mode

    return losses


#%%
losses = train_coinformer_model(model,
                                num_epochs=4,
                                learning_rate=0.001,
                                batch_size=64,
                                seq_length=99,  # we need this to be 99 because we prepend BOS token
                                num_batches=1000,
                                alpha_param=1.0,
                                beta_param=1.0,
                                pos_embed=True
                                )

# %%
data, priors = generate_data_with_p(1,1,99,num_batches=1)
data[0] -= 1
data = prepend_bos(data)
logits = model(data[0]).shape

# %%
import torch.nn.functional as F

# Apply softmax to get probabilities
probs = F.softmax(model(data[0]), dim=-1)

# Extract probability for token 0 (first element in the last dimension)
prob_token_0 = probs[:, :, 0]

# Calculate theoretical probability = (h+1)/(n+2)
# where h is the number of 0s seen so far and n is the total number of tokens seen
theoretical_probs = []
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    seq_data = data[0][i].detach().cpu().numpy()
    seq_theoretical = []
    for pos in range(len(seq_data) - 1):  # -1 because we're predicting next token
        # Count 0s in the sequence up to current position (excluding BOS token at position 0)
        h = np.sum(seq_data[1:pos+1] == 0) if pos > 0 else 0
        n = pos  # Total tokens seen (excluding BOS)
        if n == 0:
            theoretical_prob = 0.5  # Initial probability when no data is seen
        else:
            theoretical_prob = (h + 1) / (n + 2)
        seq_theoretical.append(theoretical_prob)
    theoretical_probs.append(seq_theoretical)

# Plot the probability for token 0
plt.figure(figsize=(12, 6))
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    plt.plot(prob_token_0[i].detach().cpu().numpy(), alpha=0.7, label=f'Model Sequence {i+1}', linestyle='-')
    plt.plot(theoretical_probs[i], alpha=0.7, label=f'Theoretical Sequence {i+1}', linestyle='--')

plt.title('Probability of Token 0 Across Sequence Positions')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%

def zero_positional_embeddings(model):
    """
    Zero out the positional embeddings of the model.
    
    Args:
        model: HookedTransformer model with positional embeddings
    
    Returns:
        Modified model with zeroed positional embeddings
    """
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        with torch.no_grad():
            model.pos_embed.W_pos.zero_()
    return model

# Zero the positional embeddings
model = zero_positional_embeddings(model)

# %%
import torch.nn.functional as F

# Apply softmax to get probabilities
probs = F.softmax(model(data[0]), dim=-1)

# Extract probability for token 0 (first element in the last dimension)
prob_token_0 = probs[:, :, 0]

# Calculate theoretical probability = (h+1)/(n+2)
# where h is the number of 0s seen so far and n is the total number of tokens seen
theoretical_probs = []
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    seq_data = data[0][i].detach().cpu().numpy()
    seq_theoretical = []
    for pos in range(len(seq_data) - 1):  # -1 because we're predicting next token
        # Count 0s in the sequence up to current position (excluding BOS token at position 0)
        h = np.sum(seq_data[1:pos+1] == 0) if pos > 0 else 0
        n = pos  # Total tokens seen (excluding BOS)
        if n == 0:
            theoretical_prob = 0.5  # Initial probability when no data is seen
        else:
            theoretical_prob = (h + 1) / (n + 2)
        seq_theoretical.append(theoretical_prob)
    theoretical_probs.append(seq_theoretical)

# Plot the probability for token 0
plt.figure(figsize=(12, 6))
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    plt.plot(prob_token_0[i].detach().cpu().numpy(), alpha=0.7, label=f'Model Sequence {i+1}', linestyle='-')
    plt.plot(theoretical_probs[i], alpha=0.7, label=f'Theoretical Sequence {i+1}', linestyle='--')

plt.title('Probability of Token 0 Across Sequence Positions')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%

model = deactivate_position(model)

# Store current embeddings before finetuning
initial_embed = model.embed.W_E.clone().detach()

# Freeze all parameters except embeddings
for name, param in model.named_parameters():
    if 'embed.W_E' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

#%%

losses = train_coinformer_model(model,
                                num_epochs=4,
                                learning_rate=0.0001,
                                batch_size=64,
                                seq_length=99,  # we need this to be 99 because we prepend BOS token
                                num_batches=1000,
                                alpha_param=1.0,
                                beta_param=1.0,
                                pos_embed=False,
                                )

# Compare embeddings before and after finetuning
final_embed = model.embed.W_E.clone().detach()

# Calculate L2 norms
initial_l2_norm = torch.norm(initial_embed, p=2)
final_l2_norm = torch.norm(final_embed, p=2)

# Calculate L2 norm of the difference
diff_l2_norm = torch.norm(final_embed - initial_embed, p=2)

print(f"Initial embeddings L2 norm: {initial_l2_norm:.4f}")
print(f"Final embeddings L2 norm: {final_l2_norm:.4f}")
print(f"L2 norm of embedding change: {diff_l2_norm:.4f}")
print(f"Relative change: {(diff_l2_norm / initial_l2_norm * 100):.2f}%")

# Visualize the embedding changes
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(initial_embed.cpu().numpy(), aspect='auto', cmap='viridis')
plt.title('Initial Embeddings')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(final_embed.cpu().numpy(), aspect='auto', cmap='viridis')
plt.title('Final Embeddings')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow((final_embed - initial_embed).cpu().numpy(), aspect='auto', cmap='RdBu')
plt.title('Embedding Difference')
plt.colorbar()

plt.tight_layout()
plt.show()

#%%

# Apply softmax to get probabilities
probs = F.softmax(model(data[0]), dim=-1)

# Extract probability for token 0 (first element in the last dimension)
prob_token_0 = probs[:, :, 0]

# Calculate theoretical probability = (h+1)/(n+2)
# where h is the number of 0s seen so far and n is the total number of tokens seen
theoretical_probs = []
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    seq_data = data[0][i].detach().cpu().numpy()
    seq_theoretical = []
    for pos in range(len(seq_data) - 1):  # -1 because we're predicting next token
        # Count 0s in the sequence up to current position (excluding BOS token at position 0)
        h = np.sum(seq_data[1:pos+1] == 0) if pos > 0 else 0
        n = pos  # Total tokens seen (excluding BOS)
        if n == 0:
            theoretical_prob = 0.5  # Initial probability when no data is seen
        else:
            theoretical_prob = (h + 1) / (n + 2)
        seq_theoretical.append(theoretical_prob)
    theoretical_probs.append(seq_theoretical)

# Plot the probability for token 0
plt.figure(figsize=(12, 6))
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    plt.plot(prob_token_0[i].detach().cpu().numpy(), alpha=0.7, label=f'Model Sequence {i+1}', linestyle='-')
    plt.plot(theoretical_probs[i], alpha=0.7, label=f'Theoretical Sequence {i+1}', linestyle='--')

plt.title('Probability of Token 0 Across Sequence Positions')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
