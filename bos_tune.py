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
import time
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
    d_head=32,
    n_layers=2,
    n_ctx=100,
    d_vocab=3,
    act_fn="relu",
    default_prepend_bos=False,  # Disable automatic BOS prepending since we do it manually
    normalization_type=None,
)

model = transformer_lens.HookedTransformer(coinformer_wit_bos_config).to(DEVICE)


#%%
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
    else:
        # Ensure positional embedding gradients are enabled when pos_embed=True
        for name, param in model.named_parameters():
            if "pos_embed" in name or "W_pos" in name:
                param.requires_grad = True
    

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
                                num_epochs=10,
                                learning_rate=0.001,
                                batch_size=64,
                                seq_length=99,
                                num_batches=1000,
                                alpha_param=1.0,
                                beta_param=1.0,
                                pos_embed=True
                                )

#%%

def generate_tokens(model, input_tokens, max_new_tokens=10, do_sample=True, temperature=1.0, num_samples=1):
    """
    Generate tokens from the model without requiring a tokenizer.
    
    Args:
        model: The transformer model
        input_tokens: Starting tokens as a tensor of shape [batch_size, seq_len]
        max_new_tokens: Maximum number of new tokens to generate
        do_sample: Whether to sample from the distribution (True) or use greedy decoding (False)
        temperature: Temperature for sampling (higher = more random)
        num_samples: Number of samples to generate for each input sequence
    
    Returns:
        Generated sequence including the input tokens, shape [batch_size * num_samples, seq_len + max_new_tokens]
    """
    model.eval()
    
    with torch.no_grad():
        # Repeat input tokens for each sample
        current_tokens = input_tokens.repeat_interleave(num_samples, dim=0)
        
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = model(current_tokens)
            
            # Get the logits for the last position
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Sample from the probability distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding - take the most likely token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the new token to the sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if we hit the context limit
            if current_tokens.shape[1] >= model.cfg.n_ctx:
                break
    
    return current_tokens

# Test the generation function with multiple samples
generated_sequences = generate_tokens(model, torch.tensor([[2]]).to(DEVICE), max_new_tokens=10, do_sample=True, num_samples=5)
print(f"Generated sequences shape: {generated_sequences.shape}")
for i, seq in enumerate(generated_sequences):
    print(f"Sample {i+1}: {seq[1:].tolist()}")  # Exclude BOS token

#%%


def predictive_resampling(model: transformer_lens.HookedTransformer,
    prefix: torch.Tensor = None,
    num_samples: int = 1000,
    M: int = 50,
    device: str = 'cpu',
    max_seq_length: int = 50
):
    """
    Perform predictive resampling using the Transformer model.
    
    Args:
        model: The Transformer model to use for resampling.
        prefix: Optional prefix tensor to condition the model on. If None, starts with BOS token (2).
        num_samples: Number of samples to generate.
        M: Number of samples per batch.
        device: Device to run the model on ('cpu' or 'cuda').
        max_seq_length: Maximum length of generated sequences.
    
    Returns:
        A tensor of probabilities for token 1, shape (num_samples,).
    """
    model.to(device).eval()
    
    # Set default prefix to BOS token if none provided
    if prefix is None:
        prefix = torch.tensor([[2]], device=device)
    else:
        prefix = prefix.to(device)
    
    # Generate samples
    all_probs = []
    num_batches = (num_samples + M - 1) // M  # Ceiling division
    
    for i in range(num_batches):
        current_batch_size = min(M, num_samples - i * M)
        
        # Create batch by repeating the prefix
        batch_prefix = prefix.repeat(current_batch_size, 1)
        
        # Generate sequences for this batch
        batch_sequences = generate_tokens(
            model, 
            batch_prefix, 
            max_new_tokens=max_seq_length - prefix.shape[1], 
            do_sample=True
        )
        
        # Compute logits and extract probability of token 1
        with torch.no_grad():
            logits = model(batch_sequences)
            probs = torch.softmax(logits, dim=-1)
            # Extract probability of token 1 from the last position
            prob_token_1 = probs[:, -1, 1]  # Shape: (batch_size,)
            all_probs.append(prob_token_1.cpu())
    
    return torch.cat(all_probs, dim=0)[:num_samples]


def compare_priors(
    A,
    B,
    model: transformer_lens.HookedTransformer,
    device: str = 'cpu',
    n_samples: int = 1000,
):
    
    """
    Based on the PUQ framework by Fortini and Petrone et al.
    """

    print(f"→ Starting compare_priors on device='{device}'.")

    model.to(device).eval()

    # Build a uniform grid on [0,1] and compute true prior density of P(A)
    grid = np.linspace(0, 1, 300)
    true_prior = beta(A, B).pdf(grid)

    plt.figure(figsize=(8, 5))

    # Plot true prior
    plt.plot(
        grid,
        true_prior,
        label='True Prior',
        lw=2
    )

    # Generate prior samples via Transformer predictive resampling (prefix=None)
    t1 = time.time()
    prior_samples = predictive_resampling(
        model,
        prefix=None,
        num_samples=n_samples,
        M=50,
        device=device
    ).cpu().numpy()
    print(f"  Transformer predictive resampling for prior in {time.time() - t1:.3f} sec.")

    # Overlay histogram of resampled prior P(A)
    # Use fewer bins for better visual comparison with the PDF
    n_bins = 50
    plt.hist(
        prior_samples,
        bins=n_bins,
        density=True,
        alpha=0.6,
        range=(0, 1),  # Ensure histogram covers the same range as the PDF
        label='Transformer Predictive Resampling'
    )

    # Use simple title and xlabel
    plt.title('Prior distribution comparison')
    plt.xlabel('P(A)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


#%%

compare_priors(
    A=1.0,
    B=1.0,
    model=model,
    device=DEVICE,
    n_samples = 10000
)


#%%

# Switch off gradients for all parameters and switch on gradients for positional embeddings
for name, param in model.named_parameters():
    if "pos_embed" in name or "W_pos" in name:
        param.requires_grad = True
    else:
        param.requires_grad = True

#%%
print("Gradient status after modification:")
print("=" * 50)
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

#%%
# Clear all gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        param.grad.zero_()

losses = train_coinformer_model(model,
                                num_epochs=10,
                                learning_rate=0.001,
                                batch_size=64,
                                seq_length=99,
                                num_batches=1000,
                                alpha_param=1.0,
                                beta_param=10.0,
                                pos_embed=True
                                )


#%%
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

#%%
# Generate a sequence of all ones and get model predictions
def plot_log_odds_vs_theoretical(model, seq_length=20):
    """
    Plot log odds of the model vs theoretical log odds for a sequence of all ones.
    Theoretical log odds should follow log(i+2) pattern.
    """
    # Create a sequence of all ones with BOS token
    ones_sequence = torch.ones((1, seq_length), dtype=torch.long,device="cpu")
    ones_with_bos = prepend_bos([ones_sequence])[0].to(DEVICE)
    
    # Get model predictions
    with torch.no_grad():
        logits = model(ones_with_bos)
        probs = F.softmax(logits, dim=-1)
        
        # Extract probabilities for tokens 0 and 1
        prob_token_0 = probs[0, :, 0].cpu().numpy()
        prob_token_1 = probs[0, :, 1].cpu().numpy()
        
        # Calculate log odds: log(P(1)/P(0))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        model_log_odds = np.log((prob_token_1 + epsilon) / (prob_token_0 + epsilon))
    
    # Calculate theoretical log odds
    # For a sequence of all ones, after seeing i ones, P(next=1) = (i+1)/(i+2)
    # So P(next=0) = 1/(i+2) and log odds = log((i+1)/1) = log(i+1)
    positions = np.arange(1, len(model_log_odds) + 1)  # Start from 1 since position 0 is BOS
    theoretical_log_odds = np.log(positions + 1)  # log(i+1) where i is number of ones seen
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(positions, model_log_odds, 'b-', label='Model Log Odds', linewidth=2)
    plt.plot(positions, theoretical_log_odds, 'r--', label='Theoretical Log Odds (log(i+1))', linewidth=2)
    
    plt.xlabel('Position in Sequence')
    plt.ylabel('Log Odds [log(P(1)/P(0))]')
    plt.title('Model vs Theoretical Log Odds for Sequence of All Ones')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print some values for comparison
    print("Position | Model Log Odds | Theoretical | Difference")
    print("-" * 55)
    for i in range(min(10, len(positions))):
        diff = abs(model_log_odds[i] - theoretical_log_odds[i])
        print(f"{positions[i]:8d} | {model_log_odds[i]:13.4f} | {theoretical_log_odds[i]:11.4f} | {diff:10.4f}")

# Plot for the trained model
plot_log_odds_vs_theoretical(model, seq_length=50)


# %%
# List all trained parameters of the model
print("Trained parameters of the model:")
print("=" * 50)

total_params = 0
for name, param in model.named_parameters():
    param_count = param.numel()
    total_params += param_count
    print(f"{name}: {param.shape} ({param_count:,} parameters)")

print("=" * 50)
print(f"Total parameters: {total_params:,}")

#%%

# Create a new model with the same configuration
new_model = transformer_lens.HookedTransformer(coinformer_wit_bos_config).to(DEVICE)

# Copy weights from the original model to the new model
with torch.no_grad():
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
        param2.copy_(param1.data)

# Ensure all parameters in the new model require gradients (make them independent)
for name, param in new_model.named_parameters():
    param.requires_grad = True

print("New model created with copied weights:")
print("=" * 50)
for name, param in new_model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

#%%
# Add a new learnable BOS embedding vector
embed_W_E_bos = torch.nn.Parameter(torch.randn(model.cfg.d_model, requires_grad=True, device=DEVICE))


#%%
# Add the parameter to the model so it's tracked by optimizers
new_model.register_parameter('embed_W_E_bos', embed_W_E_bos)

def hook_function(
        W_U, hook
):
    W_U[-1,:] = W_U[-1,:] + new_model.embed_W_E_bos

new_model.run_with_hooks(data[0],
                         fwd_hooks= [
                             ('hook_embed', hook_function)
                         ])


#TODO: fix this function!!
def train_coinformer_model_with_hooks(model, 
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
            logits = model.run_with_hooks(inputs,
                         fwd_hooks= [
                             ('hook_embed', hook_function)
                         ])

            
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

#%%
# Print all activation names in the new_model
print("All activation names in new_model:")
print("=" * 50)
for name in new_model.hook_dict.keys():
    print(name)

#%%
# Switch off gradients for all existing parameters
for name, param in new_model.named_parameters():
    if name != "embed_W_E_bos":
        param.requires_grad = False

print("Gradient status after adding BOS embedding:")
print("=" * 50)
for name, param in new_model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")


# %%
losses = train_coinformer_model_with_hooks(new_model,
                                num_epochs=10,
                                learning_rate=0.001,
                                batch_size=64,
                                seq_length=99,
                                num_batches=1000,
                                alpha_param=8,
                                beta_param=2,
                                pos_embed=False,
                                )

#%%
# Check that weights in new_model and model are equal for relevant layers
print("Checking if weights are equal between new_model and original model:")
print("=" * 70)

weights_equal = True
model_params = dict(model.named_parameters())
new_model_params = dict(new_model.named_parameters())

for name, param1 in model_params.items():
    if name in new_model_params:
        param2 = new_model_params[name]
        are_equal = torch.allclose(param1.data, param2.data, atol=1e-6)
        if not are_equal:
            weights_equal = False
            print(f"❌ {name}: DIFFERENT")
            print(f"   Max difference: {torch.max(torch.abs(param1.data - param2.data)).item():.8f}")
        else:
            print(f"✅ {name}: EQUAL")
    else:
        print(f"⚠️  {name}: NOT FOUND in new_model")

print("=" * 70)
if weights_equal:
    print("✅ All corresponding weights are equal between models!")
else:
    print("❌ Some weights differ between models!")

# %%
print("=" * 50)
for name, param in new_model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")
# %%
data, priors = generate_data_with_p(1,1,99,num_batches=1)
data = prepend_bos(data)
logits = new_model(data[0]).shape

# %%
import torch.nn.functional as F

# Apply softmax to get probabilities
probs = F.softmax(new_model(data[0]), dim=-1)

# Extract probability for token 0 (first element in the last dimension)
prob_token_0 = probs[:, :, 0]

# Plot the probability for token 0
plt.figure(figsize=(12, 6))
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    plt.plot(prob_token_0[i].detach().cpu().numpy(), alpha=0.7, label=f'Sequence {i+1}')

plt.title('Probability of Token 0 Across Sequence Positions')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
# Switch off gradients for all layers except embed.W_E
for name, param in new_model.named_parameters():
    if name != "embed.W_E":
        param.requires_grad = False
    else:
        param.requires_grad = True

print("Gradient status after modification:")
print("=" * 50)
for name, param in new_model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

# # %%
# # Switch off gradients for embed.W_E and switch on gradients for mlp.W_in in both layers
# for name, param in model.named_parameters():
#     if name == "embed.W_E":
#         param.requires_grad = False
#     elif "mlp.W_in" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

print("Gradient status after modification:")
print("=" * 50)
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")
# %%
losses = train_coinformer_model_with_hooks(new_model,
                                num_epochs=10,
                                learning_rate=0.001,
                                batch_size=64,
                                seq_length=99,
                                num_batches=1000,
                                bernoulli=True,
                                bernoulli_p=0.5,
                                pos_embed=False,
                                )
# %%
data, priors = generate_data_with_p(1,1,99,num_batches=1)
data = prepend_bos(data)
logits = model(data[0]).shape

# %%
import torch.nn.functional as F

# Apply softmax to get probabilities
probs = F.softmax(model(data[0]), dim=-1)

# Extract probability for token 0 (first element in the last dimension)
prob_token_0 = probs[:, :, 0]

# Plot the probability for token 0
plt.figure(figsize=(12, 6))
for i in range(prob_token_0.shape[0]):  # For each sequence in the batch
    plt.plot(prob_token_0[i].detach().cpu().numpy(), alpha=0.7, label=f'Sequence {i+1}')

plt.title('Probability of Token 0 Across Sequence Positions')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
