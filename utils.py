import torch
import matplotlib.pyplot as plt
from scipy import special
from samplers import generate_data_with_p
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def calculate_posterior_mean(dataset, alpha=1.0, beta=1.0):
    """
    Calculate the posterior mean of the Beta distribution given a dataset of binary sequences.
    
    Args:
        dataset (torch.Tensor): A tensor of shape (batch_size, seq_length) containing binary sequences.
        alpha (float): Alpha parameter for the Beta distribution.
        beta (float): Beta parameter for the Beta distribution.
    
    Returns:
        float: The posterior mean of the Beta distribution.
    """
    # Count the number of 1s in the dataset
    num_ones = dataset.sum().item()
    num_zeros = dataset.numel() - num_ones
    
    # Update alpha and beta based on observed data
    posterior_alpha = alpha + num_ones
    posterior_beta = beta + num_zeros
    
    # Calculate the posterior mean
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    return posterior_mean

def calculate_optimal_loss(alpha=1.0, beta=1.0):
    """
    Calculate the theoretical optimal cross-entropy loss when predicting sequences
    with probabilities drawn from a Beta(alpha, beta) distribution.
    
    For binary sequences, the optimal expected cross-entropy is:
    E[H(p)] = E[-p*log(p) - (1-p)*log(1-p)]
    
    Where H(p) is the entropy of a Bernoulli distribution with parameter p,
    and the expectation is over p ~ Beta(alpha, beta).

    Returns:
        float: The theoretical optimal loss value
    """
    # Expected value of p*log(p) under Beta distribution can be derived using 
    # the digamma function and properties of the Beta distribution
    expected_p_log_p = special.beta(alpha + 1, beta) / special.beta(alpha, beta) * (
        special.digamma(alpha + 1) - special.digamma(alpha + beta + 1)
    )
    
    # By symmetry for 1-p
    expected_1_p_log_1_p = special.beta(alpha, beta + 1) / special.beta(alpha, beta) * (
        special.digamma(beta + 1) - special.digamma(alpha + beta + 1)
    )
    
    # The optimal loss is the negative of the expectation of p*log(p) + (1-p)*log(1-p)
    optimal_loss = -(expected_p_log_p + expected_1_p_log_1_p)
    
    return optimal_loss

def count_ones_and_zeros(sequence):
    """
    Count the number of 1s and 0s in a binary sequence.
    
    Args:
        sequence (list): A list of binary values (0s and 1s).
    
    Returns:
        tuple: A tuple containing the count of 1s and the count of 0s.
    """
    count_ones = sum(sequence)
    count_zeros = len(sequence) - count_ones
    return count_ones, count_zeros


# Create sequences with a single 1 at different positions
def analyze_single_one_sequences(model, seq_length=10):
    """
    Analyze model predictions for sequences with a single 1 at different positions.
    
    Args:
        model: The transformer model to analyze
        seq_length: Length of sequences to generate (default: 10)
    
    Returns:
        List of predictions for each sequence
    """
    sequences = []
    sequence_names = []

    # Create sequences with all zeros except for a 1 at different positions
    for i in range(seq_length):
        seq = [0] * seq_length
        seq[i] = 1
        sequences.append(seq)
        sequence_names.append(f"1 at position {i}")

    # Get predictions for all sequences
    predictions = []
    for seq in sequences:
        pred = get_autoregressive_predictions(model, torch.tensor(seq))
        predictions.append(pred)

    # Plot all predictions in a single figure
    plt.figure(figsize=(12, 8))
    for i, preds in enumerate(predictions):
        plt.plot(range(1, len(sequences[i])), preds, marker='o', label=f"1 at pos {i}")

    plt.title('Autoregressive Predictions: Probability of Next Token Being 1')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Predicted Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return predictions

def get_autoregressive_predictions(model, data):
    """Get model's autoregressive predictions for a sequence."""
    model.eval()
    data = data.to(DEVICE)
    sequence = [data[0].item()]
    predictions = []
    
    with torch.no_grad():
        for i in range(1, len(data)):
            # Create input tensor from current sequence
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(DEVICE)
            # Get model prediction
            logits = model(input_tensor)
            probs = torch.softmax(logits[0, -1], dim=-1)
            prob_of_one = probs[1].item()
            
            # Store prediction
            predictions.append(prob_of_one)
            
            # Add actual next token for next prediction
            sequence.append(data[i].item())
    
    return predictions


def get_incremental_log_odds(
    model,
    seq_length,
    batch_size,
    alpha0,
    beta0,
    theta,
    test_data=None,
):
        # generate one batch
    if test_data is None:
        test_data, _ = generate_data_with_p(theta,
                                        batch_size=batch_size,
                                        seq_length=seq_length,
                                        num_batches=1,
                                        flip_batch=False)
        test = test_data[0]               # [B, T]
    else:
        test = test_data               # [B, T]
    # truncate if sequence length exceeds seq_length
    if test.shape[1] > seq_length:
        test = test[:, :seq_length]
    test = test.to(DEVICE)           # [B, T]
    with torch.inference_mode():  
        logits = model(test)           # [B, T, 2]
    # model log-odds diff
    ld = logits[...,1] - logits[...,0]        # [B, T]
    logit_diffs = ld[:,1:] - ld[:,:-1]        # [B, T-1]
    # bayesian log-odds diff
    B, T = test.shape
    bayes_ld = torch.zeros_like(ld)
    for b in range(B):
        for t in range(1, T):
            p = calculate_posterior_mean(test[b,:t+1], alpha0, beta0)
            bayes_ld[b,t] = torch.log(torch.tensor(p/(1-p)))
    bayesian_diffs = bayes_ld[:,1:] - bayes_ld[:,:-1]  # [B, T-1]
    return logit_diffs, bayesian_diffs

def get_log_loss(
        model,
        seq_length,
        batch_size,
        alpha0,
        beta0,
        theta,
        test_data=None,
    ):
    if test_data is None:
        test_data, _ = generate_data_with_p(theta,
                                        batch_size=batch_size,
                                        seq_length=seq_length,
                                        num_batches=1,
                                        flip_batch=False)
        test = test_data[0]            # [B, T]
    else:
        test = test_data
    test = test.to(DEVICE)           # [B, T]
    # truncate if sequence length exceeds seq_length
    if test.shape[1] > seq_length:
        test = test[:, :seq_length]
    test = test.to(DEVICE)
    with torch.inference_mode():
        logits = model(test)
    
    # Get true next tokens
    targets = test[:, 1:] # Shape: [B, T-1]
    
    # Get predicted log probabilities for the true next tokens (Transformer)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1) # Shape: [B, T-1, vocab_size]
    log_probs_for_targets = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # Shape: [B, T-1]
    
    neg_log_loss_per_sequence_transformer = -log_probs_for_targets.sum(dim=1) # Shape [B]
    avg_neg_log_loss_transformer = neg_log_loss_per_sequence_transformer.mean() # Scalar

    # Calculate Bayesian PPD log loss
    B, T = test.shape
    bayesian_log_losses_for_sequences = torch.zeros(B, device=DEVICE)

    for b in range(B):
        current_sequence_log_loss = 0.0
        for t in range(1, T): # Predict token test[b, t] given test[b, :t]
            prefix = test[b, :t] # Tokens observed so far
            next_token = test[b, t]

            # Calculate posterior mean P(next_token=1 | prefix)
            # num_ones = prefix.sum().item()
            # num_zeros = prefix.numel() - num_ones
            # posterior_alpha = alpha0 + num_ones
            # posterior_beta = beta0 + num_zeros
            # prob_next_is_one = posterior_alpha / (posterior_alpha + posterior_beta)
            
            # Use the existing calculate_posterior_mean function
            # Ensure prefix is not empty, if t=0 (first token), this is tricky, but we start from t=1
            # So prefix will have at least one element test[b,0]
            prob_next_is_one = calculate_posterior_mean(prefix, alpha0, beta0)

            if next_token == 1:
                prob_next_token = prob_next_is_one
            else:
                prob_next_token = 1.0 - prob_next_is_one
            
            # Add small epsilon to prevent log(0)
            current_sequence_log_loss -= torch.log(torch.tensor(prob_next_token, device=DEVICE) + 1e-9)
        
        bayesian_log_losses_for_sequences[b] = current_sequence_log_loss
        
    avg_bayesian_neg_log_loss = bayesian_log_losses_for_sequences.mean()
    
    return avg_neg_log_loss_transformer, avg_bayesian_neg_log_loss

def test_autoregressive_prediction(model, data, probability):
    model.eval()
    
    # Ensure data is on the correct device
    data = data.to(DEVICE)
    
    print(f"True probability used to generate sequence: {probability:.4f}")
    print(f"Original sequence: {data.tolist()}")
    
    # Start with first token for autoregressive prediction
    sequence = [data[0].item()]
    
    with torch.no_grad():
        for i in range(1, len(data)):
            # Create input tensor from current sequence
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(DEVICE)
            
            # Get model prediction
            logits = model(input_tensor)
            probs = torch.softmax(logits[0, -1], dim=-1)
            prob_of_one = probs[1].item()
            
            # Calculate posterior mean based on observed sequence so far
            observed_data = torch.tensor(sequence)
            posterior_mean = calculate_posterior_mean(observed_data)
            
            # Get actual next token
            actual_next = data[i].item()
            
            print(f"Step {i}:")
            print(f"  Sequence so far: {sequence}")
            print(f"  Model probability of next token=1: {prob_of_one:.4f}")
            print(f" Model probability of next token=0: {1 - prob_of_one:.4f}")
            print(f"  Posterior mean (Beta posterior): {posterior_mean:.4f}")
            print(f"  Actual next token: {actual_next}")
            
            # Add the actual next token to our sequence for next prediction
            sequence.append(actual_next)
        
    # Final posterior mean
    final_posterior_mean = calculate_posterior_mean(data)
    print(f"\nFinal posterior mean after observing entire sequence: {final_posterior_mean:.4f}")
    print(f"True data-generating probability: {probability:.4f}")



def get_kl_divergence(model,
                        seq_length,
                        batch_size,
                        alpha0,
                        beta0,
                        theta,
                        test_data=None):
    """
    Calculate KL divergence between transformer predictions and Bayesian posterior means.
    """
    # 1) sample one batch or use provided data
    if test_data is None:
        test_data, _ = generate_data_with_p(theta,
                                            batch_size=batch_size,
                                            seq_length=seq_length,
                                            num_batches=1,
                                            flip_batch=False)
        test = test_data[0]            # [B, T]
    else:
        test = test_data
    # truncate if sequence length exceeds seq_length
    if test.shape[1] > seq_length:
        test = test[:, :seq_length]
    test = test.to(DEVICE)           # [B, T]

    with torch.inference_mode():
        logits = model(test) # Use model directly
        if isinstance(logits, tuple):
            logits = logits[0]

    probs = torch.softmax(logits, dim=-1)  # [B, T, 2]
    model_probs_1 = probs[..., 1]    # [B, T], P_model(X_t=1 | X_0...X_{t-1})

    # 2) compute Bayesian posteriors (vectorized)
    # bayes_posterior_means[b, t] = P_Bayes(theta_mean | X_0...X_t)
    B, T = test.shape
    test_float = test.float()
    num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
    num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
    alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
    beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
    _posterior_p_values_full = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive)

    bayes_posterior_means = torch.zeros_like(model_probs_1, device=test.device) # model_probs_1 is [B,T]
    # bayes_posterior_means[:, t] for t>=1 is P(theta_mean | test[b,:t+1])
    # We are interested in the posterior predictive for the *next* token,
    # or comparing the model's prediction for current token with Bayesian posterior *after* current token.
    # The plot_kl_divergence was comparing model's P(X_t=1 | X_{<t}) with Bayesian P(theta_mean | X_{<=t})
    # Let's stick to that logic for now.
    bayes_posterior_means[:, 1:] = _posterior_p_values_full[:, 1:]
    # bayes_posterior_means[:, 0] is 0.

    # 3) compute KL divergence per position
    positions = list(range(1, seq_length)) # t from 1 to T-1
    kl_divs = []
    for t_idx in positions: # t_idx is the actual position/time step, e.g., 1 to T-1
        # p is Bayesian posterior P(theta_mean | X_0...X_{t_idx})
        p = bayes_posterior_means[:, t_idx]
        # q is Model's prediction P_model(X_{t_idx}=1 | X_0...X_{t_idx-1})
        q = model_probs_1[:, t_idx]
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        q_clamped = torch.clamp(q, epsilon, 1 - epsilon)
        p_clamped = torch.clamp(p, epsilon, 1 - epsilon)
        
        # KL for Bernoulli: p*log(p/q) + (1-p)*log((1-p)/(1-q))
        # Note: p here is the Bayesian posterior mean of theta, and q is the model's predicted probability of 1.
        # This is KL(Bernoulli(p_bayes_posterior) || Bernoulli(p_model_predictive))
        kl = p_clamped * torch.log(p_clamped / q_clamped) + (1 - p_clamped) * torch.log((1 - p_clamped) / (1 - q_clamped))
        avg_kl = kl.mean().item()
        kl_divs.append(avg_kl)
        
    return kl_divs, positions