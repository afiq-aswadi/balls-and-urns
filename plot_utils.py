import torch
import numpy as np
from torch.nn.functional import cosine_similarity, softmax, log_softmax
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from samplers import generate_data_with_p
from utils import calculate_posterior_mean, get_kl_divergence, get_incremental_log_odds, get_residual_cosine_similarity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#TODO: turn this into a class so i can just do a single call to plot all the things


def plot_beta_distribution(alpha, beta):
    """
    Plot the Beta distribution for given alpha and beta parameters.
    """
    x = np.linspace(0, 1, 1000)
    y = (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))
    y /= np.trapezoid(y, x)  # Normalize the distribution
    plt.plot(x, y)
    plt.title(f'Beta Distribution (α={alpha}, β={beta})')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def plot_proportional_log_odds(theta,
                               model,
                               seq_length=100,
                               batch_size=32,
                               alpha0=1,
                               beta0=1,
                               data=None):
    """
                            Plot the proportional log-odds between model predictions and Bayesian inference.
                            This function compares a model's predicted log probabilities with the log probabilities
                            from a Bayesian model. It plots the ratio (proportion) of the model's log-odds to the 
                            Bayesian log-odds across different positions in the sequence.
                            Parameters
                            ----------
                            theta : float
                                The true probability parameter used for data generation.
                            model : object
                                The model to evaluate, which must have a `run_with_cache` method that returns logits.
                            seq_length : int, optional
                                Length of the sequences to generate (default 100).
                            batch_size : int, optional
                                Number of sequences to generate in a batch (default 32).
                            alpha0 : float, optional
                                Alpha parameter for the Beta prior distribution (default 1).
                            beta0 : float, optional
                                Beta parameter for the Beta prior distribution (default 1).
                            data : torch.Tensor, optional
                                Pre-generated data to use instead of generating new data. If None,
                                data will be generated using the provided theta (default None).
                            Returns
                            -------
                            None
                                This function doesn't return any value but displays a plot.
                            Notes
                            -----
                            The function generates binary sequences, either using the provided data or generating
                            new data based on theta. It then computes two types of log probabilities:
                            1. Model log probabilities: Extracted from the model's logits
                            2. Bayesian log probabilities: Calculated using posterior means
                            The plot shows the ratio of model log-odds to Bayesian log-odds for each position
                            in the sequence, indicating how the model's predictions compare to optimal Bayesian inference.
    """
    # generate one batch
    if data is None:
        test_data, _ = generate_data_with_p(theta,
                                            batch_size=batch_size,
                                            seq_length=seq_length,
                                            num_batches=1,
                                            flip_batch=False)
        test = test_data[0]               # [B, T]
    else:
        test = data
    # truncate if sequence length exceeds seq_length
    if test.shape[1] > seq_length:
        test = test[:, :seq_length]
    
    test = test.to(DEVICE)               # [B, T]
    
    with torch.inference_mode():
        logits = model(test) # Use model directly
        if isinstance(logits, tuple): # Handle if model returns more than just logits
            logits = logits[0]

    # model log-probs
    model_log_probs = log_softmax(logits, dim = -1) # [B, T, VocabSize]
    
    B, T = test.shape
    # Index log probabilities for the true next tokens (vectorized)
    # model_log_probs[:, :-1, :] is [B, T-1, VocabSize]
    # test[:, 1:] is [B, T-1], needs to be [B, T-1, 1] for gather
    _true_next_log_probs_tensor = torch.gather(model_log_probs[:, :-1, :], 2, test[:, 1:].unsqueeze(-1).long()).squeeze(-1)
    true_next_log_probs = _true_next_log_probs_tensor.detach().cpu().numpy()

    # Bayesian log-likelihoods (vectorized)
    test_float = test.float() # Ensure test is float for calculations
    num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
    num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
    alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
    beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
    # posterior_p_after_seeing_current_token[b, i] = P(theta_mean | test[b, :i+1])
    posterior_p_after_seeing_current_token = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive)

    # p_for_pred[b,t] is P(theta_mean | test[b, :t+1]), used to predict test[b,t+1]
    p_for_pred = posterior_p_after_seeing_current_token[:, :-1] # Shape (B, T-1)
    actual_next_tokens = test_float[:, 1:] # Shape (B, T-1)
    # prob_of_actual_next_token[b,t] = P(test[b,t+1] | test[b,:t+1])
    prob_of_actual_next_token = p_for_pred * actual_next_tokens + (1 - p_for_pred) * (1 - actual_next_tokens)
    # Add small epsilon to log to prevent log(0)
    _bayes_ld_tensor = torch.log(prob_of_actual_next_token + 1e-10)
    bayes_ld = _bayes_ld_tensor.detach().cpu().numpy()
    
    # compute proportional log-odds
    positions = list(range(1, seq_length-1))
    proportions = []
    for t in positions:
        proportions.append(true_next_log_probs.mean(axis=0)[t]/bayes_ld.mean(axis=0)[t])
    # plot
    plt.figure(figsize=(8,4))
    plt.plot(positions, proportions, marker='o')
    plt.xlabel('Sequence Position')
    plt.ylabel('Proportional Log-Odds')
    plt.title(f'Proportional Log-Odds (θ={theta})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_proportional_log_odds_surface(theta_values,
                                        model,
                                        seq_length=100,
                                        batch_size=32,
                                        alpha0=1,
                                        beta0=1,
                                        data_list=None): # Added data_list
    """
    3D surface of proportional log-odds over a list of prior thetas.
    """
    positions = list(range(1, seq_length-1))
    M = len(theta_values)
    N = len(positions)
    prop_mat = np.zeros((M, N), dtype=float)

    for i, theta in enumerate(theta_values):
        # generate one batch or use provided data
        if data_list is None:
            test_data_for_theta, _ = generate_data_with_p(theta,
                                                batch_size=batch_size,
                                                seq_length=seq_length,
                                                num_batches=1,
                                                flip_batch=False)
            test = test_data_for_theta[0]
        else:
            test = data_list[i]
        # truncate if sequence length exceeds seq_length
        if test.shape[1] > seq_length:
            test = test[:, :seq_length]
        
        test = test.to(DEVICE)               # [B, T]
        
        with torch.inference_mode():
            logits = model(test) # Use model directly
            if isinstance(logits, tuple):
                logits = logits[0]
        
        # model log-probs
        model_log_probs = log_softmax(logits, dim = -1) # [B, T, VocabSize]
        
        B, T = test.shape
        # Index log probabilities for the true next tokens (vectorized)
        _true_next_log_probs_tensor = torch.gather(model_log_probs[:, :-1, :], 2, test[:, 1:].unsqueeze(-1).long()).squeeze(-1)
        true_next_log_probs_np = _true_next_log_probs_tensor.detach().cpu().numpy()
        
        # Bayesian log-likelihoods (vectorized)
        test_float = test.float()
        num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
        num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
        beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
        posterior_p_after_seeing_current_token = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive)

        p_for_pred = posterior_p_after_seeing_current_token[:, :-1]
        actual_next_tokens = test_float[:, 1:]
        prob_of_actual_next_token = p_for_pred * actual_next_tokens + (1 - p_for_pred) * (1 - actual_next_tokens)
        _bayes_ld_tensor = torch.log(prob_of_actual_next_token + 1e-10)
        bayes_ld_np = _bayes_ld_tensor.detach().cpu().numpy()
        
        # compute proportional log-odds for each position
        for j, pos in enumerate(positions): # pos is actual sequence position, index is pos
            # Original code used true_next_log_probs.mean(axis=0)[t] where t was also sequence position
            # If positions are 1-indexed and up to seq_length-2 (T-2), then index for np arrays is pos or pos-1
            # true_next_log_probs_np and bayes_ld_np are (B, T-1). Mean over axis 0 gives (T-1).
            # If positions are 1 to T-2, then index is pos (0 to T-3 for the array)
            # The loop `positions = list(range(1, seq_length-1))` means pos goes from 1 to T-2 (T=seq_length)
            # So index for a (T-1) length array should be `pos` if we want element `pos` (0-indexed)
            # Or `pos-1` if `positions` are 1-indexed true positions.
            # Original: true_next_log_probs.mean(axis=0)[t] where t is from positions.
            # So if positions[j] = t, we access mean_array[t].
            # Here pos is t.
            idx = pos # if pos is 0-indexed for the (T-1) array
            if pos < true_next_log_probs_np.mean(axis=0).shape[0] and pos < bayes_ld_np.mean(axis=0).shape[0]: # Defensive check
                 prop_mat[i, j] = true_next_log_probs_np.mean(axis=0)[idx] / (bayes_ld_np.mean(axis=0)[idx] + 1e-10) # add epsilon
            else: # Should not happen if positions are set correctly (0 to T-2)
                 prop_mat[i, j] = np.nan


        print(f"Processed θ={theta:.2f}")

    # plot surface
    fig = go.Figure(data=[go.Surface(
        z=prop_mat,
        x=positions,
        y=theta_values,
        colorscale='RdBu',
        opacity=0.9
    )])
    fig.update_layout(
        title='Proportional Log-Odds Surface',
        scene=dict(
            xaxis_title='Seq Position',
            yaxis_title='Theta',
            zaxis_title='Prop Log-Odds'
        ),
        width=800,
        height=600,
    )
    fig.show()

def plot_incremental_log_odds_cosine(theta,
                                     model,
                                     seq_length=100,
                                     batch_size=32,
                                     alpha0=1,
                                     beta0=1,
                                     data=None): # Added data
    """
    2D plot of cosine similarity between transformer incremental log-odds
    updates and Bayesian incremental log-odds, for a single prior theta.
    """
    # generate one batch or use provided data
    if data is None:
        test_data, _ = generate_data_with_p(theta,
                                            batch_size=batch_size,
                                            seq_length=seq_length,
                                            num_batches=1,
                                            flip_batch=False)
        test_data = test_data[0]               # [B, T]
    else:
        test_data = data
    # truncate if sequence length exceeds seq_length
    if test_data.shape[1] > seq_length:
        test_data = test_data[:, :seq_length]
    test = test_data.to(DEVICE)               # [B, T]

    with torch.inference_mode():
        logits = model(test) # Use model directly
        if isinstance(logits, tuple):
            logits = logits[0]

    # model log-odds diff
    ld = logits[...,1] - logits[...,0]        # [B, T]
    logit_diffs = ld[:,1:] - ld[:,:-1]        # [B, T-1]

    # Bayesian log-odds (vectorized)
    B, T = test.shape
    test_float = test.float()
    num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
    num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
    alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
    beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
    # posterior_p_after_seeing_current_token[b, i] = P(theta_mean | test[b, :i+1])
    posterior_p_after_seeing_current_token = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive) # Shape (B,T)

    epsilon = 1e-10 # For numerical stability
    p_clamped = torch.clamp(posterior_p_after_seeing_current_token, epsilon, 1 - epsilon)
    
    bayes_log_odds_full = torch.log(p_clamped / (1 - p_clamped)) # Shape (B,T)
    
    bayes_ld_torch = torch.zeros_like(ld, device=test.device) # ld is [B,T]
    bayes_ld_torch[:, 1:] = bayes_log_odds_full[:, 1:] # Fill for t=1 to T-1 (history test[b,:t+1])
    
    bayesian_diffs = bayes_ld_torch[:,1:] - bayes_ld_torch[:,:-1]  # [B, T-1]

    # compute cosine similarity per position
    positions = list(range(1, seq_length))
    sims = []
    for pos in positions:
        idx = pos - 1
        c = cosine_similarity(logit_diffs[:,idx],
                              bayesian_diffs[:,idx],
                              dim=0)
        sims.append(c.item())

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(positions, sims, marker='o')
    plt.xlabel('Sequence Position')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Incremental Log-Odds Cosine Sim (θ={theta})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_incremental_log_odds_cosine_surface(theta_values,
                                             model,
                                             seq_length=100,
                                             batch_size=32,
                                             alpha0=1,
                                             beta0=1,
                                             data_list=None): # Added data_list
    """
    3D surface of cosine similarity over a list of prior thetas.
    """
    positions = list(range(1, seq_length))
    M = len(theta_values)
    N = len(positions)
    sim_mat = np.zeros((M, N), dtype=float)

    for i, theta in enumerate(theta_values):
        # generate & run or use provided data
        if data_list is None:
            test_data_for_theta, _ = generate_data_with_p(theta,
                                                batch_size=batch_size,
                                                seq_length=seq_length,
                                                num_batches=1,
                                                flip_batch=False)
            test = test_data_for_theta[0]
        else:
            test = data_list[i]
        # truncate if sequence length exceeds seq_length
        if test.shape[1] > seq_length:
            test = test[:, :seq_length]
        
        test = test.to(DEVICE)               # [B, T]

        with torch.inference_mode():
            logits = model(test) # Use model directly
            if isinstance(logits, tuple):
                logits = logits[0]
        
        ld = logits[...,1] - logits[...,0]
        logit_diffs = ld[:,1:] - ld[:,:-1]

        # Bayesian log-odds (vectorized)
        B, T = test.shape
        test_float = test.float()
        num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
        num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
        beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
        posterior_p_after_seeing_current_token = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive)

        epsilon = 1e-10
        p_clamped = torch.clamp(posterior_p_after_seeing_current_token, epsilon, 1 - epsilon)
        bayes_log_odds_full = torch.log(p_clamped / (1 - p_clamped))
        
        bayes_ld_torch = torch.zeros_like(ld, device=test.device)
        bayes_ld_torch[:, 1:] = bayes_log_odds_full[:, 1:]
        bayesian_diffs = bayes_ld_torch[:,1:] - bayes_ld_torch[:,:-1]

        # fill row
        for j in range(N):
            c = cosine_similarity(logit_diffs[:,j],
                                  bayesian_diffs[:,j],
                                  dim=0)
            sim_mat[i,j] = c.item()

    # plot surface
    fig = go.Figure(data=[go.Surface(
        z=sim_mat,
        x=positions,
        y=theta_values,
        colorscale='RdBu',
        opacity=0.9
    )])
    fig.update_layout(
        title='Cosine Similarity: Transformer vs Bayesian Incremental Log-Odds',
        scene=dict(
            xaxis_title='Seq Position',
            yaxis_title='Theta',
            zaxis_title='Cosine Sim'
        ),
        width=800,
        height=600,
    )
    fig.show()

def plot_probability_diff(theta,
                          model,
                          norm: str = 'abs',
                          seq_length: int = 100,
                          batch_size: int = 32,
                          alpha0: float = 1,
                          beta0: float = 1,
                          data=None): # Added data
    """
    2D plot of difference between transformer token-1 probabilities
    and Bayesian posterior means, for a single prior theta.
    norm: 'abs'     -> mean absolute error
          'l1'      -> L1 norm over batch
          'l2'      -> L2 norm over batch
          'max'     -> max absolute error over batch
    """
    # 1) sample one batch or use provided data
    if data is None:
        test_data, _ = generate_data_with_p(theta,
                                            batch_size=batch_size,
                                            seq_length=seq_length,
                                            num_batches=1,
                                            flip_batch=False)
        test = test_data[0]               # [B, T]
    else:
        test = data
    # truncate if sequence length exceeds seq_length
    if test.shape[1] > seq_length:
        test = test[:, :seq_length]
    
    test = test.to(DEVICE)               # [B, T]

    with torch.inference_mode():
        logits = model(test) # Use model directly
        if isinstance(logits, tuple):
            logits = logits[0]
            
    probs = softmax(logits, dim=-1)[..., 1]  # [B, T], P_model(X_t=1 | X_0...X_{t-1})

    # 2) compute Bayesian posteriors (vectorized)
    # bayes[b, t] = P_Bayes(theta_mean | X_0...X_t)
    B, T = test.shape
    test_float = test.float()
    num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
    num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
    alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
    beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
    _posterior_p_values_full = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive) # Shape (B,T)

    bayes = torch.zeros_like(probs, device=test.device) # probs is [B,T]
    # bayes[b,t] for t>=1 is P(theta_mean | test[b,:t+1])
    bayes[:, 1:] = _posterior_p_values_full[:, 1:] 
    # bayes[:, 0] is 0, this means comparison for t=0 is probs[:,0] vs 0. Original loop started t from 1.
    # positions loop starts from 1. So probs[:,0] and bayes[:,0] are not used in diffs.

    # 3) compute per-position normed difference
    positions = list(range(1, seq_length))
    diffs = []
    for t in positions:
        d = probs[:, t] - bayes[:, t]
        if norm == 'abs':
            val = d.abs().mean().item()
        elif norm == 'l1':
            val = torch.norm(d, p=1).item()
        elif norm == 'l2':
            val = torch.norm(d, p=2).item()
        elif norm == 'max':
            val = torch.norm(d, p=float('inf')).item()
        else:
            raise ValueError(f"Unknown norm '{norm}'")
        diffs.append(val)

    # 4) plot
    plt.figure(figsize=(8,4))
    plt.plot(positions, diffs, marker='o')
    plt.xlabel('Sequence Position')
    plt.ylabel(f'Probability Diff ({norm})')
    plt.title(f'Prob Diff vs Bayesian (θ={theta}, norm={norm})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_probability_diff_surface(theta_values,
                                  model,
                                  norm: str = 'abs',
                                  seq_length: int = 100,
                                  batch_size: int = 32,
                                  alpha0: float = 1,
                                  beta0: float = 1,
                                  data_list=None): # Added data_list
    """
    3D surface of probability differences over a list of thetas.
    """
    positions = list(range(1, seq_length))
    M = len(theta_values)
    N = len(positions)
    mat = np.zeros((M, N), dtype=float)

    for i, theta in enumerate(theta_values):
        # sample & run or use provided data
        if data_list is None:
            test_data_for_theta, _ = generate_data_with_p(theta,
                                                batch_size=batch_size,
                                                seq_length=seq_length,
                                                num_batches=1,
                                                flip_batch=False)
            test = test_data_for_theta[0] # [B, T]
        
        else:
            test = data_list[i]
        # truncate if sequence length exceeds seq_length
        if test.shape[1] > seq_length:
            test = test[:, :seq_length]
        
        test = test.to(DEVICE)               # [B, T]

        with torch.inference_mode():   
            logits = model(test) # Use model directly
            if isinstance(logits, tuple):
                logits = logits[0]
        probs = softmax(logits, dim=-1)[..., 1]  # [B, T]

        # bayesian (vectorized)
        B, T = test.shape
        test_float = test.float()
        num_ones_cumsum_inclusive = torch.cumsum(test_float, dim=1)
        num_elements_inclusive = torch.arange(1, T + 1, device=test.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        alpha_n_inclusive = alpha0 + num_ones_cumsum_inclusive
        beta_n_inclusive = beta0 + (num_elements_inclusive - num_ones_cumsum_inclusive)
        _posterior_p_values_full = alpha_n_inclusive / (alpha_n_inclusive + beta_n_inclusive)

        bayes = torch.zeros_like(probs, device=test.device)
        bayes[:, 1:] = _posterior_p_values_full[:, 1:]

        # fill row
        for j, t in enumerate(positions):
            d = probs[:, t] - bayes[:, t]
            if norm == 'abs':
                mat[i, j] = d.abs().mean().item()
            elif norm == 'l1':
                mat[i, j] = torch.norm(d, p=1).item()
            elif norm == 'l2':
                mat[i, j] = torch.norm(d, p=2).item()
            elif norm == 'max':
                mat[i, j] = torch.norm(d, p=float('inf')).item()
            else:
                raise ValueError(f"Unknown norm '{norm}'")
        print(f"Processed θ={theta:.2f}")

    # plot surface
    fig = go.Figure(data=[go.Surface(
        z=mat,
        x=positions,
        y=theta_values,
        colorscale='RdBu',
        opacity=0.9
    )])
    fig.update_layout(
        title=f'Prob Diff Surface (norm={norm})',
        scene=dict(
            xaxis_title='Seq Position',
            yaxis_title='Theta',
            zaxis_title=f'Diff ({norm})'
        ),
        width=800,
        height=600,
    )
    fig.show()


def plot_kl_divergence(theta,
                        model,
                        seq_length=100,
                        batch_size=32,
                        alpha0=1,
                        beta0=1,
                        data=None): # Added data
    """
    2D plot of KL divergence between transformer probabilities
    and Bayesian posterior means, for a single prior theta.
    """
    kl_divs, positions = get_kl_divergence(model=model,
                                           seq_length=seq_length,
                                           batch_size=batch_size,
                                           alpha0=alpha0,
                                           beta0=beta0,
                                           theta=theta,
                                           test_data=data)

    # 4) plot
    plt.figure(figsize=(8,4))
    plt.plot(positions, kl_divs, marker='o')
    plt.xlabel('Sequence Position')
    plt.ylabel('KL Divergence')
    plt.title(f'KL Divergence: Bayesian vs Model (θ={theta})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_kl_divergence_surface(theta_values,
                                model,
                                seq_length=100,
                                batch_size=32,
                                alpha0=1,
                                beta0=1,
                                data_list=None): # Added data_list
    """
    3D surface of KL divergence over a list of thetas.
    """
    positions_ref = list(range(1, seq_length)) # Reference for x-axis of plot
    M = len(theta_values)
    N = len(positions_ref)
    kl_mat = np.zeros((M, N), dtype=float)

    for i, theta in enumerate(theta_values):
        current_data = data_list[i] if data_list else None
        kl_divs_for_theta, _ = get_kl_divergence(model=model,
                                                 seq_length=seq_length,
                                                 batch_size=batch_size,
                                                 alpha0=alpha0,
                                                 beta0=beta0,
                                                 theta=theta,
                                                 test_data=current_data)
        
        # Ensure kl_divs_for_theta has the expected length N
        # If seq_length used in get_kl_divergence matches the one for positions_ref
        # then len(kl_divs_for_theta) should be N.
        if len(kl_divs_for_theta) == N:
            kl_mat[i, :] = kl_divs_for_theta
        else:
            # Handle potential mismatch, e.g. by padding or error
            # For now, assuming they match. If not, this could be an issue.
            # This might happen if get_kl_divergence returns a different number of points
            # than expected by `positions_ref`.
            # A simple fix if shorter, pad with nan; if longer, truncate.
            # Or ensure `get_kl_divergence` always respects `seq_length` for its `positions` output.
            # The `get_kl_divergence` returns `list(range(1, seq_length))` which is `seq_length-1` elements.
            # `positions_ref` is `list(range(1, seq_length))` which is `seq_length-1` elements. So N should match.
            kl_mat[i, :len(kl_divs_for_theta)] = kl_divs_for_theta[:N]


        print(f"Processed θ={theta:.2f}")

    # plot surface
    fig = go.Figure(data=[go.Surface(
        z=kl_mat,
        x=positions_ref, # Use the reference positions
        y=theta_values,
        colorscale='RdBu',
        opacity=0.9
    )])
    fig.update_layout(
        title='KL Divergence: Bayesian vs Model',
        scene=dict(
            xaxis_title='Seq Position',
            yaxis_title='Theta',
            zaxis_title='KL Divergence'
        ),
        width=800,
        height=600,
    )
    fig.show()

def plot_incremental_log_odds(theta,
                              model,
                              seq_length=100,
                              batch_size=32,
                              alpha0=1,
                              beta0=1,
                              data=None):
    """
    2D plot of absolute differences between model incremental log-odds and Bayesian incremental log-odds.
    """
    # compute incremental log-odds diffs
    logit_diffs, bayesian_diffs = get_incremental_log_odds(
        model=model,
        seq_length=seq_length,
        batch_size=batch_size,
        alpha0=alpha0,
        beta0=beta0,
        theta=theta,
        test_data=data
    )
    # absolute differences
    abs_diff = torch.abs(logit_diffs - bayesian_diffs)  # [B, T-1]
    mean_abs_diff = abs_diff.mean(dim=0).cpu().numpy()   # [T-1]
    std_abs_diff = abs_diff.std(dim=0).cpu().numpy()     # [T-1]
    positions = list(range(1, mean_abs_diff.shape[0] + 1))
    # plot
    plt.figure(figsize=(8,4))
    plt.plot(positions, mean_abs_diff, label='Mean Absolute Difference')
    plt.fill_between(positions,
                     mean_abs_diff - std_abs_diff,
                     mean_abs_diff + std_abs_diff,
                     alpha=0.3,
                     label='±1 std')
    plt.xlabel('Sequence Position')
    plt.ylabel('Absolute Difference in Log Odds')
    plt.title(f'Absolute Incremental Log-Odds Diff (θ={theta})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_incremental_log_odds_surface(theta_values,
                                     model,
                                     seq_length=100,
                                     batch_size=32,
                                     alpha0=1,
                                     beta0=1,
                                     data_list=None):
    """
    3D surface of mean absolute differences in incremental log-odds over a list of prior thetas.
    """
    M = len(theta_values)
    N = seq_length - 1
    mat = np.zeros((M, N), dtype=float)
    # compute for each theta
    for i, theta in enumerate(theta_values):
        test_data = data_list[i] if data_list is not None else None
        logit_diffs, bayesian_diffs = get_incremental_log_odds(
            model=model,
            seq_length=seq_length,
            batch_size=batch_size,
            alpha0=alpha0,
            beta0=beta0,
            theta=theta,
            test_data=test_data
        )
        abs_diff = torch.abs(logit_diffs - bayesian_diffs)
        mean_abs_diff = abs_diff.mean(dim=0).cpu().numpy()
        mat[i, :] = mean_abs_diff
    positions = list(range(1, N + 1))
    # plot surface
    fig = go.Figure(data=[go.Surface(
        z=mat,
        x=positions,
        y=theta_values,
        colorscale='Viridis',
        opacity=0.9
    )])
    fig.update_layout(
        title='Absolute Incremental Log-Odds Diff Surface',
        scene=dict(
            xaxis_title='Seq Position',
            yaxis_title='Theta',
            zaxis_title='Mean Abs Diff'
        ),
        width=800,
        height=600,
    )
    fig.show()

def visualize_attention_patterns(theta, model, seq_length=100, data=None):
    """
    Visualize attention patterns for sequences generated with probability theta.
    Accepts parameters similar to other plot functions and generates its own data if none provided.
    Args:
        theta: float, probability parameter for data generation
        model: transformer model with run_with_cache
        seq_length: int, length of sequences to generate
        data: optional torch.Tensor of shape (B, T) to use instead of generating
    """
    # Generate or use provided data
    if data is None:
        test_batches, _ = generate_data_with_p(theta,
                                              batch_size=1,
                                              seq_length=seq_length,
                                              num_batches=1,
                                              flip_batch=False)
        sequences = test_batches[0]  # tensor [B, T]
    else:
        sequences = data  # assume tensor [B, T]
    # Ensure on correct device
    sequences = sequences.to(DEVICE)
    # Print sequence for inspection
    print("Sequence being visualized:")
    print(f"Sequence : {sequences.cpu().tolist()}")
    # Prepare sequence names
    model.eval()
    # Only analyze the first sequence and title with theta
    seq = sequences[0]
    name = f"{theta}"
    # Add batch dimension
    seq_tensor = seq.unsqueeze(0)
    # Run model with cache
    _, cache = model.run_with_cache(seq_tensor)
    n_layers = model.cfg.n_layers
    # Plot attention for each layer
    for layer_idx in range(n_layers):
        attn = cache["pattern", layer_idx, "attn"][0]
        n_heads = attn.shape[0]
        
        # Use a more reasonable layout: 2 columns, and enough rows to fit all heads
        n_cols = 2
        n_rows = (n_heads + n_cols - 1) // n_cols  # Ceiling division
        
        layer_fig, layer_axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
        
        # Handle both multi-head and single-head cases
        if hasattr(layer_axes, 'flatten'):
            flat_axes = layer_axes.flatten()
        else:
            flat_axes = [layer_axes]
        
        for head_idx in range(n_heads):
            ax = flat_axes[head_idx]
            im = ax.imshow(attn[head_idx].cpu().detach().numpy(), cmap='viridis')
            ax.set_title(f"Head {head_idx}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            plt.colorbar(im, ax=ax)
            
        # Hide unused subplots
        for h in range(n_heads, len(flat_axes)):
            flat_axes[h].axis('off')
            
        layer_fig.suptitle(f"Attention Pattern for sequence sampled from probability {name}, layer {layer_idx}", fontsize=16)
        plt.show()

def plot_log_odds_vs_theoretical_log(log_odds: torch.Tensor, theoretical_log: torch.Tensor, title: str = 'Log odds vs Theoretical log(i+2)'):
    """
    Plot log odds vs theoretical log values.

    Args:
        log_odds: Tensor of log odds values.
        theoretical_log: Tensor of theoretical log values.
        title: Title for the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(log_odds.cpu().detach().numpy(), marker='o', label='Log odds')
    plt.plot(theoretical_log.cpu().detach().numpy(), marker='x', linestyle='--', label='Theoretical log(i+2)')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_probability_distribution(probabilities: torch.Tensor, theoretical_p: float, title: str = 'Distribution of Probabilities for Class 1'):
    """
    Plot the distribution of probabilities with vertical lines for theoretical and mean probability.

    Args:
        probabilities: Tensor of probabilities for class 1.
        theoretical_p: Theoretical probability value to plot as a vertical line.
        title: Title for the plot.
    """

    prob_np = probabilities.cpu().detach().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(prob_np, bins=30, alpha=0.7, color='blue')
    plt.axvline(theoretical_p, color='red', linestyle='dashed', linewidth=2, label=f'Theoretical p = {theoretical_p:.3f}')
    plt.axvline(prob_np.mean(), color='green', linestyle='dashed', linewidth=2, label=f'Mean prob = {prob_np.mean():.3f}')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residual_cosine_similarity(seq_length, num_ones, resids, print_stats=True):
    """
    Plots the cosine similarity matrix of the given residual vectors.
    Args:
        seq_length: Length of sequences
        num_ones: Number of ones in sequences
        resids (torch.Tensor or np.ndarray): Residual vectors of shape (num_sequences, d_model)
        print_stats (bool): Whether to print statistics about the similarity matrix
    
    Returns:
        np.ndarray: Cosine similarity matrix
    """
    # Use the utility function to calculate the cosine similarity matrix
    cos_sim_matrix = get_residual_cosine_similarity(resids, print_stats=print_stats)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cos_sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"Cosine Similarity of Residual Vectors (Seq Length: {seq_length}, Num Ones: {num_ones})")
    plt.xlabel("Sequence Index")
    plt.ylabel("Sequence Index")
    plt.tight_layout()
    plt.show()

    return cos_sim_matrix
