"""
Training utilities for coinformer models.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import beta
from typing import List, Tuple, Optional
import os

from .config import ExperimentConfig
from .models import create_coinformer_model
from .samplers import generate_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def calculate_optimal_loss(alpha: float, beta_param: float) -> float:
    """
    Calculate the theoretical optimal cross-entropy loss when predicting sequences
    with probabilities drawn from a Beta(alpha, beta) distribution.
    """
    from scipy import special
    
    # Expected value of p*log(p) under Beta distribution
    expected_p_log_p = special.beta(alpha + 1, beta_param) / special.beta(alpha, beta_param) * (
        special.digamma(alpha + 1) - special.digamma(alpha + beta_param + 1)
    )
    
    # By symmetry for 1-p
    expected_1_p_log_1_p = special.beta(alpha, beta_param + 1) / special.beta(alpha, beta_param) * (
        special.digamma(beta_param + 1) - special.digamma(alpha + beta_param + 1)
    )
    
    # The optimal loss is the negative of the expectation of p*log(p) + (1-p)*log(1-p)
    optimal_loss = -(expected_p_log_p + expected_1_p_log_1_p)
    
    return optimal_loss


def train_coinformer_model(config: ExperimentConfig, verbose: bool = True, 
                          training_data: Optional[Tuple[List[torch.Tensor], List[float]]] = None) -> Tuple[torch.nn.Module, List[float]]:
    """
    Train a coinformer model according to the given configuration.
    
    Args:
        config: Experiment configuration
        verbose: Whether to print training progress
        training_data: Optional pre-generated training data as (datasets, priors). 
                      If None, generates fresh data for each epoch.
        
    Returns:
        Tuple of (trained model, list of epoch losses)
    """
    # Create model
    model = create_coinformer_model(config.model_config)
    model = model.to(DEVICE)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    # Calculate theoretical optimal loss
    if config.importance_sampling:
        optimal_loss = calculate_optimal_loss(config.importance_sampling_alpha, config.importance_sampling_beta)
    else:
        optimal_loss = calculate_optimal_loss(config.alpha, config.beta)

    for epoch in range(config.num_epochs):
        # Use pre-generated data if provided, otherwise generate new data for each epoch
        if training_data is not None:
            datasets, priors = training_data
        else:
            datasets, priors = generate_data(
                batch_size=config.batch_size, 
                seq_length=config.seq_length,
                num_batches=config.num_batches, 
                alpha=config.alpha, 
                beta=config.beta, 
                bernoulli=config.bernoulli, 
                bernoulli_p=config.bernoulli_p, 
                flip_batch=config.flip_batch,
                scale=config.scale, 
                bias=config.bias,
                use_bos_token=config.model_config.use_bos_token
            )

        epoch_loss = 0
        progress_bar = tqdm(zip(datasets, priors), desc=f"Epoch {epoch+1}/{config.num_epochs}") if verbose else zip(datasets, priors)
        
        for data_batch, prior in progress_bar:
            batch_size, seq_length = data_batch.shape
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
            
            # Apply importance sampling weighting if enabled
            if config.importance_sampling:
                weight = beta(config.importance_sampling_alpha, config.importance_sampling_beta).pdf(prior)
                loss = loss * weight

            epoch_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = epoch_loss / len(datasets)
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Theoretical Lower Bound: {optimal_loss:.4f}")

    return model, losses


def save_model_with_config(model: torch.nn.Module, config: ExperimentConfig, 
                          experiment_name: str, save_dir: str = "saved_models") -> str:
    """
    Save model with configuration information in filename.
    
    Args:
        model: Trained model
        config: Experiment configuration
        experiment_name: Name of the experiment
        save_dir: Directory to save models
        
    Returns:
        Path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create descriptive filename
    model_cfg = config.model_config
    filename_parts = [
        experiment_name,
        f"dmodel{model_cfg.d_model}",
        f"dhead{model_cfg.d_head}",
        f"layers{model_cfg.n_layers}",
        f"alpha{config.alpha}",
        f"beta{config.beta}",
    ]
    
    if model_cfg.use_bos_token:
        filename_parts.append("bos")
    if not model_cfg.use_pos_embed:
        filename_parts.append("nopos")
    if model_cfg.attn_only:
        filename_parts.append("attnonly")
    if config.importance_sampling:
        filename_parts.append("importance")
    
    filename = "_".join(filename_parts) + ".pt"
    filepath = os.path.join(save_dir, filename)
    
    torch.save(model.state_dict(), filepath)
    return filepath


def load_model_from_config(config: ExperimentConfig, filepath: str) -> torch.nn.Module:
    """
    Load a model from saved state dict using configuration.
    
    Args:
        config: Model configuration
        filepath: Path to saved model
        
    Returns:
        Loaded model
    """
    model = create_coinformer_model(config.model_config)
    model.load_state_dict(torch.load(filepath))
    model = model.to(DEVICE)
    return model