"""
Data generation utilities for coinformer experiments.
"""
import torch
import itertools
from typing import List, Tuple, Optional


def sample_from_beta(alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Sample probability from a Beta distribution with parameters alpha and beta.
    When alpha=beta=1.0, this is equivalent to a uniform distribution.
    """
    return torch.distributions.Beta(alpha, beta).sample().item()


def sample_from_bernoulli(p: float) -> float:
    """
    Sample from a Bernoulli distribution with probability p.
    """
    return torch.distributions.Bernoulli(p).sample().item()


def add_bos_token(data: torch.Tensor, bos_token_id: int = 2) -> torch.Tensor:
    """
    Add BOS token to the beginning of sequences.
    
    Args:
        data: Tensor of shape (batch_size, seq_length) with values 0 and 1
        bos_token_id: ID for BOS token (default: 2)
        
    Returns:
        Tensor of shape (batch_size, seq_length + 1) with BOS tokens prepended
    """
    batch_size, seq_length = data.shape
    bos_tokens = torch.full((batch_size, 1), bos_token_id, dtype=data.dtype, device=data.device)
    return torch.cat([bos_tokens, data], dim=1)


def generate_data_with_p(
    p: float, 
    batch_size: int = 64, 
    seq_length: int = 20, 
    num_batches: int = 100, 
    flip_batch: bool = False, 
    scale: float = 1.0, 
    bias: float = 0.0,
    use_bos_token: bool = False
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Generate batches given a fixed probability p.
    
    Args:
        p: Probability for Bernoulli distribution
        batch_size: Number of sequences in each batch
        seq_length: Length of each sequence (excluding BOS token if used)
        num_batches: Number of batches to generate
        flip_batch: If True, add the flipped version of the sequence to the dataset
        scale: Scaling factor for the generated data
        bias: Bias added to the sampled probability
        use_bos_token: If True, prepend BOS token to sequences
        
    Returns:
        Tuple of (datasets, priors)
    """
    datasets = []
    priors = []

    for _ in range(num_batches):
        # Use the provided probability p
        effective_p = scale * p + bias
        assert 0 <= effective_p <= 1, f"Effective probability {effective_p} is out of bounds [0, 1]"
        priors.append(effective_p)

        # Generate sequences of 0s and 1s based on the sampled probability
        data = torch.bernoulli(torch.full((batch_size, seq_length), effective_p))
        
        # Add BOS token if requested
        if use_bos_token:
            data = add_bos_token(data)
        
        datasets.append(data.long())
        
        # If flip_batch is True, add the flipped version of the sequence
        if flip_batch:
            if use_bos_token:
                # For BOS token case, flip only the non-BOS tokens
                flipped_data = data.clone()
                flipped_data[:, 1:] = 1 - data[:, 1:]  # Flip everything except BOS token
            else:
                flipped_data = 1 - data  # Flip 0s to 1s and 1s to 0s
            
            datasets.append(flipped_data.long())
            priors.append(1 - effective_p)  # Add 1-p for the flipped batch
    
    print(f"Probability range: [{min(priors):.3f}, {max(priors):.3f}]")
    return datasets, priors


def generate_data_with_p_list(
    p_list: List[float], 
    batch_size: int = 64, 
    seq_length: int = 20, 
    num_batches: int = 100, 
    flip_batch: bool = False, 
    scale: float = 1.0, 
    bias: float = 0.0,
    use_bos_token: bool = False
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Generate batches for a list of fixed probabilities p.
    """
    all_datasets = []
    all_priors = []

    for p_val in p_list:
        datasets, priors = generate_data_with_p(
            p=p_val,
            batch_size=batch_size,
            seq_length=seq_length,
            num_batches=num_batches,
            flip_batch=flip_batch,
            scale=scale,
            bias=bias,
            use_bos_token=use_bos_token
        )
        all_datasets.extend(datasets)
        all_priors.extend(priors)
    
    if all_priors:
        print(f"Overall probability range for p_list: [{min(all_priors):.3f}, {max(all_priors):.3f}]")
    else:
        print("No data generated as p_list might be empty or resulted in no priors.")
        
    return all_datasets, all_priors


def generate_data(
    batch_size: int = 64, 
    seq_length: int = 20, 
    num_batches: int = 100, 
    alpha: float = 1.0, 
    beta: float = 1.0, 
    bernoulli: bool = False, 
    bernoulli_p: float = 0.5, 
    flip_batch: bool = False, 
    scale: float = 1.0, 
    bias: float = 0.0,
    use_bos_token: bool = False
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Generate batches of binary sequence data using a hierarchical generative model.
    """
    datasets = []
    priors = []

    for _ in range(num_batches):
        # Sample probability from beta prior or use fixed bernoulli
        if bernoulli:
            p = sample_from_bernoulli(bernoulli_p)
        else:
            p = sample_from_beta(alpha, beta)
        
        effective_p = scale * p + bias
        assert 0 <= effective_p <= 1, f"Effective probability {effective_p} is out of bounds [0, 1]"
        priors.append(effective_p)

        # Generate sequences of 0s and 1s based on the sampled probability
        data = torch.bernoulli(torch.full((batch_size, seq_length), effective_p))
        
        # Add BOS token if requested
        if use_bos_token:
            data = add_bos_token(data)
            
        datasets.append(data.long())
        
        # If flip_batch is True, add the flipped version of the sequence
        if flip_batch:
            if use_bos_token:
                # For BOS token case, flip only the non-BOS tokens
                flipped_data = data.clone()
                flipped_data[:, 1:] = 1 - data[:, 1:]
            else:
                flipped_data = 1 - data
            
            datasets.append(flipped_data.long())
            priors.append(1 - effective_p)
    
    print(f"Probability range: [{min(priors):.3f}, {max(priors):.3f}]")
    return datasets, priors


def generate_sequential_ones(n: int, add_zero: bool = False, use_bos_token: bool = False) -> torch.Tensor:
    """
    Generate a sequence of ones followed by zeros.
    
    Args:
        n: Length of the sequence (excluding BOS token if used)
        add_zero: If True, prepend a row of zeros to the output
        use_bos_token: If True, add BOS token to sequences
        
    Returns:
        Tensor containing the sequences
    """
    mat = torch.tril(torch.ones((n, n), dtype=torch.long))
    if add_zero:
        mat = torch.cat((torch.zeros((1, n), dtype=torch.long), mat), dim=0)
    
    if use_bos_token:
        # Add BOS token to all sequences
        mat = add_bos_token(mat)
    
    return mat


def generate_all_binary_sequences_with_fixed_num_ones(
    n: int, 
    num_ones: int, 
    max_n_sequences: Optional[int] = None,
    use_bos_token: bool = False
) -> torch.Tensor:
    """
    Generate all possible binary sequences of length n with exactly num_ones ones.
    
    Args:
        n: Length of the sequence (excluding BOS token if used)
        num_ones: Number of ones in each sequence
        max_n_sequences: Maximum number of sequences to generate
        use_bos_token: If True, add BOS token to sequences
        
    Returns:
        Tensor containing all permutations
    """
    # Generate all combinations of positions for the ones
    positions_iter = itertools.combinations(range(n), num_ones)
    if max_n_sequences is not None:
        positions_iter = itertools.islice(positions_iter, max_n_sequences)
    positions_list = list(positions_iter)
    num_permutations = len(positions_list)
    
    # Initialize the output tensor
    sequences = torch.zeros((num_permutations, n), dtype=torch.long)
    
    # Fill in the tensor with 1s at the appropriate positions
    for i, positions in enumerate(positions_list):
        for pos in positions:
            sequences[i, pos] = 1
    
    if use_bos_token:
        sequences = add_bos_token(sequences)
    
    return sequences