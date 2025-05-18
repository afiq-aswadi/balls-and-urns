import torch

def sample_from_beta(alpha=1.0, beta=1.0):
    """
    Sample probability from a Beta distribution with parameters alpha and beta.
    When alpha=beta=1.0, this is equivalent to a uniform distribution.
    """
    return torch.distributions.Beta(alpha, beta).sample().item()

def sample_from_bernoulli(p):
    """
    Sample from a Bernoulli distribution with probability p.
    """
    return torch.distributions.Bernoulli(p).sample().item()


def generate_data_with_p(
    p, batch_size=64, seq_length=20, num_batches=100, flip_batch=False, scale=1.0, bias=0.0
):
    """
    Generate batches given a fixed probability p.
    Args:
        p (float): Probability for Bernoulli distribution.
        batch_size (int, optional): Number of sequences in each batch. Defaults to 64.
        seq_length (int, optional): Length of each sequence. Defaults to 20.
        num_batches (int, optional): Number of batches to generate. Defaults to 100.
        flip_batch (bool, optional): If True, add the flipped version of the sequence to the dataset.
        scale (float, optional): Scaling factor for the generated data. Defaults to 1.0.
        bias (float, optional): Bias added to the sampled probability. Defaults to 0.0.
    Returns:
        tuple:
            - datasets (list): List of tensors, where each tensor has shape (batch_size, seq_length)
              and contains binary sequences.
            - priors (list): List of probability values sampled from the Beta distribution,
              one for each batch.
    """
    datasets = []
    priors = []

    for _ in range(num_batches):
        # Use the provided probability p
        assert 0 <= scale*p+bias <= 1, f"Sampled probability {scale*p+bias} is out of bounds [0, 1]"
        priors.append(scale*p + bias)

        # Generate sequences of 0s and 1s based on the sampled probability
        data = torch.bernoulli(torch.full((batch_size, seq_length), scale*p+bias))
        datasets.append(data.long())
        
        # If flip_batch is True, add the flipped version of the sequence
        if flip_batch:
            flipped_data = 1 - data  # Flip 0s to 1s and 1s to 0s
            datasets.append(flipped_data.long())
            priors.append(1 - (scale*p + bias))  # Add 1-p for the flipped batch
    print(f"Probability range: [{min(priors):.3f}, {max(priors):.3f}]")
    return datasets, priors


def generate_data_with_p_list(
    p_list, batch_size=64, seq_length=20, num_batches=100, flip_batch=False, scale=1.0, bias=0.0
):
    """
    Generate batches for a list of fixed probabilities p.
    For each p in p_list, it calls generate_data_with_p.
    Args:
        p_list (list[float]): List of probabilities for Bernoulli distribution.
        batch_size (int, optional): Number of sequences in each batch. Defaults to 64.
        seq_length (int, optional): Length of each sequence. Defaults to 20.
        num_batches (int, optional): Number of batches to generate for each p in p_list. Defaults to 100.
        flip_batch (bool, optional): If True, add the flipped version of the sequence to the dataset.
        scale (float, optional): Scaling factor for the generated data. Defaults to 1.0.
        bias (float, optional): Bias added to the sampled probability. Defaults to 0.0.
    Returns:
        tuple:
            - all_datasets (list): List of tensors, where each tensor has shape (batch_size, seq_length)
              and contains binary sequences, aggregated from all p in p_list.
            - all_priors (list): List of probability values used, one for each batch,
              aggregated from all p in p_list.
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
            bias=bias
        )
        all_datasets.extend(datasets)
        all_priors.extend(priors)
    
    if all_priors: # Ensure all_priors is not empty before trying to print min/max
        print(f"Overall probability range for p_list: [{min(all_priors):.3f}, {max(all_priors):.3f}]")
    else:
        print("No data generated as p_list might be empty or resulted in no priors.")
        
    return all_datasets, all_priors


# Generate training data
def generate_data(batch_size=64, seq_length=20, num_batches=100, alpha=1.0, beta=1.0, bernoulli=False, bernoulli_p=0.5, flip_batch=False, scale=1.0, bias=0.0):
    """
    Generate batches of binary sequence data using a hierarchical generative model.
    For each batch, a probability p is sampled from a Beta distribution with parameters
    alpha and beta. Then, batch_size sequences of 0s and 1s are generated where each 
    element is sampled independently from a Bernoulli distribution with probability p.
    Args:
        batch_size (int, optional): Number of sequences in each batch. Defaults to 64.
        seq_length (int, optional): Length of each sequence. Defaults to 20.
        num_batches (int, optional): Number of batches to generate. Defaults to 100.
        alpha (float, optional): Alpha parameter for the Beta distribution. Defaults to 1.0.
        beta (float, optional): Beta parameter for the Beta distribution. Defaults to 1.0.
        bernoulli (bool, optional): If True, sample p from Bernoulli distribution. Defaults to False.
        bernoulli_p (float, optional): Parameter for Bernoulli distribution. Defaults to 0.5.
        flip_batch (bool, optional): If True, add the flipped version of the sequence to the dataset.
        scale (float, optional): Scaling factor for the generated data. Defaults to 1.0.
        bias (float, optional): Bias added to the sampled probability. Defaults to 0.0.
    Returns:
        tuple:
            - datasets (list): List of tensors, where each tensor has shape (batch_size, seq_length)
              and contains binary sequences.
            - priors (list): List of probability values sampled from the Beta distribution,
              one for each batch.
    """
    datasets = []
    priors = []

    for _ in range(num_batches):
        # Sample probability from beta prior
        if bernoulli:
            p = sample_from_bernoulli(bernoulli_p)
            
        else:
            p = sample_from_beta(alpha, beta)
        
        assert 0 <= scale*p+bias <= 1, f"Sampled probability {scale*p+bias} is out of bounds [0, 1]"
        priors.append(scale*p + bias)

        # Generate sequences of 0s and 1s based on the sampled probability
        data = torch.bernoulli(torch.full((batch_size, seq_length), scale*p+bias))
        datasets.append(data.long())
        
        # If flip_batch is True, add the flipped version of the sequence
        if flip_batch:
            flipped_data = 1 - data  # Flip 0s to 1s and 1s to 0s
            datasets.append(flipped_data.long())
            priors.append(1 - (scale*p + bias))  # Add 1-p for the flipped batch
    print(f"Probability range: [{min(priors):.3f}, {max(priors):.3f}]")
    return datasets, priors
