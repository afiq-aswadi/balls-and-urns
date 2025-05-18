import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import analyze_single_one_sequences
from samplers import generate_data
from model import deactivate_position
from utils import calculate_optimal_loss


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_coinformer_model(model, 
                           num_epochs: int = 3,
                           learning_rate: float = 0.001,
                           batch_size: int = 64,
                           seq_length: int = 20,
                           num_batches: int = 100,
                           alpha: float = 1.0,
                           beta: float = 1.0,
                           bernoulli: bool = False,
                           bernoulli_p: float = 0.5,
                           flip_batch: bool = False,
                           pos_embed: bool = True,
                           add_zeros_and_ones: int = 0,
                           scale: float = 1.0,
                           bias: float = 0.0):
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
                                   num_batches=num_batches, alpha=alpha, beta=beta, bernoulli=bernoulli, bernoulli_p=bernoulli_p, flip_batch=flip_batch
                                   , scale=scale, bias=bias)

        # Add the specified number of batches of all 0s and all 1s
        if add_zeros_and_ones > 0:
            for _ in range(add_zeros_and_ones):
                zeros = torch.zeros((batch_size, seq_length), dtype=torch.long)
                ones = torch.ones((batch_size, seq_length), dtype=torch.long)
                datasets.extend([zeros, ones])
                priors.extend([0.0, 1.0])

        epoch_loss = 0
        
        # Plot histogram of priors for this epoch
        # plt.figure(figsize=(8, 4))
        # plt.hist(priors, bins=10, alpha=0.7, color='blue', edgecolor='black')
        # plt.title(f'Distribution of Prior Probabilities (Epoch {epoch+1})')
        # plt.xlabel('Probability Value')
        # plt.ylabel('Frequency')
        # plt.grid(True, alpha=0.3)
        # plt.show()
        optimal_loss = calculate_optimal_loss(alpha,beta)

        for data_batch in tqdm(datasets, desc=f"Epoch {epoch+1}/{num_epochs}"):
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