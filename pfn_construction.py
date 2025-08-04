#%%
import torch
#%%

embedding_dim = 128

def random_embedding(dim):
    """
    Generate a random unit embedding vector of given dimension.
    """
    vec = torch.randn(dim)
    return vec / torch.norm(vec)  # normalize to unit vector

pos_embed = random_embedding(embedding_dim) #this direction corresponds to length of the sequence, and is in the null space of the attention head
head_embed = random_embedding(embedding_dim) #this direction corresponds to the head of the sequence
tail_embed = random_embedding(embedding_dim) #this direction corresponds to the tail of the sequence
bos_embed = random_embedding(embedding_dim) #this direction corresponds to the beginning of the sequence

#%%

def get_counting_output(last_token, head_embed, tail_embed, n_heads_or_tails):
    """
    Compute the counting output that the attention head should produce.
    This is just the token embedding times the count, without positional info.
    """
    if last_token == "head":
        return head_embed * n_heads_or_tails
    else:
        return tail_embed * n_heads_or_tails

def get_residual(last_token, pos_embed, head_embed, tail_embed, sequence_length, n_heads_or_tails):
    """
    Compute the full residual vector for the last token in the sequence.
    This includes both positional and counting components.
    """
    counting_component = get_counting_output(last_token, head_embed, tail_embed, n_heads_or_tails)
    positional_component = pos_embed * sequence_length
    return positional_component + counting_component


def get_PPD(last_token, n_heads_or_tails, sequence_length, alpha, beta):
    """
    Compute the PPD (Probability of Previous Distribution) for the last token.
    """
    prob = (alpha + n_heads_or_tails) / (sequence_length + alpha + beta)
    complement = 1 - prob 

    if last_token == "head":
        return torch.tensor([0, prob, complement])
    else:
        return torch.tensor([0, complement, prob]) 
    
#%%

class MLP(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_factor=4):
        super().__init__()
        self.layer1 = torch.nn.Linear(embedding_dim, embedding_dim * hidden_factor)
        self.layer2 = torch.nn.Linear(embedding_dim * hidden_factor, 3)  # Output probabilities for [BOS, head, tail]
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)  # Apply softmax along last dimension
        
    def forward(self, x):
        # Handle both single samples and batches
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        
        x = self.activation(self.layer1(x))
        logits = self.layer2(x)
        return self.softmax(logits)

# Initialize the MLP
mlp = MLP(embedding_dim)

# Define optimizer (we'll use MSE loss directly in training loop)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)


#%%

def generate_training_data(n_samples, alpha=1.0, beta=1.0):
    """
    Generate training data for the MLP.
    
    Args:
        n_samples: Number of training samples
        alpha, beta: Beta distribution parameters
    
    Returns:
        residuals: Input residual vectors
        ppds: Target PPD distributions
    """
    residuals = []
    ppds = []
    
    for _ in range(n_samples):
        # Randomly sample sequence parameters
        sequence_length = torch.randint(2, 100, (1,)).item()  # Random sequence length 2-99
        n_heads_or_tails = torch.randint(1, sequence_length, (1,)).item()  # Random count
        
        # Randomly choose last token type
        last_token = "head" if torch.rand(1).item() > 0.5 else "tail"
        
        # Generate residual and PPD
        residual = get_residual(last_token, pos_embed, head_embed, tail_embed, 
                              sequence_length, n_heads_or_tails)
        ppd = get_PPD(last_token, n_heads_or_tails, sequence_length, alpha, beta)
        
        residuals.append(residual)
        ppds.append(ppd)
    
    return torch.stack(residuals), torch.stack(ppds)


def train_mlp(mlp, n_epochs=1000, batch_size=32, alpha=1.0, beta=1.0):
    """
    Train the MLP to learn the PPD function for both head and tail tokens.
    """
    mlp.train()
    
    for epoch in range(n_epochs):
        # Generate fresh training data each epoch (now includes both head and tail)
        residuals, true_ppds = generate_training_data(batch_size, alpha, beta)
        
        # Forward pass
        optimizer.zero_grad()
        pred_ppds = mlp(residuals)
        
        # Compute loss (using MSE since we want to match probability distributions)
        loss = torch.nn.functional.mse_loss(pred_ppds, true_ppds)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.6f}")
    
    return mlp


#%%

# Set hyperparameters
alpha = 1.0
beta = 1.0
n_epochs = 1000
batch_size = 512  # Increased batch size for better efficiency

print("Starting training...")
trained_mlp = train_mlp(mlp, n_epochs, batch_size, alpha, beta)
print("Training completed!")

#%%

# Test the trained MLP
def test_mlp(mlp, alpha=1.0, beta=1.0, n_test_samples=10):
    """
    Test the trained MLP on examples with both head and tail tokens.
    """
    mlp.eval()
    
    print("\nTesting trained MLP on both HEAD and TAIL tokens:")
    print("=" * 60)
    
    with torch.no_grad():
        for i in range(n_test_samples):
            # Generate test case
            sequence_length = torch.randint(2, 15, (1,)).item()
            n_heads_or_tails = torch.randint(1, sequence_length, (1,)).item()
            
            # Test both head and tail cases
            for last_token in ["head", "tail"]:
                # Get true values
                residual = get_residual(last_token, pos_embed, head_embed, tail_embed, 
                                      sequence_length, n_heads_or_tails)
                true_ppd = get_PPD(last_token, n_heads_or_tails, sequence_length, alpha, beta)
                
                # Get prediction
                pred_ppd = mlp(residual.unsqueeze(0)).squeeze(0)
                
                # Calculate error
                error = torch.nn.functional.mse_loss(pred_ppd, true_ppd).item()
                
                print(f"Test {i+1} ({last_token.upper()}):")
                print(f"  Sequence length: {sequence_length}, Count: {n_heads_or_tails}")
                print(f"  True PPD:  {true_ppd.numpy()}")
                print(f"  Pred PPD:  {pred_ppd.numpy()}")
                print(f"  MSE Error: {error:.6f}")
                print()

# Run tests
test_mlp(trained_mlp, alpha, beta)

#%%

def token_embedding(tokens):
    """
    Embed tokens [0, 1, 2] to [bos, head, tail] respectively using a linear map.
    
    Args:
        tokens: Tensor of token IDs (0, 1, or 2) of shape (seq_len,) or (batch_size, seq_len)
    
    Returns:
        Embedding matrix of shape (seq_len, embedding_dim) or (batch_size, seq_len, embedding_dim)
    """
    # Create embedding matrix: [bos_embed, head_embed, tail_embed]
    embedding_matrix = torch.stack([bos_embed, head_embed, tail_embed], dim=0)
    
    # Use embedding lookup
    return torch.nn.functional.embedding(tokens, embedding_matrix)

#%%

class CountingAttentionHead(torch.nn.Module):
    def __init__(self, embedding_dim, pos_embed):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_embed = pos_embed / torch.norm(pos_embed)  # Normalize pos_embed
        
        # Initialize weight matrices with appropriate scale for the task
        self.W_Q = torch.nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.W_K = torch.nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.W_V = torch.nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        
        # Use smaller initialization to start with reasonable scales
        std = 0.1 / (embedding_dim ** 0.5)
        torch.nn.init.normal_(self.W_Q, 0, std)
        torch.nn.init.normal_(self.W_K, 0, std)
        torch.nn.init.normal_(self.W_V, 0, std)
    
    def forward(self, embeddings):
        """
        Apply attention mechanism to compute residuals for all positions with causal masking.
        
        Args:
            embeddings: (seq_len, embedding_dim) - sequence of token embeddings including positional encoding
        
        Returns:
            residuals: (seq_len, embedding_dim) - residual vectors for all positions
        """
        seq_len = embeddings.shape[0]
        
        # Compute Q, K, V
        Q = embeddings @ self.W_Q  # (seq_len, embedding_dim)
        K = embeddings @ self.W_K  # (seq_len, embedding_dim)
        V = embeddings @ self.W_V  # (seq_len, embedding_dim)
        
        # Compute attention scores for all positions with scaling
        scale = 1.0 / (self.embedding_dim ** 0.5)
        scores = torch.matmul(Q, K.T) * scale  # (seq_len, seq_len)
        
        # Apply causal mask (lower triangular matrix)
        # Each position can only attend to previous positions (including itself)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply attention weights (softmax along the key dimension)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (seq_len, seq_len)
        
        # Compute weighted values for all positions
        attended_values = torch.matmul(attn_weights, V)  # (seq_len, embedding_dim)
        
        # Return attended values for all positions
        return attended_values

#%%

def generate_sequence(seq_len):
    """
    Generate a random sequence of tokens.
    
    Args:
        seq_len: Length of the sequence
    
    Returns:
        tokens: (seq_len,) - token IDs [0=BOS, 1=HEAD, 2=TAIL]
        embeddings: (seq_len, embedding_dim) - token embeddings only
        n_heads: int - count of head tokens
        n_tails: int - count of tail tokens
    """
    # Generate tokens: first token is BOS (0), rest are randomly HEAD (1) or TAIL (2)
    tokens = torch.zeros(seq_len, dtype=torch.long)
    tokens[0] = 0  # BOS token
    tokens[1:] = torch.randint(1, 3, (seq_len-1,))  # HEAD or TAIL tokens
    
    # Get token embeddings (no positional encoding yet)
    embeddings = token_embedding(tokens)  # (seq_len, embedding_dim)
    
    # Count heads and tails (excluding BOS)
    n_heads = torch.sum(tokens[1:] == 1).item()
    n_tails = torch.sum(tokens[1:] == 2).item()
    
    return tokens, embeddings, n_heads, n_tails

#%%

def generate_attention_training_data(n_samples, min_seq_len=5, max_seq_len=100):
    """
    Generate training data for the attention head.
    
    Returns:
        input_embeddings: List of embedding sequences (with positional encoding added)
        target_counting_outputs: List of target counting output matrices (seq_len, embedding_dim)
    """
    input_embeddings = []
    target_counting_outputs = []
    
    for _ in range(n_samples):
        # Random sequence length
        seq_len = max_seq_len
        
        # Generate sequence
        tokens, embeddings, n_heads, n_tails = generate_sequence(seq_len)
        
        # Add positional encoding
        position_indices = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        positional_encoding = position_indices * pos_embed.unsqueeze(0)  # (seq_len, embedding_dim)
        embeddings_with_pos = embeddings + positional_encoding
        
        # Compute target counting outputs for every position
        target_counting_matrix = torch.zeros(seq_len, embedding_dim)
        
        for pos in range(1, seq_len):  # Skip position 0 (BOS) since it has no preceding tokens
            # Consider subsequence from 0 to pos (inclusive)
            subseq_tokens = tokens[:pos+1]
            
            # Find last token in subsequence and count heads/tails in subsequence
            last_token_in_subseq = subseq_tokens[-1].item()
            last_token_type = "head" if last_token_in_subseq == 1 else "tail"
            
            # Count heads and tails in subsequence (excluding BOS at position 0)
            subseq_heads = torch.sum(subseq_tokens[1:] == 1).item()
            subseq_tails = torch.sum(subseq_tokens[1:] == 2).item()
            
            if last_token_type == "head":
                n_heads_or_tails = subseq_heads
            else:
                n_heads_or_tails = subseq_tails
                
            # Target is just the counting output (no positional component)
            target_counting_matrix[pos] = get_counting_output(last_token_type, head_embed, tail_embed, 
                                                            n_heads_or_tails)
        
        input_embeddings.append(embeddings_with_pos)
        target_counting_outputs.append(target_counting_matrix)
    
    return input_embeddings, target_counting_outputs

#%%

def train_attention_head(attention_head, n_epochs=2000, batch_size=16):
    """
    Train the attention head to output the correct counting result for every position.
    """
    optimizer = torch.optim.Adam(attention_head.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100)
    attention_head.train()
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 300
    
    # Generate fixed training data to avoid instability
    print("Generating fixed training dataset...")
    input_embeddings, target_counting_matrices = generate_attention_training_data(batch_size * 10, min_seq_len=5, max_seq_len=30)
    
    for epoch in range(n_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        # Use random subset of fixed data each epoch
        indices = torch.randperm(len(input_embeddings))[:batch_size]
        
        for idx in indices:
            embeddings = input_embeddings[idx]
            target_counting_matrix = target_counting_matrices[idx]
            
            # Forward pass
            pred_counting_matrix = attention_head(embeddings)  # (seq_len, embedding_dim)
            
            # Compute loss for all positions except position 0 (BOS)
            valid_positions = slice(1, embeddings.shape[0])
            
            pred_valid = pred_counting_matrix[valid_positions]
            target_valid = target_counting_matrix[valid_positions]
            
            # Use MSE loss only
            loss = torch.nn.functional.mse_loss(pred_valid, target_valid)
            total_loss += loss
        
        # Average loss over batch
        avg_loss = total_loss / batch_size
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(attention_head.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return attention_head

#%%

def test_attention_head(attention_head, n_test_samples=10):
    """
    Test the trained attention head on all positions.
    """
    attention_head.eval()
    
    print("\nTesting trained attention head:")
    print("=" * 60)
    
    with torch.no_grad():
        input_embeddings, target_counting_matrices = generate_attention_training_data(n_test_samples, 8, 15)
        
        total_cos_sim = 0
        total_samples = 0
        
        for i, (embeddings, target_counting_matrix) in enumerate(zip(input_embeddings, target_counting_matrices)):
            # Get predictions for all positions
            pred_counting_matrix = attention_head(embeddings)  # (seq_len, embedding_dim)
            
            # Calculate metrics for valid positions (excluding position 0)
            valid_positions = slice(1, embeddings.shape[0])
            pred_valid = pred_counting_matrix[valid_positions]
            target_valid = target_counting_matrix[valid_positions]
            
            # MSE error
            mse_error = torch.nn.functional.mse_loss(pred_valid, target_valid).item()
            
            # Cosine similarity (most important metric)
            cos_sims = torch.nn.functional.cosine_similarity(pred_valid, target_valid, dim=1)
            avg_cos_sim = cos_sims.mean().item()
            min_cos_sim = cos_sims.min().item()
            
            # Magnitude ratios
            pred_norms = torch.norm(pred_valid, dim=1)
            target_norms = torch.norm(target_valid, dim=1)
            magnitude_ratios = pred_norms / target_norms
            avg_mag_ratio = magnitude_ratios.mean().item()
            
            total_cos_sim += avg_cos_sim
            total_samples += 1
            
            # Extract sequence info for display
            seq_len = embeddings.shape[0]
            
            print(f"Test {i+1}:")
            print(f"  Sequence length: {seq_len}")
            print(f"  Average Cosine similarity: {avg_cos_sim:.4f} (min: {min_cos_sim:.4f})")
            print(f"  Average magnitude ratio: {avg_mag_ratio:.3f}")
            print(f"  MSE Error: {mse_error:.6f}")
            
            # Show metrics for last position specifically
            if seq_len > 1:
                last_cos_sim = cos_sims[-1].item()
                last_mag_ratio = magnitude_ratios[-1].item()
                print(f"  Last position cosine sim: {last_cos_sim:.4f}")
                print(f"  Last position mag ratio: {last_mag_ratio:.3f}")
            print()
        
        overall_cos_sim = total_cos_sim / total_samples
        print(f"Overall average cosine similarity: {overall_cos_sim:.4f}")
        
        if overall_cos_sim > 0.9:
            print("🎉 Excellent performance! Directions are well aligned.")
        elif overall_cos_sim > 0.7:
            print("✅ Good performance! Directions are mostly correct.")
        elif overall_cos_sim > 0.5:
            print("⚠️  Moderate performance. Some improvement needed.")
        else:
            print("❌ Poor performance. Significant training issues.")

#%%

# Initialize and train the attention head
print("Initializing attention head...")
attention_head = CountingAttentionHead(embedding_dim, pos_embed)

print("Starting attention head training...")
trained_attention_head = train_attention_head(attention_head, n_epochs=2000, batch_size=16)
print("Attention head training completed!")

# Test the trained attention head
test_attention_head(trained_attention_head)

#%%

# Test on a specific example to see attention weights
print("\nAnalyzing attention mechanism on a specific example:")
with torch.no_grad():
    # Generate a test sequence
    tokens, embeddings, n_heads, n_tails = generate_sequence(10)
    
    # Add positional encoding
    seq_len = len(tokens)
    position_indices = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
    positional_encoding = position_indices * pos_embed.unsqueeze(0)  # (seq_len, embedding_dim)
    embeddings = embeddings + positional_encoding
    
    # Determine last token type
    last_token = tokens[-1].item()
    last_token_type = "head" if last_token == 1 else "tail"
    
    print(f"Sequence tokens: {tokens.tolist()}")
    print(f"Token meanings: 0=BOS, 1=HEAD, 2=TAIL")
    print(f"Last token type: {last_token_type}")
    print(f"Heads count: {n_heads}, Tails count: {n_tails}")
    
    # Forward pass to get all attention weights
    seq_len = embeddings.shape[0]
    Q = embeddings @ trained_attention_head.W_Q
    K = embeddings @ trained_attention_head.W_K
    V = embeddings @ trained_attention_head.W_V
    
    scale = 1.0 / (embedding_dim ** 0.5)
    scores = torch.matmul(Q, K.T) * scale  # (seq_len, seq_len)
    
    # Apply causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float('-inf'))
    
    # Get attention weights for all positions
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (seq_len, seq_len)
    
    # Focus on the last token's attention pattern
    last_token_attn = attn_weights[-1]  # (seq_len,)
    
    print(f"Last token attention weights: {last_token_attn.numpy()}")
    print(f"Sum of last token attention weights: {last_token_attn.sum().item():.6f}")
    
    # Check if attention focuses on relevant tokens
    if last_token_type == "head":
        head_positions = torch.where(tokens == 1)[0]
        if len(head_positions) > 0:
            head_attention = last_token_attn[head_positions].sum().item()
            print(f"Total attention on HEAD tokens: {head_attention:.4f}")
    else:
        tail_positions = torch.where(tokens == 2)[0]
        if len(tail_positions) > 0:
            tail_attention = last_token_attn[tail_positions].sum().item()
            print(f"Total attention on TAIL tokens: {tail_attention:.4f}")
    
    # Show attention pattern for all positions
    print(f"\nAttention patterns for all positions:")
    for i in range(seq_len):
        token_type = ["BOS", "HEAD", "TAIL"][tokens[i].item()]
        print(f"Position {i} ({token_type}): {attn_weights[i].numpy()}")


# %%

class OneLayerTransformer(torch.nn.Module):
    """
    A complete one-layer transformer that combines:
    1. Token embeddings + positional encoding
    2. Attention head (for counting)
    3. MLP (for converting to probability distribution)
    """
    def __init__(self, embedding_dim, pos_embed, head_embed, tail_embed, bos_embed, 
                 attention_head, mlp):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_embed = pos_embed
        self.head_embed = head_embed
        self.tail_embed = tail_embed
        self.bos_embed = bos_embed
        self.attention_head = attention_head
        self.mlp = mlp
        
        # Create embedding matrix for token lookup
        self.embedding_matrix = torch.stack([bos_embed, head_embed, tail_embed], dim=0)
    
    def forward(self, tokens):
        """
        Full transformer forward pass.
        
        Args:
            tokens: (seq_len,) - token IDs [0=BOS, 1=HEAD, 2=TAIL]
        
        Returns:
            probabilities: (3,) - probability distribution [P(BOS), P(HEAD), P(TAIL)]
        """
        seq_len = len(tokens)
        
        # 1. Token embeddings
        embeddings = torch.nn.functional.embedding(tokens, self.embedding_matrix)  # (seq_len, embedding_dim)
        
        # 2. Add positional encoding
        position_indices = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        positional_encoding = position_indices * self.pos_embed.unsqueeze(0)  # (seq_len, embedding_dim)
        embeddings_with_pos = embeddings + positional_encoding
        
        # 3. Attention head (produces counting outputs for all positions)
        attention_outputs = self.attention_head(embeddings_with_pos)  # (seq_len, embedding_dim)
        
        # 4. Get the output for the last position
        last_attention_output = attention_outputs[-1]  # (embedding_dim,)
        
        # 5. The attention output already contains the counting component.
        # We need to add the positional component to create the full residual
        # that the MLP was trained on.
        positional_component = self.pos_embed * seq_len
        full_residual = last_attention_output + positional_component
        
        # 6. Pass through MLP to get probability distribution
        probabilities = self.mlp(full_residual)  # (3,)
        
        return probabilities.squeeze()

# %%

# Create the complete transformer
print("Creating complete one-layer transformer...")
transformer = OneLayerTransformer(
    embedding_dim=embedding_dim,
    pos_embed=pos_embed,
    head_embed=head_embed,
    tail_embed=tail_embed,
    bos_embed=bos_embed,
    attention_head=trained_attention_head,
    mlp=trained_mlp
)

print("Transformer created successfully!")

# %%

def test_transformer_bayes_updating(transformer, alpha=1.0, beta=1.0, n_tests=20):
    """
    Test that the transformer performs optimal Bayes updating.
    Compare transformer predictions with theoretical Bayesian posteriors.
    """
    transformer.eval()
    
    print(f"\nTesting Transformer vs Optimal Bayes Updating (α={alpha}, β={beta}):")
    print("=" * 70)
    
    results = {
        'sequence_lengths': [],
        'head_counts': [],
        'tail_counts': [],
        'last_tokens': [],
        'transformer_head_probs': [],
        'transformer_tail_probs': [],
        'bayes_head_probs': [],
        'bayes_tail_probs': [],
        'errors_head': [],
        'errors_tail': []
    }
    
    with torch.no_grad():
        for i in range(n_tests):
            # Generate test sequence
            seq_len = torch.randint(5, 20, (1,)).item()
            tokens, _, n_heads, n_tails = generate_sequence(seq_len)
            
            # Get transformer prediction
            transformer_probs = transformer(tokens)  # (3,) - [P(BOS), P(HEAD), P(TAIL)]
            
            # Get theoretical Bayesian prediction
            last_token = tokens[-1].item()
            last_token_type = "head" if last_token == 1 else "tail"
            
            if last_token_type == "head":
                n_heads_or_tails = n_heads
            else:
                n_heads_or_tails = n_tails
            
            bayes_probs = get_PPD(last_token_type, n_heads_or_tails, seq_len, alpha, beta)
            
            # Debug: Let's also compute what the MLP should receive
            expected_residual = get_residual(last_token_type, pos_embed, head_embed, tail_embed, 
                                           seq_len, n_heads_or_tails)
            expected_probs = trained_mlp(expected_residual)
            
            # Store results
            results['sequence_lengths'].append(seq_len)
            results['head_counts'].append(n_heads)
            results['tail_counts'].append(n_tails)
            results['last_tokens'].append(last_token_type)
            results['transformer_head_probs'].append(transformer_probs[1].item())
            results['transformer_tail_probs'].append(transformer_probs[2].item())
            results['bayes_head_probs'].append(bayes_probs[1].item())
            results['bayes_tail_probs'].append(bayes_probs[2].item())
            results['errors_head'].append(abs(transformer_probs[1].item() - bayes_probs[1].item()))
            results['errors_tail'].append(abs(transformer_probs[2].item() - bayes_probs[2].item()))
            
            # Print detailed results for first few tests
            if i < 5:
                print(f"Test {i+1}:")
                print(f"  Sequence: {tokens.tolist()}")
                print(f"  Length: {seq_len}, Heads: {n_heads}, Tails: {n_tails}")
                print(f"  Last token: {last_token_type}")
                print(f"  Transformer probs: [BOS: {transformer_probs[0]:.4f}, HEAD: {transformer_probs[1]:.4f}, TAIL: {transformer_probs[2]:.4f}]")
                print(f"  Expected MLP out:  [BOS: {expected_probs[0]:.4f}, HEAD: {expected_probs[1]:.4f}, TAIL: {expected_probs[2]:.4f}]")
                print(f"  Bayes probs:       [BOS: {bayes_probs[0]:.4f}, HEAD: {bayes_probs[1]:.4f}, TAIL: {bayes_probs[2]:.4f}]")
                print(f"  Transformer-Bayes: [BOS: {abs(transformer_probs[0]-bayes_probs[0]):.4f}, HEAD: {results['errors_head'][-1]:.4f}, TAIL: {results['errors_tail'][-1]:.4f}]")
                print(f"  MLP-Bayes:         [BOS: {abs(expected_probs[0]-bayes_probs[0]):.4f}, HEAD: {abs(expected_probs[1]-bayes_probs[1]):.4f}, TAIL: {abs(expected_probs[2]-bayes_probs[2]):.4f}]")
                print()
    
    # Calculate summary statistics
    avg_error_head = sum(results['errors_head']) / len(results['errors_head'])
    avg_error_tail = sum(results['errors_tail']) / len(results['errors_tail'])
    max_error_head = max(results['errors_head'])
    max_error_tail = max(results['errors_tail'])
    
    print(f"Summary Statistics ({n_tests} tests):")
    print(f"  Average error (HEAD): {avg_error_head:.6f}")
    print(f"  Average error (TAIL): {avg_error_tail:.6f}")
    print(f"  Maximum error (HEAD): {max_error_head:.6f}")
    print(f"  Maximum error (TAIL): {max_error_tail:.6f}")
    print(f"  Overall average error: {(avg_error_head + avg_error_tail) / 2:.6f}")
    
    return results

# Run the test
test_results = test_transformer_bayes_updating(transformer, alpha=1.0, beta=1.0, n_tests=20)

# %%

import matplotlib.pyplot as plt
import numpy as np

def plot_bayes_updating_comparison(results):
    """
    Plot comparison between transformer predictions and optimal Bayes updating.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Transformer vs Optimal Bayes Updating', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot of HEAD probabilities
    ax1 = axes[0, 0]
    ax1.scatter(results['bayes_head_probs'], results['transformer_head_probs'], 
               alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('Bayes Optimal P(HEAD)', fontsize=12)
    ax1.set_ylabel('Transformer P(HEAD)', fontsize=12)
    ax1.set_title('HEAD Token Probabilities', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Scatter plot of TAIL probabilities
    ax2 = axes[0, 1]
    ax2.scatter(results['bayes_tail_probs'], results['transformer_tail_probs'], 
               alpha=0.7, s=60, c='green', edgecolors='black', linewidth=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
    ax2.set_xlabel('Bayes Optimal P(TAIL)', fontsize=12)
    ax2.set_ylabel('Transformer P(TAIL)', fontsize=12)
    ax2.set_title('TAIL Token Probabilities', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Error vs sequence length
    ax3 = axes[1, 0]
    head_errors = np.array(results['errors_head'])
    tail_errors = np.array(results['errors_tail'])
    total_errors = head_errors + tail_errors
    
    ax3.scatter(results['sequence_lengths'], head_errors, alpha=0.7, s=60, 
               c='blue', label='HEAD errors', edgecolors='black', linewidth=0.5)
    ax3.scatter(results['sequence_lengths'], tail_errors, alpha=0.7, s=60, 
               c='green', label='TAIL errors', edgecolors='black', linewidth=0.5)
    ax3.scatter(results['sequence_lengths'], total_errors, alpha=0.7, s=60, 
               c='red', label='Total errors', edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Sequence Length', fontsize=12)
    ax3.set_ylabel('Absolute Error', fontsize=12)
    ax3.set_title('Prediction Error vs Sequence Length', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Error distribution histogram
    ax4 = axes[1, 1]
    all_errors = head_errors.tolist() + tail_errors.tolist()
    ax4.hist(all_errors, bins=15, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
    ax4.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error: {np.mean(all_errors):.4f}')
    ax4.set_xlabel('Absolute Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display correlation coefficients
    head_corr = np.corrcoef(results['bayes_head_probs'], results['transformer_head_probs'])[0, 1]
    tail_corr = np.corrcoef(results['bayes_tail_probs'], results['transformer_tail_probs'])[0, 1]
    
    print(f"\nCorrelation Analysis:")
    print(f"  HEAD probability correlation: {head_corr:.6f}")
    print(f"  TAIL probability correlation: {tail_corr:.6f}")
    print(f"  Average correlation: {(head_corr + tail_corr) / 2:.6f}")
    
    if (head_corr > 0.95 and tail_corr > 0.95):
        print("🎉 Excellent! Transformer closely matches optimal Bayes updating.")
    elif (head_corr > 0.9 and tail_corr > 0.9):
        print("✅ Very good! Transformer performs near-optimal Bayes updating.")
    elif (head_corr > 0.8 and tail_corr > 0.8):
        print("👍 Good performance with some deviations from optimal.")
    else:
        print("⚠️  Significant deviations from optimal Bayes updating detected.")

# Generate the plot
plot_bayes_updating_comparison(test_results)

# %%
