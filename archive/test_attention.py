#%%
import torch

embedding_dim = 128

def random_embedding(dim):
    """
    Generate a random unit embedding vector of given dimension.
    """
    vec = torch.randn(dim)
    return vec / torch.norm(vec)  # normalize to unit vector

pos_embed = random_embedding(embedding_dim)
head_embed = random_embedding(embedding_dim)
tail_embed = random_embedding(embedding_dim)
bos_embed = random_embedding(embedding_dim)

def get_residual(last_token, pos_embed, head_embed, tail_embed, sequence_length, n_heads_or_tails):
    """
    Compute the residual vector for the last token in the sequence.
    """
    if last_token == "head":
        return pos_embed * sequence_length + head_embed * n_heads_or_tails
    else:
        return pos_embed * sequence_length + tail_embed * n_heads_or_tails

def token_embedding(tokens):
    """
    Embed tokens [0, 1, 2] to [bos, head, tail] respectively using a linear map.
    """
    embedding_matrix = torch.stack([bos_embed, head_embed, tail_embed], dim=0)
    return torch.nn.functional.embedding(tokens, embedding_matrix)

class CountingAttentionHead(torch.nn.Module):
    def __init__(self, embedding_dim, pos_embed):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_embed = pos_embed / torch.norm(pos_embed)  # Normalize pos_embed
        
        # Initialize weight matrices with smaller scale
        self.W_Q = torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.02)
        self.W_K = torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.02)
        self.W_V = torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.02)
        self.W_O = torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.02)
        
        # Apply initial null space constraint
        self.apply_nullspace_constraint()
        
    def apply_nullspace_constraint(self):
        """
        Ensure that pos_embed is in the null space of W_Q, W_K, W_V, W_O.
        """
        with torch.no_grad():
            for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                projection = torch.outer(W @ self.pos_embed, self.pos_embed)
                W.data = W.data - projection
    
    def forward(self, embeddings):
        """
        Apply attention mechanism to compute residual for the last token.
        """
        seq_len = embeddings.shape[0]
        
        # Compute Q, K, V
        Q = embeddings @ self.W_Q
        K = embeddings @ self.W_K
        V = embeddings @ self.W_V
        
        # Focus on the last token's query
        q_last = Q[-1]
        
        # Compute attention scores with scaling
        scale = 1.0 / (self.embedding_dim ** 0.5)
        scores = torch.matmul(q_last, K.T) * scale
        
        # Apply attention weights (softmax)
        attn_weights = torch.nn.functional.softmax(scores, dim=0)
        
        # Compute weighted value
        attended_value = torch.sum(attn_weights.unsqueeze(1) * V, dim=0)
        
        # Apply output projection
        output = attended_value @ self.W_O
        
        return output

def generate_sequence_with_positional_encoding(seq_len, pos_embed):
    """
    Generate a random sequence with positional encoding.
    """
    # Generate tokens: first token is BOS (0), rest are randomly HEAD (1) or TAIL (2)
    tokens = torch.zeros(seq_len, dtype=torch.long)
    tokens[0] = 0  # BOS token
    tokens[1:] = torch.randint(1, 3, (seq_len-1,))  # HEAD or TAIL tokens
    
    # Get token embeddings
    token_embeds = token_embedding(tokens)
    
    # Add positional encoding (same for all positions, scaled by position index)
    position_indices = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    positional_encoding = position_indices * pos_embed.unsqueeze(0)
    
    # Combine token embeddings and positional encoding
    embeddings = token_embeds + positional_encoding
    
    # Count heads and tails (excluding BOS)
    n_heads = torch.sum(tokens[1:] == 1).item()
    n_tails = torch.sum(tokens[1:] == 2).item()
    
    # Determine last token type
    last_token_type = "head" if tokens[-1].item() == 1 else "tail"
    
    return tokens, embeddings, last_token_type, n_heads, n_tails

def generate_attention_training_data(n_samples, min_seq_len=5, max_seq_len=100):
    """
    Generate training data for the attention head.
    """
    input_embeddings = []
    target_residuals = []
    
    for _ in range(n_samples):
        # Random sequence length
        seq_len = torch.randint(min_seq_len, max_seq_len + 1, (1,)).item()
        
        # Generate sequence
        tokens, embeddings, last_token_type, n_heads, n_tails = generate_sequence_with_positional_encoding(seq_len, pos_embed)
        
        # Compute target residual based on last token type and counts
        if last_token_type == "head":
            n_heads_or_tails = n_heads
        else:
            n_heads_or_tails = n_tails
            
        target_residual = get_residual(last_token_type, pos_embed, head_embed, tail_embed, seq_len, n_heads_or_tails)
        
        input_embeddings.append(embeddings)
        target_residuals.append(target_residual)
    
    return input_embeddings, target_residuals

def train_attention_head(attention_head, n_epochs=3000, batch_size=16):
    """
    Train the attention head to output the correct residual.
    """
    optimizer = torch.optim.Adam(attention_head.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200)
    attention_head.train()
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 500
    
    for epoch in range(n_epochs):
        # Generate training batch
        input_embeddings, target_residuals = generate_attention_training_data(batch_size, min_seq_len=3, max_seq_len=50)
        
        total_loss = 0
        optimizer.zero_grad()
        
        for embeddings, target_residual in zip(input_embeddings, target_residuals):
            # Forward pass
            pred_residual = attention_head(embeddings)
            
            # Compute loss (MSE + small regularization on attention weights)
            mse_loss = torch.nn.functional.mse_loss(pred_residual, target_residual)
            
            # Add small regularization to prevent collapse
            reg_loss = 0.001 * torch.norm(pred_residual)
            
            loss = mse_loss + reg_loss
            total_loss += loss
        
        # Average loss over batch
        avg_loss = total_loss / batch_size
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(attention_head.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Reapply null space constraint after each update (but less frequently)
        if epoch % 10 == 0:
            attention_head.apply_nullspace_constraint()
        
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
        if epoch % 300 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return attention_head

def test_attention_head(attention_head, n_test_samples=10):
    """
    Test the trained attention head.
    """
    attention_head.eval()
    
    print("\nTesting trained attention head:")
    print("=" * 60)
    
    with torch.no_grad():
        input_embeddings, target_residuals = generate_attention_training_data(n_test_samples, 10, 20)
        
        for i, (embeddings, target_residual) in enumerate(zip(input_embeddings, target_residuals)):
            # Get prediction
            pred_residual = attention_head(embeddings)
            
            # Calculate error
            error = torch.nn.functional.mse_loss(pred_residual, target_residual).item()
            
            # Extract sequence info for display
            seq_len = embeddings.shape[0]
            
            print(f"Test {i+1}:")
            print(f"  Sequence length: {seq_len}")
            print(f"  Target residual norm: {torch.norm(target_residual).item():.4f}")
            print(f"  Predicted residual norm: {torch.norm(pred_residual).item():.4f}")
            print(f"  MSE Error: {error:.6f}")
            print(f"  Cosine similarity: {torch.nn.functional.cosine_similarity(pred_residual, target_residual, dim=0).item():.4f}")
            print()

# Initialize and train the attention head
print("Initializing attention head...")
attention_head = CountingAttentionHead(embedding_dim, pos_embed)

print("Starting attention head training...")
trained_attention_head = train_attention_head(attention_head, n_epochs=2000, batch_size=16)
print("Attention head training completed!")

# Test the trained attention head
test_attention_head(trained_attention_head)

#%%
# Verify the null space constraint is being maintained
print("\nVerifying null space constraint:")
with torch.no_grad():
    for name, W in [("W_Q", trained_attention_head.W_Q), 
                    ("W_K", trained_attention_head.W_K), 
                    ("W_V", trained_attention_head.W_V), 
                    ("W_O", trained_attention_head.W_O)]:
        result = W @ trained_attention_head.pos_embed
        norm = torch.norm(result).item()
        print(f"{name} @ pos_embed norm: {norm:.8f}")
