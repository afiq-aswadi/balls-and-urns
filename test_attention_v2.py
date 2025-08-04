#%%
import torch

embedding_dim = 128

def random_embedding(dim):
    """Generate a random unit embedding vector of given dimension."""
    vec = torch.randn(dim)
    return vec / torch.norm(vec)

pos_embed = random_embedding(embedding_dim)
head_embed = random_embedding(embedding_dim)
tail_embed = random_embedding(embedding_dim)
bos_embed = random_embedding(embedding_dim)

def get_residual(last_token, pos_embed, head_embed, tail_embed, sequence_length, n_heads_or_tails):
    """Compute the residual vector for the last token in the sequence."""
    if last_token == "head":
        return pos_embed * sequence_length + head_embed * n_heads_or_tails
    else:
        return pos_embed * sequence_length + tail_embed * n_heads_or_tails

def token_embedding(tokens):
    """Embed tokens [0, 1, 2] to [bos, head, tail] respectively using a linear map."""
    embedding_matrix = torch.stack([bos_embed, head_embed, tail_embed], dim=0)
    return torch.nn.functional.embedding(tokens, embedding_matrix)

class CountingAttentionHead(torch.nn.Module):
    def __init__(self, embedding_dim, pos_embed):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_embed = pos_embed / torch.norm(pos_embed)
        
        # Initialize weight matrices with Xavier initialization
        self.W_Q = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.W_K = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.W_V = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.W_O = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with null space constraint built in."""
        with torch.no_grad():
            # Create a basis for the subspace orthogonal to pos_embed
            pos_embed_normalized = self.pos_embed
            
            # Generate random vectors and orthogonalize them against pos_embed
            for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                # Initialize with random values
                torch.nn.init.xavier_uniform_(W, gain=0.1)
                
                # Project out pos_embed direction from each row
                for i in range(W.shape[0]):
                    W[i] = W[i] - torch.dot(W[i], pos_embed_normalized) * pos_embed_normalized
    
    def apply_nullspace_constraint(self):
        """Ensure that pos_embed is in the null space of all weight matrices."""
        with torch.no_grad():
            pos_embed_normalized = self.pos_embed
            for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                # Project out pos_embed direction from each row
                for i in range(W.shape[0]):
                    W[i] = W[i] - torch.dot(W[i], pos_embed_normalized) * pos_embed_normalized
    
    def forward(self, embeddings):
        """Apply attention mechanism to compute residual for the last token."""
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
    """Generate a random sequence with positional encoding."""
    tokens = torch.zeros(seq_len, dtype=torch.long)
    tokens[0] = 0  # BOS token
    tokens[1:] = torch.randint(1, 3, (seq_len-1,))  # HEAD or TAIL tokens
    
    token_embeds = token_embedding(tokens)
    
    # Add positional encoding
    position_indices = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    positional_encoding = position_indices * pos_embed.unsqueeze(0)
    embeddings = token_embeds + positional_encoding
    
    # Count heads and tails (excluding BOS)
    n_heads = torch.sum(tokens[1:] == 1).item()
    n_tails = torch.sum(tokens[1:] == 2).item()
    
    last_token_type = "head" if tokens[-1].item() == 1 else "tail"
    
    return tokens, embeddings, last_token_type, n_heads, n_tails

def generate_attention_training_data(n_samples, min_seq_len=5, max_seq_len=100):
    """Generate training data for the attention head."""
    input_embeddings = []
    target_residuals = []
    
    for _ in range(n_samples):
        seq_len = torch.randint(min_seq_len, max_seq_len + 1, (1,)).item()
        tokens, embeddings, last_token_type, n_heads, n_tails = generate_sequence_with_positional_encoding(seq_len, pos_embed)
        
        if last_token_type == "head":
            n_heads_or_tails = n_heads
        else:
            n_heads_or_tails = n_tails
            
        target_residual = get_residual(last_token_type, pos_embed, head_embed, tail_embed, seq_len, n_heads_or_tails)
        
        input_embeddings.append(embeddings)
        target_residuals.append(target_residual)
    
    return input_embeddings, target_residuals

def train_attention_head(attention_head, n_epochs=2000, batch_size=16):
    """Train the attention head to output the correct residual."""
    optimizer = torch.optim.Adam(attention_head.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=150)
    attention_head.train()
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 400
    
    for epoch in range(n_epochs):
        input_embeddings, target_residuals = generate_attention_training_data(batch_size, min_seq_len=3, max_seq_len=50)
        
        total_loss = 0
        optimizer.zero_grad()
        
        for embeddings, target_residual in zip(input_embeddings, target_residuals):
            pred_residual = attention_head(embeddings)
            
            # Compute loss
            mse_loss = torch.nn.functional.mse_loss(pred_residual, target_residual)
            
            # Add small regularization
            reg_loss = 0.0001 * sum(torch.norm(W)**2 for W in [attention_head.W_Q, attention_head.W_K, attention_head.W_V, attention_head.W_O])
            
            loss = mse_loss + reg_loss
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(attention_head.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Apply null space constraint more strictly
        if epoch % 5 == 0:
            attention_head.apply_nullspace_constraint()
        
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
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return attention_head

def test_attention_head(attention_head, n_test_samples=10):
    """Test the trained attention head."""
    attention_head.eval()
    
    print("\nTesting trained attention head:")
    print("=" * 60)
    
    with torch.no_grad():
        input_embeddings, target_residuals = generate_attention_training_data(n_test_samples, 10, 20)
        
        total_error = 0
        total_cosine_sim = 0
        
        for i, (embeddings, target_residual) in enumerate(zip(input_embeddings, target_residuals)):
            pred_residual = attention_head(embeddings)
            
            error = torch.nn.functional.mse_loss(pred_residual, target_residual).item()
            cosine_sim = torch.nn.functional.cosine_similarity(pred_residual, target_residual, dim=0).item()
            
            total_error += error
            total_cosine_sim += cosine_sim
            
            seq_len = embeddings.shape[0]
            
            print(f"Test {i+1}:")
            print(f"  Sequence length: {seq_len}")
            print(f"  Target residual norm: {torch.norm(target_residual).item():.4f}")
            print(f"  Predicted residual norm: {torch.norm(pred_residual).item():.4f}")
            print(f"  MSE Error: {error:.6f}")
            print(f"  Cosine similarity: {cosine_sim:.4f}")
            print()
        
        print(f"Average MSE Error: {total_error / n_test_samples:.6f}")
        print(f"Average Cosine Similarity: {total_cosine_sim / n_test_samples:.4f}")

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

#%%
# Test on a specific example to see attention weights
print("\nAnalyzing attention mechanism on a specific example:")
with torch.no_grad():
    # Generate a test sequence
    tokens, embeddings, last_token_type, n_heads, n_tails = generate_sequence_with_positional_encoding(10, pos_embed)
    
    print(f"Sequence tokens: {tokens.tolist()}")
    print(f"Last token type: {last_token_type}")
    print(f"Heads count: {n_heads}, Tails count: {n_tails}")
    
    # Forward pass with attention weights
    Q = embeddings @ trained_attention_head.W_Q
    K = embeddings @ trained_attention_head.W_K
    V = embeddings @ trained_attention_head.W_V
    
    q_last = Q[-1]
    scale = 1.0 / (embedding_dim ** 0.5)
    scores = torch.matmul(q_last, K.T) * scale
    attn_weights = torch.nn.functional.softmax(scores, dim=0)
    
    print(f"Attention weights: {attn_weights.numpy()}")
    print(f"Sum of attention weights: {attn_weights.sum().item():.6f}")
    
    # Check if attention focuses on relevant tokens
    if last_token_type == "head":
        head_positions = torch.where(tokens == 1)[0]
        if len(head_positions) > 0:
            head_attention = attn_weights[head_positions].sum().item()
            print(f"Total attention on HEAD tokens: {head_attention:.4f}")
    else:
        tail_positions = torch.where(tokens == 2)[0]
        if len(tail_positions) > 0:
            tail_attention = attn_weights[tail_positions].sum().item()
            print(f"Total attention on TAIL tokens: {tail_attention:.4f}")
