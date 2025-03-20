import torch
import torch.nn as nn
import math

class ScalarRoPEEmbedding(nn.Module):
    """
    Module that converts scalar time steps into RoPE embeddings.
    Can be used as a component in larger neural network architectures.
    """
    def __init__(self, embedding_dim, max_position=10000, base=10000.0):
        """
        Initialize the RoPE embedding module.
        
        Args:
            embedding_dim: Size of the output embedding dimension (must be even)
            max_position: Maximum position that will be encoded
            base: Base value for frequency scaling (default: 10000.0)
        """
        super().__init__()
        
        # Ensure embedding_dim is even
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even for RoPE, got {embedding_dim}")
            
        self.embedding_dim = embedding_dim
        self.max_position = max_position
        
        # Calculate and store frequency bands
        self.register_buffer(
            "freqs",
            1.0 / (base ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        )
        
        # Pre-compute sin and cos values for all possible positions
        # This is an optimization to avoid computing these during forward pass
        positions = torch.arange(0, max_position, dtype=torch.float)
        freqs = torch.outer(positions, self.freqs)
        
        # Create and store cache of sin and cos values
        # Shape: [max_position, embedding_dim/2, 2]
        emb = torch.stack([freqs.sin(), freqs.cos()], dim=-1)
        self.register_buffer("sin_cos_cache", emb)

    def forward(self, positions):
        """
        Convert scalar positions to RoPE embeddings.
        
        Args:
            positions: Tensor of any shape containing position indices
                      Can be single scalar, 1D tensor, or batch of positions
        
        Returns:
            Tensor of shape [..., embedding_dim] with RoPE encodings
                where ... represents the input shape of positions
        """
        # Handle scalar input
        if isinstance(positions, (int, float)):
            positions = torch.tensor([positions], device=self.freqs.device)
        
        # Store original shape to reshape output correctly
        original_shape = positions.shape
        
        # Flatten the input
        flat_positions = positions.view(-1).long()
        
        # Bounds checking
        if torch.any(flat_positions >= self.max_position) or torch.any(flat_positions < 0):
            raise ValueError(f"Position values must be between 0 and {self.max_position-1}")
        
        # Get the cached sin and cos values for these positions
        # Shape: [num_flat_positions, embedding_dim/2, 2]
        sin_cos = self.sin_cos_cache[flat_positions]
        
        # Create output tensor
        embeddings = torch.zeros(
            *original_shape, self.embedding_dim, 
            device=positions.device
        )
        
        # Fill output tensor with sin and cos values
        flat_embeddings = embeddings.view(-1, self.embedding_dim)
        
        for i in range(flat_positions.shape[0]):
            for j in range(0, self.embedding_dim, 2):
                dim_idx = j // 2
                flat_embeddings[i, j] = sin_cos[i, dim_idx, 1]     # cos
                flat_embeddings[i, j+1] = sin_cos[i, dim_idx, 0]   # sin
        
        return embeddings

'''
# Example usage
if __name__ == "__main__":
    # Create the embedding module
    embed_dim = 64
    rope_module = ScalarRoPEEmbedding(embedding_dim=embed_dim)
    
    # Example 1: Single scalar
    time_step = 47
    embedding = rope_module(time_step)
    print(f"Single scalar embedding shape: {embedding.shape}")
    
    # Example 2: Batch of positions
    batch_positions = torch.tensor([1, 5, 47, 100, 500])
    batch_embeddings = rope_module(batch_positions)
    print(f"Batch embedding shape: {batch_embeddings.shape}")
    
    # Example 3: Multi-dimensional input
    matrix_positions = torch.tensor([[1, 2, 3], [4, 5, 6]])
    matrix_embeddings = rope_module(matrix_positions)
    print(f"Matrix embedding shape: {matrix_embeddings.shape}")
    
    # Example 4: Integration with a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.time_embedding = ScalarRoPEEmbedding(embedding_dim=hidden_dim)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, 1)
            
        def forward(self, x, time_steps):
            """
            Args:
                x: Input features [batch_size, input_dim]
                time_steps: Time positions [batch_size]
            """
            # Project input features
            x_proj = self.input_proj(x)
            
            # Get time embeddings
            time_emb = self.time_embedding(time_steps)
            
            # Combine feature projection with time embedding
            # A simple way is to add them, but you could also concatenate, multiply, etc.
            combined = x_proj + time_emb
            
            # Apply output projection
            return self.output_layer(combined)
    
    # Create example model and inputs
    model = SimpleModel(input_dim=32, hidden_dim=embed_dim)
    features = torch.randn(8, 32)  # [batch_size, feature_dim]
    times = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])  # [batch_size]
    
    # Run model
    output = model(features, times)
    print(f"Model output shape: {output.shape}")
'''