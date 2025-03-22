import torch
import torch.nn as nn
import math

class ScalarRoPEEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for scalar inputs.
    
    This module takes a scalar position (like day count) and produces
    a fixed-size embedding vector using RoPE principles.
    
    Args:
        embedding_dim (int): Size of the embedding vector. Must be even.
        base (float, optional): Base value for frequency calculation. Default: 10000.0
    """
    def __init__(self, embedding_dim=4, base=10000.0):
        super(ScalarRoPEEmbedding, self).__init__()
        
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")
            
        self.embedding_dim = embedding_dim
        self.base = base
        
        # Precompute frequency bands
        self.freqs = self._precompute_freqs()
        
    def _precompute_freqs(self):
        # Calculate frequency bands
        dim = self.embedding_dim // 2
        freqs = torch.exp(
            -torch.arange(0, dim, dtype=torch.float) * (math.log(self.base) / dim)
        )
        return freqs
        
    def forward(self, positions):
        """
        Calculate RoPE embeddings for scalar position values.
        
        Args:
            positions (torch.Tensor): Scalar position values (e.g., day counts)
                                     Shape: (batch_size,)
                                     
        Returns:
            torch.Tensor: RoPE embeddings with shape (batch_size, embedding_dim)
        """
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float)
            
        # Ensure positions has the right shape
        if positions.dim() == 0:  # If it's a single scalar
            positions = positions.unsqueeze(0)
            
        batch_size = positions.shape[0]
        device = positions.device
        
        # Move freqs to the correct device
        freqs = self.freqs.to(device)
        
        # Calculate embeddings
        t = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (batch_size, dim//2)
        
        # Calculate sin and cos
        sin_embeds = torch.sin(t)  # (batch_size, dim//2)
        cos_embeds = torch.cos(t)  # (batch_size, dim//2)
        
        # Interleave sin and cos to get final embeddings
        embeddings = torch.zeros((batch_size, self.embedding_dim), device=device)
        embeddings[:, 0::2] = cos_embeds
        embeddings[:, 1::2] = sin_embeds
        
        return embeddings
    
    def get_numpy_embeddings(self, positions):
        """
        Utility method to get numpy array embeddings for given positions.
        Useful for non-PyTorch applications like pandas.
        
        Args:
            positions: List or array of scalar position values
            
        Returns:
            numpy.ndarray: Array of RoPE embeddings with shape (len(positions), embedding_dim)
        """
        with torch.no_grad():
            embeddings = self.forward(torch.tensor(positions, dtype=torch.float))
            return embeddings.numpy()
