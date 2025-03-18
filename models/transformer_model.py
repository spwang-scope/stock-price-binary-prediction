import torch
import torch.nn as nn
import math
from .base_model import BaseTimeSeriesModel

class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for Transformer models.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(BaseTimeSeriesModel):
    """
    Transformer-based model for binary time series classification.
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2, 
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1):
        """
        Initialize Transformer model
        
        Args:
            input_size (int): Number of input features
            d_model (int): Dimensionality of the model
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            dim_feedforward (int): Dimensionality of feedforward network
            dropout (float): Dropout rate
        """
        super(TransformerModel, self).__init__(input_size)
        
        # Feature projection layer to transform raw input to transformer dimensions
        self.feature_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Create a transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
        
        # Initialize a learnable target token for the decoder
        self.target_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Initialize parameters using Xavier uniform initialization
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize the parameters using Xavier uniform initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence to prevent attending to future positions
        
        Args:
            sz: sequence length
        
        Returns:
            mask: mask tensor of shape [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src):
        """
        Forward pass of the transformer model
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size]
        """
        batch_size = src.size(0)
        
        # Project features to d_model dimensions
        src = self.feature_projection(src)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        
        # Create target token for decoder (batch_size, 1, d_model)
        tgt = self.target_token.expand(batch_size, 1, -1)
        
        # Generate masks
        src_mask = None  # Allow attending to all positions in the encoder
        tgt_mask = self._generate_square_subsequent_mask(1).to(src.device)  # Only one token in decoder
        
        # Forward pass through transformer
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # Extract the single token from decoder output
        output = output[:, 0, :]  # Shape: [batch_size, d_model]
        
        # Project to output dimension
        output = self.output_projection(output)  # Shape: [batch_size, 1]
        
        return output.squeeze(-1)  # Shape: [batch_size]
