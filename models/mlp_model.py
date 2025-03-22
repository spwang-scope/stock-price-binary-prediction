import torch
import torch.nn as nn
from .base_model import BaseTimeSeriesModel

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and dropout.
    Supports different input and output sizes with a projection layer when needed.
    Uses Kaiming initialization for the linear layers.
    """
    def __init__(self, in_size, out_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        # Main processing block
        self.block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Projection shortcut if dimensions don't match
        self.needs_projection = (in_size != out_size)
        if self.needs_projection:
            self.projection = nn.Linear(in_size, out_size)
            nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.zeros_(self.projection.bias)
        
        # Apply Kaiming initialization to linear layer
        nn.init.kaiming_normal_(self.block[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.block[0].bias)
        
    def forward(self, x):
        # Apply block
        block_output = self.block(x)
        
        # Apply projection if needed
        if self.needs_projection:
            return block_output + self.projection(x)
        else:
            return block_output + x


class MLPModel(BaseTimeSeriesModel):
    """
    Multi-Layer Perceptron model with progressive resizing for binary time series classification.
    This model applies pooling to the sequence dimension and then processes
    the resulting features through a series of fully connected layers with decreasing sizes.
    Each layer has half the neurons of the previous layer, creating a funnel architecture.
    All linear layers use Kaiming initialization for better gradient flow with LeakyReLU activations.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, pooling='mean'):
        """
        Initialize MLP model with progressive resizing
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of features in first hidden layer
            num_layers (int): Total number of layers including input and output layers
            dropout (float): Dropout rate applied after each non-output layer
            pooling (str): Method to handle sequence dimension ('mean' or 'last')
        """
        super(MLPModel, self).__init__(input_size)
        
        self.pooling = pooling.lower()
        
        # Validate pooling method
        if self.pooling not in ['mean', 'last']:
            raise ValueError(f"Pooling method '{pooling}' is not supported. Use 'mean' or 'last'.")
        
        # Calculate layer sizes with progressive reduction
        layer_sizes = []
        current_size = hidden_size
        for _ in range(num_layers):
            layer_sizes.append(current_size)
            current_size = max(current_size // 2, 1)  # Halve the size, ensure minimum of 1
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, layer_sizes[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Stack residual blocks with decreasing sizes
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):  # -2 because we have dedicated input and output layers
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            self.residual_blocks.append(ResidualBlock(in_size, out_size, dropout))
        
        # Output layer - use the last calculated layer size
        final_hidden_size = layer_sizes[-2] if num_layers > 1 else layer_sizes[0]
        self.output_layer = nn.Linear(final_hidden_size, 1)
        
        # Apply Kaiming initialization to all linear layers
        self._init_weights()
        
    def _init_weights(self):
        """
        Apply Kaiming initialization to input and output layers.
        ResidualBlocks initialize their own weights.
        """
        # Initialize input layer
        nn.init.kaiming_normal_(self.input_layer[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.input_layer[0].bias)
        
        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """
        Forward pass of the MLP model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size]
        """
        # Handle sequence dimension based on pooling method
        if self.pooling == 'mean':
            # Global average pooling
            x = torch.mean(x, dim=1)  # Shape: [batch_size, input_size]
        else:  # self.pooling == 'last'
            # Use the last time step
            x = x[:, -1, :]  # Shape: [batch_size, input_size]
        
        # Apply input layer
        x = self.input_layer(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply output layer
        x = self.output_layer(x)
        
        return x.squeeze()  # Shape: [batch_size]
