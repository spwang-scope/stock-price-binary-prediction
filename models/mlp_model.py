import torch
import torch.nn as nn
from .base_model import BaseTimeSeriesModel

class MLPModel(BaseTimeSeriesModel):
    """
    Multi-Layer Perceptron model for binary time series classification.
    This model applies pooling to the sequence dimension and then processes
    the resulting features through a series of fully connected layers.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, pooling='mean'):
        """
        Initialize MLP model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of features in hidden layers
            num_layers (int): Total number of linear layers (excluding the output layer)
            dropout (float): Dropout rate applied after each non-output layer
            pooling (str): Method to handle sequence dimension ('mean' or 'last')
        """
        super(MLPModel, self).__init__(input_size)
        
        self.pooling = pooling.lower()
        
        # Validate pooling method
        if self.pooling not in ['mean', 'last']:
            raise ValueError(f"Pooling method '{pooling}' is not supported. Use 'mean' or 'last'.")
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.mlp = nn.Sequential(*layers)
        
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
        
        # Apply MLP layers
        x = self.mlp(x)
        
        return x.squeeze()  # Shape: [batch_size]
