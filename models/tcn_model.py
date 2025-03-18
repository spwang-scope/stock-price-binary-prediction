import torch
import torch.nn as nn
from .base_model import BaseTimeSeriesModel

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated causal convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # Calculate padding to maintain sequence length (causal padding)
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Apply causal convolution
        out = self.conv1(x)
        
        # Apply ReLU and dropout
        out = self.relu(out)
        out = self.dropout(out)
        
        return out

class TCNModel(BaseTimeSeriesModel):
    """
    Temporal Convolutional Network for binary time series classification
    """
    def __init__(self, input_size, hidden_size=64, num_layers=3, kernel_size=3, dropout=0.2):
        """
        Initialize TCN model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of channels in TCN blocks
            num_layers (int): Number of TCN blocks
            kernel_size (int): Kernel size for convolutions
            dropout (float): Dropout rate
        """
        super(TCNModel, self).__init__(input_size)
        
        layers = []
        
        # Input layer
        layers.append(TCNBlock(input_size, hidden_size, kernel_size, dilation=1, dropout=dropout))
        
        # Hidden layers with increasing dilation
        for i in range(1, num_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(hidden_size, hidden_size, kernel_size, dilation, dropout))
        
        self.tcn_blocks = nn.Sequential(*layers)
        
        # Global pooling and classification layer
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass of the TCN model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size]
        """
        # Transpose for 1D convolution: [batch_size, input_size, seq_len]
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        out = self.tcn_blocks(x)
        
        # Global pooling
        out = self.global_pool(out).squeeze(-1)
        
        # Classification
        out = self.fc(out)
        
        return out.squeeze()
