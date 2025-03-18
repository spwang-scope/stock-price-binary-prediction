import torch
import torch.nn as nn
from .base_model import BaseTimeSeriesModel

class LSTMModel(BaseTimeSeriesModel):
    """
    LSTM-based model for binary time series classification
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of the LSTM model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_size]
        
        # Take only the last time step's output
        last_time_step = lstm_out[:, -1, :]  # shape: [batch_size, hidden_size]
        
        # Apply dropout and feed into fully connected layer
        out = self.dropout(last_time_step)
        out = self.fc(out)  # shape: [batch_size, 1]
        
        return out.squeeze()  # shape: [batch_size]
