import torch
import torch.nn as nn

class BaseTimeSeriesModel(nn.Module):
    """
    Base class for time series classification models.
    All models should inherit from this class and implement the forward method.
    """
    def __init__(self, input_size, **kwargs):
        super(BaseTimeSeriesModel, self).__init__()
        self.input_size = input_size
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size]
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def save(self, path):
        """Save model to path"""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load model from path"""
        self.load_state_dict(torch.load(path))
