import torch
import torch.nn as nn
from .base_model import BaseTimeSeriesModel

class FiLMModule(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) module.
    
    Applies gamma * features + beta (element-wise) using externally provided
    gamma and beta parameters.
    """
    def __init__(self, target_size):
        """
        Initialize FiLM module
        
        Args:
            target_size (int): Size of the features to be modulated
        """
        super(FiLMModule, self).__init__()
        self.target_size = target_size
        
    def forward(self, target, gamma, beta):
        """
        Apply FiLM conditioning with external gamma and beta parameters
        
        Args:
            target: Tensor to be modulated [batch_size, target_size]
            gamma: Scaling parameters [batch_size, target_size]
            beta: Shifting parameters [batch_size, target_size]
            
        Returns:
            Modulated tensor [batch_size, target_size]
        """
        # Apply FiLM: gamma * target + beta
        return gamma * target + beta


class FiLMResidualBlock(nn.Module):
    """
    Residual block with FiLM modulation, batch normalization, and dropout.
    Supports different input and output sizes with a projection layer when needed.
    """
    def __init__(self, in_size, out_size, dropout=0.2):
        super(FiLMResidualBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        # Main processing block
        self.block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # FiLM module (without its own parameter generation)
        self.film = FiLMModule(out_size)
        
        # Projection shortcut if dimensions don't match
        self.needs_projection = (in_size != out_size)
        if self.needs_projection:
            self.projection = nn.Linear(in_size, out_size)
            nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.zeros_(self.projection.bias)
        
        # Apply Kaiming initialization to linear layer
        nn.init.kaiming_normal_(self.block[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.block[0].bias)
        
    def forward(self, x, gamma, beta):
        """
        Forward pass with FiLM modulation
        
        Args:
            x: Input tensor [batch_size, in_size]
            gamma: Scaling parameters [batch_size, out_size]
            beta: Shifting parameters [batch_size, out_size]
            
        Returns:
            Modulated tensor [batch_size, out_size]
        """
        # Apply block
        block_output = self.block(x)
        
        # Apply FiLM modulation
        film_output = self.film(block_output, gamma, beta)
        
        # Apply projection if needed and add residual connection
        if self.needs_projection:
            return film_output + self.projection(x)
        else:
            return film_output + x


class SharedFiLMGenerator(nn.Module):
    """
    Shared network that generates FiLM parameters (gamma and beta)
    for all layers in the model.
    """
    def __init__(self, condition_size, layer_sizes):
        """
        Initialize shared FiLM parameter generator
        
        Args:
            condition_size (int): Size of the conditioning input
            layer_sizes (list): List of sizes for each layer to be modulated
        """
        super(SharedFiLMGenerator, self).__init__()
        
        self.layer_sizes = layer_sizes
        
        # Calculate total number of parameters needed (gamma and beta for each layer size)
        total_params = sum(size * 2 for size in layer_sizes)
        
        # Hidden layer for parameter generation
        hidden_size = max(condition_size * 2, 64)
        
        # Create parameter generation network
        self.condition_network = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, total_params)
        )
        
        # Initialize with Kaiming weights
        nn.init.kaiming_normal_(self.condition_network[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.condition_network[0].bias)
        nn.init.kaiming_normal_(self.condition_network[2].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.condition_network[2].bias)
        
    def forward(self, condition):
        """
        Generate FiLM parameters for all layers
        
        Args:
            condition: Tensor containing conditioning features [batch_size, condition_size]
            
        Returns:
            List of (gamma, beta) pairs for each layer
        """
        # Generate all parameters in one go
        all_params = self.condition_network(condition)  # [batch_size, total_params]
        
        # Split parameters for each layer
        param_pairs = []
        start_idx = 0
        
        for size in self.layer_sizes:
            # Calculate indices for this layer's parameters
            params_count = size * 2
            end_idx = start_idx + params_count
            
            # Extract parameters for this layer
            layer_params = all_params[:, start_idx:end_idx]
            
            # Split into gamma and beta
            gamma, beta = torch.chunk(layer_params, 2, dim=1)
            
            # Add to result list
            param_pairs.append((gamma, beta))
            
            # Update start index for next layer
            start_idx = end_idx
            
        return param_pairs


class FiLMMLPModel(BaseTimeSeriesModel):
    """
    MLP model with Feature-wise Linear Modulation (FiLM) for conditional computation.
    
    This model extends the MLPModel by adding FiLM modules that use the last 
    rope_embedding_dim features to modulate the outputs of each linear layer.
    All layers share the same network for generating FiLM parameters.
    """
    def __init__(self, input_size, rope_embedding_dim, hidden_size=64, num_layers=2, 
                 dropout=0.2, pooling='mean'):
        """
        Initialize FiLM-enhanced MLP model
        
        Args:
            input_size (int): Number of input features
            rope_embedding_dim (int): Number of features at the end to use as FiLM conditioning
            hidden_size (int): Number of features in first hidden layer
            num_layers (int): Total number of layers including input and output layers
            dropout (float): Dropout rate applied after each non-output layer
            pooling (str): Method to handle sequence dimension ('mean' or 'last')
        """
        super(FiLMMLPModel, self).__init__(input_size)
        
        self.pooling = pooling.lower()
        self.rope_embedding_dim = rope_embedding_dim
        self.feature_dim = input_size - rope_embedding_dim
        
        # Validate pooling method
        if self.pooling not in ['mean', 'last']:
            raise ValueError(f"Pooling method '{pooling}' is not supported. Use 'mean' or 'last'.")
            
        # Calculate layer sizes with progressive reduction
        layer_sizes = []
        current_size = hidden_size
        for _ in range(num_layers):
            layer_sizes.append(current_size)
            current_size = max(current_size // 2, 1)  # Halve the size, ensure minimum of 1
        
        # Input layer for regular features
        self.input_layer = nn.Sequential(
            nn.Linear(self.feature_dim, layer_sizes[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # FiLM module for input layer
        self.input_film = FiLMModule(layer_sizes[0])
        
        # Stack FiLM residual blocks with decreasing sizes
        self.residual_blocks = nn.ModuleList()
        
        # Create residual blocks
        block_layer_sizes = []  # Track layer sizes for the shared FiLM generator
        block_layer_sizes.append(layer_sizes[0])  # First for input layer
        
        for i in range(num_layers - 2):  # -2 because we have dedicated input and output layers
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            self.residual_blocks.append(
                FiLMResidualBlock(in_size, out_size, dropout)
            )
            block_layer_sizes.append(out_size)
        
        # Create shared FiLM parameter generator for all layers
        self.film_generator = SharedFiLMGenerator(
            rope_embedding_dim, 
            block_layer_sizes
        )
        
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
        Forward pass of the FiLM-enhanced MLP model
        
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
        
        # Split input into features and conditioning (rope embedding)
        features = x[:, :self.feature_dim]
        condition = x[:, self.feature_dim:]
        
        # Generate all FiLM parameters with shared generator
        film_params = self.film_generator(condition)
        
        # Apply input layer
        x = self.input_layer(features)
        
        # Apply FiLM modulation to input layer output
        input_gamma, input_beta = film_params[0]
        x = self.input_film(x, input_gamma, input_beta)
        
        # Apply FiLM residual blocks
        for i, block in enumerate(self.residual_blocks):
            # Get parameters for this block
            gamma, beta = film_params[i+1]  # +1 because input layer used index 0
            x = block(x, gamma, beta)
        
        # Apply output layer
        x = self.output_layer(x)
        
        return x.squeeze()  # Shape: [batch_size]
