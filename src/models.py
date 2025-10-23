"""
Neural network models for Deep Hedging policies.

Implements various policy architectures including MLPs and transformers
for learning optimal hedging strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Tuple


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy network for hedging.
    
    Maps state features to target position in the underlying asset.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "gelu",
                 use_layer_norm: bool = True,
                 dropout: float = 0.0,
                 q_max: float = 1.5):
        """
        Initialize MLP policy network.
        
        Args:
            input_dim: Dimension of state features
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("gelu", "relu", "tanh")
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout rate
            q_max: Maximum position size
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.q_max = q_max
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: State features of shape (batch_size, input_dim)
            
        Returns:
            Target position of shape (batch_size, 1)
        """
        output = self.network(x)
        
        # Apply tanh activation and scale to q_max
        output = torch.tanh(output) * self.q_max
        
        return output


class TransformerPolicy(nn.Module):
    """
    Transformer-based policy network for sequential hedging decisions.
    
    Processes sequences of state features to make hedging decisions.
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 q_max: float = 1.5):
        """
        Initialize transformer policy network.
        
        Args:
            input_dim: Dimension of state features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            q_max: Maximum position size
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.q_max = q_max
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: State features of shape (batch_size, seq_len, input_dim)
            mask: Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Target positions of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to output
        output = self.output_projection(x)  # (batch_size, seq_len, 1)
        
        # Apply tanh activation and scale
        output = torch.tanh(output) * self.q_max
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class EnsemblePolicy(nn.Module):
    """
    Ensemble of policy networks for improved robustness.
    
    Combines multiple policy networks to make more robust hedging decisions.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "gelu",
                 use_layer_norm: bool = True,
                 dropout: float = 0.0,
                 q_max: float = 1.5,
                 n_models: int = 5):
        """
        Initialize ensemble policy network.
        
        Args:
            input_dim: Dimension of state features
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout rate
            q_max: Maximum position size
            n_models: Number of models in ensemble
        """
        super().__init__()
        
        self.n_models = n_models
        self.q_max = q_max
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            MLPPolicy(input_dim, hidden_dims, activation, use_layer_norm, dropout, q_max)
            for _ in range(n_models)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: State features of shape (batch_size, input_dim)
            
        Returns:
            Average target position of shape (batch_size, 1)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions, dim=0), dim=0)
        
        return ensemble_pred
    
    def forward_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: State features of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (mean_prediction, prediction_std)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_models, batch_size, 1)
        
        # Compute mean and standard deviation
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred


class ResidualMLP(nn.Module):
    """
    Residual MLP policy network with skip connections.
    
    Implements residual connections for improved gradient flow.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "gelu",
                 use_layer_norm: bool = True,
                 dropout: float = 0.0,
                 q_max: float = 1.5):
        """
        Initialize residual MLP policy network.
        
        Args:
            input_dim: Dimension of state features
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout rate
            q_max: Maximum position size
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.q_max = q_max
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = ResidualBlock(
                hidden_dims[i], 
                hidden_dims[i+1], 
                activation, 
                use_layer_norm, 
                dropout
            )
            self.residual_blocks.append(block)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual network.
        
        Args:
            x: State features of shape (batch_size, input_dim)
            
        Returns:
            Target position of shape (batch_size, 1)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output layer
        output = self.output_layer(x)
        
        # Apply tanh activation and scale
        output = torch.tanh(output) * self.q_max
        
        return output


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "gelu",
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection projection if dimensions don't match
        self.skip_projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.skip_projection(x)
        
        x = self.norm1(self.linear1(x))
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.norm2(self.linear2(x))
        x = self.activation(x)
        x = self.dropout(x)
        
        return x + residual


# Factory function for creating policy networks
def create_policy(policy_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Create a policy network based on type.
    
    Args:
        policy_type: Type of policy network
        input_dim: Dimension of input features
        **kwargs: Additional arguments
        
    Returns:
        Policy network
    """
    if policy_type.lower() == "mlp":
        return MLPPolicy(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [256, 256]),
            activation=kwargs.get('activation', 'gelu'),
            use_layer_norm=kwargs.get('use_layer_norm', True),
            dropout=kwargs.get('dropout', 0.0),
            q_max=kwargs.get('q_max', 1.5)
        )
    elif policy_type.lower() == "transformer":
        return TransformerPolicy(
            input_dim=input_dim,
            d_model=kwargs.get('d_model', 256),
            n_heads=kwargs.get('n_heads', 8),
            n_layers=kwargs.get('n_layers', 4),
            dropout=kwargs.get('dropout', 0.1),
            q_max=kwargs.get('q_max', 1.5)
        )
    elif policy_type.lower() == "ensemble":
        return EnsemblePolicy(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [256, 256]),
            activation=kwargs.get('activation', 'gelu'),
            use_layer_norm=kwargs.get('use_layer_norm', True),
            dropout=kwargs.get('dropout', 0.0),
            q_max=kwargs.get('q_max', 1.5),
            n_models=kwargs.get('n_models', 5)
        )
    elif policy_type.lower() == "residual":
        return ResidualMLP(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [256, 256]),
            activation=kwargs.get('activation', 'gelu'),
            use_layer_norm=kwargs.get('use_layer_norm', True),
            dropout=kwargs.get('dropout', 0.0),
            q_max=kwargs.get('q_max', 1.5)
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
