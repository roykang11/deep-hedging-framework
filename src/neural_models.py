#!/usr/bin/env python3
"""
Advanced Neural Network Models for Deep Hedging
Implements sophisticated architectures: LSTM, Transformer, Attention-based models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AdvancedLSTMHedger(nn.Module):
    """Advanced LSTM-based hedging model with attention mechanism."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2, 
                 attention_dim=64, output_dim=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, 
                                             dropout=dropout, batch_first=True)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Position limits [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        lstm_out = lstm_out + attn_out
        
        # Output projection
        output = self.output_layers(lstm_out)
        
        return output, hidden

class TransformerHedger(nn.Module):
    """Transformer-based hedging model with sophisticated attention."""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer forward pass
        transformer_out = self.transformer(x, src_key_padding_mask=mask)
        
        # Output projection
        output = self.output_layers(transformer_out)
        
        return output

class AttentionHedger(nn.Module):
    """Attention-based hedging model with multi-scale temporal features."""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, 
                 dropout=0.1, temporal_scales=[1, 5, 10, 20]):
        super().__init__()
        
        self.temporal_scales = temporal_scales
        self.d_model = d_model
        
        # Multi-scale feature extraction
        self.scale_encoders = nn.ModuleList([
            nn.Linear(input_dim, d_model) for _ in temporal_scales
        ])
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Cross-scale attention
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale feature extraction
        scale_features = []
        for i, scale in enumerate(self.temporal_scales):
            if scale == 1:
                scale_feat = self.scale_encoders[i](x)
            else:
                # Downsample for longer scales
                pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, 
                                    stride=scale).transpose(1, 2)
                scale_feat = self.scale_encoders[i](pooled)
                # Upsample back to original length
                scale_feat = F.interpolate(scale_feat.transpose(1, 2), 
                                         size=seq_len, mode='linear', 
                                         align_corners=False).transpose(1, 2)
            scale_features.append(scale_feat)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=-1)
        
        # Temporal attention
        temp_attn, _ = self.temporal_attention(multi_scale, multi_scale, multi_scale)
        
        # Cross-scale attention
        cross_attn, _ = self.cross_attention(temp_attn, temp_attn, temp_attn)
        
        # Transformer processing
        transformer_out = self.transformer(cross_attn)
        
        # Output projection
        output = self.output_layers(transformer_out)
        
        return output

class EnsembleHedger(nn.Module):
    """Ensemble of multiple hedging models for robustness."""
    
    def __init__(self, input_dim, models_config):
        super().__init__()
        
        self.models = nn.ModuleList()
        self.weights = nn.Parameter(torch.ones(len(models_config)))
        
        for config in models_config:
            if config['type'] == 'lstm':
                model = AdvancedLSTMHedger(input_dim, **config['params'])
            elif config['type'] == 'transformer':
                model = TransformerHedger(input_dim, **config['params'])
            elif config['type'] == 'attention':
                model = AttentionHedger(input_dim, **config['params'])
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            self.models.append(model)
        
        # Softmax weights for ensemble
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x, **kwargs):
        outputs = []
        weights = self.softmax(self.weights)
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'forward') and 'hidden' in model.forward.__code__.co_varnames:
                output, _ = model(x, **kwargs)
            else:
                output = model(x, **kwargs)
            outputs.append(output * weights[i])
        
        return torch.sum(torch.stack(outputs), dim=0)

class RiskAwareLoss(nn.Module):
    """Advanced risk-aware loss function with multiple risk measures."""
    
    def __init__(self, risk_type='cvar', alpha=0.95, lambda_risk=10.0, 
                 turnover_penalty=1e-4, entropic_lambda=10.0):
        super().__init__()
        self.risk_type = risk_type
        self.alpha = alpha
        self.lambda_risk = lambda_risk
        self.turnover_penalty = turnover_penalty
        self.entropic_lambda = entropic_lambda
    
    def forward(self, hedging_errors, positions, previous_positions=None):
        """
        Compute risk-aware loss.
        
        Args:
            hedging_errors: Tensor of hedging errors (batch_size,)
            positions: Tensor of current positions (batch_size, seq_len, 1)
            previous_positions: Tensor of previous positions (batch_size, seq_len, 1)
        """
        batch_size = hedging_errors.size(0)
        
        # Base risk measure
        if self.risk_type == 'cvar':
            risk_loss = self._cvar_loss(hedging_errors)
        elif self.risk_type == 'entropic':
            risk_loss = self._entropic_loss(hedging_errors)
        elif self.risk_type == 'combined':
            cvar_loss = self._cvar_loss(hedging_errors)
            entropic_loss = self._entropic_loss(hedging_errors)
            risk_loss = 0.7 * cvar_loss + 0.3 * entropic_loss
        else:
            risk_loss = torch.mean(hedging_errors ** 2)
        
        # Turnover penalty
        turnover_loss = 0.0
        if previous_positions is not None:
            position_changes = torch.abs(positions - previous_positions)
            turnover_loss = self.turnover_penalty * torch.mean(position_changes)
        
        # Regularization
        position_penalty = 0.01 * torch.mean(torch.abs(positions))
        
        total_loss = risk_loss + turnover_loss + position_penalty
        
        return total_loss, {
            'risk_loss': risk_loss.item(),
            'turnover_loss': turnover_loss.item() if isinstance(turnover_loss, torch.Tensor) else turnover_loss,
            'position_penalty': position_penalty.item(),
            'total_loss': total_loss.item()
        }
    
    def _cvar_loss(self, errors):
        """Conditional Value at Risk loss."""
        sorted_errors, _ = torch.sort(errors, descending=True)
        n = sorted_errors.size(0)
        var_idx = int(self.alpha * n)
        var = sorted_errors[var_idx]
        cvar = torch.mean(sorted_errors[:var_idx+1])
        return cvar
    
    def _entropic_loss(self, errors):
        """Entropic risk measure loss."""
        return (1.0 / self.entropic_lambda) * torch.log(torch.mean(torch.exp(self.entropic_lambda * errors)))

class AdvancedFeatureExtractor(nn.Module):
    """Advanced feature extraction with technical indicators and market microstructure."""
    
    def __init__(self, input_dim, feature_dim=64):
        super().__init__()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Technical indicators
        self.technical_indicators = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # Moving averages
            nn.Conv1d(1, 16, kernel_size=10, padding=4),  # Longer trends
            nn.Conv1d(1, 16, kernel_size=20, padding=9),  # Long-term trends
        ])
        
        self.technical_proj = nn.Linear(48, feature_dim // 2)
        
        # Volatility features
        self.volatility_encoder = nn.Sequential(
            nn.Linear(3, feature_dim // 4),  # Rolling volatility features
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim // 4)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Basic feature extraction
        features = self.feature_layers(x)
        
        # Technical indicators (assuming first feature is price)
        price = x[:, :, 0:1].transpose(1, 2)  # (batch, 1, seq_len)
        tech_features = []
        for conv in self.technical_indicators:
            tech_feat = conv(price)
            tech_features.append(tech_feat)
        
        tech_concat = torch.cat(tech_features, dim=1).transpose(1, 2)  # (batch, seq_len, 48)
        tech_proj = self.technical_proj(tech_concat)
        
        # Volatility features
        if x.size(-1) >= 3:
            vol_features = x[:, :, -3:]  # Last 3 features as volatility
            vol_proj = self.volatility_encoder(vol_features)
        else:
            vol_proj = torch.zeros(batch_size, seq_len, self.volatility_encoder[-1].out_features, 
                                 device=x.device)
        
        # Combine all features
        combined_features = torch.cat([features, tech_proj, vol_proj], dim=-1)
        
        return combined_features

def create_advanced_models(input_dim, config):
    """Factory function to create advanced models."""
    
    models = {}
    
    # LSTM Model
    if 'lstm' in config:
        models['lstm'] = AdvancedLSTMHedger(
            input_dim=input_dim,
            **config['lstm']
        )
    
    # Transformer Model
    if 'transformer' in config:
        models['transformer'] = TransformerHedger(
            input_dim=input_dim,
            **config['transformer']
        )
    
    # Attention Model
    if 'attention' in config:
        models['attention'] = AttentionHedger(
            input_dim=input_dim,
            **config['attention']
        )
    
    # Ensemble Model
    if 'ensemble' in config:
        models['ensemble'] = EnsembleHedger(
            input_dim=input_dim,
            models_config=config['ensemble']['models']
        )
    
    return models

# Example configuration
ADVANCED_MODEL_CONFIG = {
    'lstm': {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'attention_dim': 64
    },
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1
    },
    'attention': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'temporal_scales': [1, 5, 10, 20]
    },
    'ensemble': {
        'models': [
            {'type': 'lstm', 'params': {'hidden_dim': 128, 'num_layers': 2}},
            {'type': 'transformer', 'params': {'d_model': 128, 'nhead': 4, 'num_layers': 3}},
            {'type': 'attention', 'params': {'d_model': 128, 'nhead': 4, 'num_layers': 2}}
        ]
    }
}
