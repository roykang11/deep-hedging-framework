#!/usr/bin/env python3
"""
Advanced Training Framework for Deep Hedging
Implements sophisticated training algorithms, risk-sensitive optimization, and advanced techniques
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import torch.nn.functional as F
from advanced_models import create_advanced_models, RiskAwareLoss, AdvancedFeatureExtractor
import math
from collections import defaultdict
import time

class AdvancedTrainer:
    """Advanced trainer with sophisticated optimization and risk-aware learning."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = RiskAwareLoss(
            risk_type=config.get('risk_type', 'cvar'),
            alpha=config.get('alpha', 0.95),
            lambda_risk=config.get('lambda_risk', 10.0),
            turnover_penalty=config.get('turnover_penalty', 1e-4),
            entropic_lambda=config.get('entropic_lambda', 10.0)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.training_history = defaultdict(list)
        self.best_model_state = None
        self.best_loss = float('inf')
        
        # Advanced techniques
        self.gradient_clipping = config.get('gradient_clipping', 1.0)
        self.ema_decay = config.get('ema_decay', 0.0)  # Disable EMA for now
        self.ema_model = None
        
    def _create_optimizer(self):
        """Create advanced optimizer with different strategies."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('lr', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True
            )
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 1000),
                eta_min=self.config.get('lr_min', 1e-6)
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=50,
                min_lr=1e-6
            )
        elif scheduler_type == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('lr', 1e-3),
                epochs=self.config.get('epochs', 1000),
                steps_per_epoch=self.config.get('steps_per_epoch', 100)
            )
        else:
            return None
    
    def _create_ema_model(self):
        """Create exponential moving average model."""
        if self.ema_decay > 0:
            # Create a copy of the model by cloning the state dict
            ema_model = type(self.model).__new__(type(self.model))
            ema_model.__dict__.update(self.model.__dict__)
            ema_model.load_state_dict(self.model.state_dict())
            ema_model.to(self.device)
            ema_model.eval()
            return ema_model
        return None
    
    def _update_ema(self):
        """Update exponential moving average model."""
        if self.ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _compute_advanced_features(self, prices, returns, volatilities, time_features):
        """Compute advanced features for the model."""
        batch_size, seq_len = prices.shape
        
        # Basic features
        features = []
        
        # Price features
        log_prices = np.log(prices)
        features.append(log_prices)
        
        # Return features
        features.append(returns)
        
        # Volatility features
        features.append(volatilities)
        
        # Time features
        features.append(time_features)
        
        # Technical indicators
        # RSI
        rsi = self._compute_rsi(prices)
        features.append(rsi)
        
        # MACD
        macd, macd_signal = self._compute_macd(prices)
        features.append(macd)
        features.append(macd_signal)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._compute_bollinger_bands(prices)
        features.append(bb_upper)
        features.append(bb_lower)
        features.append(bb_middle)
        
        # Volume features (if available)
        volume_features = np.ones((batch_size, seq_len, 1))  # Placeholder
        features.append(volume_features)
        
        # Market microstructure features
        bid_ask_spread = np.random.normal(0.001, 0.0001, (batch_size, seq_len, 1))
        features.append(bid_ask_spread)
        
        # Combine all features
        combined_features = np.concatenate(features, axis=-1)
        
        return torch.FloatTensor(combined_features).to(self.device)
    
    def _compute_rsi(self, prices, period=14):
        """Compute Relative Strength Index."""
        batch_size, seq_len = prices.shape
        rsi = np.zeros_like(prices)
        
        for b in range(batch_size):
            price_series = prices[b]
            deltas = np.diff(price_series)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
            avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
            
            rs = avg_gains / (avg_losses + 1e-8)
            rsi_values = 100 - (100 / (1 + rs))
            
            # Pad the beginning
            rsi[b, period:] = rsi_values
            rsi[b, :period] = 50  # Neutral RSI
        
        return rsi.reshape(batch_size, seq_len, 1)
    
    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        """Compute MACD indicator."""
        batch_size, seq_len = prices.shape
        macd = np.zeros_like(prices)
        macd_signal = np.zeros_like(prices)
        
        for b in range(batch_size):
            price_series = prices[b]
            
            # EMA calculations
            ema_fast = self._compute_ema(price_series, fast)
            ema_slow = self._compute_ema(price_series, slow)
            
            macd_values = ema_fast - ema_slow
            macd_signal_values = self._compute_ema(macd_values, signal)
            
            macd[b] = macd_values
            macd_signal[b] = macd_signal_values
        
        return (macd.reshape(batch_size, seq_len, 1), 
                macd_signal.reshape(batch_size, seq_len, 1))
    
    def _compute_ema(self, prices, period):
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _compute_bollinger_bands(self, prices, period=20, std_dev=2):
        """Compute Bollinger Bands."""
        batch_size, seq_len = prices.shape
        bb_upper = np.zeros_like(prices)
        bb_lower = np.zeros_like(prices)
        bb_middle = np.zeros_like(prices)
        
        for b in range(batch_size):
            price_series = prices[b]
            
            for i in range(period-1, seq_len):
                window = price_series[i-period+1:i+1]
                mean = np.mean(window)
                std = np.std(window)
                
                bb_middle[b, i] = mean
                bb_upper[b, i] = mean + std_dev * std
                bb_lower[b, i] = mean - std_dev * std
        
        return (bb_upper.reshape(batch_size, seq_len, 1),
                bb_lower.reshape(batch_size, seq_len, 1),
                bb_middle.reshape(batch_size, seq_len, 1))
    
    def train_epoch(self, data_loader, epoch):
        """Train for one epoch with advanced techniques."""
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        
        for batch_idx, (features, option_payoffs) in enumerate(data_loader):
            self.optimizer.zero_grad()
            
            # Unpack features
            features = features.to(self.device)
            option_payoffs = option_payoffs.to(self.device)
            
            # Extract individual features
            prices = features[:, :, 0]
            returns = features[:, :, 1]
            volatilities = features[:, :, 2]
            time_features = features[:, :, 3]
            
            # Use the features directly (already computed)
            # features is already the correct shape
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'hidden' in self.model.forward.__code__.co_varnames:
                positions, hidden = self.model(features)
            else:
                positions = self.model(features)
                hidden = None
            
            # Compute hedging errors
            hedging_errors = self._compute_hedging_errors(
                prices, positions, option_payoffs, None
            )
            
            # Compute loss
            previous_positions = None  # No previous positions for now
            
            loss, loss_dict = self.loss_fn(
                hedging_errors, positions, previous_positions
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            # Update parameters
            self.optimizer.step()
            
            # Update EMA
            self._update_ema()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                loss_components[key] += value
            
            # Logging
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()
        
        # Average losses
        avg_loss = total_loss / len(data_loader)
        for key in loss_components:
            loss_components[key] /= len(data_loader)
        
        return avg_loss, loss_components
    
    def _compute_hedging_errors(self, prices, positions, option_payoffs, transaction_costs=None):
        """Compute hedging errors with transaction costs."""
        batch_size, seq_len = prices.shape
        
        # Initialize portfolio value
        portfolio_values = torch.zeros(batch_size, device=self.device)
        hedging_errors = torch.zeros(batch_size, device=self.device)
        
        for t in range(seq_len):
            current_price = prices[:, t]
            current_position = positions[:, t, 0] if positions.dim() == 3 else positions[:, t]
            
            # Update portfolio value
            portfolio_values += current_position * current_price
            
            # Add transaction costs
            if transaction_costs is not None:
                if t > 0:
                    position_change = torch.abs(current_position - positions[:, t-1, 0])
                    transaction_cost = transaction_costs[:, t] * position_change * current_price
                    portfolio_values -= transaction_cost
        
        # Final hedging error
        hedging_errors = option_payoffs - portfolio_values
        
        return hedging_errors
    
    def validate(self, data_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        loss_components = defaultdict(float)
        
        with torch.no_grad():
            for features, option_payoffs in data_loader:
                # Unpack features
                features = features.to(self.device)
                option_payoffs = option_payoffs.to(self.device)
                
                # Extract individual features
                prices = features[:, :, 0]
                returns = features[:, :, 1]
                volatilities = features[:, :, 2]
                time_features = features[:, :, 3]
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'hidden' in self.model.forward.__code__.co_varnames:
                    positions, _ = self.model(features)
                else:
                    positions = self.model(features)
                
                # Compute hedging errors
                hedging_errors = self._compute_hedging_errors(
                    prices, positions, option_payoffs, None
                )
                
                # Compute loss
                previous_positions = None  # No previous positions for now
                
                loss, loss_dict = self.loss_fn(
                    hedging_errors, positions, previous_positions
                )
                
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    loss_components[key] += value
        
        avg_loss = total_loss / len(data_loader)
        for key in loss_components:
            loss_components[key] /= len(data_loader)
        
        return avg_loss, loss_components
    
    def train(self, train_loader, val_loader=None, epochs=1000):
        """Main training loop with advanced techniques."""
        print(f"Starting advanced training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            train_loss, train_components = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None:
                val_loss, val_components = self.validate(val_loader)
            else:
                val_loss = train_loss
                val_components = train_components
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            for key, value in train_components.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_components.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # Logging
            if epoch % self.config.get('log_interval', 50) == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch:4d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {elapsed:.1f}s')
                
                # Log detailed components
                for key, value in train_components.items():
                    print(f'  Train {key}: {value:.6f}')
                for key, value in val_components.items():
                    print(f'  Val {key}: {value:.6f}')
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f} seconds")
        print(f"Best validation loss: {self.best_loss:.6f}")
        
        return self.training_history

class CurriculumLearning:
    """Curriculum learning for progressive difficulty increase."""
    
    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0, 
                 difficulty_increase=0.05, patience=50):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = initial_difficulty
        self.difficulty_increase = difficulty_increase
        self.patience = patience
        self.no_improvement_count = 0
        self.best_loss = float('inf')
    
    def update_difficulty(self, current_loss):
        """Update difficulty based on performance."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        if (self.no_improvement_count >= self.patience and 
            self.current_difficulty < self.max_difficulty):
            self.current_difficulty = min(
                self.current_difficulty + self.difficulty_increase,
                self.max_difficulty
            )
            self.no_improvement_count = 0
            print(f"Increased difficulty to {self.current_difficulty:.2f}")
    
    def get_difficulty(self):
        """Get current difficulty level."""
        return self.current_difficulty

class AdvancedDataAugmentation:
    """Advanced data augmentation for financial time series."""
    
    def __init__(self, noise_level=0.01, time_warping=0.1, volatility_scaling=0.2):
        self.noise_level = noise_level
        self.time_warping = time_warping
        self.volatility_scaling = volatility_scaling
    
    def augment_batch(self, batch):
        """Apply data augmentation to a batch."""
        augmented_batch = {}
        
        for key, value in batch.items():
            if key in ['prices', 'returns', 'volatilities']:
                augmented_batch[key] = self._augment_series(value)
            else:
                augmented_batch[key] = value
        
        return augmented_batch
    
    def _augment_series(self, series):
        """Augment a time series."""
        # Add noise
        noise = torch.randn_like(series) * self.noise_level
        augmented = series + noise
        
        # Time warping
        if self.time_warping > 0:
            augmented = self._apply_time_warping(augmented)
        
        # Volatility scaling
        if self.volatility_scaling > 0:
            scale_factor = 1 + torch.randn(1) * self.volatility_scaling
            augmented = augmented * scale_factor
        
        return augmented
    
    def _apply_time_warping(self, series):
        """Apply time warping to series."""
        seq_len = series.size(1)
        warp_factor = 1 + torch.randn(1) * self.time_warping
        
        # Create warped indices
        indices = torch.linspace(0, seq_len - 1, int(seq_len * warp_factor))
        indices = torch.clamp(indices, 0, seq_len - 1)
        
        # Interpolate
        warped_series = torch.zeros_like(series)
        for i in range(series.size(0)):
            warped_series[i] = torch.interp(
                torch.arange(seq_len, dtype=torch.float32),
                indices,
                series[i, indices.long()]
            )
        
        return warped_series
