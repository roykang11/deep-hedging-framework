#!/usr/bin/env python3
"""
Deep Hedging Experiment with Advanced Techniques
Comprehensive experiment using sophisticated models, training, and risk measures
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_models import create_advanced_models, ADVANCED_MODEL_CONFIG
from advanced_training import AdvancedTrainer, CurriculumLearning, AdvancedDataAugmentation
from advanced_risk import AdvancedRiskMeasures, RiskSensitiveLoss, AdvancedRiskMetrics
import os
import time
from collections import defaultdict

class DeepHedgingExperiment:
    """Comprehensive deep hedging experiment with advanced techniques."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.models = {}
        
        # Create output directory
        os.makedirs('deep_results', exist_ok=True)
        os.makedirs('deep_results/models', exist_ok=True)
        os.makedirs('deep_results/plots', exist_ok=True)
        
    def generate_advanced_data(self, n_paths=50000, n_steps=200, model_type='gbm'):
        """Generate sophisticated market data with multiple features."""
        print(f"Generating advanced {model_type.upper()} data: {n_paths:,} paths, {n_steps} steps...")
        
        if model_type == 'gbm':
            return self._generate_gbm_data(n_paths, n_steps)
        elif model_type == 'heston':
            return self._generate_heston_data(n_paths, n_steps)
        elif model_type == 'jump_diffusion':
            return self._generate_jump_diffusion_data(n_paths, n_steps)
        elif model_type == 'regime_switching':
            return self._generate_regime_switching_data(n_paths, n_steps)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _generate_gbm_data(self, n_paths, n_steps):
        """Generate GBM data with advanced features."""
        S0, K, T, r, mu, sigma = 100, 100, 1.0, 0.0, 0.0, 0.2
        dt = T / n_steps
        
        # Generate paths
        log_returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt, 
            sigma * np.sqrt(dt), 
            (n_paths, n_steps)
        )
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        # Compute features
        returns = np.diff(prices, axis=1, prepend=S0)
        volatilities = self._compute_rolling_volatility(returns, window=20)
        time_features = np.linspace(0, T, n_steps).reshape(1, -1).repeat(n_paths, axis=0)
        
        # Option payoffs
        option_payoffs = np.maximum(prices[:, -1] - K, 0)
        
        return {
            'prices': prices,
            'returns': returns,
            'volatilities': volatilities,
            'time_features': time_features,
            'option_payoffs': option_payoffs,
            'strike': K,
            'risk_free_rate': r
        }
    
    def _generate_heston_data(self, n_paths, n_steps):
        """Generate Heston stochastic volatility data."""
        S0, K, T, r = 100, 100, 1.0, 0.0
        v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.3, -0.7
        dt = T / n_steps
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Generate correlated Brownian motions
        for t in range(n_steps):
            dW1 = np.random.normal(0, np.sqrt(dt), n_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n_paths)
            
            # Update variance
            v[:, t+1] = np.maximum(v[:, t] + kappa * (theta - v[:, t]) * dt + 
                                  sigma_v * np.sqrt(v[:, t]) * dW2, 0)
            
            # Update price
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v[:, t]) * dt + 
                                        np.sqrt(v[:, t]) * dW1)
        
        prices = S[:, 1:]
        returns = np.diff(prices, axis=1, prepend=S0)
        volatilities = np.sqrt(v[:, 1:])
        time_features = np.linspace(0, T, n_steps).reshape(1, -1).repeat(n_paths, axis=0)
        option_payoffs = np.maximum(prices[:, -1] - K, 0)
        
        return {
            'prices': prices,
            'returns': returns,
            'volatilities': volatilities,
            'time_features': time_features,
            'option_payoffs': option_payoffs,
            'strike': K,
            'risk_free_rate': r
        }
    
    def _generate_jump_diffusion_data(self, n_paths, n_steps):
        """Generate jump-diffusion data with Poisson jumps."""
        S0, K, T, r, mu, sigma = 100, 100, 1.0, 0.0, 0.0, 0.2
        lambda_jump, mu_jump, sigma_jump = 0.1, -0.1, 0.3
        dt = T / n_steps
        
        # Generate jump times and sizes
        jump_times = np.random.poisson(lambda_jump * dt, (n_paths, n_steps))
        jump_sizes = np.random.normal(mu_jump, sigma_jump, (n_paths, n_steps))
        
        # Generate continuous part
        continuous_returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt, 
            sigma * np.sqrt(dt), 
            (n_paths, n_steps)
        )
        
        # Add jumps
        jump_returns = jump_times * jump_sizes
        total_returns = continuous_returns + jump_returns
        
        # Compute prices
        log_prices = np.cumsum(total_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        returns = np.diff(prices, axis=1, prepend=S0)
        volatilities = self._compute_rolling_volatility(returns, window=20)
        time_features = np.linspace(0, T, n_steps).reshape(1, -1).repeat(n_paths, axis=0)
        option_payoffs = np.maximum(prices[:, -1] - K, 0)
        
        return {
            'prices': prices,
            'returns': returns,
            'volatilities': volatilities,
            'time_features': time_features,
            'option_payoffs': option_payoffs,
            'strike': K,
            'risk_free_rate': r
        }
    
    def _generate_regime_switching_data(self, n_paths, n_steps):
        """Generate regime-switching data with two volatility regimes."""
        S0, K, T, r, mu = 100, 100, 1.0, 0.0, 0.0
        sigma_low, sigma_high = 0.15, 0.35
        transition_prob = 0.05  # Probability of regime switch per step
        dt = T / n_steps
        
        # Initialize
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = S0
        regimes = np.zeros((n_paths, n_steps), dtype=int)
        
        for t in range(n_steps):
            # Regime switching
            regime_switches = np.random.random((n_paths,)) < transition_prob
            if t > 0:
                regimes[:, t] = np.where(regime_switches, 1 - regimes[:, t-1], regimes[:, t-1])
            else:
                regimes[:, t] = np.random.randint(0, 2, n_paths)
            
            # Generate returns based on regime
            sigma = np.where(regimes[:, t] == 0, sigma_low, sigma_high)
            returns = np.random.normal(
                (mu - 0.5 * sigma**2) * dt, 
                sigma * np.sqrt(dt), 
                n_paths
            )
            
            prices[:, t+1] = prices[:, t] * np.exp(returns)
        
        prices = prices[:, 1:]
        returns = np.diff(prices, axis=1, prepend=S0)
        volatilities = np.where(regimes == 0, sigma_low, sigma_high)
        time_features = np.linspace(0, T, n_steps).reshape(1, -1).repeat(n_paths, axis=0)
        option_payoffs = np.maximum(prices[:, -1] - K, 0)
        
        return {
            'prices': prices,
            'returns': returns,
            'volatilities': volatilities,
            'time_features': time_features,
            'option_payoffs': option_payoffs,
            'strike': K,
            'risk_free_rate': r,
            'regimes': regimes
        }
    
    def _compute_rolling_volatility(self, returns, window=20):
        """Compute rolling volatility."""
        n_paths, n_steps = returns.shape
        volatilities = np.zeros_like(returns)
        
        for i in range(window, n_steps):
            volatilities[:, i] = np.std(returns[:, i-window:i], axis=1)
        
        # Fill initial values
        volatilities[:, :window] = volatilities[:, window:window+1]
        
        return volatilities
    
    def create_advanced_models(self, input_dim):
        """Create advanced neural network models."""
        print("Creating advanced neural network models...")
        
        models = create_advanced_models(input_dim, ADVANCED_MODEL_CONFIG)
        
        for name, model in models.items():
            print(f"  {name}: {sum(p.numel() for p in model.parameters()):,} parameters")
            self.models[name] = model
        
        return models
    
    def prepare_data_loaders(self, data, batch_size=1024, train_ratio=0.8):
        """Prepare data loaders for training."""
        n_paths = data['prices'].shape[0]
        n_train = int(n_paths * train_ratio)
        
        # Split data
        train_data = {k: v[:n_train] if hasattr(v, '__getitem__') and len(v.shape) > 0 else v for k, v in data.items()}
        val_data = {k: v[n_train:] if hasattr(v, '__getitem__') and len(v.shape) > 0 else v for k, v in data.items()}
        
        # Create datasets
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _create_dataset(self, data):
        """Create PyTorch dataset."""
        # Convert to tensors
        prices = torch.FloatTensor(data['prices'])
        returns = torch.FloatTensor(data['returns'])
        volatilities = torch.FloatTensor(data['volatilities'])
        time_features = torch.FloatTensor(data['time_features'])
        option_payoffs = torch.FloatTensor(data['option_payoffs'])
        
        # Create features tensor
        features = torch.stack([
            prices, returns, volatilities, time_features
        ], dim=-1)
        
        return TensorDataset(features, option_payoffs)
    
    def train_advanced_models(self, train_loader, val_loader, epochs=1000):
        """Train all advanced models."""
        print("Training advanced models...")
        
        training_config = {
            'optimizer': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'gradient_clipping': 1.0,
            'ema_decay': 0.999,
            'risk_type': 'cvar',
            'alpha': 0.95,
            'lambda_risk': 10.0,
            'turnover_penalty': 1e-4,
            'epochs': epochs,
            'log_interval': 100
        }
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            trainer = AdvancedTrainer(model, training_config)
            history = trainer.train(train_loader, val_loader, epochs)
            
            # Save model
            torch.save(model.state_dict(), f'deep_results/models/{name}_model.pt')
            
            # Store results
            self.results[name] = {
                'history': history,
                'best_loss': trainer.best_loss
            }
    
    def evaluate_models(self, test_loader):
        """Evaluate all models on test data."""
        print("Evaluating models...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            model.eval()
            model.to(self.device)
            
            all_hedging_errors = []
            all_positions = []
            all_option_payoffs = []
            
            with torch.no_grad():
                for features, option_payoffs in test_loader:
                    features = features.to(self.device)
                    option_payoffs = option_payoffs.to(self.device)
                    
                    # Forward pass
                    if hasattr(model, 'forward') and 'hidden' in model.forward.__code__.co_varnames:
                        positions, _ = model(features)
                    else:
                        positions = model(features)
                    
                    # Compute hedging errors
                    hedging_errors = self._compute_hedging_errors(
                        features, positions, option_payoffs
                    )
                    
                    all_hedging_errors.append(hedging_errors.cpu().numpy())
                    all_positions.append(positions.cpu().numpy())
                    all_option_payoffs.append(option_payoffs.cpu().numpy())
            
            # Combine results
            hedging_errors = np.concatenate(all_hedging_errors)
            positions = np.concatenate(all_positions)
            option_payoffs = np.concatenate(all_option_payoffs)
            
            # Compute risk metrics
            risk_metrics = self._compute_risk_metrics(hedging_errors)
            
            evaluation_results[name] = {
                'hedging_errors': hedging_errors,
                'positions': positions,
                'option_payoffs': option_payoffs,
                'risk_metrics': risk_metrics
            }
        
        return evaluation_results
    
    def _compute_hedging_errors(self, features, positions, option_payoffs):
        """Compute hedging errors."""
        prices = features[:, :, 0]  # First feature is price
        final_prices = prices[:, -1]
        
        # Simple hedging error calculation
        # In practice, this would be more sophisticated
        portfolio_values = positions[:, -1, 0] * final_prices
        hedging_errors = option_payoffs - portfolio_values
        
        return hedging_errors
    
    def _compute_risk_metrics(self, hedging_errors):
        """Compute comprehensive risk metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = np.mean(hedging_errors)
        metrics['std'] = np.std(hedging_errors)
        metrics['skewness'] = self._compute_skewness(hedging_errors)
        metrics['kurtosis'] = self._compute_kurtosis(hedging_errors)
        
        # Risk measures
        metrics['var_95'] = np.percentile(hedging_errors, 5)
        metrics['var_99'] = np.percentile(hedging_errors, 1)
        metrics['cvar_95'] = np.mean(hedging_errors[hedging_errors <= metrics['var_95']])
        metrics['cvar_99'] = np.mean(hedging_errors[hedging_errors <= metrics['var_99']])
        
        # Advanced risk measures
        metrics['entropic_risk'] = AdvancedRiskMeasures.entropic_risk(hedging_errors)
        metrics['max_drawdown'] = self._compute_max_drawdown(hedging_errors)
        metrics['tail_ratio'] = self._compute_tail_ratio(hedging_errors)
        
        return metrics
    
    def _compute_skewness(self, data):
        """Compute skewness."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_max_drawdown(self, data):
        """Compute maximum drawdown."""
        cumulative = np.cumsum(data)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)
    
    def _compute_tail_ratio(self, data):
        """Compute tail ratio."""
        upper_tail = np.percentile(data, 95)
        lower_tail = np.percentile(data, 5)
        return upper_tail / abs(lower_tail)
    
    def generate_comprehensive_plots(self, evaluation_results):
        """Generate comprehensive plots for analysis."""
        print("Generating comprehensive plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Risk comparison plot
        self._plot_risk_comparison(evaluation_results)
        
        # 2. Model performance comparison
        self._plot_model_performance(evaluation_results)
        
        # 3. Risk-return scatter plot
        self._plot_risk_return_scatter(evaluation_results)
        
        # 4. Position analysis
        self._plot_position_analysis(evaluation_results)
        
        # 5. Training curves
        self._plot_training_curves()
        
        print("Plots saved to deep_results/plots/")
    
    def _plot_risk_comparison(self, evaluation_results):
        """Plot risk measures comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Advanced Risk Measures Comparison', fontsize=16, fontweight='bold')
        
        models = list(evaluation_results.keys())
        metrics = ['cvar_95', 'cvar_99', 'entropic_risk', 'max_drawdown']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [evaluation_results[model]['risk_metrics'][metric] for model in models]
            
            bars = ax.bar(models, values, alpha=0.8)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('deep_results/plots/risk_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance(self, evaluation_results):
        """Plot model performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(evaluation_results.keys())
        
        # Mean vs Std
        means = [evaluation_results[model]['risk_metrics']['mean'] for model in models]
        stds = [evaluation_results[model]['risk_metrics']['std'] for model in models]
        
        axes[0, 0].scatter(means, stds, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 0].annotate(model, (means[i], stds[i]), xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_xlabel('Mean Error')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].set_title('Mean vs Standard Deviation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # CVaR comparison
        cvar_95 = [evaluation_results[model]['risk_metrics']['cvar_95'] for model in models]
        cvar_99 = [evaluation_results[model]['risk_metrics']['cvar_99'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, cvar_95, width, label='CVaR 95%', alpha=0.8)
        axes[0, 1].bar(x + width/2, cvar_99, width, label='CVaR 99%', alpha=0.8)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('CVaR Value')
        axes[0, 1].set_title('CVaR Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distributions
        for i, model in enumerate(models):
            errors = evaluation_results[model]['hedging_errors']
            axes[1, 0].hist(errors, bins=50, alpha=0.6, label=model, density=True)
        axes[1, 0].set_xlabel('Hedging Error')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Error Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk-return scatter
        risks = [evaluation_results[model]['risk_metrics']['std'] for model in models]
        returns = [evaluation_results[model]['risk_metrics']['mean'] for model in models]
        
        axes[1, 1].scatter(risks, returns, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (risks[i], returns[i]), xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Risk (Std)')
        axes[1, 1].set_ylabel('Return (Mean)')
        axes[1, 1].set_title('Risk-Return Profile')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_results/plots/model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_return_scatter(self, evaluation_results):
        """Plot risk-return scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = list(evaluation_results.keys())
        risks = [evaluation_results[model]['risk_metrics']['std'] for model in models]
        returns = [evaluation_results[model]['risk_metrics']['mean'] for model in models]
        
        scatter = ax.scatter(risks, returns, s=200, alpha=0.7, c=range(len(models)), cmap='viridis')
        
        for i, model in enumerate(models):
            ax.annotate(model, (risks[i], returns[i]), xytext=(10, 10), 
                       textcoords='offset points', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Risk (Standard Deviation)', fontsize=14)
        ax.set_ylabel('Return (Mean Error)', fontsize=14)
        ax.set_title('Risk-Return Scatter Plot', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_results/plots/risk_return_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_analysis(self, evaluation_results):
        """Plot position analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Position Analysis', fontsize=16, fontweight='bold')
        
        for i, (model, results) in enumerate(evaluation_results.items()):
            positions = results['positions']
            
            # Position distribution
            ax = axes[0, 0]
            ax.hist(positions[:, -1, 0], bins=50, alpha=0.6, label=model, density=True)
            ax.set_xlabel('Final Position')
            ax.set_ylabel('Density')
            ax.set_title('Position Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Position over time (sample)
            ax = axes[0, 1]
            sample_positions = positions[:5, :, 0].T
            ax.plot(sample_positions, alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Position')
            ax.set_title('Position Evolution (Sample)')
            ax.grid(True, alpha=0.3)
            
            # Position statistics
            ax = axes[1, 0]
            pos_stats = [np.mean(positions[:, -1, 0]), np.std(positions[:, -1, 0]), 
                        np.min(positions[:, -1, 0]), np.max(positions[:, -1, 0])]
            stat_names = ['Mean', 'Std', 'Min', 'Max']
            ax.bar(stat_names, pos_stats, alpha=0.8)
            ax.set_ylabel('Position Value')
            ax.set_title(f'{model} Position Statistics')
            ax.grid(True, alpha=0.3)
            
            # Position vs Error correlation
            ax = axes[1, 1]
            final_positions = positions[:, -1, 0]
            errors = results['hedging_errors']
            ax.scatter(final_positions, errors, alpha=0.6)
            ax.set_xlabel('Final Position')
            ax.set_ylabel('Hedging Error')
            ax.set_title(f'{model} Position vs Error')
            ax.grid(True, alpha=0.3)
            
            break  # Only plot for first model to avoid clutter
        
        plt.tight_layout()
        plt.savefig('deep_results/plots/position_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self):
        """Plot training curves."""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        for model_name, results in self.results.items():
            history = results['history']
            
            # Training loss
            axes[0, 0].plot(history['train_loss'], label=f'{model_name} (train)', alpha=0.8)
            axes[0, 0].plot(history['val_loss'], label=f'{model_name} (val)', alpha=0.8)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Risk components
            if 'train_risk_loss' in history:
                axes[0, 1].plot(history['train_risk_loss'], label=f'{model_name} risk', alpha=0.8)
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Risk Loss')
                axes[0, 1].set_title('Risk Loss Component')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Turnover penalty
            if 'train_turnover_loss' in history:
                axes[1, 0].plot(history['train_turnover_loss'], label=f'{model_name} turnover', alpha=0.8)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Turnover Loss')
                axes[1, 0].set_title('Turnover Loss Component')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Position penalty
            if 'train_position_penalty' in history:
                axes[1, 1].plot(history['train_position_penalty'], label=f'{model_name} position', alpha=0.8)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Position Penalty')
                axes[1, 1].set_title('Position Penalty Component')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_results/plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_results_table(self, evaluation_results):
        """Generate comprehensive results table."""
        print("Generating results table...")
        
        results_data = []
        
        for model_name, results in evaluation_results.items():
            metrics = results['risk_metrics']
            
            results_data.append({
                'Model': model_name,
                'Mean Error': f"{metrics['mean']:.6f}",
                'Std Error': f"{metrics['std']:.6f}",
                'Skewness': f"{metrics['skewness']:.6f}",
                'Kurtosis': f"{metrics['kurtosis']:.6f}",
                'VaR 95%': f"{metrics['var_95']:.6f}",
                'CVaR 95%': f"{metrics['cvar_95']:.6f}",
                'VaR 99%': f"{metrics['var_99']:.6f}",
                'CVaR 99%': f"{metrics['cvar_99']:.6f}",
                'Entropic Risk': f"{metrics['entropic_risk']:.6f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.6f}",
                'Tail Ratio': f"{metrics['tail_ratio']:.6f}"
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv('deep_results/results_table.csv', index=False)
        
        print("\n" + "="*120)
        print("COMPREHENSIVE DEEP HEDGING RESULTS")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        return df

def main():
    """Run comprehensive deep hedging experiment."""
    print("🚀 Starting Deep Hedging Experiment with Advanced Techniques")
    print("="*70)
    
    # Configuration - Smaller for testing
    config = {
        'n_paths': 10000,
        'n_steps': 100,
        'batch_size': 512,
        'epochs': 100,
        'models': ['lstm', 'transformer', 'attention']
    }
    
    # Initialize experiment
    experiment = DeepHedgingExperiment(config)
    
    # Generate data for different models
    model_types = ['gbm', 'heston', 'jump_diffusion', 'regime_switching']
    
    for model_type in model_types:
        print(f"\n📊 Processing {model_type.upper()} model...")
        
        # Generate data
        data = experiment.generate_advanced_data(
            n_paths=config['n_paths'],
            n_steps=config['n_steps'],
            model_type=model_type
        )
        
        # Create models
        input_dim = 4  # prices, returns, volatilities, time_features
        models = experiment.create_advanced_models(input_dim)
        
        # Prepare data loaders
        train_loader, val_loader = experiment.prepare_data_loaders(
            data, batch_size=config['batch_size']
        )
        
        # Train models
        experiment.train_advanced_models(train_loader, val_loader, epochs=config['epochs'])
        
        # Evaluate models
        test_loader = val_loader  # Use validation as test for simplicity
        evaluation_results = experiment.evaluate_models(test_loader)
        
        # Generate plots
        experiment.generate_comprehensive_plots(evaluation_results)
        
        # Generate results table
        df = experiment.generate_results_table(evaluation_results)
        
        print(f"✅ {model_type.upper()} experiment completed!")
    
    print("\n🎉 All experiments completed successfully!")
    print("📁 Results saved to 'deep_results/' directory")
    print("   - Models: deep_results/models/")
    print("   - Plots: deep_results/plots/")
    print("   - Data: deep_results/results_table.csv")

if __name__ == "__main__":
    main()
