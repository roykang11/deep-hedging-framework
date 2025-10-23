"""
Training loop for Deep Hedging models.

Implements the main training pipeline with vectorized rollout, risk optimization,
and experiment tracking for learning optimal hedging policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from .data import create_generator, create_state_features
from .payoffs import create_payoff
from .models import create_policy
from .risk import create_risk_measure, RegularizedLoss
from .env import VectorizedHedgingEnv
from .baselines import create_baseline_strategy
from .eval import HedgingEvaluator


class DeepHedgingTrainer:
    """
    Main trainer class for Deep Hedging models.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize components
        self._setup_data_generator()
        self._setup_payoff_function()
        self._setup_policy_network()
        self._setup_risk_measure()
        self._setup_environment()
        self._setup_optimizer()
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
    def _setup_data_generator(self):
        """Setup price path generator."""
        model_params = self.config.model
        self.generator = create_generator(
            model_type=model_params.type,
            S0=model_params.S0,
            mu=model_params.mu,
            sigma=model_params.sigma,
            r=model_params.r,
            # Heston parameters
            v0=getattr(model_params, 'v0', 0.04),
            kappa=getattr(model_params, 'kappa', 2.0),
            theta=getattr(model_params, 'theta', 0.04),
            sigma_v=getattr(model_params, 'sigma_v', 0.3),
            rho=getattr(model_params, 'rho', -0.7)
        )
        
        self.n_steps = self.config.simulation.n_steps
        self.dt = self.config.simulation.dt
        
    def _setup_payoff_function(self):
        """Setup option payoff function."""
        model_params = self.config.model
        self.payoff_func = create_payoff(
            payoff_type="european_call",
            K=model_params.K
        )
        
    def _setup_policy_network(self):
        """Setup policy neural network."""
        # Determine input dimension from state features
        input_dim = len(self.config.state.features)
        
        self.policy = create_policy(
            policy_type=self.config.network.type,
            input_dim=input_dim,
            hidden_dims=self.config.network.hidden_dims,
            activation=self.config.network.activation,
            use_layer_norm=self.config.network.use_layer_norm,
            dropout=self.config.network.dropout,
            q_max=self.config.network.q_max
        ).to(self.device)
        
        print(f"Policy network: {self.policy}")
        print(f"Total parameters: {sum(p.numel() for p in self.policy.parameters())}")
        
    def _setup_risk_measure(self):
        """Setup risk measure."""
        self.risk_measure = create_risk_measure(
            risk_type=self.config.risk.type,
            alpha=self.config.risk.alpha,
            lambda_risk=self.config.risk.lambda_risk
        ).to(self.device)
        
        # Setup regularized loss
        self.loss_fn = RegularizedLoss(
            risk_measure=self.risk_measure,
            turnover_penalty=self.config.risk.turnover_penalty
        )
        
    def _setup_environment(self):
        """Setup hedging environment."""
        self.env = VectorizedHedgingEnv(
            payoff_func=self.payoff_func,
            kappa=self.config.costs.kappa,
            kappa_quad=self.config.costs.kappa_quad,
            r=self.config.model.r,
            dt=self.config.simulation.dt
        )
        
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        training_params = self.config.training
        
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=training_params.lr,
            weight_decay=training_params.weight_decay
        )
        
        if training_params.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_params.epochs,
                eta_min=training_params.lr_min
            )
        else:
            self.scheduler = None
            
    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        logging_config = self.config.logging
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f"runs/{int(time.time())}")
        
        # Weights & Biases
        if logging_config.use_wandb:
            wandb.init(
                project="deep-hedging",
                config=OmegaConf.to_container(self.config),
                name=f"hedging_{int(time.time())}"
            )
            
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate a batch of price paths and state features.
        
        Args:
            batch_size: Number of paths to generate
            
        Returns:
            Tuple of (prices, state_features, variances)
        """
        if self.config.model.type.lower() == "gbm":
            prices = self.generator.simulate_paths(
                n_paths=batch_size,
                n_steps=self.n_steps,
                dt=self.dt,
                device=self.device,
                seed=None  # Random seed for training
            )
            variances = None
            
        elif self.config.model.type.lower() == "heston":
            prices, variances = self.generator.simulate_paths(
                n_paths=batch_size,
                n_steps=self.n_steps,
                dt=self.dt,
                device=self.device,
                seed=None
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model.type}")
        
        # Create state features
        state_features = create_state_features(
            prices=prices,
            variances=variances,
            K=self.config.model.K,
            T=self.config.model.T,
            dt=self.dt,
            features=self.config.state.features
        )
        
        return prices, state_features, variances
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        
        # Generate training batch
        batch_size = self.config.simulation.n_paths
        prices, state_features, variances = self.generate_batch(batch_size)
        
        # Forward pass
        hedging_error, rollout_info = self.env.rollout(
            policy=self.policy,
            prices=prices,
            state_features=state_features,
            q_max=self.config.network.q_max
        )
        
        # Compute loss
        loss = self.loss_fn(
            losses=hedging_error,
            total_turnover=rollout_info['total_turnover'],
            total_costs=rollout_info['total_costs'],
            final_positions=rollout_info['final_position']
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'mean_hedging_error': hedging_error.mean().item(),
            'std_hedging_error': hedging_error.std().item(),
            'mean_costs': rollout_info['total_costs'].mean().item(),
            'mean_turnover': rollout_info['total_turnover'].mean().item(),
            'cvar_95': self.risk_measure.get_risk_value(hedging_error).item()
        }
        
        return metrics
    
    def evaluate(self, n_test_paths: int = 10000) -> Dict[str, float]:
        """
        Evaluate the trained policy.
        
        Args:
            n_test_paths: Number of test paths
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        
        with torch.no_grad():
            # Generate test batch
            prices, state_features, variances = self.generate_batch(n_test_paths)
            
            # Forward pass
            hedging_error, rollout_info = self.env.rollout(
                policy=self.policy,
                prices=prices,
                state_features=state_features,
                q_max=self.config.network.q_max
            )
            
            # Compute comprehensive metrics
            from .env import HedgingMetrics
            metrics = HedgingMetrics.compute_metrics(
                hedging_error=hedging_error,
                total_costs=rollout_info['total_costs'],
                total_turnover=rollout_info['total_turnover']
            )
            
            # Add risk-adjusted metrics
            risk_metrics = HedgingMetrics.compute_risk_adjusted_metrics(
                hedging_error=hedging_error,
                total_costs=rollout_info['total_costs'],
                risk_free_rate=self.config.model.r
            )
            metrics.update(risk_metrics)
            
        return metrics
    
    def compare_baselines(self, n_test_paths: int = 10000) -> Dict[str, Dict[str, float]]:
        """
        Compare trained policy with baseline strategies.
        
        Args:
            n_test_paths: Number of test paths
            
        Returns:
            Dictionary of baseline comparison results
        """
        results = {}
        
        # Generate test paths
        prices, state_features, variances = self.generate_batch(n_test_paths)
        
        # Evaluate learned policy
        self.policy.eval()
        with torch.no_grad():
            hedging_error, rollout_info = self.env.rollout(
                policy=self.policy,
                prices=prices,
                state_features=state_features,
                q_max=self.config.network.q_max
            )
            
            from .env import HedgingMetrics
            results['learned'] = HedgingMetrics.compute_metrics(
                hedging_error=hedging_error,
                total_costs=rollout_info['total_costs'],
                total_turnover=rollout_info['total_turnover']
            )
        
        # Evaluate baseline strategies
        baseline_strategies = ['delta', 'periodic', 'no_hedge']
        
        for strategy_type in baseline_strategies:
            strategy = create_baseline_strategy(
                strategy_type=strategy_type,
                K=self.config.model.K,
                T=self.config.model.T,
                r=self.config.model.r,
                sigma=self.config.model.sigma,
                option_type='call'
            )
            
            hedging_error, hedge_info = strategy.hedge(
                prices=prices,
                dt=self.dt,
                kappa=self.config.costs.kappa,
                kappa_quad=self.config.costs.kappa_quad,
                r=self.config.model.r
            )
            
            from .env import HedgingMetrics
            results[strategy_type] = HedgingMetrics.compute_metrics(
                hedging_error=hedging_error,
                total_costs=hedge_info['total_costs'],
                total_turnover=hedge_info['total_turnover']
            )
        
        return results
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Training step
            train_metrics = self.train_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            if epoch % self.config.logging.log_every == 0:
                print(f"Epoch {epoch}: Loss = {train_metrics['loss']:.6f}, "
                      f"CVaR = {train_metrics['cvar_95']:.6f}")
                
                # TensorBoard logging
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
                
                # Weights & Biases logging
                if self.config.logging.use_wandb:
                    wandb.log({f"train/{key}": value for key, value in train_metrics.items()}, 
                             step=epoch)
            
            # Evaluation
            if epoch % self.config.logging.eval_every == 0:
                eval_metrics = self.evaluate()
                
                print(f"Evaluation - CVaR: {eval_metrics['cvar_95']:.6f}, "
                      f"Mean Error: {eval_metrics['mean_error']:.6f}")
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f"eval/{key}", value, epoch)
                
                if self.config.logging.use_wandb:
                    wandb.log({f"eval/{key}": value for key, value in eval_metrics.items()}, 
                             step=epoch)
                
                # Save best model
                if eval_metrics['cvar_95'] < self.best_loss:
                    self.best_loss = eval_metrics['cvar_95']
                    self.save_model(f"best_model.pt")
            
            # Periodic model saving
            if epoch % self.config.logging.save_every == 0:
                self.save_model(f"model_epoch_{epoch}.pt")
        
        print("Training completed!")
        
        # Final evaluation and comparison
        print("\nFinal evaluation...")
        final_metrics = self.evaluate()
        print(f"Final CVaR: {final_metrics['cvar_95']:.6f}")
        
        print("\nBaseline comparison...")
        baseline_results = self.compare_baselines()
        
        # Print comparison table
        print("\nBaseline Comparison:")
        print("Strategy\t\tCVaR\t\tMean Error\t\tMean Costs")
        print("-" * 60)
        for strategy, metrics in baseline_results.items():
            print(f"{strategy:<15}\t{metrics['cvar_95']:.6f}\t\t{metrics['mean_error']:.6f}\t\t{metrics['mean_cost']:.6f}")
        
        # Close logging
        self.writer.close()
        if self.config.logging.use_wandb:
            wandb.finish()
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        os.makedirs("checkpoints", exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, f"checkpoints/{filename}")
        print(f"Model saved: checkpoints/{filename}")
    
    def load_model(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(f"checkpoints/{filename}", map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded: checkpoints/{filename}")


@hydra.main(version_base=None, config_path="../configs", config_name="gbm")
def main(config: DictConfig) -> None:
    """Main training function."""
    print("Deep Hedging Training")
    print("=" * 50)
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Create trainer
    trainer = DeepHedgingTrainer(config)
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
