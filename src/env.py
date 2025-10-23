"""
Portfolio environment for Deep Hedging.

Implements the portfolio simulator with transaction costs, cash management,
and hedging error computation for optimal hedging policy learning.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import numpy as np


class PortfolioEnv:
    """
    Portfolio environment for hedging derivative instruments.
    
    Manages cash B_t and position q_t in the underlying asset, with transaction costs
    and hedging error computation.
    """
    
    def __init__(self, 
                 payoff_func,
                 kappa: float = 0.001,
                 kappa_quad: float = 0.0,
                 r: float = 0.0,
                 dt: float = 0.01):
        """
        Initialize portfolio environment.
        
        Args:
            payoff_func: Option payoff function
            kappa: Proportional transaction cost
            kappa_quad: Quadratic transaction cost
            r: Risk-free rate
            dt: Time step size
        """
        self.payoff_func = payoff_func
        self.kappa = kappa
        self.kappa_quad = kappa_quad
        self.r = r
        self.dt = dt
        
    def reset(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Reset environment for new batch of episodes.
        
        Args:
            batch_size: Number of parallel episodes
            device: Device to run on
            
        Returns:
            Initial state dictionary
        """
        return {
            'cash': torch.zeros(batch_size, device=device),
            'position': torch.zeros(batch_size, device=device),
            'total_costs': torch.zeros(batch_size, device=device),
            'step': 0
        }
    
    def step(self, 
             state: Dict[str, torch.Tensor],
             action: torch.Tensor,
             prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Execute one step of the portfolio environment.
        
        Args:
            state: Current portfolio state
            action: Trading action (change in position)
            prices: Current asset prices
            
        Returns:
            Updated state dictionary
        """
        cash = state['cash']
        position = state['position']
        total_costs = state['total_costs']
        
        # Compute transaction costs
        transaction_cost = self._compute_transaction_cost(action, prices)
        
        # Update cash: B_{t+1} = B_t - Δq_t * S_t - C_t + r * B_t * dt
        cash_change = -action * prices - transaction_cost
        cash = cash + cash_change + self.r * cash * self.dt
        
        # Update position: q_{t+1} = q_t + Δq_t
        new_position = position + action
        
        # Accumulate total costs
        total_costs = total_costs + transaction_cost
        
        return {
            'cash': cash,
            'position': new_position,
            'total_costs': total_costs,
            'step': state['step'] + 1
        }
    
    def _compute_transaction_cost(self, action: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """Compute transaction costs for given action."""
        # Proportional cost: κ * S_t * |Δq_t|
        proportional_cost = self.kappa * prices * torch.abs(action)
        
        # Quadratic cost: κ_quad * S_t * (Δq_t)²
        quadratic_cost = self.kappa_quad * prices * action**2
        
        return proportional_cost + quadratic_cost
    
    def compute_hedging_error(self, 
                            final_state: Dict[str, torch.Tensor],
                            final_prices: torch.Tensor) -> torch.Tensor:
        """
        Compute terminal hedging error.
        
        Terminal hedging error: L = Φ(S_T) - (q_T * S_T + B_T)
        where Φ(S_T) is the option payoff and q_T * S_T + B_T is the portfolio value.
        
        Args:
            final_state: Final portfolio state
            final_prices: Final asset prices
            
        Returns:
            Hedging error for each path
        """
        # Portfolio value: q_T * S_T + B_T
        portfolio_value = final_state['position'] * final_prices + final_state['cash']
        
        # Option payoff
        option_payoff = self.payoff_func(final_prices)
        
        # Hedging error: positive means we owe money (shortfall)
        hedging_error = option_payoff - portfolio_value
        
        return hedging_error
    
    def compute_portfolio_value(self, state: Dict[str, torch.Tensor], prices: torch.Tensor) -> torch.Tensor:
        """Compute current portfolio value."""
        return state['position'] * prices + state['cash']


class VectorizedHedgingEnv:
    """
    Vectorized hedging environment for efficient batch processing.
    
    Processes entire batches of paths simultaneously for training.
    """
    
    def __init__(self, 
                 payoff_func,
                 kappa: float = 0.001,
                 kappa_quad: float = 0.0,
                 r: float = 0.0,
                 dt: float = 0.01):
        """
        Initialize vectorized environment.
        
        Args:
            payoff_func: Option payoff function
            kappa: Proportional transaction cost
            kappa_quad: Quadratic transaction cost
            r: Risk-free rate
            dt: Time step size
        """
        self.payoff_func = payoff_func
        self.kappa = kappa
        self.kappa_quad = kappa_quad
        self.r = r
        self.dt = dt
    
    def rollout(self, 
                policy,
                prices: torch.Tensor,
                state_features: torch.Tensor,
                q_max: float = 1.5) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform vectorized rollout of hedging policy.
        
        Args:
            policy: Neural network policy
            prices: Price paths of shape (batch_size, n_steps + 1)
            state_features: State features of shape (batch_size, n_steps, n_features)
            q_max: Maximum position size
            
        Returns:
            Tuple of (hedging_errors, rollout_info)
        """
        batch_size, n_steps_plus_1 = prices.shape
        n_steps = n_steps_plus_1 - 1
        
        device = prices.device
        
        # Initialize portfolio state
        cash = torch.zeros(batch_size, device=device)
        position = torch.zeros(batch_size, device=device)
        total_costs = torch.zeros(batch_size, device=device)
        
        # Track metrics
        total_turnover = torch.zeros(batch_size, device=device)
        position_history = torch.zeros(batch_size, n_steps, device=device)
        
        # Rollout loop
        for t in range(n_steps):
            # Get current state features
            x_t = state_features[:, t, :]  # (batch_size, n_features)
            
            # Get policy action (target position)
            q_target = policy(x_t)  # (batch_size, 1) or (batch_size,)
            q_target = q_target.squeeze(-1) if q_target.dim() > 1 else q_target
            
            # Bound target position
            q_target = torch.tanh(q_target) * q_max
            
            # Compute trading action
            action = q_target - position
            
            # Get current prices
            S_t = prices[:, t+1]  # Use next period price for trading
            
            # Compute transaction costs
            transaction_cost = self._compute_transaction_cost(action, S_t)
            
            # Update cash: B_{t+1} = B_t - Δq_t * S_t - C_t + r * B_t * dt
            cash_change = -action * S_t - transaction_cost
            cash = cash + cash_change + self.r * cash * self.dt
            
            # Update position
            position = position + action
            
            # Accumulate costs and turnover
            total_costs = total_costs + transaction_cost
            total_turnover = total_turnover + torch.abs(action)
            
            # Store position history
            position_history[:, t] = position
        
        # Compute terminal hedging error
        final_prices = prices[:, -1]
        portfolio_value = position * final_prices + cash
        option_payoff = self.payoff_func(final_prices)
        hedging_error = option_payoff - portfolio_value
        
        # Compile rollout information
        rollout_info = {
            'hedging_error': hedging_error,
            'total_costs': total_costs,
            'total_turnover': total_turnover,
            'final_position': position,
            'final_cash': cash,
            'final_portfolio_value': portfolio_value,
            'option_payoff': option_payoff,
            'position_history': position_history
        }
        
        return hedging_error, rollout_info
    
    def _compute_transaction_cost(self, action: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """Compute transaction costs for given action."""
        # Proportional cost: κ * S_t * |Δq_t|
        proportional_cost = self.kappa * prices * torch.abs(action)
        
        # Quadratic cost: κ_quad * S_t * (Δq_t)²
        quadratic_cost = self.kappa_quad * prices * action**2
        
        return proportional_cost + quadratic_cost


class HedgingMetrics:
    """Utility class for computing hedging performance metrics."""
    
    @staticmethod
    def compute_metrics(hedging_error: torch.Tensor, 
                       total_costs: torch.Tensor,
                       total_turnover: torch.Tensor,
                       alpha: float = 0.95) -> Dict[str, float]:
        """
        Compute comprehensive hedging performance metrics.
        
        Args:
            hedging_error: Terminal hedging errors
            total_costs: Total transaction costs paid
            total_turnover: Total absolute trading volume
            alpha: CVaR confidence level
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Hedging error statistics
        metrics['mean_error'] = torch.mean(hedging_error).item()
        metrics['std_error'] = torch.std(hedging_error).item()
        metrics['max_error'] = torch.max(hedging_error).item()
        metrics['min_error'] = torch.min(hedging_error).item()
        
        # Risk measures
        metrics['var_95'] = torch.quantile(hedging_error, 0.95).item()
        metrics['var_99'] = torch.quantile(hedging_error, 0.99).item()
        
        # CVaR (Expected Shortfall)
        var_alpha = torch.quantile(hedging_error, alpha)
        excess = torch.clamp(hedging_error - var_alpha, min=0.0)
        metrics[f'cvar_{int(alpha*100)}'] = (var_alpha + excess.mean() / (1 - alpha)).item()
        
        # Cost statistics
        metrics['mean_cost'] = torch.mean(total_costs).item()
        metrics['std_cost'] = torch.std(total_costs).item()
        
        # Turnover statistics
        metrics['mean_turnover'] = torch.mean(total_turnover).item()
        metrics['std_turnover'] = torch.std(total_turnover).item()
        
        # Efficiency metrics
        metrics['cost_per_unit_turnover'] = (metrics['mean_cost'] / metrics['mean_turnover'] 
                                           if metrics['mean_turnover'] > 0 else 0.0)
        
        return metrics
    
    @staticmethod
    def compute_risk_adjusted_metrics(hedging_error: torch.Tensor,
                                    total_costs: torch.Tensor,
                                    risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Compute risk-adjusted performance metrics.
        
        Args:
            hedging_error: Terminal hedging errors
            total_costs: Total transaction costs paid
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics = {}
        
        # Sharpe-like ratio (negative hedging error is good)
        excess_return = -hedging_error - risk_free_rate
        metrics['sharpe_ratio'] = (torch.mean(excess_return) / torch.std(hedging_error)).item()
        
        # Information ratio (hedging error vs. costs)
        if torch.std(total_costs) > 0:
            metrics['information_ratio'] = (torch.mean(hedging_error) / torch.std(total_costs)).item()
        else:
            metrics['information_ratio'] = 0.0
        
        # Maximum drawdown (worst hedging error)
        metrics['max_drawdown'] = torch.max(hedging_error).item()
        
        return metrics
