"""
Baseline hedging strategies for Deep Hedging comparison.

Implements traditional hedging methods including delta hedging, periodic rebalancing,
and no-hedge strategies for performance benchmarking.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from .payoffs import BlackScholesPrice
from .env import VectorizedHedgingEnv


class DeltaHedgeStrategy:
    """
    Black-Scholes delta hedging strategy.
    
    Computes the theoretical delta and trades to maintain delta-neutral position.
    """
    
    def __init__(self, 
                 K: float,
                 T: float,
                 r: float = 0.0,
                 sigma: float = 0.2,
                 option_type: str = "call"):
        """
        Initialize delta hedging strategy.
        
        Args:
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
        """
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        
        # Black-Scholes pricing model
        self.bs_model = BlackScholesPrice(r=r)
    
    def compute_delta(self, S: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute Black-Scholes delta at given time.
        
        Args:
            S: Current stock price
            t: Current time
            
        Returns:
            Delta value
        """
        time_to_maturity = max(self.T - t, 1e-6)  # Avoid division by zero
        
        if self.option_type.lower() == "call":
            return self.bs_model.call_delta(S, self.K, time_to_maturity, self.sigma)
        else:
            return self.bs_model.put_delta(S, self.K, time_to_maturity, self.sigma)
    
    def hedge(self, 
              prices: torch.Tensor,
              dt: float,
              kappa: float = 0.001,
              kappa_quad: float = 0.0,
              r: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform delta hedging on price paths.
        
        Args:
            prices: Price paths of shape (batch_size, n_steps + 1)
            dt: Time step size
            kappa: Proportional transaction cost
            kappa_quad: Quadratic transaction cost
            r: Risk-free rate
            
        Returns:
            Tuple of (hedging_errors, hedge_info)
        """
        batch_size, n_steps_plus_1 = prices.shape
        n_steps = n_steps_plus_1 - 1
        
        device = prices.device
        
        # Initialize portfolio
        cash = torch.zeros(batch_size, device=device)
        position = torch.zeros(batch_size, device=device)
        total_costs = torch.zeros(batch_size, device=device)
        total_turnover = torch.zeros(batch_size, device=device)
        
        # Track position history
        position_history = torch.zeros(batch_size, n_steps, device=device)
        
        # Hedge loop
        for t in range(n_steps):
            # Current time and prices
            current_time = t * dt
            S_t = prices[:, t+1]  # Use next period price for trading
            
            # Compute target delta
            target_delta = self.compute_delta(S_t, current_time)
            
            # Compute required position change
            action = target_delta - position
            
            # Compute transaction costs
            transaction_cost = kappa * S_t * torch.abs(action) + kappa_quad * S_t * action**2
            
            # Update cash
            cash_change = -action * S_t - transaction_cost
            cash = cash + cash_change + r * cash * dt
            
            # Update position
            position = position + action
            
            # Accumulate costs and turnover
            total_costs = total_costs + transaction_cost
            total_turnover = total_turnover + torch.abs(action)
            
            # Store position
            position_history[:, t] = position
        
        # Compute terminal hedging error
        final_prices = prices[:, -1]
        portfolio_value = position * final_prices + cash
        
        # Option payoff
        if self.option_type.lower() == "call":
            option_payoff = torch.clamp(final_prices - self.K, min=0.0)
        else:
            option_payoff = torch.clamp(self.K - final_prices, min=0.0)
        
        hedging_error = option_payoff - portfolio_value
        
        # Compile hedge information
        hedge_info = {
            'hedging_error': hedging_error,
            'total_costs': total_costs,
            'total_turnover': total_turnover,
            'final_position': position,
            'final_cash': cash,
            'final_portfolio_value': portfolio_value,
            'option_payoff': option_payoff,
            'position_history': position_history
        }
        
        return hedging_error, hedge_info


class PeriodicHedgeStrategy:
    """
    Periodic rebalancing strategy.
    
    Rebalances to delta hedge every k steps, otherwise holds position.
    """
    
    def __init__(self, 
                 K: float,
                 T: float,
                 r: float = 0.0,
                 sigma: float = 0.2,
                 option_type: str = "call",
                 rebalance_frequency: int = 10):
        """
        Initialize periodic hedging strategy.
        
        Args:
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            rebalance_frequency: How often to rebalance (every k steps)
        """
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.rebalance_frequency = rebalance_frequency
        
        # Black-Scholes pricing model
        self.bs_model = BlackScholesPrice(r=r)
    
    def compute_delta(self, S: torch.Tensor, t: float) -> torch.Tensor:
        """Compute Black-Scholes delta at given time."""
        time_to_maturity = max(self.T - t, 1e-6)
        
        if self.option_type.lower() == "call":
            return self.bs_model.call_delta(S, self.K, time_to_maturity, self.sigma)
        else:
            return self.bs_model.put_delta(S, self.K, time_to_maturity, self.sigma)
    
    def hedge(self, 
              prices: torch.Tensor,
              dt: float,
              kappa: float = 0.001,
              kappa_quad: float = 0.0,
              r: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform periodic hedging on price paths.
        
        Args:
            prices: Price paths of shape (batch_size, n_steps + 1)
            dt: Time step size
            kappa: Proportional transaction cost
            kappa_quad: Quadratic transaction cost
            r: Risk-free rate
            
        Returns:
            Tuple of (hedging_errors, hedge_info)
        """
        batch_size, n_steps_plus_1 = prices.shape
        n_steps = n_steps_plus_1 - 1
        
        device = prices.device
        
        # Initialize portfolio
        cash = torch.zeros(batch_size, device=device)
        position = torch.zeros(batch_size, device=device)
        total_costs = torch.zeros(batch_size, device=device)
        total_turnover = torch.zeros(batch_size, device=device)
        
        # Track position history
        position_history = torch.zeros(batch_size, n_steps, device=device)
        
        # Hedge loop
        for t in range(n_steps):
            # Current time and prices
            current_time = t * dt
            S_t = prices[:, t+1]  # Use next period price for trading
            
            # Decide whether to rebalance
            should_rebalance = (t % self.rebalance_frequency == 0)
            
            if should_rebalance:
                # Compute target delta
                target_delta = self.compute_delta(S_t, current_time)
                
                # Compute required position change
                action = target_delta - position
            else:
                # Hold position
                action = torch.zeros_like(position)
            
            # Compute transaction costs
            transaction_cost = kappa * S_t * torch.abs(action) + kappa_quad * S_t * action**2
            
            # Update cash
            cash_change = -action * S_t - transaction_cost
            cash = cash + cash_change + r * cash * dt
            
            # Update position
            position = position + action
            
            # Accumulate costs and turnover
            total_costs = total_costs + transaction_cost
            total_turnover = total_turnover + torch.abs(action)
            
            # Store position
            position_history[:, t] = position
        
        # Compute terminal hedging error
        final_prices = prices[:, -1]
        portfolio_value = position * final_prices + cash
        
        # Option payoff
        if self.option_type.lower() == "call":
            option_payoff = torch.clamp(final_prices - self.K, min=0.0)
        else:
            option_payoff = torch.clamp(self.K - final_prices, min=0.0)
        
        hedging_error = option_payoff - portfolio_value
        
        # Compile hedge information
        hedge_info = {
            'hedging_error': hedging_error,
            'total_costs': total_costs,
            'total_turnover': total_turnover,
            'final_position': position,
            'final_cash': cash,
            'final_portfolio_value': portfolio_value,
            'option_payoff': option_payoff,
            'position_history': position_history
        }
        
        return hedging_error, hedge_info


class NoHedgeStrategy:
    """
    No hedging strategy (benchmark).
    
    Simply holds cash and pays out the option payoff at maturity.
    """
    
    def __init__(self, K: float, option_type: str = "call"):
        """
        Initialize no hedge strategy.
        
        Args:
            K: Strike price
            option_type: "call" or "put"
        """
        self.K = K
        self.option_type = option_type
    
    def hedge(self, 
              prices: torch.Tensor,
              dt: float,
              kappa: float = 0.001,
              kappa_quad: float = 0.0,
              r: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform no hedging (just hold cash).
        
        Args:
            prices: Price paths of shape (batch_size, n_steps + 1)
            dt: Time step size
            kappa: Proportional transaction cost (unused)
            kappa_quad: Quadratic transaction cost (unused)
            r: Risk-free rate
            
        Returns:
            Tuple of (hedging_errors, hedge_info)
        """
        batch_size, n_steps_plus_1 = prices.shape
        n_steps = n_steps_plus_1 - 1
        
        device = prices.device
        
        # Initialize portfolio (all cash, no stock position)
        cash = torch.zeros(batch_size, device=device)
        position = torch.zeros(batch_size, device=device)
        total_costs = torch.zeros(batch_size, device=device)
        total_turnover = torch.zeros(batch_size, device=device)
        
        # Track position history
        position_history = torch.zeros(batch_size, n_steps, device=device)
        
        # No trading loop
        for t in range(n_steps):
            # Just accumulate risk-free interest
            cash = cash + r * cash * dt
            
            # Store position (always zero)
            position_history[:, t] = position
        
        # Compute terminal hedging error
        final_prices = prices[:, -1]
        portfolio_value = position * final_prices + cash
        
        # Option payoff
        if self.option_type.lower() == "call":
            option_payoff = torch.clamp(final_prices - self.K, min=0.0)
        else:
            option_payoff = torch.clamp(self.K - final_prices, min=0.0)
        
        hedging_error = option_payoff - portfolio_value
        
        # Compile hedge information
        hedge_info = {
            'hedging_error': hedging_error,
            'total_costs': total_costs,
            'total_turnover': total_turnover,
            'final_position': position,
            'final_cash': cash,
            'final_portfolio_value': portfolio_value,
            'option_payoff': option_payoff,
            'position_history': position_history
        }
        
        return hedging_error, hedge_info


class AdaptiveDeltaHedgeStrategy:
    """
    Adaptive delta hedging with volatility estimation.
    
    Estimates realized volatility and adjusts delta accordingly.
    """
    
    def __init__(self, 
                 K: float,
                 T: float,
                 r: float = 0.0,
                 initial_sigma: float = 0.2,
                 option_type: str = "call",
                 vol_window: int = 20):
        """
        Initialize adaptive delta hedging strategy.
        
        Args:
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            initial_sigma: Initial volatility estimate
            option_type: "call" or "put"
            vol_window: Window for volatility estimation
        """
        self.K = K
        self.T = T
        self.r = r
        self.initial_sigma = initial_sigma
        self.option_type = option_type
        self.vol_window = vol_window
        
        # Black-Scholes pricing model
        self.bs_model = BlackScholesPrice(r=r)
    
    def estimate_volatility(self, prices: torch.Tensor, t: int) -> torch.Tensor:
        """
        Estimate realized volatility from price history.
        
        Args:
            prices: Price paths of shape (batch_size, n_steps + 1)
            t: Current time step
            
        Returns:
            Estimated volatility
        """
        if t < self.vol_window:
            # Use initial volatility for early steps
            return torch.full((prices.shape[0],), self.initial_sigma, device=prices.device)
        
        # Compute log returns
        start_idx = max(0, t - self.vol_window + 1)
        price_window = prices[:, start_idx:t+2]  # Include current price
        
        log_returns = torch.log(price_window[:, 1:] / price_window[:, :-1])
        
        # Compute realized volatility
        realized_vol = torch.std(log_returns, dim=1)
        
        return realized_vol
    
    def compute_delta(self, S: torch.Tensor, t: float, sigma: torch.Tensor) -> torch.Tensor:
        """Compute Black-Scholes delta with estimated volatility."""
        time_to_maturity = max(self.T - t, 1e-6)
        
        if self.option_type.lower() == "call":
            return self.bs_model.call_delta(S, self.K, time_to_maturity, sigma)
        else:
            return self.bs_model.put_delta(S, self.K, time_to_maturity, sigma)
    
    def hedge(self, 
              prices: torch.Tensor,
              dt: float,
              kappa: float = 0.001,
              kappa_quad: float = 0.0,
              r: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform adaptive delta hedging on price paths.
        
        Args:
            prices: Price paths of shape (batch_size, n_steps + 1)
            dt: Time step size
            kappa: Proportional transaction cost
            kappa_quad: Quadratic transaction cost
            r: Risk-free rate
            
        Returns:
            Tuple of (hedging_errors, hedge_info)
        """
        batch_size, n_steps_plus_1 = prices.shape
        n_steps = n_steps_plus_1 - 1
        
        device = prices.device
        
        # Initialize portfolio
        cash = torch.zeros(batch_size, device=device)
        position = torch.zeros(batch_size, device=device)
        total_costs = torch.zeros(batch_size, device=device)
        total_turnover = torch.zeros(batch_size, device=device)
        
        # Track position history
        position_history = torch.zeros(batch_size, n_steps, device=device)
        
        # Hedge loop
        for t in range(n_steps):
            # Current time and prices
            current_time = t * dt
            S_t = prices[:, t+1]  # Use next period price for trading
            
            # Estimate volatility
            sigma_est = self.estimate_volatility(prices, t)
            
            # Compute target delta with estimated volatility
            target_delta = self.compute_delta(S_t, current_time, sigma_est)
            
            # Compute required position change
            action = target_delta - position
            
            # Compute transaction costs
            transaction_cost = kappa * S_t * torch.abs(action) + kappa_quad * S_t * action**2
            
            # Update cash
            cash_change = -action * S_t - transaction_cost
            cash = cash + cash_change + r * cash * dt
            
            # Update position
            position = position + action
            
            # Accumulate costs and turnover
            total_costs = total_costs + transaction_cost
            total_turnover = total_turnover + torch.abs(action)
            
            # Store position
            position_history[:, t] = position
        
        # Compute terminal hedging error
        final_prices = prices[:, -1]
        portfolio_value = position * final_prices + cash
        
        # Option payoff
        if self.option_type.lower() == "call":
            option_payoff = torch.clamp(final_prices - self.K, min=0.0)
        else:
            option_payoff = torch.clamp(self.K - final_prices, min=0.0)
        
        hedging_error = option_payoff - portfolio_value
        
        # Compile hedge information
        hedge_info = {
            'hedging_error': hedging_error,
            'total_costs': total_costs,
            'total_turnover': total_turnover,
            'final_position': position,
            'final_cash': cash,
            'final_portfolio_value': portfolio_value,
            'option_payoff': option_payoff,
            'position_history': position_history
        }
        
        return hedging_error, hedge_info


def create_baseline_strategy(strategy_type: str, **kwargs):
    """
    Create a baseline hedging strategy.
    
    Args:
        strategy_type: Type of strategy ("delta", "periodic", "no_hedge", "adaptive_delta")
        **kwargs: Additional arguments
        
    Returns:
        Baseline strategy instance
    """
    if strategy_type.lower() == "delta":
        return DeltaHedgeStrategy(
            K=kwargs.get('K', 100.0),
            T=kwargs.get('T', 1.0),
            r=kwargs.get('r', 0.0),
            sigma=kwargs.get('sigma', 0.2),
            option_type=kwargs.get('option_type', 'call')
        )
    elif strategy_type.lower() == "periodic":
        return PeriodicHedgeStrategy(
            K=kwargs.get('K', 100.0),
            T=kwargs.get('T', 1.0),
            r=kwargs.get('r', 0.0),
            sigma=kwargs.get('sigma', 0.2),
            option_type=kwargs.get('option_type', 'call'),
            rebalance_frequency=kwargs.get('rebalance_frequency', 10)
        )
    elif strategy_type.lower() == "no_hedge":
        return NoHedgeStrategy(
            K=kwargs.get('K', 100.0),
            option_type=kwargs.get('option_type', 'call')
        )
    elif strategy_type.lower() == "adaptive_delta":
        return AdaptiveDeltaHedgeStrategy(
            K=kwargs.get('K', 100.0),
            T=kwargs.get('T', 1.0),
            r=kwargs.get('r', 0.0),
            initial_sigma=kwargs.get('initial_sigma', 0.2),
            option_type=kwargs.get('option_type', 'call'),
            vol_window=kwargs.get('vol_window', 20)
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
