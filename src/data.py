"""
Price path simulation for Deep Hedging models.

Implements Geometric Brownian Motion (GBM) and Heston stochastic volatility models
with vectorized Euler-Maruyama simulation schemes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class GBMGenerator:
    """Geometric Brownian Motion price path generator.
    
    Simulates paths following: dS_t = μS_t dt + σS_t dW_t
    """
    
    def __init__(self, S0: float, mu: float, sigma: float, r: float = 0.0):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        
    def simulate_paths(self, 
                      n_paths: int, 
                      n_steps: int, 
                      dt: float,
                      device: str = 'cpu',
                      seed: Optional[int] = None) -> torch.Tensor:
        """
        Simulate GBM paths using Euler-Maruyama scheme.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            dt: Time step size
            device: Device to run on
            seed: Random seed
            
        Returns:
            Price paths of shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate random increments
        dW = torch.randn(n_paths, n_steps, device=device) * np.sqrt(dt)
        
        # Euler-Maruyama scheme: S_{t+1} = S_t * exp((μ - 0.5σ²)dt + σdW_t)
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dW
        
        # Log returns
        log_returns = drift + diffusion
        
        # Compute log prices
        log_prices = torch.cumsum(log_returns, dim=1)
        
        # Convert to prices and prepend initial price
        prices = torch.exp(log_prices)
        prices = torch.cat([
            torch.ones(n_paths, 1, device=device) * self.S0,
            self.S0 * prices
        ], dim=1)
        
        return prices


class HestonGenerator:
    """Heston stochastic volatility model generator.
    
    Simulates paths following:
    dS_t = μS_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t) dt + σ_v √v_t dW_t^v
    where dW_t^S dW_t^v = ρ dt
    """
    
    def __init__(self, S0: float, v0: float, mu: float, kappa: float, 
                 theta: float, sigma_v: float, rho: float, r: float = 0.0):
        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.r = r
        
    def simulate_paths(self, 
                      n_paths: int, 
                      n_steps: int, 
                      dt: float,
                      device: str = 'cpu',
                      seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate Heston paths using Euler-Maruyama scheme.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            dt: Time step size
            device: Device to run on
            seed: Random seed
            
        Returns:
            Tuple of (price_paths, variance_paths) both of shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Initialize paths
        prices = torch.zeros(n_paths, n_steps + 1, device=device)
        variances = torch.zeros(n_paths, n_steps + 1, device=device)
        
        prices[:, 0] = self.S0
        variances[:, 0] = self.v0
        
        # Generate correlated Brownian motions
        dW1 = torch.randn(n_paths, n_steps, device=device) * np.sqrt(dt)
        dW2 = torch.randn(n_paths, n_steps, device=device) * np.sqrt(dt)
        
        # Correlated increments: dW^S = dW1, dW^v = ρ dW1 + √(1-ρ²) dW2
        dWS = dW1
        dWv = self.rho * dW1 + np.sqrt(1 - self.rho**2) * dW2
        
        # Euler-Maruyama scheme
        for t in range(n_steps):
            # Variance process: dv_t = κ(θ - v_t) dt + σ_v √v_t dW_t^v
            # Ensure variance stays positive (reflection scheme)
            v_t = torch.clamp(variances[:, t], min=1e-8)
            dv = self.kappa * (self.theta - v_t) * dt + self.sigma_v * torch.sqrt(v_t) * dWv[:, t]
            variances[:, t+1] = torch.clamp(v_t + dv, min=1e-8)
            
            # Price process: dS_t = μS_t dt + √v_t S_t dW_t^S
            S_t = prices[:, t]
            sqrt_v = torch.sqrt(variances[:, t+1])
            dS = self.mu * S_t * dt + sqrt_v * S_t * dWS[:, t]
            prices[:, t+1] = S_t + dS
            
        return prices, variances


def compute_realized_volatility(prices: torch.Tensor, window: int = 20) -> torch.Tensor:
    """
    Compute rolling realized volatility from price paths.
    
    Args:
        prices: Price paths of shape (n_paths, n_steps)
        window: Rolling window size
        
    Returns:
        Rolling realized volatility of shape (n_paths, n_steps)
    """
    # Compute log returns
    log_returns = torch.log(prices[:, 1:] / prices[:, :-1])
    
    # Compute rolling standard deviation
    n_paths, n_steps = log_returns.shape
    realized_vol = torch.zeros_like(log_returns)
    
    for t in range(window, n_steps):
        window_returns = log_returns[:, t-window+1:t+1]
        realized_vol[:, t] = torch.std(window_returns, dim=1)
    
    # Fill initial values with first available volatility
    if n_steps > window:
        realized_vol[:, :window] = realized_vol[:, window:window+1]
    
    return realized_vol


def create_state_features(prices: torch.Tensor, 
                         variances: Optional[torch.Tensor] = None,
                         K: float = 100.0,
                         T: float = 1.0,
                         dt: float = 0.01,
                         features: list = None) -> torch.Tensor:
    """
    Create state features for the policy network.
    
    Args:
        prices: Price paths of shape (n_paths, n_steps + 1)
        variances: Variance paths for Heston model (optional)
        K: Strike price
        T: Time to maturity
        dt: Time step size
        features: List of feature names to include
        
    Returns:
        State features of shape (n_paths, n_steps, n_features)
    """
    if features is None:
        features = ["time_normalized", "log_price", "realized_vol", "moneyness", 
                   "time_to_maturity", "past_returns_1", "past_returns_5", "past_returns_10"]
    
    n_paths, n_steps = prices.shape
    n_steps = n_steps - 1  # Exclude initial price
    n_features = len(features)
    
    # Add variance features if Heston model
    if variances is not None:
        if "variance" not in features:
            features.append("variance")
        if "vol_of_vol" not in features:
            features.append("vol_of_vol")
        n_features = len(features)
    
    state = torch.zeros(n_paths, n_steps, n_features, device=prices.device)
    
    # Compute realized volatility
    realized_vol = compute_realized_volatility(prices)
    
    # Compute log returns for different windows
    log_returns = torch.log(prices[:, 1:] / prices[:, :-1])
    
    for t in range(n_steps):
        feature_idx = 0
        
        for feature in features:
            if feature == "time_normalized":
                state[:, t, feature_idx] = t * dt / T
                
            elif feature == "log_price":
                state[:, t, feature_idx] = torch.log(prices[:, t+1])
                
            elif feature == "realized_vol":
                state[:, t, feature_idx] = realized_vol[:, t] if t < realized_vol.shape[1] else realized_vol[:, -1]
                
            elif feature == "moneyness":
                state[:, t, feature_idx] = prices[:, t+1] / K
                
            elif feature == "time_to_maturity":
                state[:, t, feature_idx] = (T - t * dt) / T
                
            elif feature == "past_returns_1":
                if t > 0:
                    state[:, t, feature_idx] = log_returns[:, t-1] if t-1 < log_returns.shape[1] else 0.0
                else:
                    state[:, t, feature_idx] = 0.0
                    
            elif feature == "past_returns_5":
                if t >= 5:
                    window_returns = log_returns[:, t-5:t]
                    state[:, t, feature_idx] = torch.mean(window_returns, dim=1)
                else:
                    state[:, t, feature_idx] = 0.0
                    
            elif feature == "past_returns_10":
                if t >= 10:
                    window_returns = log_returns[:, t-10:t]
                    state[:, t, feature_idx] = torch.mean(window_returns, dim=1)
                else:
                    state[:, t, feature_idx] = 0.0
                    
            elif feature == "variance" and variances is not None:
                state[:, t, feature_idx] = variances[:, t+1]
                
            elif feature == "vol_of_vol" and variances is not None:
                # Approximate vol of vol from variance changes
                if t > 0:
                    vol_of_vol = torch.abs(variances[:, t+1] - variances[:, t]) / dt
                    state[:, t, feature_idx] = vol_of_vol
                else:
                    state[:, t, feature_idx] = 0.0
            
            feature_idx += 1
    
    return state


# Factory function for creating generators
def create_generator(model_type: str, **kwargs):
    """Create a price generator based on model type."""
    if model_type.lower() == "gbm":
        return GBMGenerator(
            S0=kwargs.get('S0', 100.0),
            mu=kwargs.get('mu', 0.0),
            sigma=kwargs.get('sigma', 0.2),
            r=kwargs.get('r', 0.0)
        )
    elif model_type.lower() == "heston":
        return HestonGenerator(
            S0=kwargs.get('S0', 100.0),
            v0=kwargs.get('v0', 0.04),
            mu=kwargs.get('mu', 0.0),
            kappa=kwargs.get('kappa', 2.0),
            theta=kwargs.get('theta', 0.04),
            sigma_v=kwargs.get('sigma_v', 0.3),
            rho=kwargs.get('rho', -0.7),
            r=kwargs.get('r', 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
