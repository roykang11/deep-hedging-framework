"""
Option payoff functions for Deep Hedging.

Implements various option payoffs including European calls, puts, and more exotic structures.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class EuropeanCall:
    """European call option payoff: max(S_T - K, 0)"""
    
    def __init__(self, K: float):
        self.K = K
        
    def __call__(self, S_T: torch.Tensor) -> torch.Tensor:
        """Compute payoff at maturity."""
        return torch.clamp(S_T - self.K, min=0.0)


class EuropeanPut:
    """European put option payoff: max(K - S_T, 0)"""
    
    def __init__(self, K: float):
        self.K = K
        
    def __call__(self, S_T: torch.Tensor) -> torch.Tensor:
        """Compute payoff at maturity."""
        return torch.clamp(self.K - S_T, min=0.0)


class DigitalCall:
    """Digital call option payoff: 1 if S_T > K, 0 otherwise"""
    
    def __init__(self, K: float):
        self.K = K
        
    def __call__(self, S_T: torch.Tensor) -> torch.Tensor:
        """Compute payoff at maturity."""
        return (S_T > self.K).float()


class AsianCall:
    """Asian call option with arithmetic average payoff: max(A_T - K, 0)"""
    
    def __init__(self, K: float):
        self.K = K
        
    def __call__(self, S_paths: torch.Tensor) -> torch.Tensor:
        """Compute payoff using arithmetic average of price path."""
        # S_paths shape: (n_paths, n_steps + 1)
        A_T = torch.mean(S_paths, dim=1)  # Arithmetic average
        return torch.clamp(A_T - self.K, min=0.0)


class LookbackCall:
    """Lookback call option payoff: max(S_T - S_min, 0)"""
    
    def __init__(self):
        pass
        
    def __call__(self, S_paths: torch.Tensor) -> torch.Tensor:
        """Compute payoff using minimum price over path."""
        S_min = torch.min(S_paths, dim=1)[0]  # Minimum price
        S_T = S_paths[:, -1]  # Final price
        return torch.clamp(S_T - S_min, min=0.0)


class BarrierCall:
    """Barrier call option with up-and-out barrier"""
    
    def __init__(self, K: float, B: float):
        self.K = K  # Strike
        self.B = B  # Barrier level
        
    def __call__(self, S_paths: torch.Tensor) -> torch.Tensor:
        """Compute payoff with barrier condition."""
        # Check if barrier is breached
        max_prices = torch.max(S_paths, dim=1)[0]
        barrier_hit = max_prices >= self.B
        
        # Standard call payoff
        S_T = S_paths[:, -1]
        call_payoff = torch.clamp(S_T - self.K, min=0.0)
        
        # Knock out if barrier is hit
        return torch.where(barrier_hit, torch.zeros_like(call_payoff), call_payoff)


class BlackScholesPrice:
    """Black-Scholes option pricing for baseline comparisons"""
    
    def __init__(self, r: float = 0.0):
        self.r = r
        
    def call_price(self, S: torch.Tensor, K: float, T: float, sigma: float) -> torch.Tensor:
        """Compute Black-Scholes call option price."""
        # Handle zero volatility case
        if sigma == 0.0:
            return torch.clamp(S - K * torch.exp(-self.r * T), min=0.0)
        
        # Black-Scholes formula
        d1 = (torch.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)
        
        # Normal CDF approximation (using error function)
        def norm_cdf(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        
        N_d1 = norm_cdf(d1)
        N_d2 = norm_cdf(d2)
        
        price = S * N_d1 - K * torch.exp(-self.r * T) * N_d2
        return price
        
    def call_delta(self, S: torch.Tensor, K: float, T: float, sigma: float) -> torch.Tensor:
        """Compute Black-Scholes call option delta."""
        if sigma == 0.0:
            return (S > K).float()
        
        d1 = (torch.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
        
        def norm_cdf(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        
        N_d1 = norm_cdf(d1)
        return N_d1
        
    def put_price(self, S: torch.Tensor, K: float, T: float, sigma: float) -> torch.Tensor:
        """Compute Black-Scholes put option price."""
        call_p = self.call_price(S, K, T, sigma)
        put_p = call_p - S + K * torch.exp(-self.r * T)
        return put_p
        
    def put_delta(self, S: torch.Tensor, K: float, T: float, sigma: float) -> torch.Tensor:
        """Compute Black-Scholes put option delta."""
        return self.call_delta(S, K, T, sigma) - 1.0


# Factory function for creating payoffs
def create_payoff(payoff_type: str, **kwargs):
    """Create a payoff function based on type."""
    if payoff_type.lower() == "european_call":
        return EuropeanCall(K=kwargs.get('K', 100.0))
    elif payoff_type.lower() == "european_put":
        return EuropeanPut(K=kwargs.get('K', 100.0))
    elif payoff_type.lower() == "digital_call":
        return DigitalCall(K=kwargs.get('K', 100.0))
    elif payoff_type.lower() == "asian_call":
        return AsianCall(K=kwargs.get('K', 100.0))
    elif payoff_type.lower() == "lookback_call":
        return LookbackCall()
    elif payoff_type.lower() == "barrier_call":
        return BarrierCall(K=kwargs.get('K', 100.0), B=kwargs.get('B', 120.0))
    else:
        raise ValueError(f"Unknown payoff type: {payoff_type}")


def compute_implied_volatility(market_price: float, S: float, K: float, T: float, 
                             r: float = 0.0, option_type: str = "call") -> float:
    """
    Compute implied volatility using Newton-Raphson method.
    
    Args:
        market_price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        option_type: "call" or "put"
        
    Returns:
        Implied volatility
    """
    bs = BlackScholesPrice(r=r)
    
    # Initial guess
    sigma = 0.2
    
    for _ in range(50):  # Max iterations
        if option_type.lower() == "call":
            price = bs.call_price(torch.tensor(S), K, T, sigma).item()
            vega = bs.call_vega(torch.tensor(S), K, T, sigma).item()
        else:
            price = bs.put_price(torch.tensor(S), K, T, sigma).item()
            vega = bs.put_vega(torch.tensor(S), K, T, sigma).item()
        
        if abs(vega) < 1e-8:  # Avoid division by zero
            break
            
        # Newton-Raphson update
        sigma = sigma - (price - market_price) / vega
        
        if sigma <= 0:
            sigma = 0.01  # Ensure positive volatility
            
    return sigma


# Extend BlackScholesPrice with vega calculation
class BlackScholesPriceExtended(BlackScholesPrice):
    """Extended Black-Scholes pricing with Greeks"""
    
    def call_vega(self, S: torch.Tensor, K: float, T: float, sigma: float) -> torch.Tensor:
        """Compute Black-Scholes call option vega."""
        if sigma == 0.0:
            return torch.zeros_like(S)
        
        d1 = (torch.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
        
        def norm_pdf(x):
            return torch.exp(-0.5 * x**2) / torch.sqrt(2 * torch.pi)
        
        n_d1 = norm_pdf(d1)
        vega = S * torch.sqrt(T) * n_d1
        return vega
        
    def put_vega(self, S: torch.Tensor, K: float, T: float, sigma: float) -> torch.Tensor:
        """Compute Black-Scholes put option vega."""
        return self.call_vega(S, K, T, sigma)
