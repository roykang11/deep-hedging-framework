"""
Risk measures for Deep Hedging optimization.

Implements CVaR (Conditional Value at Risk) and entropic risk measures
for robust hedging policy learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class CVaRLoss(nn.Module):
    """
    Conditional Value at Risk (CVaR) loss function.
    
    CVaR_α(L) = min_τ [τ + (1-α)^(-1) E[(L-τ)_+]]
    
    This is implemented as a learnable surrogate function where τ is optimized
    jointly with the policy parameters.
    """
    
    def __init__(self, alpha: float = 0.95, learnable_tau: bool = True):
        """
        Initialize CVaR loss function.
        
        Args:
            alpha: Confidence level (e.g., 0.95 for 95% CVaR)
            learnable_tau: Whether to make τ a learnable parameter
        """
        super().__init__()
        
        self.alpha = alpha
        self.learnable_tau = learnable_tau
        
        if learnable_tau:
            # Initialize τ as a learnable parameter
            self.tau = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('tau', torch.tensor(0.0))
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR loss.
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            CVaR loss value
        """
        # Ensure tau is positive for stability
        tau = torch.clamp(self.tau, min=-10.0, max=10.0)
        
        # Compute excess losses: (L - τ)_+
        excess = torch.clamp(losses - tau, min=0.0)
        
        # CVaR surrogate: τ + (1-α)^(-1) * E[(L-τ)_+]
        cvar_loss = tau + excess.mean() / (1.0 - self.alpha)
        
        return cvar_loss
    
    def get_cvar_value(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute actual CVaR value (not the surrogate).
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            CVaR value
        """
        # Compute Value at Risk (VaR)
        var = torch.quantile(losses, self.alpha)
        
        # Compute Expected Shortfall (CVaR)
        excess = torch.clamp(losses - var, min=0.0)
        cvar = var + excess.mean() / (1.0 - self.alpha)
        
        return cvar


class EntropicRisk(nn.Module):
    """
    Entropic risk measure.
    
    ρ_λ(L) = λ^(-1) log E[exp(λL)]
    
    Provides smooth gradients and is coherent.
    """
    
    def __init__(self, lambda_risk: float = 10.0):
        """
        Initialize entropic risk measure.
        
        Args:
            lambda_risk: Risk aversion parameter (higher = more risk averse)
        """
        super().__init__()
        
        self.lambda_risk = lambda_risk
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic risk.
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            Entropic risk value
        """
        # Compute exponential: exp(λL)
        exp_losses = torch.exp(self.lambda_risk * losses)
        
        # Compute expectation and take log
        mean_exp = exp_losses.mean()
        
        # Avoid numerical issues
        mean_exp = torch.clamp(mean_exp, min=1e-8, max=1e8)
        
        # Entropic risk: λ^(-1) log E[exp(λL)]
        entropic_risk = torch.log(mean_exp) / self.lambda_risk
        
        return entropic_risk


class SpectralRisk(nn.Module):
    """
    Spectral risk measure (generalization of CVaR).
    
    ρ_φ(L) = ∫₀¹ φ(u) * F^(-1)(u) du
    
    where φ is a spectral function and F^(-1) is the quantile function.
    """
    
    def __init__(self, spectral_weights: torch.Tensor):
        """
        Initialize spectral risk measure.
        
        Args:
            spectral_weights: Spectral function weights φ(u_i)
        """
        super().__init__()
        
        self.spectral_weights = nn.Parameter(spectral_weights)
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral risk.
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            Spectral risk value
        """
        # Sort losses to get quantiles
        sorted_losses, _ = torch.sort(losses)
        
        # Normalize spectral weights
        weights = F.softmax(self.spectral_weights, dim=0)
        
        # Compute weighted sum of quantiles
        spectral_risk = torch.sum(weights * sorted_losses)
        
        return spectral_risk


class ExpectedShortfall(nn.Module):
    """
    Expected Shortfall (ES) risk measure.
    
    ES_α(L) = E[L | L ≥ VaR_α(L)]
    
    This is the same as CVaR but computed differently.
    """
    
    def __init__(self, alpha: float = 0.95):
        """
        Initialize Expected Shortfall.
        
        Args:
            alpha: Confidence level
        """
        super().__init__()
        
        self.alpha = alpha
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute Expected Shortfall.
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            Expected Shortfall value
        """
        # Compute Value at Risk
        var = torch.quantile(losses, self.alpha)
        
        # Compute conditional expectation
        tail_losses = losses[losses >= var]
        
        if len(tail_losses) > 0:
            es = tail_losses.mean()
        else:
            es = var
        
        return es


class RiskMeasure(nn.Module):
    """
    Unified risk measure class that can switch between different risk measures.
    """
    
    def __init__(self, risk_type: str = "cvar", **kwargs):
        """
        Initialize risk measure.
        
        Args:
            risk_type: Type of risk measure ("cvar", "entropic", "expected_shortfall")
            **kwargs: Additional arguments for specific risk measures
        """
        super().__init__()
        
        self.risk_type = risk_type.lower()
        
        if self.risk_type == "cvar":
            self.risk_measure = CVaRLoss(
                alpha=kwargs.get('alpha', 0.95),
                learnable_tau=kwargs.get('learnable_tau', True)
            )
        elif self.risk_type == "entropic":
            self.risk_measure = EntropicRisk(
                lambda_risk=kwargs.get('lambda_risk', 10.0)
            )
        elif self.risk_type == "expected_shortfall":
            self.risk_measure = ExpectedShortfall(
                alpha=kwargs.get('alpha', 0.95)
            )
        elif self.risk_type == "spectral":
            spectral_weights = kwargs.get('spectral_weights', 
                                        torch.ones(10) / 10.0)
            self.risk_measure = SpectralRisk(spectral_weights)
        else:
            raise ValueError(f"Unknown risk type: {risk_type}")
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute risk measure.
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            Risk measure value
        """
        return self.risk_measure(losses)
    
    def get_risk_value(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Get the actual risk value (not surrogate for CVaR).
        
        Args:
            losses: Terminal hedging errors of shape (batch_size,)
            
        Returns:
            Risk measure value
        """
        if hasattr(self.risk_measure, 'get_cvar_value'):
            return self.risk_measure.get_cvar_value(losses)
        else:
            return self.forward(losses)


class RegularizedLoss(nn.Module):
    """
    Combined loss function with risk measure and regularization terms.
    """
    
    def __init__(self, 
                 risk_measure: RiskMeasure,
                 turnover_penalty: float = 1e-4,
                 position_penalty: float = 0.0,
                 cost_penalty: float = 0.0):
        """
        Initialize regularized loss function.
        
        Args:
            risk_measure: Risk measure to use
            turnover_penalty: Penalty on trading activity
            position_penalty: Penalty on position size
            cost_penalty: Penalty on transaction costs
        """
        super().__init__()
        
        self.risk_measure = risk_measure
        self.turnover_penalty = turnover_penalty
        self.position_penalty = position_penalty
        self.cost_penalty = cost_penalty
    
    def forward(self, 
                losses: torch.Tensor,
                total_turnover: Optional[torch.Tensor] = None,
                total_costs: Optional[torch.Tensor] = None,
                final_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute regularized loss.
        
        Args:
            losses: Terminal hedging errors
            total_turnover: Total trading volume
            total_costs: Total transaction costs
            final_positions: Final positions
            
        Returns:
            Regularized loss value
        """
        # Base risk measure
        risk_loss = self.risk_measure(losses)
        
        # Regularization terms
        reg_loss = 0.0
        
        if self.turnover_penalty > 0 and total_turnover is not None:
            reg_loss += self.turnover_penalty * total_turnover.mean()
        
        if self.position_penalty > 0 and final_positions is not None:
            reg_loss += self.position_penalty * torch.mean(final_positions**2)
        
        if self.cost_penalty > 0 and total_costs is not None:
            reg_loss += self.cost_penalty * total_costs.mean()
        
        return risk_loss + reg_loss


def compute_risk_metrics(losses: torch.Tensor, alpha: float = 0.95) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics.
    
    Args:
        losses: Terminal hedging errors
        alpha: Confidence level for VaR/CVaR
        
    Returns:
        Dictionary of risk metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = losses.mean().item()
    metrics['std'] = losses.std().item()
    metrics['min'] = losses.min().item()
    metrics['max'] = losses.max().item()
    
    # Risk measures
    metrics['var_95'] = torch.quantile(losses, 0.95).item()
    metrics['var_99'] = torch.quantile(losses, 0.99).item()
    
    # CVaR (Expected Shortfall)
    var_alpha = torch.quantile(losses, alpha)
    excess = torch.clamp(losses - var_alpha, min=0.0)
    metrics[f'cvar_{int(alpha*100)}'] = (var_alpha + excess.mean() / (1 - alpha)).item()
    
    # Higher moments
    metrics['skewness'] = torch.mean(((losses - losses.mean()) / losses.std())**3).item()
    metrics['kurtosis'] = torch.mean(((losses - losses.mean()) / losses.std())**4).item()
    
    # Tail risk metrics
    metrics['tail_expectation_95'] = losses[losses >= metrics['var_95']].mean().item() if len(losses[losses >= metrics['var_95']]) > 0 else metrics['var_95']
    metrics['tail_expectation_99'] = losses[losses >= metrics['var_99']].mean().item() if len(losses[losses >= metrics['var_99']]) > 0 else metrics['var_99']
    
    return metrics


def create_risk_measure(risk_type: str, **kwargs) -> RiskMeasure:
    """
    Create a risk measure based on type.
    
    Args:
        risk_type: Type of risk measure
        **kwargs: Additional arguments
        
    Returns:
        Risk measure instance
    """
    return RiskMeasure(risk_type, **kwargs)
