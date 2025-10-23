#!/usr/bin/env python3
"""
Advanced Risk Measures and Optimization for Deep Hedging
Implements sophisticated risk measures, robust optimization, and risk-sensitive learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
import math

class AdvancedRiskMeasures:
    """Advanced risk measures for financial optimization."""
    
    @staticmethod
    def cvar(losses, alpha=0.95):
        """Conditional Value at Risk (Expected Shortfall)."""
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)
        var_idx = int(alpha * n)
        cvar = np.mean(sorted_losses[var_idx:])
        return cvar
    
    @staticmethod
    def entropic_risk(losses, lambda_risk=10.0):
        """Entropic risk measure."""
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        return (1.0 / lambda_risk) * np.log(np.mean(np.exp(lambda_risk * losses)))
    
    @staticmethod
    def spectral_risk(losses, weights):
        """Spectral risk measure with custom weights."""
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        sorted_losses = np.sort(losses)
        return np.sum(sorted_losses * weights)
    
    @staticmethod
    def distortion_risk(losses, distortion_func):
        """Distortion risk measure with custom distortion function."""
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)
        weights = np.array([distortion_func((i+1)/n) - distortion_func(i/n) 
                           for i in range(n)])
        return np.sum(sorted_losses * weights)
    
    @staticmethod
    def wang_transform(losses, lambda_param=0.5):
        """Wang transform risk measure."""
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        from scipy.stats import norm
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)
        
        # Wang transform weights
        u = np.arange(1, n+1) / (n+1)
        wang_weights = norm.cdf(norm.ppf(u) + lambda_param)
        wang_weights = np.diff(np.concatenate([[0], wang_weights]))
        
        return np.sum(sorted_losses * wang_weights)
    
    @staticmethod
    def gini_coefficient(losses):
        """Gini coefficient for inequality measurement."""
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)
        cumsum = np.cumsum(sorted_losses)
        
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return gini

class RobustOptimizer:
    """Robust optimization with uncertainty sets."""
    
    def __init__(self, risk_measure='cvar', uncertainty_type='ellipsoidal'):
        self.risk_measure = risk_measure
        self.uncertainty_type = uncertainty_type
    
    def robust_optimize(self, objective_func, constraints, uncertainty_params):
        """Robust optimization with uncertainty sets."""
        if self.uncertainty_type == 'ellipsoidal':
            return self._ellipsoidal_optimization(objective_func, constraints, uncertainty_params)
        elif self.uncertainty_type == 'box':
            return self._box_optimization(objective_func, constraints, uncertainty_params)
        else:
            raise ValueError(f"Unknown uncertainty type: {self.uncertainty_type}")
    
    def _ellipsoidal_optimization(self, objective_func, constraints, uncertainty_params):
        """Ellipsoidal uncertainty set optimization."""
        # Implementation of ellipsoidal robust optimization
        # This is a simplified version - full implementation would be more complex
        pass
    
    def _box_optimization(self, objective_func, constraints, uncertainty_params):
        """Box uncertainty set optimization."""
        # Implementation of box robust optimization
        pass

class RiskSensitiveLoss(nn.Module):
    """Advanced risk-sensitive loss function."""
    
    def __init__(self, risk_type='cvar', alpha=0.95, lambda_risk=10.0, 
                 risk_aversion=1.0, tail_focus=0.1):
        super().__init__()
        self.risk_type = risk_type
        self.alpha = alpha
        self.lambda_risk = lambda_risk
        self.risk_aversion = risk_aversion
        self.tail_focus = tail_focus
    
    def forward(self, losses):
        """Compute risk-sensitive loss."""
        if self.risk_type == 'cvar':
            return self._cvar_loss(losses)
        elif self.risk_type == 'entropic':
            return self._entropic_loss(losses)
        elif self.risk_type == 'spectral':
            return self._spectral_loss(losses)
        elif self.risk_type == 'distortion':
            return self._distortion_loss(losses)
        elif self.risk_type == 'wang':
            return self._wang_loss(losses)
        else:
            return torch.mean(losses ** 2)
    
    def _cvar_loss(self, losses):
        """CVaR loss with tail focus."""
        sorted_losses, _ = torch.sort(losses, descending=True)
        n = len(sorted_losses)
        var_idx = int(self.alpha * n)
        
        # Focus on tail losses
        tail_losses = sorted_losses[:var_idx+1]
        cvar = torch.mean(tail_losses)
        
        # Add tail focus penalty
        tail_penalty = self.tail_focus * torch.mean(torch.relu(sorted_losses - cvar))
        
        return cvar + tail_penalty
    
    def _entropic_loss(self, losses):
        """Entropic risk loss."""
        return (1.0 / self.lambda_risk) * torch.log(torch.mean(torch.exp(self.lambda_risk * losses)))
    
    def _spectral_loss(self, losses):
        """Spectral risk loss with custom weights."""
        sorted_losses, _ = torch.sort(losses, descending=True)
        n = len(sorted_losses)
        
        # Generate spectral weights (example: exponential decay)
        weights = torch.exp(-torch.arange(n, dtype=torch.float32) / (n * 0.1))
        weights = weights / torch.sum(weights)
        
        return torch.sum(sorted_losses * weights)
    
    def _distortion_loss(self, losses):
        """Distortion risk loss."""
        sorted_losses, _ = torch.sort(losses, descending=True)
        n = len(sorted_losses)
        
        # Wang distortion function
        u = torch.arange(1, n+1, dtype=torch.float32) / (n+1)
        distortion_weights = torch.distributions.Normal(0, 1).cdf(
            torch.distributions.Normal(0, 1).icdf(u) + 0.5
        )
        distortion_weights = torch.diff(torch.cat([torch.zeros(1), distortion_weights]))
        
        return torch.sum(sorted_losses * distortion_weights)
    
    def _wang_loss(self, losses):
        """Wang transform loss."""
        sorted_losses, _ = torch.sort(losses, descending=True)
        n = len(sorted_losses)
        
        u = torch.arange(1, n+1, dtype=torch.float32) / (n+1)
        wang_weights = torch.distributions.Normal(0, 1).cdf(
            torch.distributions.Normal(0, 1).icdf(u) + 0.5
        )
        wang_weights = torch.diff(torch.cat([torch.zeros(1), wang_weights]))
        
        return torch.sum(sorted_losses * wang_weights)

class MultiObjectiveOptimizer:
    """Multi-objective optimization for risk-return trade-offs."""
    
    def __init__(self, objectives, weights=None):
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
    
    def optimize(self, model, data_loader, epochs=100):
        """Multi-objective optimization."""
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Compute all objectives
                objective_values = []
                for obj_func in self.objectives:
                    obj_value = obj_func(model, batch)
                    objective_values.append(obj_value)
                
                # Weighted sum
                weighted_loss = sum(w * obj for w, obj in zip(self.weights, objective_values))
                
                weighted_loss.backward()
                optimizer.step()
                
                total_loss += weighted_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

class RegimeAwareRisk:
    """Regime-aware risk measures that adapt to market conditions."""
    
    def __init__(self, regime_detector, risk_measures_per_regime):
        self.regime_detector = regime_detector
        self.risk_measures = risk_measures_per_regime
    
    def compute_risk(self, losses, market_features):
        """Compute regime-aware risk measure."""
        regime = self.regime_detector.detect_regime(market_features)
        risk_measure = self.risk_measures[regime]
        
        return risk_measure.compute(losses)
    
    def update_regime_models(self, new_data):
        """Update regime detection models with new data."""
        self.regime_detector.update(new_data)

class DynamicRiskBudgeting:
    """Dynamic risk budgeting across different risk factors."""
    
    def __init__(self, risk_factors, budget_weights):
        self.risk_factors = risk_factors
        self.budget_weights = budget_weights
    
    def compute_budgeted_risk(self, losses, factor_exposures):
        """Compute risk with dynamic budgeting."""
        total_risk = 0
        
        for i, factor in enumerate(self.risk_factors):
            factor_losses = losses * factor_exposures[:, i]
            factor_risk = factor.compute_risk(factor_losses)
            total_risk += self.budget_weights[i] * factor_risk
        
        return total_risk
    
    def update_budget_weights(self, performance_history):
        """Update budget weights based on performance."""
        # Implement dynamic weight updating based on performance
        pass

class StressTesting:
    """Stress testing framework for model robustness."""
    
    def __init__(self, stress_scenarios):
        self.stress_scenarios = stress_scenarios
    
    def run_stress_tests(self, model, data_loader):
        """Run stress tests on the model."""
        results = {}
        
        for scenario_name, scenario_func in self.stress_scenarios.items():
            stressed_data = scenario_func(data_loader)
            scenario_results = self._evaluate_scenario(model, stressed_data)
            results[scenario_name] = scenario_results
        
        return results
    
    def _evaluate_scenario(self, model, data_loader):
        """Evaluate model performance under stress scenario."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Forward pass
                output = model(batch['features'])
                
                # Compute loss
                loss = self._compute_stress_loss(output, batch)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _compute_stress_loss(self, output, batch):
        """Compute loss under stress conditions."""
        # Implementation depends on specific stress scenario
        return torch.mean(output ** 2)

class RiskParityOptimizer:
    """Risk parity optimization for balanced risk allocation."""
    
    def __init__(self, risk_measures):
        self.risk_measures = risk_measures
    
    def optimize_weights(self, returns, target_risk=0.1):
        """Optimize weights for risk parity."""
        n_assets = returns.shape[1]
        
        def objective(weights):
            portfolio_returns = torch.sum(returns * weights, dim=1)
            portfolio_risk = self._compute_portfolio_risk(portfolio_returns)
            
            # Risk parity objective: minimize deviation from equal risk contribution
            risk_contributions = self._compute_risk_contributions(returns, weights)
            risk_parity_penalty = torch.sum((risk_contributions - 1/n_assets) ** 2)
            
            return portfolio_risk + 10 * risk_parity_penalty
        
        # Initialize weights
        weights = torch.ones(n_assets) / n_assets
        weights.requires_grad = True
        
        optimizer = torch.optim.Adam([weights], lr=0.01)
        
        for _ in range(1000):
            optimizer.zero_grad()
            loss = objective(weights)
            loss.backward()
            optimizer.step()
            
            # Project to simplex
            weights.data = torch.softmax(weights.data, dim=0)
        
        return weights.detach()
    
    def _compute_portfolio_risk(self, portfolio_returns):
        """Compute portfolio risk."""
        return torch.std(portfolio_returns)
    
    def _compute_risk_contributions(self, returns, weights):
        """Compute risk contributions of each asset."""
        portfolio_returns = torch.sum(returns * weights, dim=1)
        portfolio_risk = torch.std(portfolio_returns)
        
        risk_contributions = []
        for i in range(returns.shape[1]):
            # Compute marginal risk contribution
            marginal_risk = torch.autograd.grad(
                portfolio_risk, weights, create_graph=True
            )[0][i]
            risk_contributions.append(weights[i] * marginal_risk)
        
        risk_contributions = torch.stack(risk_contributions)
        return risk_contributions / torch.sum(risk_contributions)

class AdvancedRiskMetrics:
    """Advanced risk metrics for comprehensive analysis."""
    
    @staticmethod
    def max_drawdown(returns):
        """Compute maximum drawdown."""
        cumulative = torch.cumsum(returns, dim=0)
        running_max = torch.cummax(cumulative, dim=0)[0]
        drawdown = cumulative - running_max
        return torch.min(drawdown)
    
    @staticmethod
    def calmar_ratio(returns, risk_free_rate=0.0):
        """Compute Calmar ratio."""
        annual_return = torch.mean(returns) * 252
        max_dd = AdvancedRiskMetrics.max_drawdown(returns)
        return (annual_return - risk_free_rate) / torch.abs(max_dd)
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0):
        """Compute Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        return torch.mean(excess_returns) / torch.std(excess_returns) * math.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.0):
        """Compute Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = torch.where(excess_returns < 0, excess_returns, 0)
        downside_deviation = torch.std(downside_returns)
        return torch.mean(excess_returns) / downside_deviation * math.sqrt(252)
    
    @staticmethod
    def omega_ratio(returns, threshold=0.0):
        """Compute Omega ratio."""
        excess_returns = returns - threshold
        positive_returns = torch.where(excess_returns > 0, excess_returns, 0)
        negative_returns = torch.where(excess_returns < 0, -excess_returns, 0)
        
        return torch.sum(positive_returns) / torch.sum(negative_returns)
    
    @staticmethod
    def tail_ratio(returns, threshold=0.05):
        """Compute tail ratio (95th percentile / 5th percentile)."""
        upper_tail = torch.quantile(returns, 1 - threshold)
        lower_tail = torch.quantile(returns, threshold)
        return upper_tail / torch.abs(lower_tail)
    
    @staticmethod
    def value_at_risk(returns, confidence_level=0.05):
        """Compute Value at Risk."""
        return torch.quantile(returns, confidence_level)
    
    @staticmethod
    def expected_shortfall(returns, confidence_level=0.05):
        """Compute Expected Shortfall (CVaR)."""
        var = AdvancedRiskMetrics.value_at_risk(returns, confidence_level)
        tail_returns = returns[returns <= var]
        return torch.mean(tail_returns)
