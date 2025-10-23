#!/usr/bin/env python3
"""
Simplified Deep Hedging Experiment Script
Generates paper-style results without matplotlib dependencies.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class SimplifiedDeepHedgingExperiment:
    """Simplified Deep Hedging experiment framework."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def simulate_gbm_paths(self, n_paths, n_steps, S0, mu, sigma, T):
        """Simulate Geometric Brownian Motion paths."""
        dt = T / n_steps
        log_returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt, 
            sigma * np.sqrt(dt), 
            (n_paths, n_steps)
        )
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        return prices
    
    def simulate_heston_paths(self, n_paths, n_steps, S0, r, v0, kappa, theta, sigma_v, rho, T):
        """Simulate Heston stochastic volatility paths."""
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
            
            # Update variance (with Feller condition)
            v[:, t+1] = np.maximum(v[:, t] + kappa * (theta - v[:, t]) * dt + 
                                  sigma_v * np.sqrt(v[:, t]) * dW2, 0)
            
            # Update price
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v[:, t]) * dt + 
                                        np.sqrt(v[:, t]) * dW1)
        
        return S[:, 1:], v[:, 1:]  # Return without initial values
    
    def black_scholes_delta(self, S, K, T, r, sigma):
        """Calculate Black-Scholes delta for call option."""
        if sigma == 0:
            return np.where(S > K, 1.0, 0.0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1)
    
    def calculate_cvar(self, losses, alpha=0.95):
        """Calculate Conditional Value at Risk."""
        var = np.percentile(losses, alpha * 100)
        cvar = np.mean(losses[losses >= var])
        return var, cvar
    
    def calculate_entropic_risk(self, losses, lambda_risk):
        """Calculate entropic risk measure."""
        return (1/lambda_risk) * np.log(np.mean(np.exp(lambda_risk * losses)))
    
    def delta_hedging_strategy(self, prices, K, T, r, sigma, kappa):
        """Implement delta hedging strategy with transaction costs."""
        n_paths, n_steps = prices.shape
        dt = T / n_steps
        
        # Initialize portfolio
        cash = np.zeros(n_paths)
        positions = np.zeros(n_paths)
        transaction_costs = np.zeros(n_paths)
        
        for t in range(n_steps):
            time_to_maturity = T - t * dt
            current_prices = prices[:, t]
            
            # Calculate delta
            delta = self.black_scholes_delta(current_prices, K, time_to_maturity, r, sigma)
            
            # Calculate position change
            position_change = delta - positions
            
            # Calculate transaction costs
            costs = kappa * current_prices * np.abs(position_change)
            transaction_costs += costs
            
            # Update positions and cash
            positions = delta
            cash -= current_prices * position_change + costs
        
        # Final payoff
        final_prices = prices[:, -1]
        option_payoff = np.maximum(final_prices - K, 0)
        portfolio_value = cash + positions * final_prices
        
        hedging_error = option_payoff - portfolio_value
        return hedging_error, transaction_costs
    
    def periodic_hedging_strategy(self, prices, K, T, r, sigma, kappa, rebalance_freq=10):
        """Implement periodic hedging strategy."""
        n_paths, n_steps = prices.shape
        dt = T / n_steps
        
        # Initialize portfolio
        cash = np.zeros(n_paths)
        positions = np.zeros(n_paths)
        transaction_costs = np.zeros(n_paths)
        
        for t in range(0, n_steps, rebalance_freq):
            time_to_maturity = T - t * dt
            current_prices = prices[:, t]
            
            # Calculate delta
            delta = self.black_scholes_delta(current_prices, K, time_to_maturity, r, sigma)
            
            # Calculate position change
            position_change = delta - positions
            
            # Calculate transaction costs
            costs = kappa * current_prices * np.abs(position_change)
            transaction_costs += costs
            
            # Update positions and cash
            positions = delta
            cash -= current_prices * position_change + costs
        
        # Final payoff
        final_prices = prices[:, -1]
        option_payoff = np.maximum(final_prices - K, 0)
        portfolio_value = cash + positions * final_prices
        
        hedging_error = option_payoff - portfolio_value
        return hedging_error, transaction_costs
    
    def no_hedge_strategy(self, prices, K):
        """No hedging strategy (cash only)."""
        final_prices = prices[:, -1]
        option_payoff = np.maximum(final_prices - K, 0)
        hedging_error = option_payoff  # No hedging, so error is just the payoff
        return hedging_error, np.zeros_like(hedging_error)
    
    def simple_neural_hedging(self, prices, K, T, r, sigma, kappa, hidden_dim=64):
        """Simplified neural hedging strategy using basic features."""
        n_paths, n_steps = prices.shape
        dt = T / n_steps
        
        # Simple neural network weights (randomly initialized)
        np.random.seed(42)
        W1 = np.random.randn(8, hidden_dim) * 0.1
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * 0.1
        b2 = np.zeros(1)
        
        # Initialize portfolio
        cash = np.zeros(n_paths)
        positions = np.zeros(n_paths)
        transaction_costs = np.zeros(n_paths)
        
        for t in range(n_steps):
            time_to_maturity = T - t * dt
            current_prices = prices[:, t]
            
            # Create features
            features = np.column_stack([
                np.full(n_paths, t / n_steps),  # Normalized time
                np.log(current_prices),  # Log price
                np.std(prices[:, max(0, t-10):t+1], axis=1) if t > 0 else np.zeros(n_paths),  # Rolling volatility
                current_prices / K,  # Moneyness
                np.full(n_paths, time_to_maturity / T),  # Normalized time to maturity
                (prices[:, t] - prices[:, max(0, t-1)]) / prices[:, max(0, t-1)] if t > 0 else np.zeros(n_paths),  # Recent return
                (np.mean(prices[:, max(0, t-5):t+1], axis=1) / current_prices - 1) if t >= 5 else np.zeros(n_paths),  # 5-step return
                (np.mean(prices[:, max(0, t-10):t+1], axis=1) / current_prices - 1) if t >= 10 else np.zeros(n_paths)  # 10-step return
            ])
            
            # Neural network forward pass
            h = np.tanh(features @ W1 + b1)
            delta = np.tanh(h @ W2 + b2).flatten()
            delta = np.clip(delta, -1.5, 1.5)  # Position limits
            
            # Calculate position change
            position_change = delta - positions
            
            # Calculate transaction costs
            costs = kappa * current_prices * np.abs(position_change)
            transaction_costs += costs
            
            # Update positions and cash
            positions = delta
            cash -= current_prices * position_change + costs
        
        # Final payoff
        final_prices = prices[:, -1]
        option_payoff = np.maximum(final_prices - K, 0)
        portfolio_value = cash + positions * final_prices
        
        hedging_error = option_payoff - portfolio_value
        return hedging_error, transaction_costs
    
    def run_experiment(self, model_type='gbm', n_paths=10000, n_steps=100):
        """Run comprehensive hedging experiment."""
        print(f"Running {model_type.upper()} experiment with {n_paths:,} paths...")
        
        # Model parameters
        if model_type == 'gbm':
            S0, K, T, r, mu, sigma = 100, 100, 1.0, 0.0, 0.0, 0.2
            prices = self.simulate_gbm_paths(n_paths, n_steps, S0, mu, sigma, T)
        elif model_type == 'heston':
            S0, K, T, r = 100, 100, 1.0, 0.0
            v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.3, -0.7
            prices, _ = self.simulate_heston_paths(n_paths, n_steps, S0, r, v0, kappa, theta, sigma_v, rho, T)
            sigma = np.sqrt(theta)  # Use long-term volatility for Black-Scholes
        else:
            raise ValueError("Model type must be 'gbm' or 'heston'")
        
        # Transaction cost parameter
        kappa = 0.001  # 10 basis points
        
        # Run different hedging strategies
        strategies = {}
        
        print("  Running delta hedging...")
        delta_errors, delta_costs = self.delta_hedging_strategy(prices, K, T, r, sigma, kappa)
        strategies['Delta Hedging'] = {'errors': delta_errors, 'costs': delta_costs}
        
        print("  Running periodic hedging...")
        periodic_errors, periodic_costs = self.periodic_hedging_strategy(prices, K, T, r, sigma, kappa, rebalance_freq=10)
        strategies['Periodic Hedging'] = {'errors': periodic_errors, 'costs': periodic_costs}
        
        print("  Running no hedge...")
        no_hedge_errors, no_hedge_costs = self.no_hedge_strategy(prices, K)
        strategies['No Hedge'] = {'errors': no_hedge_errors, 'costs': no_hedge_costs}
        
        print("  Running neural hedging...")
        neural_errors, neural_costs = self.simple_neural_hedging(prices, K, T, r, sigma, kappa)
        strategies['Neural Hedging'] = {'errors': neural_errors, 'costs': neural_costs}
        
        # Calculate performance metrics
        results = {}
        for name, data in strategies.items():
            errors = data['errors']
            costs = data['costs']
            
            # Risk measures
            var_95, cvar_95 = self.calculate_cvar(errors, 0.95)
            var_99, cvar_99 = self.calculate_cvar(errors, 0.99)
            entropic_risk = self.calculate_entropic_risk(errors, 10.0)
            
            results[name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'var_95': var_95,
                'cvar_95': cvar_95,
                'var_99': var_99,
                'cvar_99': cvar_99,
                'entropic_risk': entropic_risk,
                'mean_costs': np.mean(costs),
                'total_costs': np.sum(costs),
                'errors': errors,
                'costs': costs
            }
        
        self.results[model_type] = results
        return results
    
    def generate_results_table(self, save_dir='results'):
        """Generate comprehensive results table."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        all_results = []
        for model_type, results in self.results.items():
            for strategy, metrics in results.items():
                all_results.append({
                    'Model': model_type.upper(),
                    'Strategy': strategy,
                    'Mean Error': f"{metrics['mean_error']:.6f}",
                    'Std Error': f"{metrics['std_error']:.6f}",
                    'VaR 95%': f"{metrics['var_95']:.6f}",
                    'CVaR 95%': f"{metrics['cvar_95']:.6f}",
                    'VaR 99%': f"{metrics['var_99']:.6f}",
                    'CVaR 99%': f"{metrics['cvar_99']:.6f}",
                    'Entropic Risk': f"{metrics['entropic_risk']:.6f}",
                    'Mean Costs': f"{metrics['mean_costs']:.6f}",
                    'Total Costs': f"{metrics['total_costs']:.2f}"
                })
        
        df = pd.DataFrame(all_results)
        df.to_csv(f'{save_dir}/results_table.csv', index=False)
        
        # Print formatted table
        print("\n" + "="*120)
        print("COMPREHENSIVE RESULTS TABLE")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        return df

def main():
    """Main experiment function."""
    print("Deep Hedging: Comprehensive Experiment Suite")
    print("="*60)
    
    # Initialize experiment
    config = {'n_paths': 10000, 'n_steps': 100}
    experiment = SimplifiedDeepHedgingExperiment(config)
    
    # Run GBM experiment
    print("\n1. Running GBM Model Experiment...")
    gbm_results = experiment.run_experiment('gbm', n_paths=10000, n_steps=100)
    
    # Run Heston experiment
    print("\n2. Running Heston Model Experiment...")
    heston_results = experiment.run_experiment('heston', n_paths=10000, n_steps=100)
    
    # Generate results table
    print("\n3. Generating results table...")
    df = experiment.generate_results_table('results')
    
    print(f"\n✅ Experiment completed! Results saved to 'results/' directory")
    print("   - results_table.csv: Comprehensive results table")
    
    return experiment

if __name__ == "__main__":
    experiment = main()
