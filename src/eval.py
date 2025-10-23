"""
Evaluation and backtesting framework for Deep Hedging models.

Implements comprehensive evaluation metrics, visualization tools, and backtesting
capabilities for assessing hedging performance.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .data import create_generator, create_state_features
from .payoffs import create_payoff
from .models import create_policy
from .env import VectorizedHedgingEnv
from .baselines import create_baseline_strategy


class HedgingEvaluator:
    """
    Comprehensive evaluation framework for hedging strategies.
    """
    
    def __init__(self, config, device: str = 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = device
        
        # Setup components
        self._setup_data_generator()
        self._setup_payoff_function()
        self._setup_environment()
        
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
        
    def _setup_environment(self):
        """Setup hedging environment."""
        self.env = VectorizedHedgingEnv(
            payoff_func=self.payoff_func,
            kappa=self.config.costs.kappa,
            kappa_quad=self.config.costs.kappa_quad,
            r=self.config.model.r,
            dt=self.config.simulation.dt
        )
    
    def evaluate_policy(self, 
                       policy: nn.Module,
                       n_paths: int = 10000,
                       seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate a hedging policy.
        
        Args:
            policy: Trained policy network
            n_paths: Number of evaluation paths
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation results
        """
        policy.eval()
        
        with torch.no_grad():
            # Generate evaluation paths
            if self.config.model.type.lower() == "gbm":
                prices = self.generator.simulate_paths(
                    n_paths=n_paths,
                    n_steps=self.n_steps,
                    dt=self.dt,
                    device=self.device,
                    seed=seed
                )
                variances = None
                
            elif self.config.model.type.lower() == "heston":
                prices, variances = self.generator.simulate_paths(
                    n_paths=n_paths,
                    n_steps=self.n_steps,
                    dt=self.dt,
                    device=self.device,
                    seed=seed
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
            
            # Run policy
            hedging_error, rollout_info = self.env.rollout(
                policy=policy,
                prices=prices,
                state_features=state_features,
                q_max=self.config.network.q_max
            )
            
            # Compile results
            results = {
                'hedging_error': hedging_error,
                'total_costs': rollout_info['total_costs'],
                'total_turnover': rollout_info['total_turnover'],
                'final_position': rollout_info['final_position'],
                'final_cash': rollout_info['final_cash'],
                'final_portfolio_value': rollout_info['final_portfolio_value'],
                'option_payoff': rollout_info['option_payoff'],
                'position_history': rollout_info['position_history'],
                'prices': prices,
                'state_features': state_features
            }
            
            if variances is not None:
                results['variances'] = variances
                
            return results
    
    def evaluate_baseline(self, 
                         strategy_type: str,
                         n_paths: int = 10000,
                         seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate a baseline strategy.
        
        Args:
            strategy_type: Type of baseline strategy
            n_paths: Number of evaluation paths
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation results
        """
        # Generate evaluation paths
        if self.config.model.type.lower() == "gbm":
            prices = self.generator.simulate_paths(
                n_paths=n_paths,
                n_steps=self.n_steps,
                dt=self.dt,
                device=self.device,
                seed=seed
            )
            
        elif self.config.model.type.lower() == "heston":
            prices, variances = self.generator.simulate_paths(
                n_paths=n_paths,
                n_steps=self.n_steps,
                dt=self.dt,
                device=self.device,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model.type}")
        
        # Create baseline strategy
        strategy = create_baseline_strategy(
            strategy_type=strategy_type,
            K=self.config.model.K,
            T=self.config.model.T,
            r=self.config.model.r,
            sigma=self.config.model.sigma,
            option_type='call'
        )
        
        # Run strategy
        hedging_error, hedge_info = strategy.hedge(
            prices=prices,
            dt=self.dt,
            kappa=self.config.costs.kappa,
            kappa_quad=self.config.costs.kappa_quad,
            r=self.config.model.r
        )
        
        # Compile results
        results = {
            'hedging_error': hedging_error,
            'total_costs': hedge_info['total_costs'],
            'total_turnover': hedge_info['total_turnover'],
            'final_position': hedge_info['final_position'],
            'final_cash': hedge_info['final_cash'],
            'final_portfolio_value': hedge_info['final_portfolio_value'],
            'option_payoff': hedge_info['option_payoff'],
            'position_history': hedge_info['position_history'],
            'prices': prices
        }
        
        if self.config.model.type.lower() == "heston":
            results['variances'] = variances
            
        return results
    
    def compute_performance_metrics(self, results: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            results: Evaluation results
            
        Returns:
            Dictionary of performance metrics
        """
        hedging_error = results['hedging_error']
        total_costs = results['total_costs']
        total_turnover = results['total_turnover']
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_error'] = hedging_error.mean().item()
        metrics['std_error'] = hedging_error.std().item()
        metrics['min_error'] = hedging_error.min().item()
        metrics['max_error'] = hedging_error.max().item()
        
        # Risk measures
        metrics['var_95'] = torch.quantile(hedging_error, 0.95).item()
        metrics['var_99'] = torch.quantile(hedging_error, 0.99).item()
        
        # CVaR (Expected Shortfall)
        for alpha in [0.95, 0.99]:
            var_alpha = torch.quantile(hedging_error, alpha)
            excess = torch.clamp(hedging_error - var_alpha, min=0.0)
            metrics[f'cvar_{int(alpha*100)}'] = (var_alpha + excess.mean() / (1 - alpha)).item()
        
        # Higher moments
        mean_err = hedging_error.mean()
        std_err = hedging_error.std()
        if std_err > 0:
            metrics['skewness'] = torch.mean(((hedging_error - mean_err) / std_err)**3).item()
            metrics['kurtosis'] = torch.mean(((hedging_error - mean_err) / std_err)**4).item()
        else:
            metrics['skewness'] = 0.0
            metrics['kurtosis'] = 0.0
        
        # Cost and turnover metrics
        metrics['mean_cost'] = total_costs.mean().item()
        metrics['std_cost'] = total_costs.std().item()
        metrics['mean_turnover'] = total_turnover.mean().item()
        metrics['std_turnover'] = total_turnover.std().item()
        
        # Efficiency metrics
        if metrics['mean_turnover'] > 0:
            metrics['cost_per_unit_turnover'] = metrics['mean_cost'] / metrics['mean_turnover']
        else:
            metrics['cost_per_unit_turnover'] = 0.0
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = (-metrics['mean_error'] / metrics['std_error']) if metrics['std_error'] > 0 else 0.0
        
        return metrics
    
    def plot_hedging_error_distribution(self, 
                                      results: Dict[str, torch.Tensor],
                                      title: str = "Hedging Error Distribution",
                                      save_path: Optional[str] = None):
        """
        Plot hedging error distribution.
        
        Args:
            results: Evaluation results
            title: Plot title
            save_path: Path to save plot
        """
        hedging_error = results['hedging_error'].cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(hedging_error, bins=50, alpha=0.7, density=True, edgecolor='black')
        plt.xlabel('Hedging Error')
        plt.ylabel('Density')
        plt.title('Hedging Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(2, 2, 2)
        stats.probplot(hedging_error, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal)')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_errors = np.sort(hedging_error)
        p = np.linspace(0, 1, len(sorted_errors))
        plt.plot(sorted_errors, p)
        plt.xlabel('Hedging Error')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 4)
        plt.boxplot(hedging_error, vert=True)
        plt.ylabel('Hedging Error')
        plt.title('Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_position_evolution(self, 
                              results: Dict[str, torch.Tensor],
                              n_sample_paths: int = 10,
                              title: str = "Position Evolution",
                              save_path: Optional[str] = None):
        """
        Plot position evolution for sample paths.
        
        Args:
            results: Evaluation results
            n_sample_paths: Number of sample paths to plot
            title: Plot title
            save_path: Path to save plot
        """
        position_history = results['position_history'].cpu().numpy()
        prices = results['prices'].cpu().numpy()
        
        # Select random sample paths
        n_paths = position_history.shape[0]
        sample_indices = np.random.choice(n_paths, min(n_sample_paths, n_paths), replace=False)
        
        time_steps = np.arange(position_history.shape[1]) * self.dt
        
        plt.figure(figsize=(15, 10))
        
        # Position evolution
        plt.subplot(2, 1, 1)
        for i, idx in enumerate(sample_indices):
            plt.plot(time_steps, position_history[idx], alpha=0.7, label=f'Path {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Position Evolution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Price evolution
        plt.subplot(2, 1, 2)
        for i, idx in enumerate(sample_indices):
            plt.plot(time_steps, prices[idx, 1:], alpha=0.7, label=f'Path {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Price Evolution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_risk_cost_frontier(self, 
                              results_dict: Dict[str, Dict[str, torch.Tensor]],
                              title: str = "Risk-Cost Frontier",
                              save_path: Optional[str] = None):
        """
        Plot risk-cost frontier comparing different strategies.
        
        Args:
            results_dict: Dictionary of results for different strategies
            title: Plot title
            save_path: Path to save plot
        """
        strategies = list(results_dict.keys())
        cvar_values = []
        cost_values = []
        turnover_values = []
        
        for strategy in strategies:
            metrics = self.compute_performance_metrics(results_dict[strategy])
            cvar_values.append(metrics['cvar_95'])
            cost_values.append(metrics['mean_cost'])
            turnover_values.append(metrics['mean_turnover'])
        
        plt.figure(figsize=(15, 5))
        
        # CVaR vs Cost
        plt.subplot(1, 3, 1)
        plt.scatter(cost_values, cvar_values, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            plt.annotate(strategy, (cost_values[i], cvar_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Mean Transaction Cost')
        plt.ylabel('CVaR 95%')
        plt.title('CVaR vs Transaction Cost')
        plt.grid(True, alpha=0.3)
        
        # CVaR vs Turnover
        plt.subplot(1, 3, 2)
        plt.scatter(turnover_values, cvar_values, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            plt.annotate(strategy, (turnover_values[i], cvar_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Mean Turnover')
        plt.ylabel('CVaR 95%')
        plt.title('CVaR vs Turnover')
        plt.grid(True, alpha=0.3)
        
        # Cost vs Turnover
        plt.subplot(1, 3, 3)
        plt.scatter(turnover_values, cost_values, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            plt.annotate(strategy, (turnover_values[i], cost_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Mean Turnover')
        plt.ylabel('Mean Transaction Cost')
        plt.title('Transaction Cost vs Turnover')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_table(self, 
                               results_dict: Dict[str, Dict[str, torch.Tensor]]) -> pd.DataFrame:
        """
        Create a comprehensive performance comparison table.
        
        Args:
            results_dict: Dictionary of results for different strategies
            
        Returns:
            DataFrame with performance metrics
        """
        strategies = list(results_dict.keys())
        metrics_list = []
        
        for strategy in strategies:
            metrics = self.compute_performance_metrics(results_dict[strategy])
            metrics['strategy'] = strategy
            metrics_list.append(metrics)
        
        df = pd.DataFrame(metrics_list)
        
        # Reorder columns
        strategy_col = df.pop('strategy')
        df.insert(0, 'strategy', strategy_col)
        
        return df
    
    def run_stress_test(self, 
                       policy: nn.Module,
                       stress_params: Dict[str, float],
                       n_paths: int = 5000) -> Dict[str, Dict[str, float]]:
        """
        Run stress tests with modified market parameters.
        
        Args:
            policy: Trained policy network
            stress_params: Dictionary of stress test parameters
            n_paths: Number of test paths
            
        Returns:
            Dictionary of stress test results
        """
        stress_results = {}
        
        # Baseline results
        baseline_results = self.evaluate_policy(policy, n_paths=n_paths)
        stress_results['baseline'] = self.compute_performance_metrics(baseline_results)
        
        # Stress tests
        for stress_name, stress_value in stress_params.items():
            # Create modified config
            modified_config = self.config.copy()
            
            if stress_name == 'volatility':
                modified_config.model.sigma = stress_value
            elif stress_name == 'transaction_cost':
                modified_config.costs.kappa = stress_value
            elif stress_name == 'drift':
                modified_config.model.mu = stress_value
            else:
                print(f"Unknown stress parameter: {stress_name}")
                continue
            
            # Create modified evaluator
            modified_evaluator = HedgingEvaluator(modified_config, self.device)
            
            # Evaluate policy
            stress_results_eval = modified_evaluator.evaluate_policy(policy, n_paths=n_paths)
            stress_results[stress_name] = modified_evaluator.compute_performance_metrics(stress_results_eval)
        
        return stress_results
    
    def generate_report(self, 
                       results_dict: Dict[str, Dict[str, torch.Tensor]],
                       save_dir: str = "results"):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results_dict: Dictionary of results for different strategies
            save_dir: Directory to save report
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Create performance table
        performance_df = self.create_performance_table(results_dict)
        performance_df.to_csv(f"{save_dir}/performance_metrics.csv", index=False)
        
        # Plot hedging error distributions
        for strategy, results in results_dict.items():
            self.plot_hedging_error_distribution(
                results, 
                title=f"Hedging Error Distribution - {strategy}",
                save_path=f"{save_dir}/hedging_error_{strategy}.png"
            )
        
        # Plot position evolution
        for strategy, results in results_dict.items():
            self.plot_position_evolution(
                results,
                title=f"Position Evolution - {strategy}",
                save_path=f"{save_dir}/position_evolution_{strategy}.png"
            )
        
        # Plot risk-cost frontier
        self.plot_risk_cost_frontier(
            results_dict,
            title="Risk-Cost Frontier",
            save_path=f"{save_dir}/risk_cost_frontier.png"
        )
        
        print(f"Evaluation report saved to {save_dir}/")
        print("\nPerformance Summary:")
        print(performance_df.round(6))
