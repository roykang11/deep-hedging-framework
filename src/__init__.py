"""
Deep Hedging: Learning Optimal Hedging Policies with Transaction Costs

A comprehensive framework for learning optimal hedging strategies for derivative
instruments using deep reinforcement learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Deep Hedging Team"

# Import main modules
from .data import create_generator, create_state_features
from .payoffs import create_payoff
from .models import create_policy
from .risk import create_risk_measure
from .env import VectorizedHedgingEnv
from .baselines import create_baseline_strategy
from .eval import HedgingEvaluator
from .train import DeepHedgingTrainer

__all__ = [
    'create_generator',
    'create_state_features', 
    'create_payoff',
    'create_policy',
    'create_risk_measure',
    'VectorizedHedgingEnv',
    'create_baseline_strategy',
    'HedgingEvaluator',
    'DeepHedgingTrainer'
]
