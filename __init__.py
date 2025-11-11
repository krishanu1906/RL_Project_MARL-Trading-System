"""
Multi-Agent Reinforcement Learning Stock Trading Framework
"""

__version__ = "1.0.0"
__author__ = "MARL Trading System"

from .data_handler import DataHandler
from .environment import MultiAgentTradingEnv
from .agent import TradingAgent, create_training_environment, VecEnvWrapper
from .backtesting import Backtester

__all__ = [
    'DataHandler',
    'MultiAgentTradingEnv',
    'TradingAgent',
    'create_training_environment',
    'VecEnvWrapper',
    'Backtester'
]