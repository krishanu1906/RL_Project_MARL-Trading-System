"""
Multi-Agent Trading Environment using PettingZoo Parallel API
"""
import numpy as np
import pandas as pd
from pettingzoo import ParallelEnv
from gymnasium import spaces
import functools


class MultiAgentTradingEnv(ParallelEnv):
    """Custom multi-agent trading environment"""
    
    metadata = {"name": "multi_agent_trading_v0", "render_modes": []}
    
    def __init__(self, data_dict, num_agents=3, initial_balance=10000, cash_penalty_coef=0.0001, render_mode=None):
        """
        Initialize the multi-agent trading environment
        
        Args:
            data_dict (dict): Dictionary of dataframes with ticker data
            num_agents (int): Number of trading agents
            initial_balance (float): Initial cash balance per agent
            cash_penalty_coef (float): Penalty coefficient for holding cash
            render_mode (str): Rendering mode 
        """
        self.render_mode = render_mode
        self.data_dict = data_dict
        self.tickers = list(data_dict.keys())
        self._num_agents = num_agents  
        self.initial_balance = initial_balance
        self.cash_penalty_coef = cash_penalty_coef
        
        sample_df = list(data_dict.values())[0]
        self.num_features = len(sample_df.columns)
        self.max_steps = len(sample_df)
        
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        super().__init__()
        
       
        self.current_step = 0
        self.agent_portfolios = {}
        self.agent_positions = {}
        self.previous_values = {}
        
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        obs_size = self.num_features + 3
        
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(obs_size,), 
                dtype=np.float32
            ) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        
        for agent in self.agents:
            self.agent_portfolios[agent] = {
                'cash': self.initial_balance,
                'shares': 0,
                'ticker': np.random.choice(self.tickers)  
            }
            self.agent_positions[agent] = []
        
        observations = self._get_observations()
        
        for agent in self.agents:
            self.previous_values[agent] = self._get_portfolio_value(agent)
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """Execute one step in the environment"""
        if not actions:
            return {}, {}, {}, {}, {}
        
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent, action in actions.items():
            if agent not in self.agents:
                continue
            
            ticker = self.agent_portfolios[agent]['ticker']
            current_price = self._get_current_price(ticker)
            
            if action == 1:  # Buy
                self._execute_buy(agent, current_price)
            elif action == 2:  # Sell
                self._execute_sell(agent, current_price)
            
            current_value = self._get_portfolio_value(agent)
            value_change = current_value - self.previous_values[agent]
            
            cash_held = self.agent_portfolios[agent]['cash']
            cash_penalty = -self.cash_penalty_coef * cash_held
            
            rewards[agent] = value_change + cash_penalty
            self.previous_values[agent] = current_value
            
            
            is_done = self.current_step >= self.max_steps - 1
            terminations[agent] = is_done
            truncations[agent] = is_done
            
            infos[agent] = {
                'portfolio_value': current_value,
                'cash': self.agent_portfolios[agent]['cash'],
                'shares': self.agent_portfolios[agent]['shares'],
                'ticker': ticker
            }
        
        self.current_step += 1
        
        observations = self._get_observations()
        
        
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_observations(self):
        """Get current observations for all agents"""
        observations = {}
        
        for agent in self.agents:
            ticker = self.agent_portfolios[agent]['ticker']
            
            if self.current_step < self.max_steps:
                market_data = self.data_dict[ticker].iloc[self.current_step].values
            else:
                market_data = np.zeros(self.num_features)
            
            portfolio_state = np.array([
                self.agent_portfolios[agent]['cash'] / self.initial_balance,
                self.agent_portfolios[agent]['shares'],
                self._get_portfolio_value(agent) / self.initial_balance
            ])
            
            obs = np.concatenate([market_data, portfolio_state]).astype(np.float32)
            observations[agent] = obs
        
        return observations
    
    def _get_current_price(self, ticker):
        """Get current closing price for a ticker"""
        if self.current_step < self.max_steps:
            return self.data_dict[ticker].iloc[self.current_step]['close']
        return 0
    
    def _get_portfolio_value(self, agent):
        """Calculate total portfolio value (cash + shares value)"""
        ticker = self.agent_portfolios[agent]['ticker']
        current_price = self._get_current_price(ticker)
        
        cash = self.agent_portfolios[agent]['cash']
        shares = self.agent_portfolios[agent]['shares']
        
        return cash + (shares * current_price)
    
    def _execute_buy(self, agent, current_price):
        """Execute a buy order (buy 1 share if enough cash)"""
        cash = self.agent_portfolios[agent]['cash']
        
        if cash >= current_price and current_price > 0:
            self.agent_portfolios[agent]['cash'] -= current_price
            self.agent_portfolios[agent]['shares'] += 1
            self.agent_positions[agent].append({
                'action': 'buy',
                'price': current_price,
                'step': self.current_step
            })
    
    def _execute_sell(self, agent, current_price):
        """Execute a sell order (sell 1 share if available)"""
        shares = self.agent_portfolios[agent]['shares']
        
        if shares > 0 and current_price > 0:
            self.agent_portfolios[agent]['cash'] += current_price
            self.agent_portfolios[agent]['shares'] -= 1
            self.agent_positions[agent].append({
                'action': 'sell',
                'price': current_price,
                'step': self.current_step
            })
    
    def render(self):
        """Render the environment (not implemented for headless trading)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass