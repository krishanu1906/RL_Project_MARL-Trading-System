"""
Backtesting Module with QuantStats Integration and Monte Carlo Simulation
"""
import numpy as np
import pandas as pd
import yfinance as yf
import quantstats as qs
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """Backtests trained agents and analyzes performance"""
    
    def __init__(self, model, test_data, initial_balance=10000, num_agents=3):
        """
        Initialize backtester
        
        Args:
            model: Trained PPO model
            test_data (dict): Dictionary of test dataframes
            initial_balance (float): Initial cash balance
            num_agents (int): Number of agents
        """
        self.model = model
        self.test_data = test_data
        self.initial_balance = initial_balance
        self.num_agents = num_agents
        self.tickers = list(test_data.keys())
        
        self.portfolio_values = []
        self.dates = []
        self.trades = []
        self.agent_portfolios = []
        
    def run_backtest(self):
        """Run backtest on test data"""
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        
        # Initialize portfolios for each agent
        portfolios = []
        for i in range(self.num_agents):
            portfolios.append({
                'cash': self.initial_balance,
                'shares': 0,
                'ticker': np.random.choice(self.tickers)
            })
        
        # Get dates from first ticker
        dates = self.test_data[self.tickers[0]].index
        self.dates = dates
        
        # Simulate trading for each timestep
        for step_idx, date in enumerate(dates):
            daily_value = 0
            
            for agent_idx, portfolio in enumerate(portfolios):
                ticker = portfolio['ticker']
                
                # Get observation for this agent
                market_data = self.test_data[ticker].iloc[step_idx].values
                portfolio_state = np.array([
                    portfolio['cash'] / self.initial_balance,
                    portfolio['shares'],
                    (portfolio['cash'] + portfolio['shares'] * self.test_data[ticker].iloc[step_idx]['close']) / self.initial_balance
                ])
                obs = np.concatenate([market_data, portfolio_state]).astype(np.float32)
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Get current price
                current_price = self.test_data[ticker].iloc[step_idx]['close']
                
                # Execute action
                if action == 1 and portfolio['cash'] >= current_price:  # Buy
                    portfolio['cash'] -= current_price
                    portfolio['shares'] += 1
                    self.trades.append({
                        'date': date,
                        'agent': agent_idx,
                        'ticker': ticker,
                        'action': 'buy',
                        'price': current_price,
                        'shares': 1
                    })
                elif action == 2 and portfolio['shares'] > 0:  # Sell
                    portfolio['cash'] += current_price
                    portfolio['shares'] -= 1
                    self.trades.append({
                        'date': date,
                        'agent': agent_idx,
                        'ticker': ticker,
                        'action': 'sell',
                        'price': current_price,
                        'shares': 1
                    })
                
                # Calculate portfolio value
                agent_value = portfolio['cash'] + portfolio['shares'] * current_price
                daily_value += agent_value
            
            # Average portfolio value across all agents
            self.portfolio_values.append(daily_value / self.num_agents)
        
        self.agent_portfolios = portfolios
        
        print(f"✓ Backtest completed")
        print(f"  Total days: {len(dates)}")
        print(f"  Total trades: {len(self.trades)}")
        print(f"  Final portfolio value: ${self.portfolio_values[-1]:.2f}")
        print(f"  Initial balance: ${self.initial_balance:.2f}")
        print(f"  Total return: {((self.portfolio_values[-1] / self.initial_balance - 1) * 100):.2f}%")
        
        return self.portfolio_values, self.dates, self.trades
    
    def calculate_returns(self):
        """Calculate returns series"""
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().fillna(0)
        
        # Remove timezone information to avoid compatibility issues with QuantStats
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        
        return returns
    
    def get_benchmark_returns(self, benchmark_ticker="SPY"):
        """
        Get benchmark returns for comparison
        
        Args:
            benchmark_ticker (str): Ticker symbol for benchmark (default: SPY for S&P 500)
        """
        try:
            start_date = self.dates[0]
            end_date = self.dates[-1]
            
            # Download benchmark data
            benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
            
            # Standardize column names
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in benchmark_data.columns.values]
            benchmark_data.columns = benchmark_data.columns.str.lower()
            
            # Calculate returns
            benchmark_returns = benchmark_data['close'].pct_change().fillna(0)
            
            # Remove timezone information
            if benchmark_returns.index.tz is not None:
                benchmark_returns.index = benchmark_returns.index.tz_localize(None)
            
            # Align with strategy dates
            benchmark_returns = benchmark_returns.reindex(self.dates, method='ffill').fillna(0)
            
            return benchmark_returns
            
        except Exception as e:
            print(f"Warning: Could not fetch benchmark data: {str(e)}")
            return None
    
    def analyze_performance(self, benchmark_ticker="SPY"):
        """
        Analyze performance using QuantStats
        
        Args:
            benchmark_ticker (str): Benchmark ticker for comparison
        """
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        returns = self.calculate_returns()
        benchmark_returns = self.get_benchmark_returns(benchmark_ticker)
        
        # Calculate key metrics
        metrics = {}
        
        # Total return
        total_return = (self.portfolio_values[-1] / self.initial_balance - 1) * 100
        metrics['Total Return (%)'] = total_return
        
        # Sharpe Ratio
        if benchmark_returns is not None:
            sharpe = qs.stats.sharpe(returns, benchmark_returns.mean())
        else:
            sharpe = qs.stats.sharpe(returns)
        metrics['Sharpe Ratio'] = sharpe
        
        # Max Drawdown
        max_dd = qs.stats.max_drawdown(returns) * 100
        metrics['Max Drawdown (%)'] = max_dd
        
        # Win Rate
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            if 'action' in trades_df.columns:
                buy_trades = trades_df[trades_df['action'] == 'buy']
                sell_trades = trades_df[trades_df['action'] == 'sell']
                
                # Simple win rate calculation
                if len(sell_trades) > 0:
                    winning_trades = sum(sell_trades['price'] > buy_trades['price'].mean())
                    win_rate = (winning_trades / len(sell_trades)) * 100
                    metrics['Win Rate (%)'] = win_rate
        
        # Volatility
        volatility = qs.stats.volatility(returns) * 100
        metrics['Volatility (%)'] = volatility
        
        # Sortino Ratio
        sortino = qs.stats.sortino(returns)
        metrics['Sortino Ratio'] = sortino
        
        # Calmar Ratio
        calmar = qs.stats.calmar(returns)
        metrics['Calmar Ratio'] = calmar
        
        # Print metrics
        print("\nKey Metrics:")
        print("-" * 60)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric:.<40} {value:>10.4f}")
            else:
                print(f"{metric:.<40} {value:>10}")
        
        if benchmark_returns is not None:
            bench_return = (benchmark_returns + 1).prod() - 1
            print(f"\nBenchmark ({benchmark_ticker}) Return: {bench_return * 100:.2f}%")
            print(f"Alpha: {(total_return / 100 - bench_return) * 100:.2f}%")
        
        return metrics
    
    def monte_carlo_simulation(self, num_simulations=1000, num_days=252):
        """
        Run Monte Carlo simulation for risk analysis
        
        Args:
            num_simulations (int): Number of simulation paths
            num_days (int): Number of days to simulate
            
        Returns:
            tuple: (simulated_paths, percentiles)
        """
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION")
        print("="*60)
        
        returns = self.calculate_returns()
        
        # Calculate mean and std of returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        print(f"Running {num_simulations} simulations for {num_days} days...")
        print(f"Historical mean return: {mean_return:.6f}")
        print(f"Historical std return: {std_return:.6f}")
        
        # Initialize simulations
        simulations = np.zeros((num_simulations, num_days))
        
        # Starting value
        starting_value = self.portfolio_values[-1]
        
        # Run simulations
        for i in range(num_simulations):
            daily_returns = np.random.normal(mean_return, std_return, num_days)
            price_series = starting_value * (1 + daily_returns).cumprod()
            simulations[i] = price_series
        
        # Calculate percentiles
        percentiles = {
            '5th': np.percentile(simulations, 5, axis=0),
            '25th': np.percentile(simulations, 25, axis=0),
            '50th': np.percentile(simulations, 50, axis=0),
            '75th': np.percentile(simulations, 75, axis=0),
            '95th': np.percentile(simulations, 95, axis=0)
        }
        
        # Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        final_values = simulations[:, -1]
        var_95 = np.percentile(final_values, 5)
        cvar_95 = final_values[final_values <= var_95].mean()
        
        print(f"\nRisk Metrics (95% confidence):")
        print(f"  Value at Risk (VaR): ${var_95:.2f}")
        print(f"  Conditional VaR (CVaR): ${cvar_95:.2f}")
        print(f"  Potential Loss: ${starting_value - var_95:.2f} ({((var_95/starting_value - 1) * 100):.2f}%)")
        
        return simulations, percentiles
    
    def get_trades_dataframe(self):
        """Return trades as a DataFrame"""
        if len(self.trades) > 0:
            return pd.DataFrame(self.trades)
        return pd.DataFrame()