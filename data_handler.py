"""
Data Handler Module
Fetches historical OHLCV data and enriches with technical indicators
"""
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
import numpy as np


class DataHandler:
    """Handles data fetching and technical indicator enrichment"""
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize DataHandler
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        
    def fetch_data(self):
        """Fetch OHLCV data for all tickers"""
        print(f"Fetching data for {len(self.tickers)} ticker(s)...")
        
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if df.empty:
                    print(f"Warning: No data retrieved for {ticker}")
                    continue
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
                
                df.columns = [col.replace(f'_{ticker}', '') for col in df.columns]
                df.columns = df.columns.str.lower()
                
                self.data[ticker] = df
                print(f"✓ Fetched {len(df)} records for {ticker}")
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
        
        return self.data
    
    def add_technical_indicators(self):
        """Add comprehensive technical indicators to the data"""
        print("Adding technical indicators...")
        
        for ticker in self.data.keys():
            df = self.data[ticker].copy()
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns for {ticker}")
                continue
            
            df = dropna(df)
            
            try:
                df = add_all_ta_features(
                    df, 
                    open="open", 
                    high="high", 
                    low="low", 
                    close="close", 
                    volume="volume",
                    fillna=True
                )
                
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                
                df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
                
                self.data[ticker] = df
                print(f"✓ Added {len(df.columns)} features for {ticker}")
                
            except Exception as e:
                print(f"Error adding indicators for {ticker}: {str(e)}")
        
        return self.data
    
    def get_processed_data(self):
        """Get the fully processed data"""
        return self.data
    
    def split_train_test(self, train_ratio=0.8):
        """
        Split data into training and testing sets
        
        Args:
            train_ratio (float): Ratio of data to use for training
            
        Returns:
            tuple: (train_data, test_data) dictionaries
        """
        train_data = {}
        test_data = {}
        
        for ticker, df in self.data.items():
            split_idx = int(len(df) * train_ratio)
            train_data[ticker] = df.iloc[:split_idx].copy()
            test_data[ticker] = df.iloc[split_idx:].copy()
        
        return train_data, test_data