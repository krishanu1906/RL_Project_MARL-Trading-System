"""
Streamlit Web Application for Multi-Agent Reinforcement Learning Trading Framework
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_handler import DataHandler
from src.environment import MultiAgentTradingEnv
from src.agent import TradingAgent, create_training_environment
from src.backtesting import Backtester


st.set_page_config(
    page_title="MARL Stock Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    st.markdown('<div class="main-header">🤖 Multi-Agent Reinforcement Learning Stock Trading System</div>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Stock Selection")
        default_tickers = "AAPL,GOOGL,MSFT"
        tickers_input = st.text_input("Tickers (comma-separated)", default_tickers)
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        
        st.subheader("Date Range")
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2)
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        num_agents = st.slider("Number of Agents", 1, 10, 3)
        initial_balance = st.number_input("Initial Balance ($)", 1000, 100000, 10000, step=1000)
        train_timesteps = st.number_input("Training Timesteps", 10000, 1000000, 100000, step=10000)
        
        # Advanced parameters
        with st.expander("Advanced Settings"):
            learning_rate = st.number_input("Learning Rate", 0.00001, 0.01, 0.0001, format="%.5f")
            gamma = st.slider("Gamma (Discount Factor)", 0.9, 0.999, 0.995, step=0.001)
            n_steps = st.number_input("N Steps", 512, 4096, 2048, step=256)
            ent_coef = st.number_input("Entropy Coefficient", 0.0, 0.1, 0.01, format="%.3f")
            cash_penalty = st.number_input("Cash Penalty Coefficient", 0.0, 0.001, 0.0001, format="%.5f")
        
        # Train/test split
        train_ratio = st.slider("Train/Test Split", 0.5, 0.9, 0.8, step=0.05)
        
        # Run button
        run_button = st.button("🚀 Run Training & Backtest", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Data Loading
            status_text.text("📊 Loading and processing data...")
            progress_bar.progress(10)
            
            data_handler = DataHandler(
                tickers=tickers,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            data_dict = data_handler.fetch_data()
            
            if not data_dict:
                st.error("❌ Failed to fetch data. Please check your ticker symbols and date range.")
                return
            
            data_dict = data_handler.add_technical_indicators()
            train_data, test_data = data_handler.split_train_test(train_ratio)
            
            progress_bar.progress(20)
            
            # Display data info
            st.success(f"✅ Loaded data for {len(tickers)} ticker(s)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Days", len(list(train_data.values())[0]))
            with col2:
                st.metric("Testing Days", len(list(test_data.values())[0]))
            with col3:
                st.metric("Features per Stock", len(list(train_data.values())[0].columns))
            
            # Step 2: Environment Creation
            status_text.text("🏗️ Creating multi-agent environment...")
            progress_bar.progress(30)
            
            trading_env = MultiAgentTradingEnv(
                data_dict=train_data,
                num_agents=num_agents,
                initial_balance=initial_balance,
                cash_penalty_coef=cash_penalty
            )
            
            wrapped_env = create_training_environment(trading_env)
            
            progress_bar.progress(40)
            
            # Step 3: Agent Training
            status_text.text("🤖 Training PPO agents...")
            progress_bar.progress(50)
            
            agent = TradingAgent(
                env=wrapped_env,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                ent_coef=ent_coef
            )
            
            with st.expander("📋 Training Logs", expanded=False):
                log_container = st.empty()
                
            model = agent.train(total_timesteps=int(train_timesteps))
            
            progress_bar.progress(70)
            st.success("✅ Training completed!")
            
            # Step 4: Backtesting
            status_text.text("📈 Running backtest on test data...")
            progress_bar.progress(80)
            
            backtester = Backtester(
                model=model,
                test_data=test_data,
                initial_balance=initial_balance,
                num_agents=num_agents
            )
            
            portfolio_values, dates, trades = backtester.run_backtest()
            
            progress_bar.progress(90)
            
            # Step 5: Performance Analysis
            status_text.text("📊 Analyzing performance...")
            
            metrics = backtester.analyze_performance(benchmark_ticker="SPY")
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display Results
            st.markdown('<div class="section-header">📊 Backtest Results</div>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Return",
                    f"{metrics.get('Total Return (%)', 0):.2f}%",
                    delta=f"{metrics.get('Total Return (%)', 0):.2f}%"
                )
            with col2:
                st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.4f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
            with col4:
                st.metric("Volatility", f"{metrics.get('Volatility (%)', 0):.2f}%")
            
            # Portfolio value chart
            st.subheader("Portfolio Value Over Time")
            
            fig_portfolio = go.Figure()
            
            fig_portfolio.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            
            fig_portfolio.add_hline(
                y=initial_balance,
                line_dash="dash",
                line_color="red",
                annotation_text="Initial Balance"
            )
            
            fig_portfolio.update_layout(
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_portfolio, use_container_width=True)
            
            # Trades visualization
            st.subheader("Trading Activity")
            
            trades_df = backtester.get_trades_dataframe()
            
            if len(trades_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trades by action
                    action_counts = trades_df['action'].value_counts()
                    fig_actions = px.pie(
                        values=action_counts.values,
                        names=action_counts.index,
                        title="Trades by Action",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig_actions, use_container_width=True)
                
                with col2:
                    # Trades by agent
                    agent_counts = trades_df['agent'].value_counts()
                    fig_agents = px.bar(
                        x=agent_counts.index,
                        y=agent_counts.values,
                        title="Trades by Agent",
                        labels={'x': 'Agent', 'y': 'Number of Trades'},
                        color=agent_counts.values,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_agents, use_container_width=True)
                
                # Recent trades table
                st.subheader("Recent Trades")
                display_trades = trades_df.tail(20).copy()
                display_trades['date'] = pd.to_datetime(display_trades['date']).dt.strftime('%Y-%m-%d')
                display_trades['price'] = display_trades['price'].apply(lambda x: f"${x:.2f}")
                st.dataframe(display_trades, use_container_width=True, hide_index=True)
                
            else:
                st.warning("⚠️ No trades were executed during the backtest period. Consider adjusting the reward function or training parameters.")
            
            # Monte Carlo Simulation
            st.markdown('<div class="section-header">🎲 Monte Carlo Risk Analysis</div>', unsafe_allow_html=True)
            
            num_simulations = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
            num_days = st.slider("Simulation Days", 30, 365, 252, step=30)
            
            if st.button("Run Monte Carlo Simulation"):
                with st.spinner("Running simulations..."):
                    simulations, percentiles = backtester.monte_carlo_simulation(
                        num_simulations=num_simulations,
                        num_days=num_days
                    )
                
                # Plot simulation results
                fig_mc = go.Figure()
                
                # Add sample paths (limited to 100 for visualization)
                sample_size = min(100, num_simulations)
                for i in range(sample_size):
                    fig_mc.add_trace(go.Scatter(
                        y=simulations[i],
                        mode='lines',
                        line=dict(color='lightgray', width=0.5),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add percentiles
                for pct_name, pct_values in percentiles.items():
                    fig_mc.add_trace(go.Scatter(
                        y=pct_values,
                        mode='lines',
                        name=f'{pct_name} Percentile',
                        line=dict(width=2)
                    ))
                
                fig_mc.update_layout(
                    title=f"Monte Carlo Simulation ({num_simulations} paths, {num_days} days)",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=600
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
            
            # Performance metrics table
            st.markdown('<div class="section-header">📈 Detailed Metrics</div>', unsafe_allow_html=True)
            
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            metrics_df.index.name = 'Metric'
            st.dataframe(metrics_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure your parameters in the sidebar and click 'Run Training & Backtest' to start!")
        
        st.markdown("""
        ### 🎯 How to Use This System
        
        1. **Select Stocks**: Enter comma-separated ticker symbols (e.g., AAPL,GOOGL,MSFT)
        2. **Set Date Range**: Choose your training and testing period
        3. **Configure Parameters**: 
           - Number of agents to train
           - Initial balance for each agent
           - Training timesteps (more = better but slower)
        4. **Advanced Settings** (optional): Fine-tune the PPO hyperparameters
        5. **Run**: Click the button to start training and backtesting
        
        ### 🧠 What This System Does
        
        - **Multi-Agent Learning**: Multiple AI agents learn to trade simultaneously
        - **PPO Algorithm**: Uses Proximal Policy Optimization for stable learning
        - **Technical Indicators**: Automatically adds 80+ technical indicators
        - **Risk Analysis**: Includes Monte Carlo simulation for risk assessment
        - **Benchmark Comparison**: Compares performance against S&P 500
        
        ### 📊 Key Features
        
        - Custom reward shaping with cash penalty to encourage active trading
        - Shared-policy architecture for efficient learning
        - Comprehensive backtesting with QuantStats integration
        - Interactive visualizations with Plotly
        - Real-time training progress monitoring
        """)
        
        # Example configuration
        with st.expander("💡 Example Configuration"):
            st.markdown("""
            **Beginner Setup:**
            - Tickers: AAPL,MSFT
            - Date Range: Last 2 years
            - Agents: 3
            - Initial Balance: $10,000
            - Training Timesteps: 50,000
            
            **Advanced Setup:**
            - Tickers: AAPL,GOOGL,MSFT,TSLA,NVDA
            - Date Range: Last 3 years
            - Agents: 5
            - Initial Balance: $50,000
            - Training Timesteps: 200,000
            - Learning Rate: 0.0001
            - Gamma: 0.995
            """)


if __name__ == "__main__":
    main()
