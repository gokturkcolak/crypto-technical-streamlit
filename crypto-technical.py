# ============================================
# CRYPTO TECHNICAL ANALYSIS - STREAMLIT DASHBOARD
# Advanced Technical Analysis Tool for Cryptocurrencies
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import talib
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Crypto Technical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        background-color: #f4f4f4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'token_data' not in st.session_state:
    st.session_state.token_data = None
if 'basic_info' not in st.session_state:
    st.session_state.basic_info = None

# Helper functions
def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Basic calculations
    df['return'] = df['price'].pct_change()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['vol_change'] = df['volume'].pct_change()
    
    # Moving averages
    df['SMA_10'] = talib.SMA(df['price'].values, timeperiod=10)
    df['SMA_50'] = talib.SMA(df['price'].values, timeperiod=50)
    df['EMA_12'] = talib.EMA(df['price'].values, timeperiod=12)
    df['EMA_26'] = talib.EMA(df['price'].values, timeperiod=26)
    
    # Momentum indicators
    df['momentum_5'] = df['price'] - df['price'].shift(5)
    df['momentum_10'] = df['price'] - df['price'].shift(10)
    
    # Volatility
    df['volatility_10'] = df['log_return'].rolling(window=10).std()
    df['volatility_30'] = df['log_return'].rolling(window=30).std()
    
    # Volume indicators
    df['vol_sma_10'] = talib.SMA(df['volume'].values, timeperiod=10)
    df['volume_spike'] = df['volume'] / df['vol_sma_10']
    
    # RSI
    df['rsi'] = talib.RSI(df['price'].values, timeperiod=14)
    
    # MACD
    df['macd_line'] = df['EMA_12'] - df['EMA_26']
    df['signal_line'] = talib.EMA(df['macd_line'].values, timeperiod=9)
    
    return df

def search_tokens(search_term):
    """Search for tokens using CoinGecko API"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url, timeout=10)
        all_coins = response.json()
        
        search_term = search_term.lower()
        matching_tokens = [
            coin for coin in all_coins 
            if search_term in coin['name'].lower() or search_term in coin['symbol'].lower()
        ]
        
        return matching_tokens[:20]  # Limit to 20 results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def get_token_info(token_id):
    """Get basic token information"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error getting token info: {str(e)}")
        return None

def get_historical_data(token_id, start_date, end_date):
    """Get historical data from CoinGecko"""
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())
        
        url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': from_timestamp,
            'to': to_timestamp
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        # Create DataFrame
        prices_data = []
        for i, (timestamp, price) in enumerate(data['prices']):
            date = datetime.fromtimestamp(timestamp / 1000).date()
            volume = data['total_volumes'][i][1] if i < len(data['total_volumes']) else 0
            market_cap = data['market_caps'][i][1] if i < len(data['market_caps']) else 0
            
            prices_data.append({
                'date': date,
                'price': round(price, 6),
                'volume': round(volume, 2),
                'market_cap': round(market_cap, 2)
            })
        
        df = pd.DataFrame(prices_data)
        
        # Convert to daily data if too granular
        if len(df) > 100:
            df = df.groupby('date').agg({
                'price': 'last',
                'volume': 'sum',
                'market_cap': 'last'
            }).reset_index()
        
        return df
    except Exception as e:
        st.error(f"Error getting historical data: {str(e)}")
        return None

# Sidebar
st.sidebar.title("🔧 Configuration")

# Token search
search_term = st.sidebar.text_input("Search Token:", placeholder="Type token name to search...")

if st.sidebar.button("🔍 Search Tokens", use_container_width=True):
    if search_term:
        with st.spinner("Searching tokens..."):
            matching_tokens = search_tokens(search_term)
            if matching_tokens:
                st.session_state.matching_tokens = matching_tokens
                st.sidebar.success(f"Found {len(matching_tokens)} matching tokens")
            else:
                st.sidebar.error("No matching tokens found")
    else:
        st.sidebar.error("Please enter a search term")

# Token selection
token_options = {
    "Ethereum": "ethereum",
    "Solana": "solana" 
}

if 'matching_tokens' in st.session_state:
    for token in st.session_state.matching_tokens:
        token_options[f"{token['name']} ({token['symbol'].upper()})"] = token['id']

selected_token_name = st.sidebar.selectbox("Select Token:", list(token_options.keys()))
selected_token_id = token_options[selected_token_name]

# Date selection
start_date = st.sidebar.date_input("Start Date:", value=datetime.now() - timedelta(days=90))
end_date = st.sidebar.date_input("End Date:", value=datetime.now())

# Analysis button
if st.sidebar.button("🚀 Run Analysis", use_container_width=True, type="primary"):
    if selected_token_id == "":
        st.sidebar.error("Please select a token")
    elif start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
    else:
        with st.spinner("Running analysis..."):
            # Get token info
            progress_bar = st.progress(0)
            progress_bar.progress(20, "Getting token information...")
            
            basic_info = get_token_info(selected_token_id)
            if basic_info:
                st.session_state.basic_info = basic_info
                
                progress_bar.progress(50, "Getting historical data...")
                
                # Get historical data
                historical_data = get_historical_data(selected_token_id, start_date, end_date)
                if historical_data is not None:
                    progress_bar.progress(70, "Calculating technical indicators...")
                    
                    # Calculate technical indicators
                    token_data = calculate_technical_indicators(historical_data)
                    st.session_state.token_data = token_data
                    
                    progress_bar.progress(100, "Complete!")
                    st.sidebar.success("Technical analysis completed!")
                else:
                    st.sidebar.error("Failed to get historical data")
            else:
                st.sidebar.error("Token not found")
            
            progress_bar.empty()

# Download button
if st.session_state.token_data is not None:
    csv = st.session_state.token_data.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{selected_token_id}_technical_analysis_{datetime.now().strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Main content
st.title("📈 Crypto Technical Analysis Dashboard")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & Configuration", 
    "💰 Price Analysis", 
    "📈 Technical Indicators", 
    "📊 Volume Analysis", 
    "📋 Statistics"
])

with tab1:
    st.header("Token Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Settings")
        st.info(f"**Selected Token:** {selected_token_name}")
        st.info(f"**Date Range:** {start_date} to {end_date}")
        st.info(f"**Days:** {(end_date - start_date).days}")
        
    with col2:
        st.subheader("Token Information")
        if st.session_state.basic_info:
            info = st.session_state.basic_info
            st.markdown(f"""
            **Token Name:** {info['name']}  
            **Symbol:** {info['symbol'].upper()}  
            **Current Price:** ${info['market_data']['current_price']['usd']:,.6f}  
            **Market Cap:** ${info['market_data']['market_cap']['usd']:,.0f}  
            **24h Change:** {info['market_data']['price_change_percentage_24h']:.2f}%  
            **24h Volume:** ${info['market_data']['total_volume']['usd']:,.0f}  
            **All-time High:** ${info['market_data']['ath']['usd']:,.6f}  
            **All-time Low:** ${info['market_data']['atl']['usd']:,.6f}
            """)
        else:
            st.write("Click 'Run Analysis' to see token information")

with tab2:
    st.header("Price Analysis")
    
    if st.session_state.token_data is not None:
        data = st.session_state.token_data
        
        # Price chart with moving averages
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['date'], y=data['price'],
            mode='lines', name='Price',
            line=dict(color='black', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['date'], y=data['SMA_10'],
            mode='lines', name='SMA 10',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['date'], y=data['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title=f"Price Analysis - {st.session_state.basic_info['name']}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns chart
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=data['date'], y=data['return'],
                mode='lines', name='Daily Returns',
                line=dict(color='green')
            ))
            fig_returns.update_layout(
                title="Daily Returns",
                xaxis_title="Date",
                yaxis_title="Return",
                yaxis_tickformat=".2%",
                height=400
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Price statistics
            st.subheader("Price Statistics")
            current_price = data['price'].iloc[-1]
            min_price = data['price'].min()
            max_price = data['price'].max()
            avg_price = data['price'].mean()
            median_price = data['price'].median()
            total_return = ((current_price / data['price'].iloc[0]) - 1) * 100
            avg_daily_return = data['return'].mean() * 100
            
            st.markdown(f"""
            **Current Price:** ${current_price:.6f}  
            **Min Price:** ${min_price:.6f}  
            **Max Price:** ${max_price:.6f}  
            **Average Price:** ${avg_price:.6f}  
            **Median Price:** ${median_price:.6f}  
            
            **Total Return:** {total_return:.2f}%  
            **Average Daily Return:** {avg_daily_return:.4f}%
            """)
    else:
        st.info("Please run analysis to see price data")

with tab3:
    st.header("Technical Indicators")
    
    if st.session_state.token_data is not None:
        data = st.session_state.token_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Momentum indicators
            fig_momentum = go.Figure()
            fig_momentum.add_trace(go.Scatter(
                x=data['date'], y=data['momentum_5'],
                mode='lines', name='Momentum 5',
                line=dict(color='green')
            ))
            fig_momentum.add_trace(go.Scatter(
                x=data['date'], y=data['momentum_10'],
                mode='lines', name='Momentum 10',
                line=dict(color='orange')
            ))
            fig_momentum.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Line")
            fig_momentum.update_layout(
                title="Momentum Indicators",
                xaxis_title="Date",
                yaxis_title="Momentum",
                height=400
            )
            st.plotly_chart(fig_momentum, use_container_width=True)
        
        with col2:
            # Volatility analysis
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=data['date'], y=data['volatility_10'],
                mode='lines', name='Volatility 10-day',
                line=dict(color='purple')
            ))
            fig_vol.add_trace(go.Scatter(
                x=data['date'], y=data['volatility_30'],
                mode='lines', name='Volatility 30-day',
                line=dict(color='red')
            ))
            fig_vol.update_layout(
                title="Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility",
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data['date'], y=data['rsi'],
            mode='lines', name='RSI',
            line=dict(color='blue')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
        fig_rsi.update_layout(
            title="RSI (Relative Strength Index)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.info("Please run analysis to see technical indicators")

with tab4:
    st.header("Volume Analysis")
    
    if st.session_state.token_data is not None:
        data = st.session_state.token_data
        
        # Volume chart
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=data['date'], y=data['volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        fig_volume.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume (USD)",
            height=500
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume spike indicator
            fig_spike = go.Figure()
            fig_spike.add_trace(go.Scatter(
                x=data['date'], y=data['volume_spike'],
                mode='lines', name='Volume Spike',
                line=dict(color='darkblue')
            ))
            fig_spike.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="High Volume (2x)")
            fig_spike.add_hline(y=1, line_dash="solid", line_color="gray", annotation_text="Average (1x)")
            fig_spike.update_layout(
                title="Volume Spike Indicator",
                xaxis_title="Date",
                yaxis_title="Volume / SMA(10)",
                height=400
            )
            st.plotly_chart(fig_spike, use_container_width=True)
        
        with col2:
            # Volume statistics
            st.subheader("Volume Statistics")
            avg_volume = data['volume'].mean()
            max_volume = data['volume'].max()
            min_volume = data['volume'].min()
            volume_std = data['volume'].std()
            
            st.markdown(f"""
            **Average Volume:** ${avg_volume:,.0f}  
            **Max Volume:** ${max_volume:,.0f}  
            **Min Volume:** ${min_volume:,.0f}  
            **Volume Std Dev:** ${volume_std:,.0f}
            """)
    else:
        st.info("Please run analysis to see volume data")

with tab5:
    st.header("Statistics")
    
    if st.session_state.token_data is not None:
        data = st.session_state.token_data
        
        # Data table
        st.subheader("Data Table")
        display_data = data[['date', 'price', 'volume', 'return', 'SMA_10', 'SMA_50', 'rsi', 'volatility_10']].copy()
        st.dataframe(display_data, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Summary statistics
            st.subheader("Summary Statistics")
            st.markdown(f"""
            **Total Observations:** {len(data)}  
            **Date Range:** {data['date'].min()} to {data['date'].max()}  
            **Data Points:** {len(data)} days  
            
            **Current Indicators:**  
            **RSI:** {data['rsi'].iloc[-1]:.2f}  
            **10-day Volatility:** {data['volatility_10'].iloc[-1]:.4f}  
            **Volume Spike:** {data['volume_spike'].iloc[-1]:.2f}
            """)
        
        with col2:
            # Risk metrics
            st.subheader("Risk Metrics")
            returns = data['return'].dropna()
            
            if len(returns) > 0:
                var_95 = returns.quantile(0.05)
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                daily_vol = returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                max_drawdown = returns.min()
                
                st.markdown(f"""
                **Volatility (Daily):** {daily_vol * 100:.4f}%  
                **Volatility (Annualized):** {annual_vol * 100:.2f}%  
                **VaR (95%):** {var_95 * 100:.4f}%  
                **Sharpe Ratio:** {sharpe_ratio:.4f}  
                **Max Drawdown:** {max_drawdown * 100:.2f}%
                """)
            else:
                st.write("Insufficient data for risk metrics")
    else:
        st.info("Please run analysis to see statistics")