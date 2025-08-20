# pages/1_Portfolio_Optimizer.py - Enhanced Portfolio Optimization Interface
# Updated to work with the corrected optimizer.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import (
    PortfolioOptimizer,
    create_efficient_frontier_plot,
    create_portfolio_composition_chart,
    create_risk_return_analysis,
    create_performance_analytics,
    create_capm_analysis_chart
)

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer", 
    layout="wide", 
    page_icon="📊"
)

# Enhanced CSS styling inspired by bibliography page
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }

        /* Hero Section */
        .optimizer-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }

        .optimizer-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: shine 3s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .optimizer-hero h1 {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -1px;
            position: relative;
            z-index: 1;
        }
        
        .optimizer-hero p {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 1rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
            position: relative;
            z-index: 1;
        }

        .hero-stats {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 2rem;
            position: relative;
            z-index: 1;
        }

        .hero-stat {
            background: rgba(255,255,255,0.2);
            padding: 1rem 1.5rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
            transition: transform 0.3s ease;
        }

        .hero-stat:hover {
            transform: translateY(-5px);
        }

        .stat-number {
            font-size: 1.8rem;
            font-weight: 800;
            display: block;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        /* Configuration Section */
        .config-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            box-shadow: 0 15px 35px rgba(0,0,0,0.08);
            border-left: 5px solid #667eea;
        }

        .config-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 2rem;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .input-group {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border-top: 4px solid #667eea;
            transition: all 0.3s ease;
        }

        .input-group:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }

        .input-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Method Selection Cards */
        .method-selection {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .method-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 2px solid transparent;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .method-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .method-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        
        .method-card.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.2);
        }
        
        .method-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .method-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .method-description {
            color: #666;
            text-align: center;
            line-height: 1.6;
            font-size: 0.95rem;
        }

        /* Quick Start Section */
        .quick-start {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            border-left: 5px solid #28a745;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }

        .quick-start-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .quick-start-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        /* Results Section */
        .results-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            border-left: 5px solid #27ae60;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }

        .results-title {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 2rem;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        /* Metric Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .metric-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border-top: 4px solid #667eea;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, transparent, rgba(102, 126, 234, 0.05));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }

        .metric-card:hover::before {
            opacity: 1;
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #667eea;
            display: block;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }

        .metric-label {
            color: #6c757d;
            font-size: 1rem;
            font-weight: 600;
            position: relative;
            z-index: 1;
        }

        .metric-change {
            font-size: 0.85rem;
            margin-top: 0.25rem;
            position: relative;
            z-index: 1;
        }

        .metric-positive { color: #27ae60; }
        .metric-negative { color: #e74c3c; }

        /* Chart Container */
        .chart-container {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin: 2rem 0;
            border-top: 4px solid #667eea;
        }

        /* Status Messages */
        .status-success {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(40, 167, 69, 0.2);
            animation: slideIn 0.5s ease;
        }

        .status-error {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(220, 53, 69, 0.2);
            animation: slideIn 0.5s ease;
        }

        .status-warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(255, 193, 7, 0.2);
            animation: slideIn 0.5s ease;
        }

        .status-info {
            background: linear-gradient(135deg, #d1ecf1 0%, #b8e6f0 100%);
            color: #0c5460;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            border-left: 4px solid #17a2b8;
            margin: 1rem 0;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(23, 162, 184, 0.2);
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Enhanced Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 0.5rem;
            border-radius: 15px;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 10px;
            color: #666;
            font-weight: 600;
            transition: all 0.3s ease;
            text-align: center;
            flex: 1;
            max-width: 200px;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        /* Data Quality Indicators */
        .quality-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .quality-high { background: #28a745; }
        .quality-medium { background: #ffc107; }
        .quality-low { background: #dc3545; }

        /* Responsive Design */
        @media (max-width: 768px) {
            .optimizer-hero h1 { font-size: 2.5rem; }
            .optimizer-hero p { font-size: 1.1rem; }
            .hero-stats { gap: 1rem; }
            .method-selection { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
            .quick-start-grid { grid-template-columns: 1fr; }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'selected_method' not in st.session_state:
    st.session_state.selected_method = 'max_sharpe'
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False

# Hero Section with enhanced styling
st.markdown("""
    <div class="optimizer-hero">
        <h1>📊 Portfolio Optimizer</h1>
        <p>Professional portfolio optimization using Modern Portfolio Theory with real-time market data, comprehensive risk analytics, and interactive visualizations</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="stat-number">4</span>
                <span class="stat-label">Optimization Methods</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">Real-Time</span>
                <span class="stat-label">Market Data</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">CAPM</span>
                <span class="stat-label">Analysis</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">Interactive</span>
                <span class="stat-label">Visualizations</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Back Button with enhanced styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🏠 Back to Home", help="Return to main dashboard", use_container_width=True):
        st.switch_page("streamlit_app.py")

# Status Display Function
def show_status():
    if st.session_state.optimization_results and st.session_state.optimizer:
        st.markdown('<div class="status-success">✅ Portfolio optimization active! Explore results in the tabs below.</div>', unsafe_allow_html=True)
    elif st.session_state.data_fetched:
        st.markdown('<div class="status-info">ℹ️ Data loaded successfully! Configure settings and click optimize.</div>', unsafe_allow_html=True)
    elif 'tickers_input' in st.session_state and len(st.session_state.get('tickers_input', '').split(',')) >= 2:
        st.markdown('<div class="status-warning">⚠️ Ready to fetch data! Enter tickers and start optimization.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ Enter 2 or more ticker symbols to begin portfolio optimization.</div>', unsafe_allow_html=True)

show_status()

# Configuration Section
st.markdown("""
    <div class="config-section">
        <div class="config-title">⚙️ Portfolio Configuration</div>
    </div>
""", unsafe_allow_html=True)

# Asset Selection with enhanced UI
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title">📈 Asset Selection</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        value=st.session_state.get('tickers_input', 'AAPL, MSFT, GOOGL, AMZN, TSLA'),
        help="Enter 2-10 stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
        placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA"
    )
    st.session_state.tickers_input = tickers_input

with col2:
    lookback_years = st.selectbox(
        "📅 Historical Period:",
        options=[1, 2, 3, 5],
        index=2,
        help="Years of historical data for analysis"
    )

with col3:
    include_rf = st.checkbox(
        "💰 Risk-Free Rate", 
        value=True, 
        help="Use current Treasury rate for Sharpe ratio calculations"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Parse and validate tickers
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
valid_input = len(tickers) >= 2

# Quick Start Section
if not st.session_state.optimization_results:
    st.markdown("""
        <div class="quick-start">
            <div class="quick-start-title">🚀 Quick Start Portfolios</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💻 Tech Giants", help="Load technology stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "AAPL, MSFT, GOOGL, AMZN, TSLA"
            st.rerun()
    
    with col2:
        if st.button("🏛️ Blue Chips", help="Load blue chip stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "JNJ, PG, KO, WMT, JPM"
            st.rerun()
    
    with col3:
        if st.button("🔥 FAANG", help="Load FAANG stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "META, AAPL, AMZN, NFLX, GOOGL"
            st.rerun()

# Optimization Method Selection
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title">🎯 Optimization Method</div>', unsafe_allow_html=True)

# Method selection with cards
methods_info = {
    'max_sharpe': {
        'title': 'Maximum Sharpe Ratio',
        'description': 'Optimize for the best risk-adjusted return. Maximizes return per unit of risk.',
        'icon': '📈',
        'color': '#667eea'
    },
    'min_variance': {
        'title': 'Minimum Variance',
        'description': 'Minimize portfolio risk and volatility. Best for conservative investors.',
        'icon': '🛡️',
        'color': '#28a745'
    },
    'target_return': {
        'title': 'Target Return',
        'description': 'Achieve specific return with minimum risk. Set your return goal.',
        'icon': '🎯',
        'color': '#ffc107'
    },
    'target_volatility': {
        'title': 'Target Risk',
        'description': 'Achieve specific risk level with maximum return. Control your risk exposure.',
        'icon': '⚖️',
        'color': '#e74c3c'
    }
}

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for i, (method_key, method_info) in enumerate(methods_info.items()):
    with cols[i]:
        button_type = "primary" if st.session_state.selected_method == method_key else "secondary"
        if st.button(
            f"{method_info['icon']} {method_info['title']}", 
            help=method_info['description'],
            key=f"method_{method_key}",
            use_container_width=True,
            type=button_type
        ):
            st.session_state.selected_method = method_key

st.markdown('</div>', unsafe_allow_html=True)

# Display selected method with enhanced styling
selected_method = st.session_state.selected_method
selected_info = methods_info[selected_method]

st.markdown(f"""
    <div class="status-info">
        <strong>{selected_info['icon']} Selected Method: {selected_info['title']}</strong><br>
        {selected_info['description']}
    </div>
""", unsafe_allow_html=True)

# Method-specific parameters
target_return = None
target_volatility = None

if selected_method == 'target_return':
    st.markdown("### 🎯 Target Return Configuration")
    target_return = st.slider(
        "Target Annual Return:",
        min_value=0.01,
        max_value=0.50,
        value=0.12,
        step=0.01,
        format="%.1f%%",
        help="Desired annual return percentage"
    )
    st.info(f"🎯 Target: {target_return*100:.1f}% annual return")

elif selected_method == 'target_volatility':
    st.markdown("### ⚖️ Target Volatility Configuration")
    target_volatility = st.slider(
        "Target Annual Volatility:",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01,
        format="%.1f%%",
        help="Desired annual volatility percentage"
    )
    st.info(f"⚖️ Target: {target_volatility*100:.1f}% annual volatility")

# Advanced Options
with st.expander("🔧 Advanced Options", expanded=False):
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Analysis Options")
        show_capm = st.checkbox("📈 CAPM Analysis", value=True, help="Show Capital Asset Pricing Model analysis")
        frontier_points = st.slider("🎯 Efficient Frontier Points", 25, 100, 50, help="Number of points for efficient frontier")
        
    with col2:
        st.markdown("#### ⚖️ Weight Constraints")
        min_weight = st.slider("⬇️ Minimum Asset Weight", 0.0, 0.2, 0.0, step=0.01, format="%.1f%%")
        max_weight = st.slider("⬆️ Maximum Asset Weight", 0.2, 1.0, 1.0, step=0.01, format="%.1f%%")

# Validation messages with enhanced styling
if not valid_input:
    st.markdown('<div class="status-error">❌ Please enter at least 2 valid ticker symbols separated by commas</div>', unsafe_allow_html=True)
elif len(tickers) > 10:
    st.markdown('<div class="status-warning">⚠️ Using more than 10 assets may slow down optimization</div>', unsafe_allow_html=True)

# Main Action Buttons
st.markdown("### 🚀 Portfolio Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button(
        "🚀 Fetch Data & Optimize Portfolio", 
        disabled=not valid_input, 
        help="Fetch market data and run portfolio optimization", 
        use_container_width=True,
        type="primary"
    ):
        with st.spinner("🔄 Initializing portfolio optimizer..."):
            try:
                # Initialize optimizer
                st.info("🔧 Initializing optimizer...")
                optimizer = PortfolioOptimizer(tickers, lookback_years)
                
                # Display debug info
                debug_info = optimizer.get_debug_info()
                st.info(f"📊 Analysis period: {lookback_years} years | Trading days: {debug_info['temporal_consistency']['trading_days']}")
                
                # Validate tickers first
                valid_tickers, invalid_tickers = optimizer.validate_tickers()
                
                if invalid_tickers:
                    st.warning(f"⚠️ Invalid ticker format detected: {', '.join(invalid_tickers)}")
                
                if len(valid_tickers) < 2:
                    st.error("❌ Need at least 2 valid tickers for portfolio optimization")
                    st.stop()
                
                # Fetch data
                st.info("📡 Fetching market data...")
                success, error_info = optimizer.fetch_data()
                
                if not success:
                    st.markdown(f'<div class="status-error">❌ Failed to fetch data: {error_info}</div>', unsafe_allow_html=True)
                    
                    # Show detailed failure analysis
                    if hasattr(optimizer, 'failed_tickers') and optimizer.failed_tickers:
                        st.warning(f"⚠️ Failed to fetch data for: {', '.join(optimizer.failed_tickers)}")
                        
                        # Check for common issues
                        if len(optimizer.failed_tickers) == len(tickers):
                            st.error("💥 **All tickers failed!** This usually indicates:")
                            st.markdown("""
                            - 🌐 Internet connectivity issues
                            - 📊 Yahoo Finance API problems  
                            - ❌ Invalid ticker symbols
                            - 🌍 Regional access restrictions
                            """)
                        elif len(optimizer.failed_tickers) > len(tickers) * 0.5:
                            st.warning("⚠️ **Most tickers failed.** Common causes:")
                            st.markdown("""
                            - 🔄 Network issues or rate limiting
                            - 📝 Ticker symbol format problems
                            - 📉 Try fewer tickers at once
                            """)
                    
                    # Provide troubleshooting suggestions
                    with st.expander("🔧 Troubleshooting Guide", expanded=True):
                        st.markdown("""
                        **Most Common Issues:**
                        
                        🔤 **Invalid Ticker Symbols:**
                        - Make sure you're using the correct stock symbols (AAPL, not Apple Inc.)
                        - Remove any spaces or special characters
                        - Use US stock symbols (NYSE, NASDAQ)
                        
                        🌐 **Connection Issues:**
                        - Check your internet connection
                        - Yahoo Finance servers might be busy - try again in 1-2 minutes
                        - Try fewer tickers at once (2-3 maximum)
                        
                        📊 **Market Data Availability:**
                        - Some stocks may not have sufficient historical data
                        - Try major blue-chip stocks: AAPL, MSFT, JNJ, PG, KO
                        - Avoid recently listed companies or penny stocks
                        
                        🚀 **Quick Solutions:**
                        """)
                        
                        # Add quick solution buttons
                        sol_col1, sol_col2, sol_col3 = st.columns(3)
                        
                        with sol_col1:
                            if st.button("🏛️ Try Blue Chips", help="Use reliable blue chip stocks"):
                                st.session_state.tickers_input = "AAPL, MSFT, JNJ, PG, KO"
                                st.rerun()
                        
                        with sol_col2:
                            if st.button("📈 Try Top ETFs", help="Use popular ETFs"):
                                st.session_state.tickers_input = "SPY, QQQ, VTI"
                                st.rerun()
                        
                        with sol_col3:
                            if st.button("🏦 Try Finance Sector", help="Use financial stocks"):
                                st.session_state.tickers_input = "JPM, BAC, WFC"
                                st.rerun()
                else:
                    st.session_state.data_fetched = True
                    
                    # Show data fetch results
                    if error_info:  # Some tickers failed
                        st.markdown(f'<div class="status-warning">⚠️ Could not fetch data for: {error_info}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="status-success">✅ Successfully loaded: {", ".join(optimizer.tickers)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-success">🎉 Successfully loaded all {len(optimizer.tickers)} assets!</div>', unsafe_allow_html=True)
                    
                    # Display data quality information
                    if optimizer.data_quality_info:
                        st.markdown("#### 📊 Data Quality Report")
                        quality_cols = st.columns(len(optimizer.tickers))
                        
                        for i, ticker in enumerate(optimizer.tickers):
                            if ticker in optimizer.data_quality_info:
                                quality = optimizer.data_quality_info[ticker]['quality_score']
                                quality_class = "quality-high" if quality > 0.8 else "quality-medium" if quality > 0.6 else "quality-low"
                                
                                with quality_cols[i]:
                                    st.markdown(f"""
                                        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; margin: 0.5rem;">
                                            <div><span class="quality-indicator {quality_class}"></span><strong>{ticker}</strong></div>
                                            <div>Quality: {quality:.1%}</div>
                                            <div style="font-size: 0.8rem; color: #666;">
                                                {optimizer.data_quality_info[ticker]['total_points']} data points
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                    
                    # Get risk-free rate
                    if include_rf:
                        with st.spinner("💰 Fetching risk-free rate..."):
                            rf_rate = optimizer.get_risk_free_rate()
                            st.info(f"💰 Current risk-free rate: {rf_rate:.2%} (annual)")
                    
                    # Run optimization
                    st.info(f"🎯 Running {methods_info[selected_method]['title']} optimization...")
                    
                    with st.spinner("⚡ Optimizing portfolio..."):
                        result = optimizer.optimize_portfolio(
                            method=selected_method,
                            target_return=target_return,
                            target_volatility=target_volatility,
                            min_weight=min_weight,
                            max_weight=max_weight,
                            include_risk_free=include_rf 
                        )
                    
                    if result['success']:
                        st.session_state.optimizer = optimizer
                        st.session_state.optimization_results = result
                        
                        st.markdown('<div class="status-success">🎉 Portfolio optimization completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Show optimization details
                        if 'optimization_details' in result:
                            details = result['optimization_details']
                            iterations = details.get('iterations', 'N/A')
                            st.info(f"✅ Optimization converged in {iterations} iterations")
                            
                            # Show additional details if using risk-free asset
                            if include_rf and 'capital_allocation_line' in details:
                                rf_weight = result.get('rf_weight', 0)
                                risky_weight = result.get('risky_weight', 1)
                                
                                if rf_weight > 0.01:
                                    st.info(f"💰 Portfolio allocation: {rf_weight:.1%} risk-free asset, {risky_weight:.1%} risky portfolio")
                                elif risky_weight > 1.01:
                                    st.info(f"📈 Using leverage: {risky_weight:.1%} in risky assets (borrowing at risk-free rate)")
                                
                                if details.get('leverage_used', False):
                                    st.warning("⚠️ Portfolio uses leverage - additional risk considerations apply")
                        
                        st.rerun()
                    else:
                        st.markdown(f'<div class="status-error">❌ Optimization failed: {result["error"]}</div>', unsafe_allow_html=True)
                        
                        # Provide optimization troubleshooting
                        with st.expander("🔧 Optimization Troubleshooting", expanded=True):
                            st.markdown("""
                            **Possible issues and solutions:**
                            
                            🎯 **Target too aggressive:** Try more realistic return/risk targets
                            
                            ⚖️ **Constraint conflicts:** Check min/max weight constraints
                            
                            📊 **Market data issues:** Some assets may have insufficient data
                            
                            🔄 **Try different method:** Switch to Maximum Sharpe or Minimum Variance
                            """)
                        
            except Exception as e:
                st.markdown(f'<div class="status-error">💥 Unexpected error: {str(e)}</div>', unsafe_allow_html=True)
                
                with st.expander("🐛 Error Details", expanded=False):
                    st.code(str(e))

with col2:
    if st.button("🗑️ Clear Results", help="Clear current optimization results", use_container_width=True, type="secondary"):
        # Clear all session state
        for key in ['optimizer', 'optimization_results', 'data_fetched']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Display Results Section
if st.session_state.optimization_results and st.session_state.optimizer:
    optimizer = st.session_state.optimizer
    result = st.session_state.optimization_results
    
    # Results Section with enhanced styling
    st.markdown("""
        <div class="results-section">
            <div class="results-title">🏆 Optimization Results</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Key Metrics Grid
    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['expected_return']*100:.1f}%</span>
                <div class="metric-label">Expected Return</div>
                <div class="metric-change">📈 Annualized</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['volatility']*100:.1f}%</span>
                <div class="metric-label">Volatility</div>
                <div class="metric-change">📊 Annual Risk</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sharpe_color = "metric-positive" if result['sharpe_ratio'] > 1 else "metric-negative" if result['sharpe_ratio'] < 0.5 else ""
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['sharpe_ratio']:.2f}</span>
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-change {sharpe_color}">⚖️ Risk-Adjusted</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        diversification = result['diversification_ratio']
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{diversification:.1f}</span>
                <div class="metric-label">Diversification</div>
                <div class="metric-change">🔄 Effective Assets</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        max_dd = abs(result['max_drawdown']) * 100
        dd_color = "metric-positive" if max_dd < 10 else "metric-negative" if max_dd > 20 else ""
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{max_dd:.1f}%</span>
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-change {dd_color}">📉 Historical</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional metrics if available
    if 'sortino_ratio' in result:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{result['sortino_ratio']:.2f}</span>
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-change">📉 Downside Risk</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cvar = abs(result.get('cvar_95', 0)) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{cvar:.1f}%</span>
                    <div class="metric-label">CVaR (95%)</div>
                    <div class="metric-change">⚠️ Tail Risk</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            concentration = result.get('concentration', 0)
            st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{concentration:.3f}</span>
                    <div class="metric-label">Concentration</div>
                    <div class="metric-change">📊 HHI Index</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Tabbed Results Display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Composition",
        "📈 Efficient Frontier", 
        "⚠️ Risk Analysis",
        "📉 Performance Analytics",
        "🎯 CAPM Analysis"
    ])

    with tab1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Portfolio composition visualization
        composition_fig = create_portfolio_composition_chart(
            optimizer.tickers, 
            result['weights'],
            result.get('rf_weight') if include_rf else None
        )
        st.plotly_chart(composition_fig, use_container_width=True)
        
        # Enhanced allocation tables
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 Asset Allocation")
            allocation_data = []
            for i, ticker in enumerate(optimizer.tickers):
                weight = result['weights'][i]
                allocation_data.append({
                    'Asset': ticker,
                    'Weight': f"{weight:.1%}",
                    'Value ($100K)': f"${weight*100000:,.0f}",
                    'Quality': f"{optimizer.data_quality_info.get(ticker, {}).get('quality_score', 0):.1%}" if optimizer.data_quality_info else "N/A"
                })
            
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚠️ Risk Attribution")
            risk_data = []
            for i, ticker in enumerate(optimizer.tickers):
                risk_contrib = result['risk_contribution'][i] if len(result['risk_contribution']) > i else 0
                marginal_risk = result['marginal_contribution'][i] if len(result['marginal_contribution']) > i else 0
                
                risk_data.append({
                    'Asset': ticker,
                    'Risk Contribution': f"{risk_contrib*100:.1f}%",
                    'Marginal Risk': f"{marginal_risk:.3f}",
                    'Risk Rank': i + 1
                })
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, hide_index=True, use_container_width=True)
        
        # Display Capital Allocation Line information if using risk-free asset
        if include_rf and result.get('rf_weight') is not None:
            st.markdown("#### 💰 Capital Allocation Line Position")
            
            rf_weight = result.get('rf_weight', 0)
            risky_weight = result.get('risky_weight', 1)
            tangency_return = result.get('tangency_return', result['expected_return'])
            tangency_volatility = result.get('tangency_volatility', result['volatility'])
            tangency_sharpe = result.get('tangency_sharpe', result['sharpe_ratio'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Risk-Free Allocation",
                    f"{rf_weight:.1%}",
                    help="Portion invested in risk-free asset"
                )
            
            with col2:
                st.metric(
                    "Risky Portfolio Allocation", 
                    f"{risky_weight:.1%}",
                    "🔄 Leveraged" if risky_weight > 1 else "💼 Conservative" if risky_weight < 1 else "⚖️ Full Investment",
                    help="Portion invested in risky asset portfolio"
                )
            
            with col3:
                st.metric(
                    "Tangency Portfolio Sharpe",
                    f"{tangency_sharpe:.3f}",
                    help="Sharpe ratio of the optimal risky portfolio"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Generate efficient frontier with enhanced parameters
        with st.spinner("📈 Generating efficient frontier..."):
            frontier_fig = create_efficient_frontier_plot(
                optimizer, 
                result, 
                include_risk_free=include_rf
            )
        
        if frontier_fig:
            st.plotly_chart(frontier_fig, use_container_width=True)
        else:
            st.warning("⚠️ Could not generate efficient frontier. Try reducing the number of assets or using different optimization method.")
        
        # Enhanced frontier statistics
        st.markdown("#### 📊 Frontier Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Position", 
                "Optimal", 
                "✅ On efficient frontier",
                help="Portfolio lies on the efficient frontier"
            )
        
        with col2:
            # Calculate equal weight portfolio for comparison
            equal_weights = np.ones(len(optimizer.tickers)) / len(optimizer.tickers)
            equal_weight_metrics = optimizer.portfolio_metrics(equal_weights)
            improvement = ((result['sharpe_ratio'] - equal_weight_metrics['sharpe_ratio']) / equal_weight_metrics['sharpe_ratio'] * 100) if equal_weight_metrics['sharpe_ratio'] != 0 else 0
            st.metric(
                "Sharpe Improvement", 
                f"{improvement:.1f}%", 
                "📈 vs Equal Weight",
                help="Improvement over equal weight portfolio"
            )
        
        with col3:
            st.metric(
                "Optimization Method", 
                result['method'].replace('_', ' ').title(),
                help="Method used for optimization"
            )
        
        with col4:
            if 'optimization_details' in result:
                iterations = result['optimization_details'].get('iterations', 0)
                converged = result['optimization_details'].get('converged', False)
                st.metric(
                    "Convergence", 
                    f"{iterations} iterations",
                    "✅ Successful" if converged else "⚠️ Warning",
                    help="Optimization convergence details"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        risk_analysis_fig = create_risk_return_analysis(optimizer, result['weights'])
        st.plotly_chart(risk_analysis_fig, use_container_width=True)
        
        # Enhanced risk metrics summary
        st.markdown("#### ⚠️ Risk Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_95 = abs(result['var_95']) * 100
            st.metric("VaR (95%)", f"{var_95:.2f}%", "📉 Daily", help="Value at Risk at 95% confidence level (daily)")
        
        with col2:
            var_99 = abs(result['var_99']) * 100
            st.metric("VaR (99%)", f"{var_99:.2f}%", "📉 Daily", help="Value at Risk at 99% confidence level (daily)")
        
        with col3:
            max_weight = np.max(result['weights'])
            st.metric("Max Weight", f"{max_weight:.1%}", "📊 Largest Position", help="Weight of largest position")
        
        with col4:
            min_weight = np.min(result['weights'])
            st.metric("Min Weight", f"{min_weight:.1%}", "📊 Smallest Position", help="Weight of smallest position")
        
        # Risk decomposition analysis
        st.markdown("#### 🔍 Risk Decomposition")
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual asset risks
            individual_risks = []
            for i, ticker in enumerate(optimizer.tickers):
                asset_vol = np.sqrt(optimizer.cov_matrix_annual.iloc[i, i]) * 100
                individual_risks.append({
                    'Asset': ticker,
                    'Individual Risk': f"{asset_vol:.1f}%",
                    'Weight': f"{result['weights'][i]:.1%}",
                    'Weighted Risk': f"{result['weights'][i] * asset_vol:.1f}%"
                })
            
            risks_df = pd.DataFrame(individual_risks)
            st.dataframe(risks_df, hide_index=True, use_container_width=True)
        
        with col2:
            # Portfolio vs individual asset risk comparison
            portfolio_vol = result['volatility'] * 100
            weighted_avg_vol = sum(result['weights'][i] * np.sqrt(optimizer.cov_matrix_annual.iloc[i, i]) * 100 
                                 for i in range(len(optimizer.tickers)))
            diversification_benefit = weighted_avg_vol - portfolio_vol
            
            st.metric("Portfolio Risk", f"{portfolio_vol:.1f}%", help="Actual portfolio volatility")
            st.metric("Weighted Average Risk", f"{weighted_avg_vol:.1f}%", help="Risk without diversification benefit")
            st.metric("Diversification Benefit", f"{diversification_benefit:.1f}%", 
                     "🛡️ Risk Reduction", help="Risk reduction due to diversification")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        performance_fig = create_performance_analytics(optimizer, result['weights'])
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Enhanced performance statistics
        portfolio_returns = (optimizer.returns * result['weights']).sum(axis=1)
        
        st.markdown("#### 📊 Performance Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (1 + portfolio_returns).prod() - 1
            st.metric(
                "Total Return", 
                f"{total_return*100:.1f}%", 
                f"📅 {lookback_years} Year Period",
                help=f"Total return over {lookback_years} year period"
            )
        
        with col2:
            annualized_return = (1 + total_return) ** (1/lookback_years) - 1
            st.metric(
                "Annualized Return", 
                f"{annualized_return*100:.1f}%",
                "📈 CAGR",
                help="Compound annual growth rate"
            )
        
        with col3:
            winning_days = (portfolio_returns > 0).sum() / len(portfolio_returns)
            st.metric(
                "Winning Days", 
                f"{winning_days*100:.1f}%",
                "📈 Positive Days",
                help="Percentage of positive return days"
            )
        
        with col4:
            volatility_realized = portfolio_returns.std() * np.sqrt(252)
            st.metric(
                "Realized Volatility", 
                f"{volatility_realized*100:.1f}%",
                "📊 Historical",
                help="Historical volatility of the portfolio"
            )
        
        # Additional performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            skewness = portfolio_returns.skew()
            skew_interpretation = "📈 Positive Skew" if skewness > 0 else "📉 Negative Skew" if skewness < -0.1 else "⚖️ Symmetric"
            st.metric(
                "Skewness", 
                f"{skewness:.2f}",
                skew_interpretation,
                help="Asymmetry of return distribution"
            )
        
        with col2:
            kurtosis = portfolio_returns.kurtosis()
            kurt_interpretation = "⚠️ Fat Tails" if kurtosis > 1 else "📊 Normal Tails"
            st.metric(
                "Kurtosis", 
                f"{kurtosis:.2f}",
                kurt_interpretation,
                help="Tail heaviness of return distribution"
            )
        
        with col3:
            avg_daily_return = portfolio_returns.mean() * 100
            st.metric(
                "Avg Daily Return", 
                f"{avg_daily_return:.3f}%",
                "📅 Daily Average",
                help="Average daily return"
            )
        
        with col4:
            best_day = portfolio_returns.max() * 100
            worst_day = portfolio_returns.min() * 100
            st.metric(
                "Best Day", 
                f"{best_day:.2f}%",
                f"📉 Worst: {worst_day:.2f}%",
                help="Best and worst single day performance"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        if show_capm:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            with st.spinner("🎯 Calculating CAPM metrics..."):
                capm_metrics = optimizer.calculate_capm_metrics()
            
            if camp_metrics:
                camp_fig = create_capm_analysis_chart(capm_metrics)
                if capm_fig:
                    st.plotly_chart(capm_fig, use_container_width=True)
                
                # Enhanced CAPM summary
                st.markdown("#### 🎯 CAPM Analysis Summary")
                
                # Create enhanced CAPM table
                capm_data = []
                for ticker in optimizer.tickers:
                    if ticker in capm_metrics:
                        metrics = camp_metrics[ticker]
                        capm_data.append({
                            'Asset': ticker,
                            'Beta': f"{metrics['beta']:.3f}",
                            'Alpha': f"{metrics['alpha']:.2%}",
                            'Expected Return': f"{metrics['expected_return']:.2%}",
                            'Actual Return': f"{metrics['actual_return']:.2%}",
                            'R-Squared': f"{metrics['r_squared']:.3f}",
                            'Correlation': f"{metrics.get('correlation', 0):.3f}",
                            'Systematic Risk': f"{metrics.get('systematic_risk', 0):.2%}",
                            'Total Risk': f"{metrics.get('total_risk', 0):.2%}"
                        })
                
                if capm_data:
                    capm_df = pd.DataFrame(camp_data)
                    st.dataframe(capm_df, hide_index=True, use_container_width=True)
                    
                    # CAPM interpretation
                    st.markdown("#### 📚 CAPM Interpretation")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **📊 Beta Analysis:**
                        - **Beta > 1:** More volatile than market 📈
                        - **Beta < 1:** Less volatile than market 🛡️
                        - **Beta = 1:** Same volatility as market ⚖️
                        
                        **🎯 R-Squared Analysis:**
                        - **R² > 0.7:** Strong market relationship 🔗
                        - **R² < 0.3:** Weak market relationship 🔄
                        """)
                    
                    with col2:
                        st.markdown("""
                        **⭐ Alpha Analysis:**
                        - **Alpha > 0:** Outperforming expectations 📈
                        - **Alpha < 0:** Underperforming expectations 📉
                        - **Alpha ≈ 0:** Performing as expected ⚖️
                        
                        **⚠️ Risk Decomposition:**
                        - **Systematic:** Market-related risk 🌍
                        - **Idiosyncratic:** Company-specific risk 🏢
                        """)
                    
                    # Portfolio CAPM summary
                    st.markdown("#### 📊 Portfolio CAPM Summary")
                    
                    # Calculate portfolio beta and alpha
                    portfolio_beta = sum(result['weights'][i] * camp_metrics[ticker]['beta'] 
                                       for i, ticker in enumerate(optimizer.tickers) 
                                       if ticker in capm_metrics)
                    
                    portfolio_alpha = sum(result['weights'][i] * camp_metrics[ticker]['alpha'] 
                                        for i, ticker in enumerate(optimizer.tickers) 
                                        if ticker in capm_metrics)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        beta_interpretation = "📈 Aggressive" if portfolio_beta > 1.2 else "🛡️ Defensive" if portfolio_beta < 0.8 else "⚖️ Market-like"
                        st.metric(
                            "Portfolio Beta",
                            f"{portfolio_beta:.3f}",
                            beta_interpretation,
                            help="Portfolio sensitivity to market movements"
                        )
                    
                    with col2:
                        alpha_interpretation = "⭐ Outperforming" if portfolio_alpha > 0.02 else "📉 Underperforming" if portfolio_alpha < -0.02 else "⚖️ Market-like"
                        st.metric(
                            "Portfolio Alpha",
                            f"{portfolio_alpha:.2%}",
                            alpha_interpretation,
                            help="Portfolio excess return above market expectations"
                        )
                    
                    with col3:
                        # Calculate portfolio systematic risk
                        portfolio_systematic_risk = sum(result['weights'][i] * capm_metrics[ticker].get('systematic_risk', 0) 
                                                       for i, ticker in enumerate(optimizer.tickers) 
                                                       if ticker in camp_metrics)
                        systematic_pct = (portfolio_systematic_risk / result['volatility']) * 100 if result['volatility'] > 0 else 0
                        
                        st.metric(
                            "Systematic Risk %",
                            f"{systematic_pct:.1f}%",
                            "🌍 Market-driven",
                            help="Percentage of portfolio risk from market movements"
                        )
                        
            else:
                st.markdown('<div class="status-warning">⚠️ CAPM analysis not available - market data could not be fetched</div>', unsafe_allow_html=True)
                
                st.markdown("""
                **Why CAPM might not be available:**
                - 📊 Market benchmark data unavailable
                - 📅 Insufficient overlapping data between assets and market
                - 🌐 Network connectivity issues
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-info">ℹ️ CAPM analysis disabled. Enable in Advanced Options to view detailed beta and alpha analysis.</div>', unsafe_allow_html=True)

# Enhanced Tips & Best Practices Section
with st.expander("💡 Tips & Best Practices", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 Optimization Methods Guide

        **📈 Maximum Sharpe Ratio**  
        - Best for: General portfolio optimization  
        - Goal: Maximize risk-adjusted returns  
        - Ideal for: Most investors seeking balance  

        **🛡️ Minimum Variance**  
        - Best for: Conservative investors  
        - Goal: Minimize portfolio risk  
        - Ideal for: Risk-averse investors, stable income needs  

        **🎯 Target Return**  
        - Best for: Specific return goals  
        - Goal: Achieve target with minimum risk  
        - Ideal for: Pension funds, endowments  

        **⚖️ Target Volatility**  
        - Best for: Risk budgeting  
        - Goal: Maximize return for specific risk  
        - Ideal for: Risk-managed strategies  
        """)

    with col2:
        st.markdown("""
        #### 📊 Key Metrics Explained

        **📈 Expected Return**: Annualized expected portfolio return  
        **📊 Volatility**: Annual portfolio standard deviation (risk)  
        **⚖️ Sharpe Ratio**: Return per unit of risk (higher is better)  
        **📉 Sortino Ratio**: Risk-adjusted return using downside deviation  
        **⚠️ VaR**: Maximum expected loss at confidence level  
        **💥 CVaR**: Average loss beyond VaR threshold  
        **📉 Max Drawdown**: Largest peak-to-trough decline  
        **🔄 Diversification Ratio**: Effective number of independent positions  
        **📊 Beta**: Sensitivity to market movements  
        **⭐ Alpha**: Excess return above market expectations  
        """)

# Enhanced Performance Tips Section
with st.expander("🚀 Performance Tips", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Optimization Best Practices
        
        **📊 Data Quality**
        - Use liquid, established stocks
        - Ensure sufficient historical data (2+ years)
        - Check data quality indicators
        
        **⚖️ Constraint Setting**
        - Set reasonable min/max weights
        - Consider liquidity constraints
        - Account for transaction costs
        
        **🔄 Regular Rebalancing**
        - Review portfolios quarterly
        - Rebalance when weights drift >5%
        - Consider market regime changes
        """)
    
    with col2:
        st.markdown("""
        #### ⚠️ Risk Management
        
        **🛡️ Diversification**
        - Use assets from different sectors
        - Include international exposure
        - Consider alternative asset classes
        
        **📊 Risk Monitoring**
        - Track VaR and CVaR regularly
        - Monitor correlation changes
        - Set stop-loss levels
        
        **🎯 Target Setting**
        - Set realistic return targets
        - Consider risk capacity
        - Account for market conditions
        """)

# Enhanced Data Sources and Methodology Section
with st.expander("📚 Data Sources & Methodology", expanded=False):
    st.markdown("""
    #### 📊 Data Sources
    - **Stock Prices**: Yahoo Finance API
    - **Risk-Free Rate**: FRED (Federal Reserve Economic Data)
    - **Market Benchmark**: S&P 500 Index (^GSPC)
    
    #### 🔬 Methodology
    - **Mean Returns**: Annualized from daily returns (252 trading days)
    - **Covariance Matrix**: Annualized from daily return covariance
    - **VaR/CVaR**: Historical simulation method (daily basis)
    - **Optimization**: Sequential Least Squares Programming (SLSQP)
    - **Risk Attribution**: Marginal contribution decomposition
    
    #### ⚠️ Important Notes
    - **Temporal Consistency**: All metrics properly annualized
    - **Risk-Free Integration**: Uses current Treasury rates
    - **CAPM Analysis**: Requires market benchmark data
    - **Historical Data**: Past performance doesn't guarantee future results
    """)

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;">
        <p style="font-size: 1.2rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                📊 QuantRisk Analytics
            </span>
        </p>
        <p style="font-size: 1rem; color: #667eea; margin-bottom: 0.5rem;">Portfolio Optimization Platform</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Built with Modern Portfolio Theory & Enhanced Risk Analytics</p>
        <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 1rem;">© 2025 | ⚡ Powered by Streamlit & 🔬 Advanced Quantitative Methods</p>
        
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #ddd;">
            <p style="font-size: 0.85rem; color: #888; line-height: 1.4;">
                <strong>🔧 Technical Features:</strong><br>
                ✅ Temporal Consistency | ✅ Risk-Free Asset Integration | ✅ CAPM Analysis<br>
                ✅ VaR/CVaR Methodology | ✅ Risk Attribution | ✅ Efficient Frontier
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Add final validation check for results
if st.session_state.optimization_results and st.session_state.optimizer:
    # Show validation warnings if any
    warnings = []
    try:
        from optimizer import validate_optimization_result
        warnings = validate_optimization_result(
            st.session_state.optimization_results,
            st.session_state.optimization_results['weights'],
            st.session_state.optimizer
        )
    except Exception:
        pass
    
    if warnings:
        with st.expander("⚠️ Validation Warnings", expanded=False):
            for warning in warnings:
                st.warning(f"⚠️ {warning}")
            
            st.info("""
            These warnings indicate potential issues with the optimization results.
            Consider adjusting constraints or trying a different optimization method.
            """)
