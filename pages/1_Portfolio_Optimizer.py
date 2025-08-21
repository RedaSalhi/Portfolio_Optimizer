# pages/1_Portfolio_Optimizer.py - Enhanced Portfolio Optimization Interface
# Updated to work with the corrected optimizer.py featuring improved efficient frontier

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
    create_capm_analysis_chart,
    validate_optimization_result
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
        <p>Professional portfolio optimization using Modern Portfolio Theory with real-time market data, comprehensive risk analytics, and enhanced efficient frontier visualization for unlimited tickers</p>
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
                <span class="stat-number">Enhanced</span>
                <span class="stat-label">Efficient Frontier</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">Unlimited</span>
                <span class="stat-label">Assets</span>
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
        num_assets = len(st.session_state.optimizer.tickers)
        st.markdown(f'<div class="status-success">✅ Portfolio optimization active with {num_assets} assets! Enhanced efficient frontier with random portfolios visualization available below.</div>', unsafe_allow_html=True)
    elif st.session_state.data_fetched:
        st.markdown('<div class="status-info">ℹ️ Data loaded successfully! Configure settings and click optimize to see the enhanced efficient frontier.</div>', unsafe_allow_html=True)
    elif 'tickers_input' in st.session_state and len(st.session_state.get('tickers_input', '').split(',')) >= 2:
        st.markdown('<div class="status-warning">⚠️ Ready to fetch data! Enter tickers and start optimization to generate the full efficient frontier.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ Enter 2 or more ticker symbols to begin portfolio optimization with enhanced efficient frontier analysis.</div>', unsafe_allow_html=True)

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
        help="Enter 2-20 stock symbols separated by commas. The enhanced efficient frontier supports unlimited assets with optimized visualization.",
        placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA"
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
        help="Use current Treasury rate for Sharpe ratio calculations and Capital Allocation Line"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Parse and validate tickers
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
valid_input = len(tickers) >= 2

# Enhanced information about efficient frontier capabilities
if len(tickers) > 2:
    st.markdown(f"""
        <div class="status-info">
            🎯 <strong>Enhanced Analysis Ready:</strong> With {len(tickers)} assets, you'll get:
            <br>• 📊 Complete efficient frontier with {len(tickers)} assets optimization
            <br>• 🌈 Random portfolios visualization colored by Sharpe ratio
            <br>• 📈 Individual asset risk-return positioning
            <br>• ⚖️ Capital Allocation Line (if risk-free rate enabled)
            <br>• 🎯 Multiple portfolio benchmarks (equal-weight, min-variance, tangency)
        </div>
    """, unsafe_allow_html=True)

# Quick Start Section with enhanced portfolios
if not st.session_state.optimization_results:
    st.markdown("""
        <div class="quick-start">
            <div class="quick-start-title">🚀 Quick Start Portfolios</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💻 Tech Giants", help="Load technology stocks portfolio (6 assets)", use_container_width=True):
            st.session_state.tickers_input = "AAPL, MSFT, GOOGL, AMZN, TSLA, META"
            st.rerun()
    
    with col2:
        if st.button("🏛️ Blue Chips", help="Load blue chip stocks portfolio (8 assets)", use_container_width=True):
            st.session_state.tickers_input = "JNJ, PG, KO, WMT, JPM, UNH, HD, MCD"
            st.rerun()
    
    with col3:
        if st.button("🔥 FAANG+", help="Load extended FAANG portfolio (7 assets)", use_container_width=True):
            st.session_state.tickers_input = "META, AAPL, AMZN, NFLX, GOOGL, TSLA, NVDA"
            st.rerun()
            
    with col4:
        if st.button("🌍 Diversified", help="Load diversified sector portfolio (10 assets)", use_container_width=True):
            st.session_state.tickers_input = "SPY, QQQ, IWM, EFA, VNQ, TLT, GLD, VTI, XLF, XLE"
            st.rerun()

# Optimization Method Selection
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title">🎯 Optimization Method</div>', unsafe_allow_html=True)

# Method selection with cards
methods_info = {
    'max_sharpe': {
        'title': 'Maximum Sharpe Ratio',
        'description': 'Optimize for the best risk-adjusted return. Finds the tangency portfolio on the efficient frontier.',
        'icon': '📈',
        'color': '#667eea'
    },
    'min_variance': {
        'title': 'Minimum Variance',
        'description': 'Minimize portfolio risk and volatility. Finds the leftmost point on the efficient frontier.',
        'icon': '🛡️',
        'color': '#28a745'
    },
    'target_return': {
        'title': 'Target Return',
        'description': 'Achieve specific return with minimum risk. Optimizes along the efficient frontier for your return goal.',
        'icon': '🎯',
        'color': '#ffc107'
    },
    'target_volatility': {
        'title': 'Target Risk',
        'description': 'Achieve specific risk level with maximum return. Controls volatility while maximizing expected return.',
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
        help="Desired annual return percentage - will be optimized on the efficient frontier"
    )
    st.info(f"🎯 Target: {target_return*100:.1f}% annual return - Portfolio will be optimized to achieve this return with minimum risk")

elif selected_method == 'target_volatility':
    st.markdown("### ⚖️ Target Volatility Configuration")
    target_volatility = st.slider(
        "Target Annual Volatility:",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01,
        format="%.1f%%",
        help="Desired annual volatility percentage - will maximize return for this risk level"
    )
    st.info(f"⚖️ Target: {target_volatility*100:.1f}% annual volatility - Portfolio will maximize return for this risk level")

# Advanced Options with enhanced efficient frontier settings
with st.expander("🔧 Advanced Options", expanded=False):
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Analysis Options")
        show_capm = st.checkbox("📈 CAPM Analysis", value=True, help="Show Capital Asset Pricing Model analysis with beta and alpha")
        frontier_points = st.slider("🎯 Efficient Frontier Points", 25, 150, 75, help="Number of points for efficient frontier calculation (more points = smoother curve)")
        show_random_portfolios = st.checkbox("🌈 Random Portfolios", value=True, help="Show random portfolio cloud colored by Sharpe ratio")
        random_portfolios_count = st.slider("🎲 Random Portfolios Count", 100, 2000, 800, help="Number of random portfolios to generate")
        
    with col2:
        st.markdown("#### ⚖️ Weight Constraints")
        min_weight = st.slider("⬇️ Minimum Asset Weight", 0.0, 0.3, 0.0, step=0.01, format="%.1f%%", help="Minimum weight for any single asset")
        max_weight = st.slider("⬆️ Maximum Asset Weight", 0.2, 1.0, 1.0, step=0.01, format="%.1f%%", help="Maximum weight for any single asset")
        
        st.markdown("#### 🎨 Visualization Options")
        chart_height = st.slider("📏 Chart Height", 400, 800, 600, help="Height of the efficient frontier chart")
        
        if len(tickers) > 10:
            st.info("💡 With many assets, consider increasing frontier points for smoother visualization")

# Validation messages with enhanced styling
if not valid_input:
    st.markdown('<div class="status-error">❌ Please enter at least 2 valid ticker symbols separated by commas</div>', unsafe_allow_html=True)
elif len(tickers) > 20:
    st.markdown('<div class="status-warning">⚠️ Using more than 20 assets may slow down optimization. Consider reducing the number or increasing frontier points for better visualization.</div>', unsafe_allow_html=True)
elif len(tickers) >= 10:
    st.markdown('<div class="status-info">ℹ️ Large portfolio detected! The enhanced efficient frontier will show comprehensive risk-return analysis with all assets optimized.</div>', unsafe_allow_html=True)

# Main Action Buttons
st.markdown("### 🚀 Portfolio Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button(
        "🚀 Fetch Data & Optimize Portfolio", 
        disabled=not valid_input, 
        help="Fetch market data and run portfolio optimization with enhanced efficient frontier", 
        use_container_width=True,
        type="primary"
    ):
        with st.spinner("🔄 Initializing enhanced portfolio optimizer..."):
            try:
                # Initialize optimizer
                st.info("🔧 Initializing optimizer with enhanced efficient frontier capabilities...")
                optimizer = PortfolioOptimizer(tickers, lookback_years)
                
                # Display debug info
                debug_info = optimizer.get_debug_info()
                st.info(f"📊 Analysis period: {lookback_years} years | Expected trading days: ~{252 * lookback_years}")
                
                # Validate tickers first
                valid_tickers, invalid_tickers = optimizer.validate_tickers()
                
                if invalid_tickers:
                    st.warning(f"⚠️ Invalid ticker format detected: {', '.join(invalid_tickers)}")
                
                if len(valid_tickers) < 2:
                    st.error("❌ Need at least 2 valid tickers for portfolio optimization")
                    st.stop()
                
                # Fetch data
                st.info("📡 Fetching market data for efficient frontier analysis...")
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
                        - Try fewer tickers at once (2-5 maximum)
                        
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
                                st.session_state.tickers_input = "AAPL, MSFT, JNJ, PG, KO, WMT"
                                st.rerun()
                        
                        with sol_col2:
                            if st.button("📈 Try Top ETFs", help="Use popular ETFs"):
                                st.session_state.tickers_input = "SPY, QQQ, VTI, IWM"
                                st.rerun()
                        
                        with sol_col3:
                            if st.button("🏦 Try Finance Sector", help="Use financial stocks"):
                                st.session_state.tickers_input = "JPM, BAC, WFC, C"
                                st.rerun()
                else:
                    st.session_state.data_fetched = True
                    
                    # Show data fetch results
                    if error_info:  # Some tickers failed
                        st.markdown(f'<div class="status-warning">⚠️ Could not fetch data for: {error_info}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="status-success">✅ Successfully loaded: {", ".join(optimizer.tickers)} ({len(optimizer.tickers)} assets for enhanced analysis)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-success">🎉 Successfully loaded all {len(optimizer.tickers)} assets for comprehensive efficient frontier analysis!</div>', unsafe_allow_html=True)
                    
                    # Display enhanced data quality information
                    if optimizer.data_quality_info:
                        st.markdown("#### 📊 Data Quality Report")
                        quality_cols = st.columns(min(len(optimizer.tickers), 6))  # Limit columns for better display
                        
                        for i, ticker in enumerate(optimizer.tickers[:6]):  # Show first 6 for display purposes
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
                        
                        if len(optimizer.tickers) > 6:
                            st.info(f"📊 Showing quality for first 6 assets. Total assets loaded: {len(optimizer.tickers)}")
                    
                    # Get risk-free rate
                    if include_rf:
                        with st.spinner("💰 Fetching risk-free rate for CAL analysis..."):
                            rf_rate = optimizer.get_risk_free_rate()
                            st.info(f"💰 Current risk-free rate: {rf_rate:.2%} (annual) - will be used for Sharpe ratio and Capital Allocation Line")
                    
                    # Run optimization
                    st.info(f"🎯 Running {methods_info[selected_method]['title']} optimization with {len(optimizer.tickers)} assets...")
                    
                    with st.spinner("⚡ Optimizing portfolio weights across all assets..."):
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
                        
                        st.markdown(f'<div class="status-success">🎉 Portfolio optimization completed successfully with {len(optimizer.tickers)} assets! Enhanced efficient frontier ready for analysis.</div>', unsafe_allow_html=True)
                        
                        # Show optimization details
                        if 'optimization_details' in result:
                            details = result['optimization_details']
                            iterations = details.get('iterations', 'N/A')
                            converged = details.get('converged', False)
                            st.info(f"✅ Optimization {'converged' if converged else 'completed'} in {iterations} iterations")
                            
                            # Show weight distribution summary
                            weights = result['weights']
                            max_weight_asset = optimizer.tickers[np.argmax(weights)]
                            min_weight_asset = optimizer.tickers[np.argmin(weights)]
                            st.info(f"📊 Weight range: {max_weight_asset} ({np.max(weights):.1%}) to {min_weight_asset} ({np.min(weights):.1%})")
                            
                            # Show additional details if using risk-free asset
                            if include_rf:
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
                        st.markdown(f'<div class="status-error">❌ Optimization failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
                        
                        # Provide optimization troubleshooting
                        with st.expander("🔧 Optimization Troubleshooting", expanded=True):
                            st.markdown("""
                            **Possible issues and solutions:**
                            
                            🎯 **Target too aggressive:** Try more realistic return/risk targets
                            
                            ⚖️ **Constraint conflicts:** Check min/max weight constraints - ensure min_weight * n_assets < 1
                            
                            📊 **Market data issues:** Some assets may have insufficient or poor quality data
                            
                            🔄 **Try different method:** Switch to Maximum Sharpe or Minimum Variance
                            
                            📈 **Large portfolio issues:** With many assets, try relaxing weight constraints
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
    st.markdown(f"""
        <div class="results-section">
            <div class="results-title">🏆 Optimization Results ({len(optimizer.tickers)} Assets)</div>
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
        "📈 Enhanced Efficient Frontier", 
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
        
        # Weight distribution analysis
        st.markdown("#### 📈 Weight Distribution Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        weights = result['weights']
        with col1:
            st.metric("Highest Weight", f"{np.max(weights):.1%}", optimizer.tickers[np.argmax(weights)])
        with col2:
            st.metric("Lowest Weight", f"{np.min(weights):.1%}", optimizer.tickers[np.argmin(weights)])
        with col3:
            st.metric("Average Weight", f"{np.mean(weights):.1%}", f"vs Equal: {1/len(weights):.1%}")
        with col4:
            weight_std = np.std(weights)
            st.metric("Weight Std Dev", f"{weight_std:.1%}", "📊 Concentration")
        
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
        
        # Enhanced info about the visualization
        st.markdown(f"""
            <div class="status-info">
                🎯 <strong>Enhanced Efficient Frontier Analysis</strong> with {len(optimizer.tickers)} assets:
                <br>• 📊 {frontier_points} points on the efficient frontier
                <br>• 🌈 Random portfolios visualization{'✓' if show_random_portfolios else '✗'}
                <br>• 📈 Capital Allocation Line {'✓' if include_rf else '✗'}
                <br>• ⚖️ Weight constraints: {min_weight:.1%} - {max_weight:.1%}
            </div>
        """, unsafe_allow_html=True)
        
        # Generate enhanced efficient frontier
        with st.spinner("📈 Generating enhanced efficient frontier with all assets..."):
            # Pass enhanced parameters to the plotting function
            frontier_fig = create_efficient_frontier_plot(
                optimizer, 
                result, 
                include_risk_free=include_rf,
                frontier_points=frontier_points,
                show_random_portfolios=show_random_portfolios
            )
            
            # Update chart height if specified
            if frontier_fig and 'chart_height' in locals():
                frontier_fig.update_layout(height=chart_height)
        
        if frontier_fig:
            st.plotly_chart(frontier_fig, use_container_width=True)
            
            # Enhanced frontier insights
            st.markdown("#### 💡 Efficient Frontier Insights")
            st.markdown(f"""
            **🎯 Portfolio Analysis with {len(optimizer.tickers)} Assets:**
            - **Red Line**: Efficient frontier showing optimal risk-return combinations
            - **Random Points**: {f'{random_portfolios_count} random portfolios' if show_random_portfolios else 'Hidden'} colored by Sharpe ratio
            - **Blue Points**: Individual assets showing their risk-return profiles
            - **Star**: Your optimized portfolio position
            - **Diamond**: Equal-weight portfolio benchmark
            {"- **Dashed Line**: Capital Allocation Line from risk-free rate" if include_rf else ""}
            
            **📊 Key Observations:**
            - Portfolio lies on the efficient frontier (optimal risk-return combination)
            - Diversification benefit clearly visible vs individual assets
            - {f"Using {np.count_nonzero(result['weights'] > 0.01)}/{len(optimizer.tickers)} assets significantly (>1%)" if len(optimizer.tickers) > 5 else "All assets contribute to the portfolio"}
            """)
        else:
            st.warning("⚠️ Could not generate efficient frontier. Try reducing the number of assets, adjusting constraints, or using a different optimization method.")
            
            # Troubleshooting for frontier generation
            with st.expander("🔧 Efficient Frontier Troubleshooting"):
                st.markdown("""
                **Common issues:**
                - **Too many assets**: Try with fewer assets (5-10) first
                - **Tight constraints**: Relax min/max weight constraints
                - **Data quality**: Check that all assets have good quality data
                - **Optimization method**: Try Maximum Sharpe or Minimum Variance first
                """)
        
        # Enhanced frontier statistics
        st.markdown("#### 📊 Frontier Analysis Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Position", 
                "Efficient", 
                "✅ On frontier",
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
            active_assets = np.count_nonzero(result['weights'] > 0.01)
            st.metric(
                "Active Assets", 
                f"{active_assets}/{len(optimizer.tickers)}",
                f"{active_assets/len(optimizer.tickers):.1%} utilized",
                help="Assets with weight > 1%"
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
            st.metric("Max Weight", f"{max_weight:.1%}", f"📊 {optimizer.tickers[np.argmax(result['weights'])]}", help="Weight of largest position")
        
        with col4:
            min_weight = np.min(result['weights'])
            st.metric("Min Weight", f"{min_weight:.1%}", f"📊 {optimizer.tickers[np.argmin(result['weights'])]}", help="Weight of smallest position")
        
        # Enhanced risk decomposition analysis for large portfolios
        st.markdown("#### 🔍 Risk Decomposition Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Top risk contributors
            risk_contributions = result['risk_contribution']
            top_risk_indices = np.argsort(risk_contributions)[-5:][::-1]  # Top 5 risk contributors
            
            top_risks = []
            for idx in top_risk_indices:
                if idx < len(optimizer.tickers):
                    ticker = optimizer.tickers[idx]
                    risk_contrib = risk_contributions[idx] * 100
                    weight = result['weights'][idx] * 100
                    top_risks.append({
                        'Asset': ticker,
                        'Risk Contribution': f"{risk_contrib:.1f}%",
                        'Weight': f"{weight:.1f}%",
                        'Risk/Weight Ratio': f"{risk_contrib/max(weight, 0.01):.2f}"
                    })
            
            st.markdown("**🔝 Top Risk Contributors**")
            top_risks_df = pd.DataFrame(top_risks)
            st.dataframe(top_risks_df, hide_index=True, use_container_width=True)
        
        with col2:
            # Portfolio vs individual asset risk comparison
            portfolio_vol = result['volatility'] * 100
            individual_vols = np.sqrt(np.diag(optimizer.cov_matrix_annual)) * 100
            weighted_avg_vol = sum(result['weights'][i] * individual_vols[i] for i in range(len(optimizer.tickers)))
            diversification_benefit = weighted_avg_vol - portfolio_vol
            
            st.markdown("**🛡️ Diversification Analysis**")
            st.metric("Portfolio Risk", f"{portfolio_vol:.1f}%", help="Actual portfolio volatility")
            st.metric("Weighted Average Risk", f"{weighted_avg_vol:.1f}%", help="Risk without diversification benefit")
            st.metric("Diversification Benefit", f"{diversification_benefit:.1f}%", 
                     "🛡️ Risk Reduction", help="Risk reduction due to diversification")
            
            # Correlation insights
            corr_matrix = optimizer.returns[optimizer.tickers].corr()
