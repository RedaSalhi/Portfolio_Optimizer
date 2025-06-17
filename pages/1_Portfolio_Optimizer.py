# pages/1_Portfolio_Optimizer.py - Enhanced Portfolio Optimization Interface

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
    page_icon="📈"
)

# Enhanced CSS styling consistent with other pages
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
        }
        
        .optimizer-hero h1 {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -1px;
        }
        
        .optimizer-hero p {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 1rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        .hero-stats {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 2rem;
        }

        .hero-stat {
            background: rgba(255,255,255,0.2);
            padding: 1rem 1.5rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }

        .stat-number {
            font-size: 1.5rem;
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
        }

        .input-group {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border-top: 4px solid #667eea;
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

        /* Method Selection */
        .method-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
        }
        
        .method-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
            text-align: center;
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
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 2rem;
            text-align: center;
        }

        /* Metric Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            display: block;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            color: #6c757d;
            font-size: 0.95rem;
            font-weight: 500;
        }

        .metric-change {
            font-size: 0.85rem;
            margin-top: 0.25rem;
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
        }

        /* Status Messages */
        .status-success {
            background: #d4edda;
            color: #155724;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
            font-weight: 500;
        }

        .status-error {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
            font-weight: 500;
        }

        .status-warning {
            background: #fff3cd;
            color: #856404;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
            font-weight: 500;
        }

        .status-info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #17a2b8;
            margin: 1rem 0;
            font-weight: 500;
        }

        /* Buttons */
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
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        /* Secondary Buttons */
        .secondary-button .stButton > button {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
        }

        .danger-button .stButton > button {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
        }

        /* Loading Animation */
        .loading-container {
            text-align: center;
            padding: 3rem;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .optimizer-hero h1 { font-size: 2.5rem; }
            .optimizer-hero p { font-size: 1.1rem; }
            .hero-stats { gap: 1rem; }
            .method-grid { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
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

# Hero Section
st.markdown("""
    <div class="optimizer-hero">
        <h1>Advanced Portfolio Optimizer</h1>
        <p>Professional portfolio optimization using Modern Portfolio Theory with real-time market data and comprehensive risk analytics</p>
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

# Back Button
if st.button("Back to Home", help="Return to main dashboard", use_container_width=False):
    st.switch_page("streamlit_app.py")

# Status Display
def show_status():
    if st.session_state.optimization_results and st.session_state.optimizer:
        st.markdown('<div class="status-success">Portfolio optimization active! Explore results in the tabs below.</div>', unsafe_allow_html=True)
    elif 'tickers_input' in st.session_state and len(st.session_state.get('tickers_input', '').split(',')) >= 2:
        st.markdown('<div class="status-info">Ready to optimize! Configure settings and click optimize.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">Enter 2 or more ticker symbols to begin portfolio optimization.</div>', unsafe_allow_html=True)

show_status()

# Configuration Section
st.markdown("""
    <div class="config-section">
        <div class="config-title">Portfolio Configuration</div>
    </div>
""", unsafe_allow_html=True)

# Asset Selection
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title">Asset Selection</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        value=st.session_state.get('tickers_input', 'AAPL, MSFT, GOOGL, AMZN, TSLA'),
        help="Enter 2-10 stock symbols separated by commas"
    )
    st.session_state.tickers_input = tickers_input

with col2:
    lookback_years = st.selectbox(
        "Historical data period:",
        options=[1, 2, 3, 5],
        index=2,
        help="Years of historical data for analysis"
    )

with col3:
    include_rf = st.checkbox("Include Risk-Free Rate", value=True, help="Use current Treasury rate for Sharpe ratio calculations")

st.markdown('</div>', unsafe_allow_html=True)

# Parse and validate tickers
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
valid_input = len(tickers) >= 2

# Optimization Method Selection
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title">Optimization Method</div>', unsafe_allow_html=True)

# Method selection columns
col1, col2, col3, col4 = st.columns(4)

methods = {
    'max_sharpe': {
        'title': 'Maximum Sharpe Ratio',
        'description': 'Optimize for the best risk-adjusted return',
        'icon': 'M'
    },
    'min_variance': {
        'title': 'Minimum Variance',
        'description': 'Minimize portfolio risk and volatility',
        'icon': 'V'
    },
    'target_return': {
        'title': 'Target Return',
        'description': 'Achieve specific return with minimum risk',
        'icon': 'R'
    },
    'target_volatility': {
        'title': 'Target Risk',
        'description': 'Achieve specific risk with maximum return',
        'icon': 'T'
    }
}

with col1:
    if st.button("Maximum Sharpe Ratio", help="Optimize for best risk-adjusted return", key="sharpe_btn"):
        st.session_state.selected_method = 'max_sharpe'

with col2:
    if st.button("Minimum Variance", help="Minimize portfolio risk", key="variance_btn"):
        st.session_state.selected_method = 'min_variance'

with col3:
    if st.button("Target Return", help="Achieve specific return target", key="return_btn"):
        st.session_state.selected_method = 'target_return'

with col4:
    if st.button("Target Risk", help="Achieve specific risk level", key="risk_btn"):
        st.session_state.selected_method = 'target_volatility'

st.markdown('</div>', unsafe_allow_html=True)

# Display selected method
selected_method = st.session_state.selected_method
st.info(f"Selected Method: **{methods[selected_method]['title']}** - {methods[selected_method]['description']}")

# Method-specific parameters
target_return = None
target_volatility = None

if selected_method == 'target_return':
    target_return = st.slider(
        "Target Annual Return:",
        min_value=0.05,
        max_value=0.30,
        value=0.12,
        step=0.01,
        format="%.1%",
        help="Desired annual return percentage"
    )
elif selected_method == 'target_volatility':
    target_volatility = st.slider(
        "Target Annual Volatility:",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
        format="%.1%",
        help="Desired annual volatility percentage"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Advanced Options
with st.expander("Advanced Options", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        show_capm = st.checkbox("CAPM Analysis", value=True, help="Show Capital Asset Pricing Model analysis")
        min_weight = st.slider("Minimum Asset Weight", 0.0, 0.2, 0.0, step=0.01, format="%.1%")
    with col2:
        max_weight = st.slider("Maximum Asset Weight", 0.2, 1.0, 1.0, step=0.01, format="%.1%")
        frontier_points = st.slider("Efficient Frontier Points", 50, 200, 100)

# Validation messages
if not valid_input:
    st.markdown('<div class="status-error">Please enter at least 2 valid ticker symbols</div>', unsafe_allow_html=True)
elif len(tickers) > 10:
    st.markdown('<div class="status-warning">Using more than 10 assets may slow down optimization</div>', unsafe_allow_html=True)

# Quick Start Options
if not st.session_state.optimization_results:
    st.markdown("### Quick Start Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Tech Portfolio", help="Load technology stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "AAPL, MSFT, GOOGL, AMZN, TSLA"
            st.rerun()
    
    with col2:
        if st.button("Diversified Portfolio", help="Load diversified ETF portfolio", use_container_width=True):
            st.session_state.tickers_input = "SPY, QQQ, VTI, BND, VEA"
            st.rerun()
    
    with col3:
        if st.button("FAANG Portfolio", help="Load FAANG stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "META, AAPL, AMZN, NFLX, GOOGL"
            st.rerun()

# Main Optimization Button
st.markdown("### Run Optimization")

optimize_col1, optimize_col2 = st.columns([3, 1])

with optimize_col1:
    if st.button("Optimize Portfolio", disabled=not valid_input, help="Run portfolio optimization with current settings", use_container_width=True):
        with st.spinner("Fetching data and optimizing portfolio..."):
            try:
                # Initialize optimizer
                optimizer = PortfolioOptimizer(tickers, lookback_years)
                
                # Fetch data
                success, error_info = optimizer.fetch_data()
                
                if not success:
                    st.markdown(f'<div class="status-error">Failed to fetch data: {error_info}</div>', unsafe_allow_html=True)
                    
                    # Provide troubleshooting suggestions
                    with st.expander("Troubleshooting Suggestions", expanded=True):
                        st.markdown("""
                        **Common solutions:**
                        - Verify ticker symbols are correct (e.g., AAPL not Apple)
                        - Try popular stocks: AAPL, MSFT, GOOGL, AMZN, TSLA
                        - Ensure tickers are from major exchanges (NYSE, NASDAQ)
                        - Check your internet connection
                        - Use Quick Start options above
                        """)

# Tips and Guidelines
                else:
                    # Show data fetch results
                    if error_info:  # Some tickers failed
                        st.markdown(f'<div class="status-warning">Could not fetch data for: {error_info}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="status-success">Successfully loaded: {", ".join(optimizer.tickers)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-success">Successfully loaded all {len(optimizer.tickers)} assets</div>', unsafe_allow_html=True)
                    
                    # Get risk-free rate
                    if include_rf:
                        rf_rate = optimizer.get_risk_free_rate()
                        st.info(f"Current risk-free rate: {rf_rate:.2%}")
                    
                    # Run optimization
                    result = optimizer.optimize_portfolio(
                        method=selected_method,
                        target_return=target_return,
                        target_volatility=target_volatility,
                        min_weight=min_weight,
                        max_weight=max_weight
                    )
                    
                    if result['success']:
                        st.session_state.optimizer = optimizer
                        st.session_state.optimization_results = result
                        st.markdown('<div class="status-success">Portfolio optimization completed successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown(f'<div class="status-error">Optimization failed: {result["error"]}</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f'<div class="status-error">Unexpected error: {str(e)}</div>', unsafe_allow_html=True)

with optimize_col2:
    if st.button("Clear Results", help="Clear current optimization results", use_container_width=True):
        if 'optimizer' in st.session_state:
            del st.session_state.optimizer
        if 'optimization_results' in st.session_state:
            del st.session_state.optimization_results
        st.rerun()

# Display Results
if st.session_state.optimization_results and st.session_state.optimizer:
    optimizer = st.session_state.optimizer
    result = st.session_state.optimization_results
    
    # Results Section
    st.markdown("""
        <div class="results-section">
            <div class="results-title">Optimization Results</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['expected_return']*100:.1f}%</span>
                <div class="metric-label">Expected Return</div>
                <div class="metric-change">Annualized</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['volatility']*100:.1f}%</span>
                <div class="metric-label">Volatility</div>
                <div class="metric-change">Annual Risk</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['sharpe_ratio']:.2f}</span>
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-change">Risk-Adjusted</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        diversification = result['diversification_ratio']
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{diversification:.1f}</span>
                <div class="metric-label">Diversification</div>
                <div class="metric-change">Effective Assets</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        max_dd = abs(result['max_drawdown']) * 100
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{max_dd:.1f}%</span>
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-change">Historical</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Tabbed Results Display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Composition",
        "Efficient Frontier", 
        "Risk Analysis",
        "Performance Analytics",
        "CAPM Analysis"
    ])
    
    with tab1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Portfolio composition visualization
        composition_fig = create_portfolio_composition_chart(optimizer.tickers, result['weights'])
        st.plotly_chart(composition_fig, use_container_width=True)
        
        # Detailed allocation table
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Asset Allocation")
            allocation_data = []
            for i, ticker in enumerate(optimizer.tickers):
                allocation_data.append({
                    'Asset': ticker,
                    'Weight': f"{result['weights'][i]:.1%}",
                    'Value ($100K)': f"${result['weights'][i]*100000:,.0f}"
                })
            
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Risk Attribution")
            risk_data = []
            for i, ticker in enumerate(optimizer.tickers):
                risk_data.append({
                    'Asset': ticker,
                    'Risk Contribution': f"{result['risk_contribution'][i]:.1%}",
                    'Marginal Risk': f"{result['marginal_contribution'][i]:.3f}"
                })
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, hide_index=True, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        with st.spinner("Generating efficient frontier..."):
            frontier_fig = create_efficient_frontier_plot(optimizer, result)
            if frontier_fig:
                st.plotly_chart(frontier_fig, use_container_width=True)
            else:
                st.warning("Could not generate efficient frontier")
        
        # Frontier statistics
        st.subheader("Frontier Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Position", "Optimal", "On efficient frontier")
        with col2:
            improvement = (result['sharpe_ratio'] - optimizer.mean_returns.mean()/optimizer.returns.std().mean()) / (optimizer.mean_returns.mean()/optimizer.returns.std().mean()) * 100
            st.metric("Sharpe Improvement", f"{improvement:.1f}%", "vs Equal Weight")
        with col3:
            st.metric("Optimization Method", result['method'].replace('_', ' ').title())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        risk_analysis_fig = create_risk_return_analysis(optimizer, result['weights'])
        st.plotly_chart(risk_analysis_fig, use_container_width=True)
        
        # Risk metrics summary
        st.subheader("Risk Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"{abs(result['var_95'])*100:.2f}%", "Daily")
        with col2:
            st.metric("VaR (99%)", f"{abs(result['var_99'])*100:.2f}%", "Daily")
        with col3:
            concentration = np.sum(result['weights']**2)
            st.metric("Concentration", f"{concentration:.3f}", "Herfindahl Index")
        with col4:
            max_weight = np.max(result['weights'])
            st.metric("Max Weight", f"{max_weight:.1%}", "Largest Position")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        performance_fig = create_performance_analytics(optimizer, result['weights'])
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Performance statistics
        portfolio_returns = (optimizer.returns * result['weights']).sum(axis=1)
        
        st.subheader("Performance Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (1 + portfolio_returns).prod() - 1
            st.metric("Total Return", f"{total_return:.1%}", f"{lookback_years} Year Period")
        
        with col2:
            annualized_return = (1 + total_return) ** (1/lookback_years) - 1
            st.metric("Annualized Return", f"{annualized_return:.1%}")
        
        with col3:
            winning_days = (portfolio_returns > 0).sum() / len(portfolio_returns)
            st.metric("Winning Days", f"{winning_days:.1%}")
        
        with col4:
            avg_daily_return = portfolio_returns.mean()
            st.metric("Avg Daily Return", f"{avg_daily_return:.3%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        if show_capm:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            capm_metrics = optimizer.calculate_capm_metrics()
            
            if capm_metrics:
                capm_fig = create_capm_analysis_chart(capm_metrics)
                if capm_fig:
                    st.plotly_chart(capm_fig, use_container_width=True)
                
                # CAPM summary table
                st.subheader("CAPM Summary")
                capm_data = []
                for ticker in optimizer.tickers:
                    if ticker in capm_metrics:
                        capm_data.append({
                            'Asset': ticker,
                            'Beta': f"{capm_metrics[ticker]['beta']:.3f}",
                            'Alpha': f"{capm_metrics[ticker]['alpha']:.2%}",
                            'Expected Return': f"{capm_metrics[ticker]['expected_return']:.2%}",
                            'R-Squared': f"{capm_metrics[ticker]['r_squared']:.3f}"
                        })
                
                if capm_data:
                    capm_df = pd.DataFrame(capm_data)
                    st.dataframe(capm_df, hide_index=True, use_container_width=True)
            else:
                st.warning("CAPM analysis not available - market data could not be fetched")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("CAPM analysis disabled. Enable in Advanced Options to view.")

# Debug Section for troubleshooting
with st.expander("Debug & Troubleshooting", expanded=False):
    st.markdown("### Test Individual Tickers")
    debug_ticker = st.text_input("Enter a single ticker to test:", value="AAPL")
    
    if st.button("Test Ticker", key="debug_test"):
        if debug_ticker.strip():
            with st.spinner(f"Testing {debug_ticker.upper()}..."):
                try:
                    from optimizer import PortfolioOptimizer
                    test_optimizer = PortfolioOptimizer([debug_ticker.upper()])
                    success, message = test_optimizer.quick_test_ticker(debug_ticker.upper())
                    
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                        
                except Exception as e:
                    st.error(f"❌ Error testing {debug_ticker.upper()}: {str(e)}")
        else:
            st.warning("Please enter a ticker symbol")
    
    st.markdown("""
    ### Common Issues & Solutions
    
    **"No valid ticker symbols found"**
    - Check internet connection
    - Verify ticker symbols are correct (e.g., AAPL not Apple)
    - Try well-known tickers: AAPL, MSFT, GOOGL, AMZN, TSLA
    
    **Buttons not working**
    - Try refreshing the page
    - Clear browser cache
    - Use the Quick Start options above
    
    **Data fetching errors**
    - Yahoo Finance may be temporarily unavailable
    - Try again in a few minutes
    - Use fewer tickers (2-5 assets work best)
    """)
with st.expander("Tips & Information", expanded=False):
    st.markdown("""
    ### Optimization Methods
    
    **Maximum Sharpe Ratio**: Finds the portfolio with the best risk-adjusted return. This is often considered the optimal portfolio for risk-averse investors.
    
    **Minimum Variance**: Finds the portfolio with the lowest possible risk, regardless of return. Good for very conservative investors.
    
    **Target Return**: Finds the minimum risk portfolio that achieves a specific return target. Useful when you have a return requirement.
    
    **Target Risk**: Finds the maximum return portfolio for a specific risk level. Good when you have a risk budget.
    
    ### Key Metrics Explained
    
    - **Expected Return**: Annualized expected portfolio return based on historical data
    - **Volatility**: Annualized portfolio standard deviation (risk measure)
    - **Sharpe Ratio**: Risk-adjusted return measure (return per unit of risk)
    - **VaR (Value at Risk)**: Maximum expected loss over a given time period at a certain confidence level
    - **Maximum Drawdown**: Largest peak-to-trough decline in portfolio value
    - **Diversification Ratio**: Effective number of independent bets in the portfolio
    
    ### Important Considerations
    
    - Results are based on historical data and may not predict future performance
    - Consider transaction costs, taxes, and rebalancing frequency in real implementations
    - Market conditions can change rapidly, requiring periodic reoptimization
    - Diversification does not guarantee profits or protect against losses
    
    ### Data Requirements
    
    - Minimum 2 assets required for optimization
    - At least 100 trading days of historical data per asset
    - Valid ticker symbols from major exchanges (NYSE, NASDAQ, etc.)
    - Stable internet connection for real-time data fetching
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>QuantRisk Analytics</strong> | Advanced Portfolio Optimization Platform</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Built with Modern Portfolio Theory | Real-time Market Data | Professional Risk Analytics</p>
    </div>
""", unsafe_allow_html=True)