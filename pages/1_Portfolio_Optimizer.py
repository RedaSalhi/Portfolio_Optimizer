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
    page_icon=""
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

        /* Method Selection Buttons */
        .method-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: 2px solid transparent;
            padding: 1.2rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
            cursor: pointer;
            width: 100%;
            margin: 0.5rem 0;
        }

        .method-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        }

        .method-btn.selected {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        }

        /* Secondary Buttons */
        .secondary-button .stButton > button {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
        }

        .success-button .stButton > button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        }

        .danger-button .stButton > button {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
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

        /* Loading Animation */
        .loading-container {
            text-align: center;
            padding: 3rem;
        }

        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 6px solid #f3f3f3;
            border-top: 6px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 2rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
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

        /* Advanced Options */
        .advanced-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 16px;
            margin: 1rem 0;
            border: 2px dashed #667eea;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .optimizer-hero h1 { font-size: 2.5rem; }
            .optimizer-hero p { font-size: 1.1rem; }
            .hero-stats { gap: 1rem; }
            .method-selection { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
            .quick-start-grid { grid-template-columns: 1fr; }
        }

        @media (max-width: 480px) {
            .config-section, .results-section, .input-group { padding: 1.5rem; }
            .metric-card { padding: 1.5rem; }
            .metric-value { font-size: 1.8rem; }
        }

        /* Tooltips */
        .tooltip {
            position: relative;
            cursor: help;
        }

        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #2c3e50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            white-space: nowrap;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        /* Enhanced Expander */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            border: 2px solid #667eea;
            font-weight: 600;
            color: #2c3e50;
        }

        .streamlit-expanderContent {
            background: white;
            border-radius: 0 0 12px 12px;
            border: 2px solid #667eea;
            border-top: none;
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
        <h1>Portfolio Optimizer</h1>
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
    import streamlit as st

    # CSS for styling all buttons inside Streamlit
    st.markdown("""
        <style>
            button[kind="secondary"] {
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
                color: white !important;
                border: none !important;
                padding: 1rem 2rem !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4) !important;
                width: 100% !important;
                position: relative !important;
                overflow: hidden !important;
            }

            button[kind="secondary"]:hover {
                transform: translateY(-3px) !important;
                box-shadow: 0 10px 30px rgba(108, 117, 125, 0.6) !important;
            }

            button[kind="secondary"]::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s ease;
            }

            button[kind="secondary"]:hover::before {
                left: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.button("← Back to Home", help="Return to main dashboard", use_container_width=True):
        st.switch_page("streamlit_app.py")


# Status Display Function
def show_status():
    if st.session_state.optimization_results and st.session_state.optimizer:
        st.markdown('<div class="status-success"> Portfolio optimization active! Explore results in the tabs below.</div>', unsafe_allow_html=True)
    elif st.session_state.data_fetched:
        st.markdown('<div class="status-info"> Data loaded successfully! Configure settings and click optimize.</div>', unsafe_allow_html=True)
    elif 'tickers_input' in st.session_state and len(st.session_state.get('tickers_input', '').split(',')) >= 2:
        st.markdown('<div class="status-warning"> Ready to fetch data! Enter tickers and start optimization.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning"> Enter 2 or more ticker symbols to begin portfolio optimization.</div>', unsafe_allow_html=True)

show_status()

# Configuration Section
st.markdown("""
    <div class="config-section">
        <div class="config-title"> Portfolio Configuration</div>
    </div>
""", unsafe_allow_html=True)

# Asset Selection with enhanced UI
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title"> Asset Selection</div>', unsafe_allow_html=True)

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
        " Historical Period:",
        options=[1, 2, 3, 5],
        index=2,
        help="Years of historical data for analysis"
    )

with col3:
    include_rf = st.checkbox(
        " Risk-Free Rate", 
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
            <div class="quick-start-title"> Quick Start Portfolios</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(" Tech Giants", help="Load technology stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "AAPL, MSFT, GOOGL, AMZN, TSLA"
            st.rerun()
    
    with col2:
        if st.button(" Blue Chips", help="Load blue chip stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "JNJ, PG, KO, WMT, JPM"
            st.rerun()
    
    with col3:
        if st.button(" Diversified ETFs", help="Load diversified ETF portfolio", use_container_width=True):
            st.session_state.tickers_input = "SPY, QQQ, VTI, BND, VEA"
            st.rerun()
    
    with col4:
        if st.button(" FAANG", help="Load FAANG stocks portfolio", use_container_width=True):
            st.session_state.tickers_input = "META, AAPL, AMZN, NFLX, GOOGL"
            st.rerun()

# Optimization Method Selection
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown('<div class="input-title"> Optimization Method</div>', unsafe_allow_html=True)

# Method selection with cards
st.markdown("""
    <style>
        div[data-testid="stButton"] > button.method-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: 2px solid transparent;
            padding: 1.2rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
            cursor: pointer;
            width: 100%;
            margin: 0.5rem 0;
            text-align: center;
        }

        div[data-testid="stButton"] > button.method-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        }

        div[data-testid="stButton"] > button.method-btn.selected {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        }

        div[data-testid="stButton"] > button.analysis-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 1.2rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stButton"] > button.analysis-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(40, 167, 69, 0.6);
        }

        div[data-testid="stButton"] > button.analysis-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }

        div[data-testid="stButton"] > button.analysis-btn:hover::before {
            left: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Method selection with cards
st.markdown('<div class="method-selection">', unsafe_allow_html=True)

methods_info = {
    'max_sharpe': {
        'title': 'Maximum Sharpe Ratio',
        'description': 'Optimize for the best risk-adjusted return. Maximizes return per unit of risk.',
        'icon': '',
        'color': '#667eea'
    },
    'min_variance': {
        'title': 'Minimum Variance',
        'description': 'Minimize portfolio risk and volatility. Best for conservative investors.',
        'icon': '',
        'color': '#28a745'
    },
    'target_return': {
        'title': 'Target Return',
        'description': 'Achieve specific return with minimum risk. Set your return goal.',
        'icon': '',
        'color': '#ffc107'
    },
    'target_volatility': {
        'title': 'Target Risk',
        'description': 'Achieve specific risk level with maximum return. Control your risk exposure.',
        'icon': '',
        'color': '#e74c3c'
    }
}

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for i, (method_key, method_info) in enumerate(methods_info.items()):
    with cols[i]:
        if st.button(
            f"{method_info['title']}", 
            help=method_info['description'],
            key=f"method_{method_key}",
            use_container_width=True,
            type="primary" if st.session_state.selected_method == method_key else "secondary"
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
    st.markdown("### Target Return Configuration")
    target_return = st.slider(
        "Target Annual Return:",
        min_value=0.05,
        max_value=0.30,
        value=0.12,
        step=0.01,
        format="%.1f%%",
        help="Desired annual return percentage"
    )
    st.info(f"Target: {target_return*100:.1f}% annual return")

elif selected_method == 'target_volatility':
    st.markdown("### Target Volatility Configuration")
    target_volatility = st.slider(
        "Target Annual Volatility:",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
        format="%.1f%%",
        help="Desired annual volatility percentage"
    )
    st.info(f"Target: {target_volatility*100:.1f}% annual volatility")

st.markdown('</div>', unsafe_allow_html=True)

# Advanced Options
with st.expander(" Advanced Options", expanded=False):
    st.markdown('<div class="advanced-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Analysis Options")
        show_capm = st.checkbox(" CAPM Analysis", value=True, help="Show Capital Asset Pricing Model analysis")
        frontier_points = st.slider(" Efficient Frontier Points", 25, 100, 50, help="Number of points for efficient frontier")
        
    with col2:
        st.markdown("#### Weight Constraints")
        min_weight = st.slider(" Minimum Asset Weight", 0.0, 0.2, 0.0, step=0.01, format="%.1f%%")
        max_weight = st.slider(" Maximum Asset Weight", 0.2, 1.0, 1.0, step=0.01, format="%.1f%%")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Validation messages with enhanced styling
if not valid_input:
    st.markdown('<div class="status-error"> Please enter at least 2 valid ticker symbols separated by commas</div>', unsafe_allow_html=True)
elif len(tickers) > 10:
    st.markdown('<div class="status-warning"> Using more than 10 assets may slow down optimization</div>', unsafe_allow_html=True)

# Main Action Buttons
st.markdown("### Portfolio Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button(
        "Fetch Data & Optimize Portfolio", 
        disabled=not valid_input, 
        help="Fetch market data and run portfolio optimization", 
        use_container_width=True,
        type="primary"
    ):
        with st.spinner(" Initializing portfolio optimizer..."):
            try:
                # Initialize optimizer
                st.info(" Initializing optimizer...")
                optimizer = PortfolioOptimizer(tickers, lookback_years)
                
                # Validate tickers first
                valid_tickers, invalid_tickers = optimizer.validate_tickers()
                
                if invalid_tickers:
                    st.warning(f" Invalid ticker format detected: {', '.join(invalid_tickers)}")
                
                if len(valid_tickers) < 2:
                    st.error(" Need at least 2 valid tickers for portfolio optimization")
                    st.stop()
                
                # Fetch data
                st.info(" Fetching market data...")
                success, error_info = optimizer.fetch_data()
                
                if not success:
                    st.markdown(f'<div class="status-error"> Failed to fetch data: {error_info}</div>', unsafe_allow_html=True)
                    
                    # Show detailed failure analysis
                    if hasattr(optimizer, 'failed_tickers') and optimizer.failed_tickers:
                        st.warning(f" Failed to fetch data for: {', '.join(optimizer.failed_tickers)}")
                        
                        # Check for common issues
                        if len(optimizer.failed_tickers) == len(tickers):
                            st.error(" **All tickers failed!** This usually indicates:")
                            st.markdown("""
                            - Internet connectivity issues
                            - Yahoo Finance API problems  
                            - Invalid ticker symbols
                            - Regional access restrictions
                            """)
                        elif len(optimizer.failed_tickers) > len(tickers) * 0.5:
                            st.warning(" **Most tickers failed.** Common causes:")
                            st.markdown("""
                            - Network issues or rate limiting
                            - Ticker symbol format problems
                            - Try fewer tickers at once
                            """)
                    
                    # Provide troubleshooting suggestions
                    with st.expander(" Troubleshooting Guide", expanded=True):
                        st.markdown("""
                        **Most Common Issues:**
                        
                         **Invalid Ticker Symbols:**
                        - Make sure you're using the correct stock symbols (AAPL, not Apple Inc.)
                        - Remove any spaces or special characters
                        - Use US stock symbols (NYSE, NASDAQ)
                        
                         **Connection Issues:**
                        - Check your internet connection
                        - Yahoo Finance servers might be busy - try again in 1-2 minutes
                        - Try fewer tickers at once (2-3 maximum)
                        
                         **Market Data Availability:**
                        - Some stocks may not have sufficient historical data
                        - Try major blue-chip stocks: AAPL, MSFT, JNJ, PG, KO
                        - Avoid recently listed companies or penny stocks
                        
                         **Quick Solutions:**
                        """)
                        
                        # Add quick solution buttons
                        sol_col1, sol_col2, sol_col3 = st.columns(3)
                        
                        with sol_col1:
                            if st.button(" Try Blue Chips", help="Use reliable blue chip stocks"):
                                st.session_state.tickers_input = "AAPL, MSFT, JNJ, PG, KO"
                                st.rerun()
                        
                        with sol_col2:
                            if st.button(" Try Top ETFs", help="Use popular ETFs"):
                                st.session_state.tickers_input = "SPY, QQQ, VTI"
                                st.rerun()
                        
                        with sol_col3:
                            if st.button(" Try Finance Sector", help="Use financial stocks"):
                                st.session_state.tickers_input = "JPM, BAC, WFC"
                                st.rerun()
                else:
                    st.session_state.data_fetched = True
                    
                    # Show data fetch results
                    if error_info:  # Some tickers failed
                        st.markdown(f'<div class="status-warning"> Could not fetch data for: {error_info}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="status-success"> Successfully loaded: {", ".join(optimizer.tickers)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-success"> Successfully loaded all {len(optimizer.tickers)} assets!</div>', unsafe_allow_html=True)
                    
                    # Display data quality information
                    if optimizer.data_quality_info:
                        st.markdown("#### Data Quality Report")
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
                        with st.spinner(" Fetching risk-free rate..."):
                            rf_rate = optimizer.get_risk_free_rate()
                            st.info(f" Current risk-free rate: {rf_rate:.2%}")
                    
                    # Run optimization
                    st.info(f" Running {methods_info[selected_method]['title']} optimization...")
                    
                    with st.spinner(" Optimizing portfolio..."):
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
                            st.info(f"✅ Optimization converged in {details.get('iterations', 'N/A')} iterations")
                        
                        st.rerun()
                    else:
                        st.markdown(f'<div class="status-error">❌ Optimization failed: {result["error"]}</div>', unsafe_allow_html=True)
                        
                        # Provide optimization troubleshooting
                        with st.expander(" Optimization Troubleshooting", expanded=True):
                            st.markdown("""
                            **Possible issues and solutions:**
                            
                            **Target too aggressive:** Try more realistic return/risk targets
                            
                            **Constraint conflicts:** Check min/max weight constraints
                            
                            **Market data issues:** Some assets may have insufficient data
                            
                            **Try different method:** Switch to Maximum Sharpe or Minimum Variance
                            """)
                        
            except Exception as e:
                st.markdown(f'<div class="status-error">💥 Unexpected error: {str(e)}</div>', unsafe_allow_html=True)
                
                with st.expander("🐛 Error Details", expanded=False):
                    st.code(str(e))
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="danger-button">', unsafe_allow_html=True)
    if st.button("Clear Results", help="Clear current optimization results", use_container_width=True, type="secondary"):
        # Clear all session state
        for key in ['optimizer', 'optimization_results', 'data_fetched']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Display Results Section
if st.session_state.optimization_results and st.session_state.optimizer:
    optimizer = st.session_state.optimizer
    result = st.session_state.optimization_results
    
    # Results Section with enhanced styling
    st.markdown("""
        <div class="results-section">
            <div class="results-title"> Optimization Results</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Key Metrics Grid
    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        ret_change = "+Optimized" if result['expected_return'] > 0 else ""
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['expected_return']*100:.1f}%</span>
                <div class="metric-label">Expected Return</div>
                <div class="metric-change"> Annualized</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['volatility']*100:.1f}%</span>
                <div class="metric-label">Volatility</div>
                <div class="metric-change"> Annual Risk</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sharpe_color = "metric-positive" if result['sharpe_ratio'] > 1 else "metric-negative" if result['sharpe_ratio'] < 0.5 else ""
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{result['sharpe_ratio']:.2f}</span>
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-change {sharpe_color}"> Risk-Adjusted</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        diversification = result['diversification_ratio']
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{diversification:.1f}</span>
                <div class="metric-label">Diversification</div>
                <div class="metric-change"> Effective Assets</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        max_dd = abs(result['max_drawdown']) * 100
        dd_color = "metric-positive" if max_dd < 10 else "metric-negative" if max_dd > 20 else ""
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value">{max_dd:.1f}%</span>
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-change {dd_color}"> Historical</div>
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
                    <div class="metric-change"> Downside Risk</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cvar = abs(result.get('cvar_95', 0)) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{cvar:.1f}%</span>
                    <div class="metric-label">CVaR (95%)</div>
                    <div class="metric-change"> Tail Risk</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            concentration = result.get('concentration', 0)
            st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{concentration:.3f}</span>
                    <div class="metric-label">Concentration</div>
                    <div class="metric-change"> HHI Index</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Tabbed Results Display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Composition",
        "Efficient Frontier", 
        "Risk Analysis",
        "Performance Analytics",
        "CAPM Analysis"
    ])

    # Add custom CSS for tab styling
    st.markdown("""
        <style>
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
        </style>
    """, unsafe_allow_html=True)

    with tab1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Portfolio composition visualization
        composition_fig = create_portfolio_composition_chart(
            optimizer.tickers, 
            result['weights'],
            result.get('rf_weight') if selected_method in ['target_return', 'target_volatility'] else None
        )
        st.plotly_chart(composition_fig, use_container_width=True)
        
        # Enhanced allocation tables
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Asset Allocation")
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
            st.markdown("#### Risk Attribution")
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
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        frontier_fig = create_efficient_frontier_plot(optimizer, result)
        if frontier_fig:
            st.plotly_chart(frontier_fig, use_container_width=True)
        else:
            st.warning(" Could not generate efficient frontier. Try reducing the number of assets or using different optimization method.")
        
        # Enhanced frontier statistics
        st.markdown("#### Frontier Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Position", 
                "Optimal", 
                "On efficient frontier",
                help="Portfolio lies on the efficient frontier"
            )
        
        with col2:
            equal_weight_return = optimizer.mean_returns.mean()
            equal_weight_sharpe = (equal_weight_return - optimizer.rf_rate) / optimizer.returns.mean(axis=1).std()
            improvement = ((result['sharpe_ratio'] - equal_weight_sharpe) / equal_weight_sharpe * 100) if equal_weight_sharpe != 0 else 0
            st.metric(
                "Sharpe Improvement", 
                f"{improvement:.1f}%", 
                "vs Equal Weight",
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
                st.metric(
                    "Convergence", 
                    f"{iterations} iterations",
                    " Successful" if result['optimization_details'].get('converged', False) else " Warning",
                    help="Optimization convergence details"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        risk_analysis_fig = create_risk_return_analysis(optimizer, result['weights'])
        st.plotly_chart(risk_analysis_fig, use_container_width=True)
        
        # Enhanced risk metrics summary
        st.markdown("#### Risk Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_95 = abs(result['var_95']) * 100
            st.metric("VaR (95%)", f"{var_95:.2f}%", "Daily", help="Value at Risk at 95% confidence level")
        
        with col2:
            var_99 = abs(result['var_99']) * 100
            st.metric("VaR (99%)", f"{var_99:.2f}%", "Daily", help="Value at Risk at 99% confidence level")
        
        with col3:
            max_weight = np.max(result['weights'])
            st.metric("Max Weight", f"{max_weight:.1%}", "Largest Position", help="Weight of largest position")
        
        with col4:
            min_weight = np.min(result['weights'])
            st.metric("Min Weight", f"{min_weight:.1%}", "Smallest Position", help="Weight of smallest position")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        performance_fig = create_performance_analytics(optimizer, result['weights'])
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Enhanced performance statistics
        portfolio_returns = (optimizer.returns * result['weights']).sum(axis=1)
        
        st.markdown("#### Performance Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (1 + portfolio_returns).prod() - 1
            st.metric(
                "Total Return", 
                f"{total_return*100:.1f}%", 
                f"{lookback_years} Year Period",
                help=f"Total return over {lookback_years} year period"
            )
        
        with col2:
            annualized_return = (1 + total_return) ** (1/lookback_years) - 1
            st.metric(
                "Annualized Return", 
                f"{annualized_return*100:.1f}%",
                help="Compound annual growth rate"
            )
        
        with col3:
            winning_days = (portfolio_returns > 0).sum() / len(portfolio_returns)
            st.metric(
                "Winning Days", 
                f"{winning_days*100:.1f}%",
                help="Percentage of positive return days"
            )
        
        with col4:
            volatility_realized = portfolio_returns.std() * np.sqrt(252)
            st.metric(
                "Realized Volatility", 
                f"{volatility_realized*100:.1f}%",
                help="Historical volatility of the portfolio"
            )
        
        # Additional performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            skewness = portfolio_returns.skew()
            skew_interpretation = "Positive Skew" if skewness > 0 else "Negative Skew" if skewness < 0 else "Symmetric"
            st.metric(
                "Skewness", 
                f"{skewness:.2f}",
                skew_interpretation,
                help="Asymmetry of return distribution"
            )
        
        with col2:
            kurtosis = portfolio_returns.kurtosis()
            kurt_interpretation = "Fat Tails" if kurtosis > 0 else "Thin Tails"
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
                help="Average daily return"
            )
        
        with col4:
            best_day = portfolio_returns.max() * 100
            st.metric(
                "Best Day", 
                f"{best_day:.2f}%",
                help="Best single day performance"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        if show_capm:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            with st.spinner(" Calculating CAPM metrics..."):
                capm_metrics = optimizer.calculate_capm_metrics()
            
            if capm_metrics:
                capm_fig = create_capm_analysis_chart(capm_metrics)
                if capm_fig:
                    st.plotly_chart(capm_fig, use_container_width=True)
                
                # Enhanced CAPM summary
                st.markdown("#### CAPM Analysis Summary")
                
                # Create enhanced CAPM table
                capm_data = []
                for ticker in optimizer.tickers:
                    if ticker in capm_metrics:
                        metrics = capm_metrics[ticker]
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
                    capm_df = pd.DataFrame(capm_data)
                    st.dataframe(capm_df, hide_index=True, use_container_width=True)
                    
                    # CAPM interpretation
                    st.markdown("#### CAPM Interpretation")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **Beta Analysis:**
                        - **Beta > 1:** More volatile than market
                        - **Beta < 1:** Less volatile than market  
                        - **Beta = 1:** Same volatility as market
                        """)
                    
                    with col2:
                        st.markdown("""
                        **Alpha Analysis:**
                        - **Alpha > 0:** Outperforming expectations
                        - **Alpha < 0:** Underperforming expectations
                        - **Alpha ≈ 0:** Performing as expected
                        """)
            else:
                st.markdown('<div class="status-warning"> CAPM analysis not available - market data could not be fetched</div>', unsafe_allow_html=True)
                
                st.markdown("""
                **Why CAPM might not be available:**
                - Market benchmark data unavailable
                - Insufficient overlapping data between assets and market
                - Network connectivity issues
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-info"> CAPM analysis disabled. Enable in Advanced Options to view detailed beta and alpha analysis.</div>', unsafe_allow_html=True)

# Debug and Troubleshooting Section
with st.expander(" Debug & Troubleshooting", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Test Individual Tickers")
        debug_ticker = st.text_input("Enter a single ticker to test:", value="AAPL", help="Test if a ticker is valid and has data")
        
        if st.button(" Test Ticker", key="debug_test"):
            if debug_ticker.strip():
                with st.spinner(f"Testing {debug_ticker.upper()}..."):
                    try:
                        test_optimizer = PortfolioOptimizer([debug_ticker.upper()])
                        
                        # Try the simple test first
                        success, message, raw_data = test_optimizer.simple_ticker_test(debug_ticker.upper())
                        
                        if success:
                            st.success(f" Basic Test: {message}")
                            
                            # Show raw data structure
                            st.markdown(" Raw Data Structure")
                            st.write("**Data Shape:**", raw_data.shape)
                            st.write("**Columns:**", list(raw_data.columns))
                            st.write("**Index Type:**", type(raw_data.index).__name__)
                            st.write("**First few rows:**")
                            st.dataframe(raw_data.head())
                            
                            # Now try full extraction test
                            full_success, full_message = test_optimizer.quick_test_ticker(debug_ticker.upper())
                            
                            if full_success:
                                st.success(f" Full Test: {full_message}")
                            else:
                                st.error(f" Full Test Failed: {full_message}")
                                
                        else:
                            st.error(f" Basic Test Failed: {message}")
                            
                    except Exception as e:
                        st.error(f" Critical Error: {str(e)}")
                        
                        # Show detailed traceback for debugging
                        st.markdown(" Error Details", expanded=False)
                        import traceback
                        st.code(traceback.format_exc())
                            
            else:
                st.warning("Please enter a ticker symbol")
    
    with col2:
        st.markdown("#### System Information")
        
        if st.session_state.optimizer:
            optimizer = st.session_state.optimizer
            debug_info = optimizer.get_debug_info()
            
            st.info(f"""
            **Current Status:**
            - Assets requested: {len(debug_info['tickers_requested'])}
            - Assets loaded: {len(debug_info['successful_tickers'])}
            - Failed assets: {len(debug_info['failed_tickers'])}
            - Data points: {debug_info['data_points']}
            - Date range: {debug_info['data_date_range']['start'] if debug_info['data_date_range'] else 'N/A'} to {debug_info['data_date_range']['end'] if debug_info['data_date_range'] else 'N/A'}
            - Risk-free rate: {debug_info['risk_free_rate']:.2%}
            - Market data: {'Available' if debug_info['market_data_available'] else 'Not available'}
            """)
            
            if debug_info['failed_tickers']:
                st.warning(f"Failed tickers: {', '.join(debug_info['failed_tickers'])}")
                
            if debug_info['successful_tickers']:
                st.success(f"Successful tickers: {', '.join(debug_info['successful_tickers'])}")
        else:
            st.info("No optimizer initialized yet")
    
    # Add a data fetch test section
    st.markdown("#### Data Fetch Test")
    col1, col2 = st.columns(2)
    
    with col1:
        test_ticker = st.text_input("Test a ticker:", value="AAPL", help="Test individual ticker data availability")
        
    with col2:
        if st.button(" Test Data Fetch", key="test_fetch"):
            if test_ticker.strip():
                with st.spinner(f"Testing data fetch for {test_ticker.upper()}..."):
                    try:
                        import yfinance as yf
                        import pandas as pd
                        
                        # Simple test
                        data = yf.download(test_ticker.upper(), period='30d', progress=False)
                        
                        if data.empty:
                            st.error(f" No data available for {test_ticker.upper()}")
                        else:
                            # Test our data extraction method
                            try:
                                temp_optimizer = PortfolioOptimizer([test_ticker.upper()])
                                price_series = temp_optimizer._extract_price_series(data, test_ticker.upper())
                                
                                if price_series is not None and len(price_series) > 0:
                                    st.success(f" Successfully fetched and extracted {len(price_series)} price points for {test_ticker.upper()}")
                                    
                                    # Show basic info - handle both single and multi-index columns
                                    try:
                                        if isinstance(data.columns, pd.MultiIndex):
                                            columns_list = [f"{col[0]}" for col in data.columns]
                                        else:
                                            columns_list = list(data.columns)
                                        
                                        latest_price = price_series.iloc[-1]
                                        
                                        st.info(f"""
                                        **Data Info:**
                                        - Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}
                                        - Columns: {', '.join(columns_list)}
                                        - Latest price: ${latest_price:.2f}
                                        - Data extraction: Success
                                        """)
                                    except Exception as info_error:
                                        st.info(f"Data extracted successfully but couldn't parse details: {str(info_error)}")
                                        
                                else:
                                    st.warning(f" Data fetched but price extraction failed for {test_ticker.upper()}")
                                    
                            except Exception as extract_error:
                                st.warning(f" Data fetched but extraction test failed: {str(extract_error)}")
                                st.success(f" Basic fetch successful for {test_ticker.upper()} ({len(data)} rows)")
                            
                    except Exception as e:
                        st.error(f" Error testing {test_ticker.upper()}: {str(e)}")
            else:
                st.warning("Please enter a ticker symbol")

# Tips and Information Section
with st.expander(" Tips & Best Practices", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Optimization Methods Guide
        
        **Maximum Sharpe Ratio**
        - Best for: General portfolio optimization
        - Goal: Maximize risk-adjusted returns
        - Ideal for: Most investors seeking balance
        
        **Minimum Variance**
        - Best for: Conservative investors
        - Goal: Minimize portfolio risk
        - Ideal for: Risk-averse investors, stable income needs
        
        **Target Return**
        - Best for: Specific return goals
        - Goal: Achieve target with minimum risk
        - Ideal for: Pension funds, endowments
        
        **Target Volatility**
        - Best for: Risk budgeting
        - Goal: Maximize return for specific risk
        - Ideal for: Risk-managed strategies
        """)
    
    with col2:
        st.markdown("""
        #### Key Metrics Explained
        
        **Expected Return**: Annualized expected portfolio return
        **Volatility**: Annual portfolio standard deviation (risk)
        **Sharpe Ratio**: Return per unit of risk (higher is better)
        **Sortino Ratio**: Risk-adjusted return using downside deviation
        **VaR**: Maximum expected loss at confidence level
        **CVaR**: Average loss beyond VaR threshold
        **Max Drawdown**: Largest peak-to-trough decline
        **Diversification Ratio**: Effective number of independent positions
        **Beta**: Sensitivity to market movements
        **Alpha**: Excess return above market expectations
        """)
    
    st.markdown("""
    #### Important Considerations
    
     **Data Limitations**
    - Results based on historical data (past performance ≠ future results)
    - Market conditions change rapidly
    - Consider transaction costs and taxes in real implementation
    
     **Portfolio Implementation**
    - Rebalance periodically (quarterly/annually)
    - Consider market impact of trades
    - Account for bid-ask spreads and commissions
    - Monitor portfolio drift from target weights
    
     **Risk Management**
    - Diversification doesn't guarantee profits
    - Monitor correlation changes over time
    - Consider regime changes and black swan events
    - Use multiple risk metrics for comprehensive analysis
    
     **Practical Usage**
    - Start with 3-6 well-known assets
    - Use 2-3 years of data for stable results
    - Test with paper trading before real implementation
    - Consider your investment timeline and liquidity needs
    """)

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;">
        <p style="font-size: 1.2rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                QuantRisk Analytics
            </span>
        </p>
        <p style="font-size: 1rem; color: #667eea; margin-bottom: 0.5rem;">Advanced Portfolio Optimization Platform</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Built with Modern Portfolio Theory | Real-time Market Data | Professional Risk Analytics</p>
        <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 1rem;">© 2025 | Powered by Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)