# var_app.py

import streamlit as st
# Add these imports at the top (replace the existing src imports):
from src.parametric import (
    compute_parametric_var, 
    create_var_gauge, 
    plot_interactive_return_distribution,  # FIXED
    create_risk_dashboard,
    plot_animated_pnl_vs_var
)
from src.fixed_income import (
    compute_fixed_income_var, 
    create_bond_analytics_dashboard,
    create_yield_scenario_analysis,  # FIXED
    create_yield_curve_analysis
)
from src.portfolio import (
    compute_portfolio_var, 
    create_portfolio_risk_dashboard,
    create_risk_attribution_treemap,
    create_correlation_network,
    plot_enhanced_portfolio_pnl_vs_var
)
from src.monte_carlo import (
    compute_monte_carlo_var, 
    create_monte_carlo_dashboard,  # FIXED
    create_realtime_simulation_progress,
    create_3d_simulation_visualization,
    plot_enhanced_correlation_matrix
)
import pandas as pd
import numpy as np
import time

# Page config
st.set_page_config(page_title="QuantRisk VaR Analytics", layout="wide", page_icon="")

# Enhanced CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }

        /* Hero Section */
        .var-hero {
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

        .var-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: shine 4s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .var-hero h1 {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -1px;
            position: relative;
            z-index: 1;
        }
        
        .var-hero p {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 1rem;
            max-width: 700px;
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

        /* Mode Selection Cards */
        .mode-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }

        .mode-card {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 2px solid transparent;
            cursor: pointer;
            height: 100%;
            position: relative;
            overflow: hidden;
        }

        .mode-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .mode-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        
        .mode-icon {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
            display: block;
            text-align: center;
        }
        
        .mode-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .mode-description {
            color: #666;
            text-align: center;
            line-height: 1.6;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .mode-features {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .mode-features li {
            padding: 0.3rem 0;
            color: #555;
            font-size: 0.9rem;
            position: relative;
            padding-left: 1.5rem;
        }

        .mode-features li::before {
            content: "✓";
            color: #27ae60;
            font-weight: bold;
            position: absolute;
            left: 0;
        }

        /* Results Container */
        .results-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            border-left: 5px solid #667eea;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }

        .results-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 2rem;
            text-align: center;
        }

        /* Selected Mode */
        .selected-mode {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 15px;
            text-align: center;
            font-weight: 600;
            margin: 2rem 0;
            box-shadow: 0 8px 25px rgba(39, 174, 96, 0.3);
            position: relative;
            overflow: hidden;
        }

        .selected-mode::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            animation: slide 2s infinite;
        }

        @keyframes slide {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        /* Input Styling */
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
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        /* Chart Containers */
        .chart-container {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin: 2rem 0;
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
            .var-hero h1 { font-size: 2.5rem; }
            .var-hero p { font-size: 1.1rem; }
            .hero-stats { gap: 1rem; }
            .mode-grid { grid-template-columns: 1fr; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="var-hero">
        <h1>QuantRisk VaR Analytics</h1>
        <p>Interactive Risk Assessment Platform with Real-Time Analytics</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="stat-number">4</span>
                <span class="stat-label">VaR Methods</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">Real-Time</span>
                <span class="stat-label">Analytics</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">3D</span>
                <span class="stat-label">Visualizations</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">Interactive</span>
                <span class="stat-label">Dashboards</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Back Button
if st.button("Back to Home", help="Return to main dashboard"):
    st.switch_page("streamlit_app.py")

# Mode Selection
st.markdown("## Select Risk Analysis Method")

st.markdown('<div class="mode-grid">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="mode-card">
            <span class="mode-icon"></span>
            <div class="mode-title">Single Asset Analytics</div>
            <div class="mode-description">Parametric VaR with interactive gauges and real-time risk monitoring</div>
            <ul class="mode-features">
                <li>Real-time VaR gauges</li>
                <li>Interactive return distributions</li>
                <li>Risk dashboard</li>
                <li>Animated P&L analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1a, col1b = st.columns(2)
    with col1a:
        if st.button("Equity VaR", key="equity_btn"):
            st.session_state.selected_mode = "Equity Analytics"
    with col1b:
        if st.button("Bond VaR", key="bond_btn"):
            st.session_state.selected_mode = "Fixed Income"

with col2:
    st.markdown("""
        <div class="mode-card">
            <span class="mode-icon"></span>
            <div class="mode-title">Portfolio Analytics</div>
            <div class="mode-description">Comprehensive portfolio risk with 3D visualizations and Monte Carlo simulations</div>
            <ul class="mode-features">
                <li>3D risk visualization</li>
                <li>Real-time simulations</li>
                <li>Risk attribution</li>
                <li>Interactive dashboards</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col2a, col2b = st.columns(2)
    with col2a:
        if st.button("Portfolio VaR", key="portfolio_btn"):
            st.session_state.selected_mode = "Portfolio Analytics"
    with col2b:
        if st.button("Interactive Monte Carlo", key="mc_btn"):
            st.session_state.selected_mode = "Interactive Monte Carlo"

st.markdown('</div>', unsafe_allow_html=True)

# Display selected mode
mode = st.session_state.get("selected_mode", None)

if mode:
    st.markdown(f"""
        <div class="selected-mode">
            Active Analysis: <strong>{mode}</strong>
            <br><small>Interactive analytics enabled</small>
        </div>
    """, unsafe_allow_html=True)


if mode == "Equity Analytics":
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown('<div class="input-title">Equity Risk Configuration</div>', unsafe_allow_html=True)

    # Simplified inputs for demonstration
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        position = st.number_input("Position Size ($)", value=1000000, step=100000)
    with col2:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
    with col3:
        enable_advanced = st.checkbox("Enable Analytics", value=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Run VaR Analysis", key="run_advanced_equity"):
        if ticker:
            with st.spinner("Computing parametric VaR with interactive analytics..."):
                try:
                    results = compute_parametric_var([ticker], confidence_level=confidence, position_size=position)
                    
                    for res in results:
                        if 'error' in res:
                            st.error(f"❌ {res['ticker']}: {res['error']}")
                        else:
                            st.markdown('<div class="results-section">', unsafe_allow_html=True)
                            st.markdown(f'<div class="results-title">Analytics for {res["ticker"]}</div>', unsafe_allow_html=True)
                            
                            # Key Metrics Row
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("1-Day VaR", f"${res['VaR']:.0f}", f"{abs(res['VaR'])/position*100:.2f}% of Portfolio")
                            with col2:
                                st.metric("Daily Volatility", f"{res['daily_volatility']:.2%}", f"Annual: {res['daily_volatility']*np.sqrt(252):.1%}")
                            with col3:
                                st.metric("VaR Breaches", f"{res['num_exceedances']}", f"{res['exceedance_pct']:.1f}% of days")
                            with col4:
                                sharpe = res.get('sharpe_ratio', 0)
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return")

                            # Interactive VaR Gauge
                            st.markdown("### Real-Time Risk Gauge")
                            gauge_fig = create_var_gauge(res['VaR'], position, confidence)
                            st.plotly_chart(gauge_fig, use_container_width=True)                            

                            if enable_advanced:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    try:
                                        st.markdown("### Interactive Return Distribution")
                                        dist_fig = plot_interactive_return_distribution(res['df'], res['VaR'], confidence)
                                        if dist_fig:
                                            st.plotly_chart(dist_fig, use_container_width=True)
                                        else:
                                            st.info("Interactive distribution temporarily unavailable")
                                    except Exception as e:
                                        st.warning(f"Distribution chart issue: {str(e)}")
                                
                                with col2:
                                    try:
                                        st.markdown("### Enhanced P&L Analysis")
                                        pnl_fig = plot_animated_pnl_vs_var(res['df'], res['VaR'], confidence)
                                        if pnl_fig:
                                            st.plotly_chart(pnl_fig, use_container_width=True)
                                        else:
                                            st.info("Enhanced P&L chart temporarily unavailable")
                                    except Exception as e:
                                        st.warning(f"P&L chart issue: {str(e)}")

                                # FIXED: Risk Dashboard with error handling
                                try:
                                    st.markdown("### Comprehensive Risk Dashboard")
                                    dashboard_fig = create_risk_dashboard([res])
                                    if dashboard_fig:
                                        st.plotly_chart(dashboard_fig, use_container_width=True)
                                    else:
                                        st.info("Risk dashboard temporarily unavailable")
                                except Exception as e:
                                    st.warning(f"Dashboard issue: {str(e)}")
                                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Try using a well-known ticker like AAPL, MSFT, or GOOG")



elif mode == "Fixed Income":
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown('<div class="input-title">Fixed Income Analytics</div>', unsafe_allow_html=True)

    ticker = st.text_input("Bond Yield Ticker", value="DGS10").upper()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        maturity = st.number_input("Bond Maturity (Years)", min_value=1, max_value=30, value=10)
    with col2:
        position = st.number_input("Position Size ($)", value=1000000, step=100000)
    with col3:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
    with col4:
        advanced_bond = st.checkbox("Duration & Convexity", value=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Run Bond VaR", key="run_advanced_bond"):
        with st.spinner("Computing fixed income VaR with duration/convexity analytics..."):
            try:
                results = compute_fixed_income_var([ticker], maturity=maturity, confidence_level=confidence, position_size=position)
                
                for res in results:
                    if 'error' in res:
                        st.error(f"❌ {res['ticker']}: {res['error']}")
                    else:
                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        st.markdown(f'<div class="results-title">Bond Analytics for {res["ticker"]}</div>', unsafe_allow_html=True)
                        
                        # Enhanced Metrics with FIXED key access
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            var_display = res.get('VaR_linear', res.get('VaR', 0))
                            st.metric("Linear VaR", f"${var_display:.0f}")
                        with col2:
                            var_quad = res.get('VaR_quadratic', var_display * 1.1)
                            st.metric("Quadratic VaR", f"${var_quad:.0f}")
                        with col3:
                            duration = res.get('duration', 0)
                            st.metric("Duration", f"{duration:.2f} years")
                        with col4:
                            convexity = res.get('convexity', 0)
                            st.metric("Convexity", f"{convexity:.2f}")
                        with col5:
                            st.metric("Current YTM", f"{res['ytm']:.2%}")

                        if advanced_bond and 'duration' in res and res['duration'] > 0:
                            # Bond Analytics Dashboard (FIXED with error handling)
                            try:
                                st.markdown("### Bond Risk Analytics Dashboard")
                                bond_dashboard = create_bond_analytics_dashboard(res)
                                if bond_dashboard:
                                    st.plotly_chart(bond_dashboard, use_container_width=True)
                                else:
                                    st.info("Dashboard temporarily unavailable - displaying basic metrics above")
                            except Exception as e:
                                st.warning(f"Dashboard display issue: {str(e)}")

                            # 3D Yield Scenario Analysis (FIXED with error handling)
                            try:
                                st.markdown("### 3D Yield Scenario Analysis")
                                scenario_fig = create_yield_scenario_analysis(res)
                                if scenario_fig:
                                    st.plotly_chart(scenario_fig, use_container_width=True)
                                else:
                                    st.info("3D analysis temporarily unavailable - using 2D approximation")
                                    
                                    # Fallback: Simple 2D scenario analysis
                                    import plotly.graph_objects as go
                                    yield_changes = np.linspace(-200, 200, 50)
                                    pnl_values = []
                                    for dy in yield_changes:
                                        duration_effect = -res['duration'] * (dy/10000) * res['price'] * position
                                        pnl_values.append(duration_effect)
                                    
                                    fig_2d = go.Figure()
                                    fig_2d.add_trace(go.Scatter(x=yield_changes, y=pnl_values, mode='lines', name='Duration Effect'))
                                    fig_2d.update_layout(title="2D Yield Scenario (Duration Only)", 
                                                        xaxis_title="Yield Change (bps)", 
                                                        yaxis_title="P&L ($)")
                                    st.plotly_chart(fig_2d, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Scenario analysis issue: {str(e)}")

                        st.markdown('</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try using a different bond ticker (e.g., ^IRX, ^TNX) or check your internet connection")


elif mode == "Portfolio Analytics":
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown('<div class="input-title">Portfolio Risk Analytics</div>', unsafe_allow_html=True)
    
    # Simplified portfolio setup
    equity_tickers_input = st.text_input("Equity Tickers (comma-separated)", value="AAPL,MSFT,GOOG").upper()
    bond_tickers_input = st.text_input("Bond Tickers (comma-separated)", value="DGS10").upper()
    
    equity_tickers = [t.strip() for t in equity_tickers_input.split(',') if t.strip()]
    bond_tickers = [t.strip() for t in bond_tickers_input.split(',') if t.strip()]
    
    total_assets = len(equity_tickers) + len(bond_tickers)
    if total_assets > 0:
        equal_weight = 1.0 / total_assets
        equity_weights = [equal_weight] * len(equity_tickers)
        bond_weights = [equal_weight] * len(bond_tickers)
        
        st.info(f"Using equal weights: {equal_weight:.1%} per asset ({len(equity_tickers)} equities, {len(bond_tickers)} bonds)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            position = st.number_input("Portfolio Value ($)", value=1000000, step=100000)
        with col2:
            maturity = st.slider("Bond Maturity (Years)", 1, 30, 10)
        with col3:
            confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
        with col4:
            advanced_portfolio = st.checkbox("Analytics", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Run Portfolio Analysis", key="run_advanced_portfolio"):
            with st.spinner("Computing portfolio VaR with risk attribution..."):
                try:
                    results = compute_portfolio_var(
                        equity_tickers, equity_weights, bond_tickers, bond_weights,
                        confidence_level=confidence, position_size=position, maturity=maturity
                    )
                
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)
                    st.markdown('<div class="results-title">Portfolio Risk Analysis</div>', unsafe_allow_html=True)
                    
                    # Key Portfolio Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Portfolio VaR", f"${results['var_portfolio']:.0f}", f"{abs(results['var_portfolio'])/position*100:.2f}%")
                    with col2:
                        diversification_benefit = results['weighted_var_sum'] - results['var_portfolio']
                        st.metric("Diversification Benefit", f"${diversification_benefit:.0f}", f"{diversification_benefit/results['weighted_var_sum']*100:.1f}%")
                    with col3:
                        st.metric("Portfolio Volatility", f"{results['volatility']*np.sqrt(252):.1%}", "Annualized")
                    with col4:
                        st.metric("VaR Breaches", f"{results['exceedances']}", f"{results['exceedance_pct']:.1f}%")

                    if advanced_portfolio:
                        # Portfolio Risk Dashboard
                        st.markdown("### Interactive Portfolio Risk Dashboard")
                        portfolio_dashboard = create_portfolio_risk_dashboard(results)
                        if portfolio_dashboard:
                            st.plotly_chart(portfolio_dashboard, use_container_width=True)

                        # Risk Attribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Risk Attribution Treemap")
                            treemap_fig = create_risk_attribution_treemap(results)
                            if treemap_fig:
                                st.plotly_chart(treemap_fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### Correlation Network")
                            network_fig = create_correlation_network(results)
                            if network_fig:
                                st.plotly_chart(network_fig, use_container_width=True)

                        # Enhanced P&L Analysis
                        st.markdown("### Enhanced Portfolio P&L Analysis")
                        pnl_fig = plot_enhanced_portfolio_pnl_vs_var(
                            results['return_df'][['PnL', 'VaR_Breach']], 
                            results['var_portfolio'], 
                            confidence
                        )
                        st.plotly_chart(pnl_fig, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif mode == "Interactive Monte Carlo":
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown('<div class="input-title">Interactive Monte Carlo Simulation</div>', unsafe_allow_html=True)
    
    tickers_input = st.text_input("Asset Tickers (comma-separated)", value="AAPL,MSFT,GOOG,TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if tickers:
        num_assets = len(tickers)
        weights = [1.0/num_assets] * num_assets
        
        st.info(f"Portfolio: {len(tickers)} assets with equal {100/num_assets:.1f}% weights")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            position = st.number_input("Portfolio Value ($)", value=1000000, step=100000)
        with col2:
            sims = st.number_input("Simulations", value=10000, step=1000, min_value=1000, max_value=50000)
        with col3:
            confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
        with col4:
            realtime_sim = st.checkbox("Real-time Simulation", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Run Interactive Monte Carlo", key="run_interactive_mc"):
            # Check for problematic tickers
            rate_like = ["^IRX", "DTB3", "DTB6", "DTB12", "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2",
                         "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]

            if any(t in rate_like for t in tickers):
                st.warning("⚠️ **Rate Ticker Detected**: Consider using bond ETFs (SHV, BIL, TLT) instead of rate tickers for better simulation accuracy.")
            else:
                if realtime_sim and sims <= 20000:
                    # Real-time simulation with progress
                    st.markdown("### Real-Time Monte Carlo Simulation")
                    
                    progress_placeholder = st.empty()
                    
                    # Simulate real-time progress (simplified for demonstration)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(1, 11):
                        progress = i * 0.1
                        progress_bar.progress(progress)
                        status_text.text(f'Running simulation... {i*10}% complete')
                        time.sleep(0.2)  # Simulate computation time
                    
                    status_text.text('Simulation complete! Generating results...')
                
                # Run actual Monte Carlo
                with st.spinner("Computing Monte Carlo VaR with analytics..."):
                    try:
                        results = compute_monte_carlo_var(
                            tickers=tickers, weights=weights, portfolio_value=position,
                            num_simulations=sims, confidence_level=confidence
                        )

                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        st.markdown('<div class="results-title">Interactive Monte Carlo Results</div>', unsafe_allow_html=True)
                        
                        # Key Results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Monte Carlo VaR", f"${results['VaR_dollar']:,.0f}", f"{results['VaR_pct']:.2%} of Portfolio")
                        with col2:
                            st.metric("Simulations", f"{results['num_simulations']:,}", "Completed")
                        with col3:
                            st.metric("VaR Breaches", f"{results['num_exceedances']}", f"{results['exceedance_pct']:.1f}%")
                        with col4:
                            worst_case = np.min(results['simulated_returns']) * position
                            st.metric("Worst Case", f"${worst_case:,.0f}", "Maximum Loss")

                        try:
                            st.markdown("### Interactive Monte Carlo Dashboard")
                            mc_dashboard = create_monte_carlo_dashboard(results)
                            if mc_dashboard:
                                st.plotly_chart(mc_dashboard, use_container_width=True)
                            else:
                                st.info("Monte Carlo dashboard temporarily unavailable - showing basic results above")
                        except Exception as e:
                            st.warning(f"Dashboard display issue: {str(e)}")

                        # FIXED: Enhanced Correlation Analysis
                        try:
                            st.markdown("### Enhanced Correlation Analysis")
                            corr_fig = plot_enhanced_correlation_matrix(results['returns'])
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                            else:
                                st.info("Correlation matrix temporarily unavailable")
                        except Exception as e:
                            st.warning(f"Correlation analysis issue: {str(e)}")
                                
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Try using fewer simulations or well-known tickers like AAPL, MSFT, GOOG")
                        
# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>QuantRisk Analytics</strong> | Interactive Value-at-Risk Platform</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Real-Time Analytics • 3D Visualizations • Interactive Dashboards • Professional Risk Management</p>
    </div>
""", unsafe_allow_html=True)
