# pages/1_Portfolio_Optimizer.py - Enhanced Interactive Portfolio Optimizer

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from optimizer import (
    optimize_portfolio_advanced,
    create_interactive_efficient_frontier,
    create_3d_risk_return_time_surface,
    create_portfolio_composition_sunburst,
    create_risk_attribution_dashboard,
    create_performance_attribution_dashboard
)

# Set page configuration
st.set_page_config(page_title="Advanced Portfolio Optimizer", layout="wide", page_icon="📈")

# Enhanced CSS styling
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
            animation: shine 4s infinite;
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
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
            position: relative;
            z-index: 1;
        }

        .hero-features {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 2rem;
            position: relative;
            z-index: 1;
        }

        .hero-feature {
            background: rgba(255,255,255,0.2);
            padding: 1rem 1.5rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }

        .feature-icon {
            font-size: 1.5rem;
            font-weight: 800;
            display: block;
        }

        .feature-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        /* Configuration Section */
        .config-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2.5rem;
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

        .config-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border-top: 4px solid #667eea;
            transition: all 0.3s ease;
        }

        .config-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.12);
        }

        .card-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
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

        /* Method Selection */
        .method-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .method-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }

        .method-card:hover {
            transform: translateY(-3px);
            border-color: #667eea;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        }

        .method-card.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 1rem 2.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        /* Chart containers */
        .chart-container {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin: 2rem 0;
        }

        .chart-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 0 24px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .optimizer-hero h1 { font-size: 2.5rem; }
            .optimizer-hero p { font-size: 1.1rem; }
            .config-card { padding: 1.5rem; }
            .hero-features { gap: 1rem; }
            .method-grid { grid-template-columns: 1fr; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="optimizer-hero">
        <h1>📈 Advanced Portfolio Optimizer</h1>
        <p>Professional-grade portfolio optimization with interactive analytics and real-time visualization</p>
        <div class="hero-features">
            <div class="hero-feature">
                <span class="feature-icon">🎯</span>
                <span class="feature-label">Multiple Optimization Methods</span>
            </div>
            <div class="hero-feature">
                <span class="feature-icon">🌐</span>
                <span class="feature-label">3D Visualizations</span>
            </div>
            <div class="hero-feature">
                <span class="feature-icon">⚡</span>
                <span class="feature-label">Real-Time Analytics</span>
            </div>
            <div class="hero-feature">
                <span class="feature-icon">🎛️</span>
                <span class="feature-label">Interactive Dashboards</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Back navigation
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")

# Configuration Section
st.markdown("""
    <div class="config-section">
        <div class="config-title">⚙️ Advanced Portfolio Configuration</div>
    </div>
""", unsafe_allow_html=True)

# Asset Selection
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🏢 Asset Universe</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    tickers_str = st.text_input(
        "Enter Asset Tickers (comma-separated)", 
        value="AAPL, MSFT, GOOG, AMZN, TSLA",
        help="Enter stock tickers separated by commas"
    )
with col2:
    advanced_features = st.checkbox("🔬 Advanced Features", value=True, help="Enable 3D visualizations and advanced analytics")

st.markdown('</div>', unsafe_allow_html=True)

# Optimization Method Selection
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🎯 Optimization Method</div>', unsafe_allow_html=True)

optimization_methods = {
    'max_sharpe': '📈 Maximum Sharpe Ratio',
    'min_variance': '🛡️ Minimum Variance', 
    'max_return': '🚀 Maximum Return',
    'target_return': '🎯 Target Return',
    'target_volatility': '📊 Target Volatility'
}

selected_method = st.selectbox(
    "Choose optimization objective:",
    options=list(optimization_methods.keys()),
    format_func=lambda x: optimization_methods[x],
    index=0
)

# Method-specific parameters
method_params = {}
if selected_method == 'target_return':
    method_params['target_return'] = st.slider("🎯 Target Annual Return", 0.0, 0.5, 0.15, step=0.01, format="%.1%")
elif selected_method == 'target_volatility':
    method_params['target_volatility'] = st.slider("📊 Target Annual Volatility", 0.05, 0.5, 0.20, step=0.01, format="%.1%")

st.markdown('</div>', unsafe_allow_html=True)

# Advanced Options
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🔧 Advanced Options</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    include_risk_free = st.checkbox("🏦 Include Risk-Free Asset", value=True)
with col2:
    use_sp500 = st.checkbox("📊 CAPM Analysis (S&P 500)", value=True)
with col3:
    num_portfolios = st.number_input("🎲 Simulations", min_value=1000, max_value=50000, value=25000, step=5000)

# Constraints
st.markdown("#### 📋 Portfolio Constraints")
col1, col2 = st.columns(2)
with col1:
    max_weight = st.slider("📏 Maximum Asset Weight", 0.1, 1.0, 1.0, step=0.05, format="%.0%")
with col2:
    min_weight = st.slider("📐 Minimum Asset Weight", 0.0, 0.2, 0.0, step=0.01, format="%.1%")

constraints = {
    'max_weight': max_weight,
    'min_weight': min_weight
}

st.markdown('</div>', unsafe_allow_html=True)

# Run Optimization Button
if st.button("🚀 Run Advanced Portfolio Optimization"):
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    
    if not tickers:
        st.error("❌ Please enter at least one ticker symbol")
    elif len(tickers) < 2:
        st.error("❌ Please enter at least 2 ticker symbols for portfolio optimization")
    else:
        with st.spinner("🔄 Running advanced portfolio optimization... Please wait."):
            try:
                # Run optimization
                results = optimize_portfolio_advanced(
                    tickers=tickers,
                    expected_return=method_params.get('target_return'),
                    expected_std=method_params.get('target_volatility'),
                    include_risk_free=include_risk_free,
                    use_sp500=use_sp500,
                    optimization_method=selected_method,
                    constraints=constraints,
                    num_portfolios=num_portfolios
                )

                # Results Section
                st.markdown("""
                    <div class="results-section">
                        <div class="results-title">📊 Advanced Optimization Results</div>
                    </div>
                """, unsafe_allow_html=True)

                st.success("✅ Portfolio optimization completed successfully!")
                
                # Key Metrics Dashboard
                st.markdown("### 📈 Portfolio Performance Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    expected_return = results['portfolio_return']
                    st.metric("📈 Expected Return", f"{expected_return:.1%}", f"Annual return")
                
                with col2:
                    volatility = results['portfolio_volatility'] 
                    st.metric("📊 Volatility", f"{volatility:.1%}", f"Annual risk")
                
                with col3:
                    sharpe = results['sharpe_ratio']
                    st.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}", f"Risk-adjusted return")
                
                with col4:
                    diversification = results.get('diversification_ratio', 1.0)
                    st.metric("🌐 Diversification", f"{diversification:.2f}", f"Benefit ratio")
                
                with col5:
                    max_dd = results.get('max_drawdown', 0)
                    st.metric("📉 Max Drawdown", f"{max_dd:.1%}", f"Worst loss period")

                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🎯 Portfolio Composition", 
                    "📈 Efficient Frontier", 
                    "🎛️ Risk Analytics", 
                    "📊 Performance Analysis",
                    "🌐 3D Visualization"
                ])

                with tab1:
                    # Portfolio Composition Analysis
                    st.markdown("### 🥧 Optimal Portfolio Weights")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Interactive sunburst chart
                        sunburst_fig = create_portfolio_composition_sunburst(tickers, results['optimal_weights'])
                        st.plotly_chart(sunburst_fig, use_container_width=True)
                    
                    with col2:
                        # Weights table with additional metrics
                        weights_data = []
                        for i, ticker in enumerate(tickers):
                            weight = results['optimal_weights'][i]
                            expected_ret = results['mean_returns'][i]
                            volatility = np.sqrt(results['cov_matrix'][i, i])
                            
                            weights_data.append({
                                "Asset": ticker,
                                "Weight": f"{weight:.1%}",
                                "Expected Return": f"{expected_ret:.1%}",
                                "Volatility": f"{volatility:.1%}",
                                "Risk Contribution": f"{results.get('percent_risk_contrib', [0]*len(tickers))[i]:.1%}"
                            })
                        
                        st.dataframe(weights_data, use_container_width=True)

                with tab2:
                    # Interactive Efficient Frontier
                    st.markdown("### 📈 Interactive Efficient Frontier")
                    
                    frontier_fig = create_interactive_efficient_frontier(results, highlight_portfolios=True)
                    st.plotly_chart(frontier_fig, use_container_width=True)
                    
                    # Special portfolios comparison
                    if 'max_sharpe_portfolio' in results:
                        st.markdown("#### 🏆 Key Portfolio Comparison")
                        
                        special_portfolios = []
                        for name, key in [("Max Sharpe", "max_sharpe_portfolio"), 
                                         ("Min Variance", "min_variance_portfolio"),
                                         ("Max Return", "max_return_portfolio")]:
                            if key in results:
                                portfolio = results[key]
                                special_portfolios.append({
                                    "Portfolio": name,
                                    "Return": f"{portfolio['return']:.1%}",
                                    "Volatility": f"{portfolio['volatility']:.1%}",
                                    "Sharpe Ratio": f"{portfolio['sharpe']:.2f}"
                                })
                        
                        if special_portfolios:
                            st.dataframe(special_portfolios, use_container_width=True)

                with tab3:
                    # Risk Attribution Dashboard
                    st.markdown("### 🎛️ Comprehensive Risk Analysis")
                    
                    risk_dashboard = create_risk_attribution_dashboard(results)
                    if risk_dashboard:
                        st.plotly_chart(risk_dashboard, use_container_width=True)
                    
                    # CAPM Analysis (if available)
                    if 'asset_capm_metrics' in results and use_sp500:
                        st.markdown("#### 📊 CAPM Analysis Results")
                        
                        capm_data = []
                        for ticker in tickers:
                            capm_metrics = results['asset_capm_metrics'].get(ticker, {})
                            capm_data.append({
                                "Asset": ticker,
                                "Alpha": f"{capm_metrics.get('alpha', 0):.2%}",
                                "Beta": f"{capm_metrics.get('beta', 1.0):.2f}",
                                "R-Squared": f"{capm_metrics.get('r_squared', 0):.1%}",
                                "Expected Return (CAPM)": f"{capm_metrics.get('expected_return', 0):.1%}"
                            })
                        
                        st.dataframe(capm_data, use_container_width=True)
                        
                        # Portfolio CAPM metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            portfolio_alpha = results.get('portfolio_alpha', 0)
                            st.metric("Portfolio Alpha", f"{portfolio_alpha:.2%}")
                        with col2:
                            portfolio_beta = results.get('portfolio_beta', 1.0)
                            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
                        with col3:
                            portfolio_r2 = results.get('portfolio_r_squared', 0)
                            st.metric("Portfolio R²", f"{portfolio_r2:.1%}")

                with tab4:
                    # Performance Attribution
                    st.markdown("### 📊 Performance Attribution Analysis")
                    
                    if 'portfolio_returns_series' in results:
                        performance_dashboard = create_performance_attribution_dashboard(results)
                        if performance_dashboard:
                            st.plotly_chart(performance_dashboard, use_container_width=True)
                        
                        # Performance statistics
                        st.markdown("#### 📈 Performance Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            total_return = results.get('total_return', 0)
                            st.metric("Total Return", f"{total_return:.1%}")
                        with col2:
                            annual_return = results.get('annual_return', 0)
                            st.metric("Annual Return", f"{annual_return:.1%}")
                        with col3:
                            annual_vol = results.get('annual_volatility', 0)
                            st.metric("Annual Volatility", f"{annual_vol:.1%}")
                        with col4:
                            max_dd = results.get('max_drawdown', 0)
                            st.metric("Max Drawdown", f"{max_dd:.1%}")
                    else:
                        st.info("📊 Performance attribution requires historical data analysis. Run optimization to generate time series data.")

                with tab5:
                    # 3D Visualization
                    if advanced_features:
                        st.markdown("### 🌐 3D Risk-Return-Time Analysis")
                        
                        viz_3d = create_3d_risk_return_time_surface(results)
                        if viz_3d:
                            st.plotly_chart(viz_3d, use_container_width=True)
                            
                            st.markdown("#### 🎛️ 3D Visualization Controls")
                            st.info("""
                                🖱️ **Interactive Controls:**
                                - **Rotate**: Click and drag to rotate the 3D view
                                - **Zoom**: Use mouse wheel or pinch to zoom in/out
                                - **Pan**: Hold Shift and drag to pan the view
                                - **Hover**: Hover over points to see detailed information
                            """)
                        else:
                            st.info("🌐 3D visualization temporarily unavailable. Showing 2D analysis above.")
                    else:
                        st.info("🔬 Enable 'Advanced Features' in the configuration to access 3D visualizations.")

                # Capital Allocation Analysis (if applicable)
                if include_risk_free and ('risky_weight' in results):
                    st.markdown("### 💰 Capital Allocation Strategy")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risky_weight = results['risky_weight']
                        st.metric("🎯 Risky Asset Weight", f"{risky_weight:.1%}")
                    
                    with col2:
                        risk_free_weight = results['risk_free_weight']
                        st.metric("🏦 Risk-Free Weight", f"{risk_free_weight:.1%}")
                    
                    with col3:
                        leverage = "Yes" if risky_weight > 1.0 else "No"
                        st.metric("⚡ Leverage", leverage)
                    
                    if risky_weight > 1.0:
                        st.warning("⚠️ **Leveraged Portfolio**: This allocation requires borrowing at the risk-free rate.")
                    elif risky_weight < 1.0:
                        st.info("💡 **Conservative Portfolio**: Part of the allocation is in risk-free assets.")

                # Summary and Insights
                st.markdown("### 💡 Key Insights & Recommendations")
                
                insights = []
                
                # Diversification insight
                if diversification > 1.5:
                    insights.append("✅ **Well Diversified**: Your portfolio benefits significantly from diversification.")
                elif diversification < 1.2:
                    insights.append("⚠️ **Limited Diversification**: Consider adding more uncorrelated assets.")
                
                # Risk insight
                if volatility > 0.25:
                    insights.append("🔥 **High Risk**: This portfolio has elevated volatility. Consider risk management.")
                elif volatility < 0.10:
                    insights.append("🛡️ **Conservative**: Low-risk portfolio suitable for conservative investors.")
                
                # Return insight
                if expected_return > 0.15:
                    insights.append("🚀 **High Return Potential**: Strong expected returns, but verify assumptions.")
                elif expected_return < 0.05:
                    insights.append("📉 **Low Return Expectation**: Consider if returns meet your objectives.")
                
                # Sharpe ratio insight
                if sharpe > 1.5:
                    insights.append("⭐ **Excellent Risk-Adjusted Returns**: Outstanding Sharpe ratio.")
                elif sharpe < 0.5:
                    insights.append("📊 **Poor Risk-Adjusted Returns**: Low Sharpe ratio suggests inefficient risk-taking.")
                
                for insight in insights:
                    st.markdown(insight)
                
                if not insights:
                    st.markdown("📊 **Balanced Portfolio**: Your optimization results appear well-balanced across key metrics.")

            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.info("""
                    💡 **Troubleshooting Tips:**
                    - Ensure all ticker symbols are valid and actively traded
                    - Check your internet connection for data retrieval
                    - Try with well-known large-cap stocks (AAPL, MSFT, GOOG)
                    - Reduce the number of simulations if the process times out
                    - Adjust constraints if optimization fails to converge
                """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>QuantRisk Analytics</strong> | Advanced Portfolio Optimization Platform</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Modern Portfolio Theory • CAPM • Interactive Analytics • Professional-Grade Optimization</p>
    </div>
""", unsafe_allow_html=True)