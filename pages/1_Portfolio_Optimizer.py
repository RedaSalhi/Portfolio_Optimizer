# pages/1_Portfolio_Optimizer.py - Enhanced Interactive Portfolio Optimizer

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from optimizer import (
    PortfolioOptimizer,
    create_efficient_frontier_plot,
    create_portfolio_composition_chart,
    create_risk_return_scatter,
    create_correlation_heatmap,
    create_performance_chart
)

# Set page configuration
st.set_page_config(
    page_title="Advanced Portfolio Optimizer", 
    layout="wide", 
    page_icon="📈"
)

# Enhanced CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { displa    if show_capm:
        with st.expander("📊 CAPM Analysis", expanded=False):
            capm_metrics = optimizer.calculate_capm_metrics()
            
            if capm_metrics:
                capm_data = []
                for ticker in optimizer.tickers:
                    if ticker in capm_metrics:
                        capm_data.append({
                            'Asset': ticker,
                            'Beta': f"{capm_metrics[ticker]['beta']:.2f}",
                            'Alpha': f"{capm_metrics[ticker]['alpha']:.2%}",
                            'Expected Return (CAPM)': f"{capm_metrics[ticker]['expected_return']:.1%}"
                        })tant; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }

        /* Hero Section */
        .optimizer-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        
        .optimizer-hero h1 {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .optimizer-hero p {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 1rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        /* Configuration Cards */
        .config-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
        }

        .config-title {
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
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin: 0.5rem 0;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            display: block;
        }

        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }

        /* Back Button */
        .back-button .stButton > button {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
            width: auto;
            padding: 0.8rem 1.5rem;
        }

        /* Chart containers */
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }

        /* Success/Error messages */
        .success-msg {
            background: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }

        .error-msg {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
        }

        .warning-msg {
            background: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }

        .reset-button {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            width: 100%;
            margin-top: 1rem;
        }
        
        .reset-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(231, 76, 60, 0.5);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .optimizer-hero h1 { font-size: 2.2rem; }
            .optimizer-hero p { font-size: 1rem; }
            .config-card { padding: 1.5rem; }
        }
    </style>
""", unsafe_allow_html=True)

# Header with Reset Button
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("""
        <div class="optimizer-hero">
            <h1>📈 Advanced Portfolio Optimizer</h1>
            <p>Professional portfolio optimization using Modern Portfolio Theory with real-time market data</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    st.markdown('<div class="reset-button">', unsafe_allow_html=True)
    if st.button("🔄 Reset", help="Clear all data and start fresh", key="reset_main"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Reset complete!")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize tickers from session state or as empty list
tickers = st.session_state.get('tickers', [])

# Initialize show_capm from session state or as False by default
show_capm = st.session_state.get('show_capm', False)

# Status Indicator
if st.session_state.get('optimization_results') and st.session_state.get('optimizer'):
    st.success("🎉 Portfolio optimization active! View results below or use tabs to explore.")
elif len(tickers) >= 2:
    st.info("✅ Ready to optimize! Click the 'Optimize Portfolio' button to get started.")
elif len(tickers) == 1:
    st.warning("⚠️ Need at least 2 assets for portfolio optimization.")
else:
    st.info("💡 Enter 2 or more ticker symbols to begin portfolio optimization.")

# Back Button
st.markdown('<div class="back-button">', unsafe_allow_html=True)
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Configuration Section
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="config-title">🏢 Portfolio Configuration</div>', unsafe_allow_html=True)

# Asset Selection
col1, col2 = st.columns([3, 1])
with col1:
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        value="AAPL, MSFT, GOOG, AMZN, TSLA",
        help="Enter 2-10 stock symbols separated by commas"
    )
with col2:
    lookback_years = st.selectbox(
        "Historical data period:",
        options=[1, 2, 3, 5],
        index=2,
        help="Years of historical data for analysis"
    )

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

# Optimization Method Selection
st.markdown("#### 🎯 Optimization Method")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📈 Maximum Sharpe Ratio", help="Optimize for best risk-adjusted return"):
        st.session_state.opt_method = 'max_sharpe'

with col2:
    if st.button("🛡️ Minimum Variance", help="Minimize portfolio risk"):
        st.session_state.opt_method = 'min_variance'

with col3:
    if st.button("🎯 Target Return", help="Achieve specific return target"):
        st.session_state.opt_method = 'target_return'

with col4:
    if st.button("📊 Target Risk", help="Achieve specific risk level"):
        st.session_state.opt_method = 'target_volatility'

# Method-specific parameters
opt_method = st.session_state.get('opt_method', 'max_sharpe')

target_return = None
target_volatility = None

if opt_method == 'target_return':
    target_return = st.slider(
        "🎯 Target Annual Return:",
        min_value=0.05,
        max_value=0.30,
        value=0.12,
        step=0.01,
        format="%.1%"
    )
elif opt_method == 'target_volatility':
    target_volatility = st.slider(
        "📊 Target Annual Volatility:",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
        format="%.1%"
    )

# Advanced Options
with st.expander("⚙️ Advanced Options", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        include_rf = st.checkbox("Include Risk-Free Rate", value=True)
        show_capm = st.checkbox("Show CAPM Analysis", value=True)
    with col2:
        efficient_frontier_points = st.slider("Efficient Frontier Points", 50, 500, 100)
        min_weight = st.slider("Minimum Asset Weight", 0.0, 0.2, 0.0, step=0.01, format="%.1%")

st.markdown('</div>', unsafe_allow_html=True)

# Validation
if len(tickers) < 2:
    st.markdown('<div class="error-msg">❌ Please enter at least 2 valid ticker symbols</div>', unsafe_allow_html=True)
elif len(tickers) > 10:
    st.markdown('<div class="warning-msg">⚠️ Using more than 10 assets may slow down optimization</div>', unsafe_allow_html=True)

# Run Optimization Button
if st.button("🚀 Optimize Portfolio", disabled=(len(tickers) < 2)):
    with st.spinner("🔄 Fetching data and optimizing portfolio..."):
        try:
            # Initialize optimizer
            optimizer = PortfolioOptimizer(tickers, lookback_years)
            
            # Fetch data
            success, error_info = optimizer.fetch_data()
            
            if not success:
                st.error(f"❌ Failed to fetch data: {error_info}")
                
                # Provide helpful suggestions
                st.markdown("""
                **💡 Troubleshooting suggestions:**
                - Check if ticker symbols are correct (e.g., AAPL not Apple)
                - Try using popular stocks: AAPL, MSFT, GOOGL, AMZN, TSLA
                - Ensure tickers are from major exchanges (NYSE, NASDAQ)
                - Check your internet connection
                - Use the debug tool below to test individual tickers
                """)
                
            else:
                # Show data fetch results
                if error_info:  # Some tickers failed
                    st.warning(f"⚠️ Could not fetch data for: {', '.join(error_info)}")
                    st.info(f"✅ Successfully loaded data for: {', '.join(optimizer.tickers)}")
                else:
                    st.success(f"✅ Successfully loaded data for all {len(optimizer.tickers)} assets")
                
                # Get risk-free rate if requested
                if include_rf:
                    rf_rate = optimizer.get_risk_free_rate()
                    st.info(f"📈 Current risk-free rate: {rf_rate:.2%}")
                
                # Run optimization
                result = optimizer.optimize_portfolio(
                    method=opt_method,
                    target_return=target_return,
                    target_volatility=target_volatility
                )
                
                if result['success']:
                    st.session_state.optimizer = optimizer
                    st.session_state.optimization_results = result
                    
                    st.markdown('<div class="success-msg">🎉 Portfolio optimization completed successfully!</div>', unsafe_allow_html=True)
                else:
                    st.error(f"❌ Optimization failed: {result['error']}")
                    
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("💡 Try using well-known ticker symbols like AAPL, MSFT, GOOGL, AMZN, TSLA")

# Quick fallback buttons (outside the main optimization)
if st.session_state.get('optimization_results') is None:
    st.markdown("#### 🚀 Quick Start Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Sample Portfolio (Tech Giants)", key="tech_sample"):
            with st.spinner("Loading tech portfolio..."):
                fallback_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
                optimizer = PortfolioOptimizer(fallback_tickers, lookback_years)
                success, error_info = optimizer.fetch_data()
                if success:
                    result = optimizer.optimize_portfolio(method=opt_method)
                    if result['success']:
                        st.session_state.optimizer = optimizer
                        st.session_state.optimization_results = result
                        st.success("✅ Tech portfolio loaded successfully!")
                        st.rerun()
                else:
                    st.error("❌ Failed to load sample portfolio")
    
    with col2:
        if st.button("🏦 Try Sample Portfolio (Diversified)", key="div_sample"):
            with st.spinner("Loading diversified portfolio..."):
                fallback_tickers = ["SPY", "BND", "VTI", "SCHG", "VEA"]
                optimizer = PortfolioOptimizer(fallback_tickers, lookback_years)
                success, error_info = optimizer.fetch_data()
                if success:
                    result = optimizer.optimize_portfolio(method=opt_method)
                    if result['success']:
                        st.session_state.optimizer = optimizer
                        st.session_state.optimization_results = result
                        st.success("✅ Diversified portfolio loaded successfully!")
                        st.rerun()
                else:
                    st.error("❌ Failed to load sample portfolio")

# Display Results
if st.session_state.optimization_results and st.session_state.optimizer:
    optimizer = st.session_state.optimizer
    result = st.session_state.optimization_results
    
    # Results header with clear button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
            <div class="results-section">
                <h3 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">📊 Optimization Results</h3>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Results", help="Clear current results", key="clear_results"):
            del st.session_state.optimization_results
            del st.session_state.optimizer
            st.rerun()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        expected_return_pct = result['expected_return'] * 100
        st.markdown("""
            <div class="metric-card">
                <span class="metric-value">{}</span>
                <div class="metric-label">Expected Annual Return</div>
            </div>
        """.format(f"{expected_return_pct:.1f}%"), unsafe_allow_html=True)
    
    with col2:
        volatility_pct = result['volatility'] * 100
        st.markdown("""
            <div class="metric-card">
                <span class="metric-value">{}</span>
                <div class="metric-label">Annual Volatility</div>
            </div>
        """.format(f"{volatility_pct:.1f}%"), unsafe_allow_html=True)
    
    with col3:
        sharpe_val = result['sharpe_ratio']
        st.markdown("""
            <div class="metric-card">
                <span class="metric-value">{}</span>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        """.format(f"{sharpe_val:.2f}"), unsafe_allow_html=True)
    
    with col4:
        diversification = 1 / np.sum(result['weights'] ** 2)
        st.markdown("""
            <div class="metric-card">
                <span class="metric-value">{}</span>
                <div class="metric-label">Diversification Ratio</div>
            </div>
        """.format(f"{diversification:.1f}"), unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🥧 Portfolio Weights",
        "📈 Efficient Frontier", 
        "🎯 Risk-Return Analysis",
        "🔗 Correlations",
        "📊 Performance"
    ])
    
    with tab1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Portfolio composition
        col1, col2 = st.columns([2, 1])
        
        with col1:
            composition_fig = create_portfolio_composition_chart(optimizer.tickers, result['weights'])
            st.plotly_chart(composition_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Detailed Allocation")
            weights_df = pd.DataFrame({
                'Asset': optimizer.tickers,
                'Weight': [f"{w:.1%}" for w in result['weights']],
                'Dollar Amount': [f"${w*1000000:,.0f}" for w in result['weights']]
            })
            st.dataframe(weights_df, hide_index=True, use_container_width=True)
            
            # Risk contribution
            portfolio_variance = result['volatility'] ** 2
            marginal_contrib = np.dot(optimizer.cov_matrix.values, result['weights']) / result['volatility']
            risk_contrib = result['weights'] * marginal_contrib
            
            st.markdown("#### ⚠️ Risk Contribution")
            risk_df = pd.DataFrame({
                'Asset': optimizer.tickers,
                'Risk %': [f"{rc:.1%}" for rc in risk_contrib]
            })
            st.dataframe(risk_df, hide_index=True, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        with st.spinner("Generating efficient frontier..."):
            frontier_fig = create_efficient_frontier_plot(optimizer)
            if frontier_fig:
                st.plotly_chart(frontier_fig, use_container_width=True)
            else:
                st.warning("Could not generate efficient frontier")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        risk_return_fig = create_risk_return_scatter(optimizer, result['weights'])
        st.plotly_chart(risk_return_fig, use_container_width=True)
        
        # Individual asset statistics
        st.markdown("#### 📊 Individual Asset Statistics")
        asset_stats = []
        for i, ticker in enumerate(optimizer.tickers):
            weight_val = result['weights'][i]
            return_val = optimizer.mean_returns[ticker]
            vol_val = np.sqrt(optimizer.cov_matrix.iloc[i,i])
            sharpe_val = (return_val - optimizer.rf_rate) / vol_val
            
            asset_stats.append({
                'Asset': ticker,
                'Weight': f"{weight_val:.1%}",
                'Expected Return': f"{return_val:.1%}",
                'Volatility': f"{vol_val:.1%}",
                'Sharpe Ratio': f"{sharpe_val:.2f}"
            })
        
        asset_df = pd.DataFrame(asset_stats)
        st.dataframe(asset_df, hide_index=True, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        corr_fig = create_correlation_heatmap(optimizer)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Correlation insights
        corr_matrix = optimizer.returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
        min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Correlation", f"{avg_corr:.3f}")
        with col2:
            st.metric("Maximum Correlation", f"{max_corr:.3f}")
        with col3:
            st.metric("Minimum Correlation", f"{min_corr:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        performance_fig = create_performance_chart(optimizer, result['weights'])
        if performance_fig:
            st.plotly_chart(performance_fig, use_container_width=True)
        
        # Performance statistics
        portfolio_returns = (optimizer.returns * result['weights']).sum(axis=1)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_return = (1 + portfolio_returns).prod() - 1
            st.metric("Total Return", f"{total_return:.1%}")
        
        with col2:
            max_drawdown = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().expanding().max() - 1).min()
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
        
        with col3:
            winning_days = (portfolio_returns > 0).sum() / len(portfolio_returns)
            st.metric("Winning Days", f"{winning_days:.1%}")
        
        with col4:
            var_95 = np.percentile(portfolio_returns, 5)
            st.metric("VaR (95%)", f"{var_95:.2%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # CAPM Analysis (if requested)
    if show_capm:
        with st.expander("📊 CAPM Analysis", expanded=False):
            capm_metrics = optimizer.calculate_capm_metrics()
            
            if capm_metrics:
                capm_data = []
                for ticker in optimizer.tickers:
                    if ticker in capm_metrics:
                        capm_data.append({
                            'Asset': ticker,
                            'Beta': f"{capm_metrics[ticker]['beta']:.2f}",
                            'Alpha': f"{capm_metrics[ticker]['alpha']:.2%}",
                            'Expected Return (CAPM)': f"{capm_metrics[ticker]['expected_return']:.1%}"
                        })
                
                if capm_data:
                    capm_df = pd.DataFrame(capm_data)
                    st.dataframe(capm_df, hide_index=True, use_container_width=True)
            else:
                st.warning("CAPM analysis not available")

# Tips and Information
with st.expander("💡 Tips & Information", expanded=False):
    st.markdown("""
    ### 🎯 Optimization Methods
    - **Maximum Sharpe Ratio**: Finds the portfolio with the best risk-adjusted return
    - **Minimum Variance**: Finds the portfolio with the lowest possible risk
    - **Target Return**: Finds the minimum risk portfolio for a specific return target
    - **Target Risk**: Finds the maximum return portfolio for a specific risk level
    
    ### 📊 Key Metrics
    - **Expected Return**: Annualized expected portfolio return based on historical data
    - **Volatility**: Annualized portfolio standard deviation (risk measure)
    - **Sharpe Ratio**: Risk-adjusted return measure (higher is better)
    - **Diversification Ratio**: Measure of portfolio diversification effectiveness
    
    ### ⚠️ Important Notes
    - Results are based on historical data and may not predict future performance
    - Consider transaction costs and taxes in real implementations
    - Rebalancing frequency affects actual performance
    - Market conditions can change rapidly
    
    ### 🔧 Troubleshooting Data Issues
    - **Use valid ticker symbols**: Check symbols on Yahoo Finance or similar sites
    - **Try well-known stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA usually work
    - **Check internet connection**: Data fetching requires stable internet
    - **Avoid delisted/exotic tickers**: Stick to major exchange-listed stocks
    - **Wait and retry**: Sometimes data providers have temporary issues
    - **Use the Reset button**: Clear all data if you encounter persistent errors
    - **Try sample portfolios**: Use the Quick Start options for testing
    
    ### 🚨 Common Errors and Solutions
    - **"No valid data"**: Check ticker symbols and internet connection
    - **"CAPM analysis not available"**: Market data issue, still usable without CAPM
    - **"Optimization failed"**: Try different optimization method or check constraints
    - **JavaScript errors**: Use the Reset button and try again with simpler inputs
    """)

# Debug Section
with st.expander("🐛 Debug Data Fetching", expanded=False):
    st.markdown("### Test Individual Tickers")
    test_ticker = st.text_input("Enter a single ticker to test:", value="AAPL")
    
    if st.button("🧪 Test Ticker"):
        if test_ticker.strip():
            with st.spinner(f"Testing {test_ticker.upper()}..."):
                try:
                    # Use the quick test function
                    from optimizer import PortfolioOptimizer
                    test_optimizer = PortfolioOptimizer([test_ticker.upper()])
                    success, message = test_optimizer.quick_test_ticker(test_ticker.upper())
                    
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                        
                except Exception as e:
                    st.error(f"❌ Error testing {test_ticker.upper()}: {str(e)}")
        else:
            st.warning("Please enter a ticker symbol")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>QuantRisk Analytics</strong> | Professional Portfolio Optimization</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Built with Modern Portfolio Theory • Real-time Market Data • Interactive Analytics</p>
    </div>
""", unsafe_allow_html=True)