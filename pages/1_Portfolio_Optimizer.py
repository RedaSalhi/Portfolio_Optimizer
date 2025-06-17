# pages/1_Portfolio_Optimizer.py

import streamlit as st
import plotly.graph_objects as go
from optimizer import optimize_portfolio

# Set page configuration
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

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

        .hero-badges {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }

        .hero-badge {
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Input Sections */
        .input-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }
        
        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            text-align: center;
            justify-content: center;
        }

        /* Configuration Cards */
        .config-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
        }
        
        .config-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.12);
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
        .results-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin: 2rem 0;
            border-left: 5px solid #667eea;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }

        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }

        /* Chart Containers */
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

        /* Back Button */
        .back-button button {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4) !important;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .optimizer-hero h1 { font-size: 2.5rem; }
            .optimizer-hero p { font-size: 1.1rem; }
            .config-card { padding: 1.5rem; }
            .hero-badges { gap: 0.5rem; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="optimizer-hero">
        <h1>📈 Portfolio Optimizer</h1>
        <p>Harness the power of Modern Portfolio Theory and CAPM to build optimal investment portfolios</p>
        <div class="hero-badges">
            <span class="hero-badge">Markowitz Theory</span>
            <span class="hero-badge">Efficient Frontier</span>
            <span class="hero-badge">CAPM Analysis</span>
            <span class="hero-badge">Risk Optimization</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Back navigation
st.markdown('<div class="back-button">', unsafe_allow_html=True)
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")
st.markdown('</div>', unsafe_allow_html=True)

# Input Section
st.markdown("""
    <div class="input-section">
        <div class="section-title">⚙️ Portfolio Configuration</div>
    </div>
""", unsafe_allow_html=True)

# Portfolio Configuration
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="config-title">🏢 Asset Selection</div>', unsafe_allow_html=True)

tickers_str = st.text_input(
    "Enter Asset Tickers (comma-separated)", 
    value="AAPL, MSFT, GOOG, TSLA",
    help="Enter stock tickers separated by commas"
)
st.markdown('</div>', unsafe_allow_html=True)

# Optimization Target
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="config-title">🎯 Optimization Target</div>', unsafe_allow_html=True)

target_option = st.radio(
    "Choose your optimization objective:",
    ("None (Maximum Sharpe Ratio)", "Target Return", "Target Volatility")
)

expected_return_val, expected_std_val = None, None
if target_option == "Target Return":
    expected_return_val = st.slider("Set Target Annual Return", 0.0, 1.0, 0.15, step=0.01)
    st.write(f"Target Return: {expected_return_val:.1%}")
elif target_option == "Target Volatility":
    expected_std_val = st.slider("Set Target Annual Volatility", 0.0, 1.0, 0.20, step=0.01)
    st.write(f"Target Volatility: {expected_std_val:.1%}")
st.markdown('</div>', unsafe_allow_html=True)

# Advanced Options
st.markdown('<div class="config-card">', unsafe_allow_html=True)
st.markdown('<div class="config-title">🔧 Advanced Options</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    include_risk_free = st.checkbox("🏦 Include Risk-Free Asset (^IRX as proxy)", value=True)
with col2:
    use_sp500 = st.checkbox("📊 Use S&P 500 as Market Proxy", value=True)
st.markdown('</div>', unsafe_allow_html=True)

# Run Optimization Button
if st.button("🚀 Optimize Portfolio"):
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    
    if not tickers:
        st.error("❌ Please enter at least one ticker symbol")
    elif len(tickers) < 2:
        st.error("❌ Please enter at least 2 ticker symbols for portfolio optimization")
    else:
        with st.spinner("🔄 Optimizing portfolio... Please wait."):
            try:
                weights, capm, betas, alphas, w, R_target, sigma_target, fig = optimize_portfolio(
                    tickers, expected_return_val, expected_std_val, include_risk_free, use_sp500
                )

                # Results Section
                st.markdown("""
                    <div class="results-container">
                        <div class="section-title">📊 Optimization Results</div>
                    </div>
                """, unsafe_allow_html=True)

                st.success("✅ Portfolio optimization completed successfully!")
                
                # Key Metrics
                st.markdown("### 📈 Portfolio Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Assets", len(tickers))
                with col2:
                    portfolio_return = sum(w * capm[t] for w, t in zip(weights, tickers))
                    st.metric("Expected Return", f"{portfolio_return:.1%}")
                with col3:
                    st.metric("Max Weight", f"{max(weights):.1%}")
                with col4:
                    diversification = 1 - sum(w**2 for w in weights)
                    st.metric("Diversification", f"{diversification:.1%}")

                # Portfolio Weights Visualization
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">🥧 Optimal Portfolio Weights</div>', unsafe_allow_html=True)
                
                # Simple pie chart without complex formatting
                fig_weights = go.Figure(data=[go.Pie(
                    labels=tickers,
                    values=weights,
                    hole=0.4,
                    textinfo='label+percent'
                )])
                
                fig_weights.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=400
                )
                
                st.plotly_chart(fig_weights, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # CAPM Analysis
                st.markdown("### 📊 CAPM Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Expected Returns & Weights")
                    for i, ticker in enumerate(tickers):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write(f"**{ticker}**")
                        with col_b:
                            st.write(f"{capm[ticker]*100:.2f}%")
                        with col_c:
                            st.write(f"{weights[i]*100:.1f}%")
                
                with col2:
                    st.markdown("#### 📊 Risk Metrics")
                    for ticker in tickers:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write(f"**{ticker}**")
                        with col_b:
                            st.write(f"β: {betas[ticker]:.3f}")
                        with col_c:
                            st.write(f"α: {alphas[ticker]:.4f}")

                # Capital Allocation (if applicable)
                if w is not None and include_risk_free:
                    st.markdown("### 💰 Capital Allocation Strategy")
                    
                    risk_free_weight = 1 - w
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🏦 Asset Allocation")
                        st.write(f"**Risk-Free Asset:** {risk_free_weight*100:.1f}%")
                        for i, ticker in enumerate(tickers):
                            risky_allocation = w * weights[i]
                            st.write(f"**{ticker}:** {risky_allocation*100:.1f}%")
                    
                    with col2:
                        # Simple capital allocation chart
                        st.markdown("#### 📊 Allocation Breakdown")
                        labels = ['Risk-Free'] + tickers
                        values = [risk_free_weight] + [w * wt for wt in weights]
                        
                        fig_alloc = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.3
                        )])
                        
                        fig_alloc.update_layout(
                            margin=dict(t=20, b=20, l=20, r=20),
                            height=300
                        )
                        
                        st.plotly_chart(fig_alloc, use_container_width=True)

                    # Portfolio target metrics
                    if R_target and sigma_target:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Target Return", f"{R_target:.2%}")
                        with col2:
                            st.metric("📊 Target Volatility", f"{sigma_target:.2%}")

                # Efficient Frontier
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">📈 Efficient Frontier Analysis</div>', unsafe_allow_html=True)
                st.pyplot(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Summary Information
                st.info("""
                    💡 **Key Insights:**
                    - Portfolio weights show optimal allocation for maximum risk-adjusted returns
                    - CAPM analysis provides risk-adjusted expected returns based on market correlation
                    - Beta measures sensitivity to market movements (>1 = more volatile than market)
                    - Alpha shows risk-adjusted excess return (positive = outperforming expectations)
                    - Efficient frontier displays optimal risk-return combinations
                """)

            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.info("""
                    💡 **Troubleshooting:**
                    - Ensure ticker symbols are valid (try AAPL, MSFT, GOOG)
                    - Check internet connection for data retrieval
                    - Reduce number of assets if optimization fails
                """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>QuantRisk Analytics</strong> | Advanced Portfolio Optimization</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Modern Portfolio Theory • CAPM • Efficient Frontier Analysis</p>
    </div>
""", unsafe_allow_html=True)