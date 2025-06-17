# var_app.py

import streamlit as st
from src.parametric import compute_parametric_var, plot_return_distribution, plot_pnl_vs_var
from src.fixed_income import compute_fixed_income_var, plot_yield_change_distribution, plot_pnl_vs_var as plot_fixed_pnl
from src.portfolio import compute_portfolio_var, plot_correlation_matrix, plot_individual_distributions, plot_portfolio_pnl_vs_var
from src.monte_carlo import compute_monte_carlo_var, plot_simulated_returns, plot_correlation_matrix as plot_mc_corr, plot_monte_carlo_pnl_vs_var
import pandas as pd

# Page config
st.set_page_config(page_title="Value at Risk Analytics", layout="wide", page_icon="⚡")

# Enhanced CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }

        /* Hero Section */
        .var-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        
        .var-hero h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .var-hero p {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 0;
        }

        /* Mode Selection Cards */
        .mode-selection {
            margin: 2rem 0;
        }
        
        .mode-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 2px solid transparent;
            cursor: pointer;
            height: 100%;
        }
        
        .mode-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 48px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        
        .mode-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
            text-align: center;
        }
        
        .mode-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.8rem;
            text-align: center;
        }
        
        .mode-description {
            color: #666;
            text-align: center;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        /* Input Sections */
        .input-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            border-left: 4px solid #667eea;
        }
        
        .section-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Asset Input Cards */
        .asset-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        
        .asset-card:hover {
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }

        /* Results Section */
        .results-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 16px;
            margin: 2rem 0;
        }
        
        .var-metric {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .var-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #e74c3c;
            margin-bottom: 0.5rem;
        }
        
        .var-label {
            color: #666;
            font-size: 1rem;
            font-weight: 500;
        }

        /* Status Messages */
        .selected-mode {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
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
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }

        /* Back Button */
        .back-button {
            margin-bottom: 1rem;
        }
        
        .back-button button {
            background: #6c757d !important;
            color: white !important;
        }

        /* Charts and Plots */
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .var-hero h1 { font-size: 2.2rem; }
            .var-hero p { font-size: 1.1rem; }
            .mode-card { padding: 1.5rem; }
            .input-section { padding: 1.5rem; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="var-hero">
        <h1>⚡ Value-at-Risk Analytics</h1>
        <p>Professional risk assessment using multiple VaR methodologies</p>
    </div>
""", unsafe_allow_html=True)

# Back Button
with st.container():
    st.markdown('<div class="back-button">', unsafe_allow_html=True)
    if st.button("🔙 Back to Home", help="Return to main dashboard"):
        st.switch_page("streamlit_app.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Mode Selection
st.markdown("""
    <div class="section-title">
        Select Risk Analysis Method
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="mode-card">
            <span class="mode-icon">📈</span>
            <div class="mode-title">Single Asset Analysis</div>
            <div class="mode-description">Parametric VaR for individual stocks or fixed income instruments</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1a, col1b = st.columns(2)
    with col1a:
        if st.button("Equity VaR", key="equity_btn", help="Parametric VaR for stocks"):
            st.session_state.selected_mode = "One Asset (Parametric)"
    with col1b:
        if st.button("Bond VaR", key="bond_btn", help="PV01-based VaR for bonds"):
            st.session_state.selected_mode = "One Asset (Fixed Income)"

with col2:
    st.markdown("""
        <div class="mode-card">
            <span class="mode-icon">📊</span>
            <div class="mode-title">Portfolio Analysis</div>
            <div class="mode-description">Multi-asset portfolio risk assessment with correlation analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    col2a, col2b = st.columns(2)
    with col2a:
        if st.button("Portfolio VaR", key="portfolio_btn", help="Mixed portfolio analysis"):
            st.session_state.selected_mode = "Portfolio (Equity + Bonds) (Variance-Covariance)"
    with col2b:
        if st.button("Monte Carlo", key="mc_btn", help="Monte Carlo simulation"):
            st.session_state.selected_mode = "Multiple Assets (Monte Carlo)"

# Display selected mode
mode = st.session_state.get("selected_mode", None)

if mode:
    st.markdown(f"""
        <div class="selected-mode">
            Active Analysis: <strong>{mode}</strong>
        </div>
    """, unsafe_allow_html=True)

# Mode-specific interfaces
if mode == "One Asset (Parametric)":
    st.markdown("""
        <div class="input-section">
            <div class="section-title">Equity Parametric VaR Configuration</div>
        </div>
    """, unsafe_allow_html=True)

    num_assets = st.number_input("Number of Stocks", min_value=1, max_value=10, value=2, step=1)
    tickers = []

    for i in range(num_assets):
        st.markdown('<div class="asset-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            default_value = "AAPL" if i == 0 else "MSFT" if i == 1 else ""
            ticker = st.text_input(f"Stock Ticker {i + 1}", value=default_value, key=f"param_ticker_{i}").upper()
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Asset {i + 1}**")
        tickers.append(ticker.strip())
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        position = st.number_input("Position Size per Asset ($)", value=100000, step=10000)
    with col2:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)

    if st.button("Run VaR Analysis", key="run_equity_var"):
        with st.spinner("Computing parametric VaR..."):
            results = compute_parametric_var(tickers, confidence_level=confidence, position_size=position)

            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            for res in results:
                with st.expander(f"Results for {res['ticker']}", expanded=True):
                    if 'error' in res:
                        st.error(f"❌ {res['ticker']}: {res['error']}")
                        continue

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">${res['VaR']:.0f}</div>
                                <div class="var-label">1-Day VaR ({int(confidence * 100)}%)</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{res['daily_volatility']:.2%}</div>
                                <div class="var-label">Daily Volatility</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{res['num_exceedances']}</div>
                                <div class="var-label">VaR Breaches ({res['exceedance_pct']:.1f}%)</div>
                            </div>
                        """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**Return Distribution**")
                        st.pyplot(plot_return_distribution(res['df']))
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**PnL vs. VaR**")
                        st.pyplot(plot_pnl_vs_var(res['df'], res['VaR'], confidence))
                        st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif mode == "One Asset (Fixed Income)":
    st.markdown("""
        <div class="input-section">
            <div class="section-title">Fixed Income VaR (PV01 Method)</div>
        </div>
    """, unsafe_allow_html=True)

    num_bonds = st.number_input("Number of Bond Instruments", min_value=1, value=2, step=1)
    bond_tickers = []

    for i in range(num_bonds):
        st.markdown('<div class="asset-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            default_value = "DGS10" if i == 0 else "^IRX" if i == 1 else ""
            ticker = st.text_input(f"Bond Yield Ticker {i + 1}", value=default_value, key=f"bond_ticker_{i}")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Bond {i + 1}**")
        bond_tickers.append(ticker.strip().upper())
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        maturity = st.number_input("Bond Maturity (Years)", min_value=1, max_value=30, value=10)
    with col2:
        position = st.number_input("Position Size ($)", value=1000000, step=100000)
    with col3:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)

    if st.button("Run Fixed Income VaR", key="run_bond_var"):
        with st.spinner("Computing PV01-based VaR..."):
            results = compute_fixed_income_var(
                bond_tickers, maturity=maturity, confidence_level=confidence, position_size=position
            )

            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            for res in results:
                with st.expander(f"Results for {res['ticker']}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">${res['VaR']:.0f}</div>
                                <div class="var-label">1-Day VaR ({int(confidence * 100)}%)</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{res['ytm']:.2%}</div>
                                <div class="var-label">Current YTM</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{res['vol_bps']:.1f}</div>
                                <div class="var-label">Volatility (bps)</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{res['exceedances']}</div>
                                <div class="var-label">Breaches ({res['exceedance_pct']:.1f}%)</div>
                            </div>
                        """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**Yield Change Distribution**")
                        fig1 = plot_yield_change_distribution(pd.DataFrame({'Yield_Change_bps': res['yield_changes']}))
                        st.pyplot(fig1)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**PnL vs. VaR**")
                        fig2 = plot_pnl_vs_var(res['df'], res['VaR'], confidence)
                        st.pyplot(fig2)
                        st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif mode == "Portfolio (Equity + Bonds) (Variance-Covariance)":
    st.markdown("""
        <div class="input-section">
            <div class="section-title">Portfolio Parametric VaR</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        num_eq = st.number_input("Number of Equity Assets", min_value=0, value=2, step=1)
    with col2:
        num_bond = st.number_input("Number of Bond Instruments", min_value=0, value=1, step=1)
    
    total_assets = num_eq + num_bond
    if total_assets == 0:
        st.error("❌ Please select at least one asset (equity or bond)")
        st.stop()
    
    default_weight = 100.0 / total_assets
    
    # Equity Holdings
    eq_tickers, eq_weights = [], []
    if num_eq > 0:
        st.markdown("### Equity Holdings")
        for i in range(num_eq):
            st.markdown('<div class="asset-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                eq_ticker = st.text_input(f"Stock Ticker {i+1}", key=f"eq_ticker_{i}").upper()
            with col2:
                eq_weight = st.number_input(f"Weight (%)", min_value=0.0, max_value=100.0,
                                            value=default_weight, step=1.0, key=f"eq_weight_{i}")
            eq_tickers.append(eq_ticker)
            eq_weights.append(eq_weight / 100)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Bond Holdings
    bond_tickers, bond_weights = [], []
    if num_bond > 0:
        st.markdown("### Bond Holdings")
        for i in range(num_bond):
            st.markdown('<div class="asset-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                bond_ticker = st.text_input(f"Bond Ticker {i+1}", key=f"bond_ticker_{i}").upper()
            with col2:
                bond_weight = st.number_input(f"Weight (%)", min_value=0.0, max_value=100.0,
                                              value=default_weight, step=1.0, key=f"bond_weight_{i}")
            bond_tickers.append(bond_ticker)
            bond_weights.append(bond_weight / 100)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Weight validation
    total_weight = sum(eq_weights) + sum(bond_weights)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Portfolio Weight", f"{total_weight * 100:.1f}%")
    with col2:
        st.metric("Equity Allocation", f"{sum(eq_weights) * 100:.1f}%")
    with col3:
        st.metric("Bond Allocation", f"{sum(bond_weights) * 100:.1f}%")
    
    if abs(total_weight - 1.0) > 0.01:
        st.error("❌ Total weights must sum to 100%. Please adjust your allocations.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        position = st.number_input("Portfolio Value ($)", value=1000000, step=100000)
    with col2:
        maturity = st.slider("Bond Maturity (Years)", 1, 30, 10)
    with col3:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)

    if st.button("Run Portfolio VaR Analysis", key="run_portfolio_var"):
        with st.spinner("Computing portfolio VaR..."):
            results = compute_portfolio_var(
                eq_tickers, eq_weights, bond_tickers, bond_weights,
                confidence_level=confidence, position_size=position, maturity=maturity
            )
        
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            # Portfolio VaR Results
            with st.expander("Portfolio VaR Results", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="var-metric">
                            <div class="var-value">${results['var_portfolio']:.0f}</div>
                            <div class="var-label">Portfolio VaR ({int(confidence * 100)}%)</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="var-metric">
                            <div class="var-value">${results['weighted_var_sum']:.0f}</div>
                            <div class="var-label">Sum of Individual VaRs</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="var-metric">
                            <div class="var-value">{results['exceedances']}</div>
                            <div class="var-label">VaR Breaches ({results['exceedance_pct']:.1f}%)</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Portfolio Analytics (separate expander)
            return_df = results['return_df']
            asset_names = results['asset_names']
            
            with st.expander("Portfolio Analytics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("**Asset Correlation Matrix**")
                    fig_corr = plot_correlation_matrix(return_df[asset_names])
                    st.pyplot(fig_corr)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("**Portfolio PnL vs. VaR**")
                    fig_pnl = plot_portfolio_pnl_vs_var(
                        return_df[['PnL', 'VaR_Breach']], results['var_portfolio'], confidence
                    )
                    st.pyplot(fig_pnl)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("**Individual Asset Distributions**")
                fig_hists = plot_individual_distributions(return_df[asset_names])
                st.pyplot(fig_hists)
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

elif mode == "Multiple Assets (Monte Carlo)":
    st.markdown("""
        <div class="input-section">
            <div class="section-title">Monte Carlo Portfolio VaR</div>
        </div>
    """, unsafe_allow_html=True)

    num_assets = st.number_input("Number of Assets", min_value=2, value=3, step=1)

    tickers, weights = [], []
    for i in range(num_assets):
        st.markdown('<div class="asset-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input(f"Asset Ticker {i + 1}", key=f"mc_ticker_{i}").upper()
        with col2:
            weight = st.number_input(f"Weight (%)", min_value=0.0, max_value=100.0, 
                                     value=100.0 / num_assets, step=1.0, key=f"mc_weight_{i}")
        tickers.append(ticker)
        weights.append(weight / 100)
        st.markdown('</div>', unsafe_allow_html=True)

    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        st.error(f"❌ Total weights must sum to 100%. Currently: {total_weight*100:.1f}%")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        position = st.number_input("Portfolio Value ($)", value=1000000, step=100000)
    with col2:
        sims = st.number_input("Simulations", value=10000, step=1000, min_value=1000)
    with col3:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)

    if st.button("Run Monte Carlo Analysis", key="run_mc_var"):
        # Check for rate-like tickers
        rate_like = ["^IRX", "DTB3", "DTB6", "DTB12", "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2",
                     "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30", "FEDFUNDS", "SOFR", "OBFR",
                     "USD3MTD156N", "T5YIE", "T10YIE", "T5YIFR", "DSWP2", "DSWP10"]

        if any(t in rate_like for t in tickers):
            st.warning("""
                ⚠️ **Rate Ticker Detected**: Some tickers represent interest rates (not tradable assets). 
                Monte Carlo VaR works with asset prices. Consider using bond ETFs like SHV, BIL, TLT instead.
            """)
        else:
            with st.spinner("Running Monte Carlo simulation..."):
                results = compute_monte_carlo_var(
                    tickers=tickers, weights=weights, portfolio_value=position,
                    num_simulations=sims, confidence_level=confidence
                )

                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                with st.expander("Monte Carlo VaR Results", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">${results['VaR_dollar']:,.0f}</div>
                                <div class="var-label">Monte Carlo VaR</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{results['VaR_pct']:.2%}</div>
                                <div class="var-label">VaR as % of Portfolio</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="var-metric">
                                <div class="var-value">{results['num_exceedances']}</div>
                                <div class="var-label">Breaches ({results['exceedance_pct']:.1f}%)</div>
                            </div>
                        """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**Simulated Returns**")
                        st.pyplot(plot_simulated_returns(results['simulated_returns'], results['VaR_pct'], confidence))
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**Correlation Matrix**")
                        st.pyplot(plot_mc_corr(results['returns']))
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("**Historical PnL vs VaR**")
                        st.pyplot(plot_monte_carlo_pnl_vs_var(results['pnl_df'], results['VaR_dollar'], confidence))
                        st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>QuantRisk Analytics</strong> | Advanced Value-at-Risk Modeling</p>
    </div>
""", unsafe_allow_html=True)