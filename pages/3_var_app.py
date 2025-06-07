# var_app.py

import streamlit as st
from src.parametric import compute_parametric_var, plot_return_distribution, plot_pnl_vs_var
from src.fixed_income import compute_fixed_income_var, plot_yield_change_distribution, plot_pnl_vs_var as plot_fixed_pnl
from src.portfolio import compute_portfolio_var, plot_correlation_matrix, plot_individual_distributions, plot_portfolio_pnl_vs_var
from src.monte_carlo import compute_monte_carlo_var, plot_simulated_returns, plot_correlation_matrix as plot_mc_corr, plot_monte_carlo_pnl_vs_var
import pandas as pd

# Page config
st.set_page_config(page_title="Value at Risk App", layout="wide")

# Hide sidebar and native headers/footers, add styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }

        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f4e79;
            margin-bottom: 1.5rem;
        }

        .section-header {
            font-size: 1.4rem;
            font-weight: 600;
            color: #333;
            margin-top: 2rem;
        }

        .button-box {
            border: 1px solid #DDD;
            border-radius: 12px;
            padding: 25px;
            background-color: #f0f4f8;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }

        .button-box:hover {
            box-shadow: 3px 3px 10px rgba(0,0,0,0.15);
        }

        .mode-label {
            margin-top: 1rem;
            font-size: 1.1rem;
            color: #555;
            text-align: center;
        }

        .back-button {
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Value at Risk Interactive App</div>', unsafe_allow_html=True)

# Back Button
with st.container():
    st.markdown('<div class="back-button">', unsafe_allow_html=True)
    if st.button("🔙 Back to Home"):
        st.switch_page("streamlit_app.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Model Selection
st.markdown('<div class="section-header">Select a VaR scenario to analyze:</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    with st.container():
        with st.container():
            st.markdown('<div class="button-box">', unsafe_allow_html=True)
            if st.button("One Asset (Stock)", use_container_width=True):
                st.session_state.selected_mode = "One Asset (Parametric)"
            if st.button("One Asset (Fixed Income)", use_container_width=True): 
                st.session_state.selected_mode = "One Asset (Fixed Income)"
            st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="button-box">', unsafe_allow_html=True)
        if st.button("Portfolio (Equity + Bonds)", use_container_width=True): 
            st.session_state.selected_mode = "Portfolio (Equity + Bonds) (Variance-Covariance)"
        if st.button("Portfolio (Monte Carlo)", use_container_width=True):
            st.session_state.selected_mode = "Multiple Assets (Monte Carlo)"
        st.markdown('</div>', unsafe_allow_html=True)

# Display selected mode
mode = st.session_state.get("selected_mode", None)

if mode:
    st.markdown("---")
    st.success(f"📌 Selected Mode: **{mode}**")




if mode == "One Asset (Parametric)":
    st.header("Parametric VaR for Stocks")
    st.subheader("Configure Parameters")
    tickers_input = st.text_input("Enter Tickers (comma-separated, e.g., AAPL, MSFT, SPY)", value="AAPL, MSFT")
    position = st.number_input("Position Size per Asset ($)", value=100)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run VaR Analysis"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        results = compute_parametric_var(tickers, confidence_level=confidence, position_size=position)
        for res in results:
            with st.expander(f"Results for {res['ticker']}"):
                if 'error' in res:
                    st.error(f"{res['ticker']}: {res['error']}")
                    continue
                st.write(f"1-Day VaR ({int(confidence * 100)}%): ${res['VaR']:.2f}")
                st.write(f"Volatility: {res['daily_volatility']:.4%}")
                st.write(f"Exceedances: {res['num_exceedances']} ({res['exceedance_pct']:.2f}%)")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.pyplot(plot_return_distribution(res['df']))
                with col2:  
                    st.pyplot(plot_pnl_vs_var(res['df'], res['VaR'], confidence))



elif mode == "One Asset (Fixed Income)":
    st.header("Fixed Income VaR using PV01 Approximation")

    st.subheader("Configure Analysis Parameters")
    tickers = st.text_input("Enter Bond Yield Tickers (comma-separated, e.g., DGS10, DGS2)", value="DGS10")
    maturity = st.number_input("Bond Maturity (Years)", min_value=1, max_value=30, value=10)
    position = st.number_input("Position Size ($)", value=100, step=100)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Fixed Income VaR"):
        ticker_list = [t.strip() for t in tickers.split(",")]
        results = compute_fixed_income_var(ticker_list, maturity=maturity, confidence_level=confidence, position_size=position)
        for res in results:
            with st.expander(f"Results for {res['ticker']}"):
                st.write(f"Latest YTM: {res['ytm']:.4%}")
                st.write(f"Yield Volatility (bps): {res['vol_bps']:.4f}")
                st.write(f"1-Day VaR ({int(confidence * 100)}%): ${res['VaR']:.2f}")
                st.write(f"Exceedances: {res['exceedances']} ({res['exceedance_pct']:.2f}%)")
                col1, col2 = st.columns([1, 1])
                with col1:
                    # Plot yield changes histogram
                    fig1 = plot_yield_change_distribution(pd.DataFrame({
                        'Yield_Change_bps': res['yield_changes']
                    }))
                    st.pyplot(fig1)
                with col2:
                    # Plot PnL vs VaR line
                    fig2 = plot_pnl_vs_var(res['df'], res['VaR'], confidence)
                    st.pyplot(fig2)





if mode == "Portfolio (Equity + Bonds) (Variance-Covariance)":
    st.header("Portfolio Parametric VaR")

    
    st.subheader("Configure Portfolio")
    eq_tickers = st.text_input("Equity Tickers (comma-separated)", value="AAPL, MSFT").split(",")
    eq_weights = st.text_input("Equity Weights (comma-separated)", value="0.5, 0.5").split(",")
    bond_tickers = st.text_input("Bond Tickers (FRED/Yahoo Finance codes, comma-separated)", value="DGS10").split(",")
    bond_weights = st.text_input("Bond Weights (comma-separated)", value="1").split(",")

    position = st.number_input("Portfolio Notional Value ($)", value=100)
    maturity = st.slider("Bond Maturity (Years)", 1, 30, 10)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Portfolio VaR"):
        eq_tickers = [t.strip().upper() for t in eq_tickers]
        eq_weights = [float(w) for w in eq_weights]
        bond_tickers = [t.strip().upper() for t in bond_tickers]
        bond_weights = [float(w) for w in bond_weights]

        results = compute_portfolio_var(eq_tickers, eq_weights,
                                    bond_tickers, bond_weights,
                                    confidence_level=confidence,
                                    position_size=position,
                                    maturity=maturity)


        with st.expander("Portfolio VaR Results"):
            st.write(f"1-Day Portfolio VaR ({int(confidence * 100)}%): ${results['var_portfolio']:.2f}")
            st.write(f"Sum of Weighted Individual VaRs: ${results['weighted_var_sum']:.2f}")
            f"Portfolio Daily Volatility: {results['volatility']:.4%} (daily std of log returns)"
            st.write(f"VaR Breaches: {results['exceedances']} ({results['exceedance_pct']:.2f}%)")
            
            return_df = results['return_df']
            asset_names = results['asset_names']
    
        with st.expander("Diagnostics & Visuals"):

            # Individual return histograms (log returns)
            fig_hists = plot_individual_distributions(return_df[asset_names])
            st.pyplot(fig_hists)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # Correlation matrix (log returns)
                fig_corr = plot_correlation_matrix(return_df[asset_names])
                st.pyplot(fig_corr)
            with col2:
                # Portfolio PnL vs VaR
                fig_pnl = plot_portfolio_pnl_vs_var(return_df[['PnL', 'VaR_Breach']], results['var_portfolio'], confidence)
                st.pyplot(fig_pnl)
            
            
            
elif mode == "Multiple Assets (Monte Carlo)":
    st.markdown("""
        <style>
            .section-title {
                font-size: 1.8rem;
                font-weight: 700;
                color: #1f4e79;
                margin-bottom: 1.2rem;
                text-align: center;
            }

            .asset-box {
                background-color: #f8f9fa;
                padding: 1rem 1.5rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                box-shadow: 1px 1px 4px rgba(0,0,0,0.06);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Monte Carlo Portfolio VaR</div>', unsafe_allow_html=True)

    st.markdown("### 🧮 Define Portfolio Assets")

    num_assets = st.number_input("Number of Assets", min_value=2, max_value=10, value=4, step=1)

    tickers, weights = [], []
    for i in range(num_assets):
        with st.container():
            st.markdown('<div class="asset-box">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker = st.text_input(f"Ticker {i + 1}", key=f"ticker_{i}").upper()
            with col2:
                weight = st.number_input(f"Weight (%)", min_value=0.0, max_value=100.0, value=100.0 / num_assets, step=1.0, key=f"weight_{i}")
            st.markdown('</div>', unsafe_allow_html=True)
            tickers.append(ticker)
            weights.append(weight / 100)  # Convert to decimals

    if abs(sum(weights) - 1.0) > 0.01:
        st.error(f"Total weights must sum to 100%. Currently: {sum(weights)*100:.2f}%")
        st.stop()

    st.markdown("### ⚙️ Simulation Parameters")
    position = st.number_input("💰 Position Size ($)", value=1000, step=100)
    sims = st.number_input("🎲 Number of Simulations", value=10000, step=1000)
    confidence = st.slider("📉 Confidence Level", 0.90, 0.99, 0.95)

    if st.button("🚀 Run Analysis"):
        rate_like = ['^IRX', 'DGS10', 'DGS2', 'DGS30', 'DTB3', 'DTB6', 'DTB12']
        if any(t in rate_like for t in tickers):
            st.info(
                "ℹ️ Note: Tickers like ‘^IRX‘ or ‘DGS10‘ represent **interest rates**, not tradable asset prices. "
                "Monte Carlo VaR simulates price-based returns, so rates should not be included directly as assets. "
                "Consider using bond ETFs (like ‘SHV‘, ‘BIL‘, ‘TLT‘) instead."
            )
        else:
            results = compute_monte_carlo_var(
                tickers=tickers,
                weights=weights,
                portfolio_value=position,
                num_simulations=sims,
                confidence_level=confidence
            )

            with st.expander("📊 Portfolio VaR Results", expanded=True):
                st.success(f"Monte Carlo VaR: ${results['VaR_dollar']:,.2f} ({results['VaR_pct']:.4%})")
                st.info(f"Exceedances: {results['num_exceedances']} ({results['exceedance_pct']:.2f}%)")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.pyplot(plot_simulated_returns(results['simulated_returns'], results['VaR_pct'], confidence))
                with col2:
                    st.pyplot(plot_mc_corr(results['returns']))
                with col3:
                    st.pyplot(plot_monte_carlo_pnl_vs_var(results['pnl_df'], results['VaR_dollar'], confidence))

        
                
                
                
