import streamlit as st
from src.parametric import compute_parametric_var, plot_return_distribution, plot_pnl_vs_var
from src.fixed_income import compute_fixed_income_var, plot_yield_change_distribution, plot_pnl_vs_var as plot_fixed_pnl
from src.portfolio import compute_portfolio_var, plot_correlation_matrix, plot_individual_distributions, plot_portfolio_pnl_vs_var
from src.monte_carlo import compute_monte_carlo_var, plot_simulated_returns, plot_correlation_matrix as plot_mc_corr, plot_monte_carlo_pnl_vs_var
import pandas as pd

st.set_page_config(page_title="Value at Risk App", layout="wide")
st.title("📊 Value at Risk Interactive App")


if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")


st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .center-box {
            border: 1px solid #DDD;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            background-color: #f9f9f9;
            transition: all 0.2s ease;
        }
        .center-box:hover {
            box-shadow: 3px 3px 10px rgba(0,0,0,0.15);
        }
    </style>
""", unsafe_allow_html=True)

# Centralized model selector using columns
st.markdown("### Select a VaR scenario to analyze:")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("One Asset (Stock)"):
        st.session_state.selected_mode = "One Asset (Parametric)"
    if st.button("One Asset (Fixed Income)"): 
        st.session_state.selected_mode = "One Asset (Fixed Income)"

with col2:
    if st.button("Portfolio of (Equity + Bonds) Assets (Variance-Covariance)"): 
        st.session_state.selected_mode = "Portfolio (Equity + Bonds) (Variance-Covariance)"
    if st.button("Portfolio of Assets (Monte Carlo)"):
        st.session_state.selected_mode = "Multiple Assets (Monte Carlo)"

# Get selected mode (persist between reruns)
mode = st.session_state.get("selected_mode", None)

if mode:
    st.markdown("---")
    st.subheader(f"📌 Selected: {mode}")



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
            with st.expander(f"📊 Results for {res['ticker']}"):
                if 'error' in res:
                    st.error(f"{res['ticker']}: {res['error']}")
                    continue
                st.write(f"🔹 1-Day VaR ({int(confidence * 100)}%): ${res['VaR']:.2f}")
                st.write(f"🔹 Volatility: {res['daily_volatility']:.4%}")
                st.write(f"🔹 Exceedances: {res['num_exceedances']} ({res['exceedance_pct']:.2f}%)")

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
            with st.expander(f"📊 Results for {res['ticker']}"):
                st.write(f"🔹 Latest YTM: {res['ytm']:.4%}")
                st.write(f"🔹 Yield Volatility (bps): {res['vol_bps']:.4f}")
                st.write(f"🔹 1-Day VaR ({int(confidence * 100)}%): ${res['VaR']:.2f}")
                st.write(f"🔹 Exceedances: {res['exceedances']} ({res['exceedance_pct']:.2f}%)")
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
    st.header("📦 Portfolio Parametric VaR")

    
    st.subheader("🧮 Configure Portfolio")
    eq_tickers = st.text_input("Equity Tickers (comma-separated)", value="AAPL, MSFT").split(",")
    eq_weights = st.text_input("Equity Weights (comma-separated)", value="0.5, 0.5").split(",")
    bond_tickers = st.text_input("Bond Tickers (FRED codes, comma-separated)", value="DGS10").split(",")
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


        with st.expander("📉 Portfolio VaR Results"):
            st.write(f"1-Day Portfolio VaR ({int(confidence * 100)}%): ${results['var_portfolio']:.2f}")
            st.write(f"Sum of Weighted Individual VaRs: ${results['weighted_var_sum']:.2f}")
            f"Portfolio Daily Volatility: {results['volatility']:.4%} (daily std of log returns)"
            st.write(f"VaR Breaches: {results['exceedances']} ({results['exceedance_pct']:.2f}%)")
            
            return_df = results['return_df']
            asset_names = results['asset_names']
    
        with st.expander("🧪 Diagnostics & Visuals"):

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
    st.header("🌺 Monte Carlo Portfolio VaR")
    tickers = st.text_input("Tickers (comma-separated)", "GLD,TLT,SPY,EURUSD=X").split(',')
    weights_str = st.text_input("Weights (comma-separated)", "0.25,0.25,0.25,0.25")
    weights = list(map(float, weights_str.split(',')))
    position = st.number_input("Position Size ($)", value=100, step=100)

    if len(tickers) != len(weights):
        st.error("The number of tickers must match the number of weights.")
        st.stop()
    
    sims = st.number_input("Number of Simulations", value=10_000)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Analysis"):
        # Warn if rates might be incorrectly included
        rate_like = ['^IRX', 'DGS10', 'DGS2', 'DGS30', 'DTB3', 'DTB6', 'DTB12']
        if any(ticker.strip().upper() in rate_like for ticker in tickers):
            st.info(
                "ℹ️ Note: Tickers like `^IRX` or `DGS10` represent **interest rates**, not tradable asset prices. "
                "Monte Carlo VaR simulates price-based returns, so rates should not be included directly as assets. "
                "Consider using bond ETFs (like `SHV`, `BIL`, `TLT`) instead."
            )
        else: 
            results = compute_monte_carlo_var(
                tickers=tickers,
                weights=weights,
                portfolio_value=position,
                num_simulations=sims,
                confidence_level=confidence
            )
            with st.expander("📉 Portfolio VaR Results"):
                st.write(f"Monte Carlo VaR: ${results['VaR_dollar']:,.2f} ({results['VaR_pct']:.4%})")
                st.write(f"Exceedances: {results['num_exceedances']} ({results['exceedance_pct']:.2f}%)")
        
                st.pyplot(plot_simulated_returns(results['simulated_returns'], results['VaR_pct'], confidence))
                st.pyplot(plot_mc_corr(results['returns']))
                st.pyplot(plot_monte_carlo_pnl_vs_var(results['pnl_df'], results['VaR_dollar'], confidence))
