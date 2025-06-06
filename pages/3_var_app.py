import streamlit as st
from src.parametric import compute_parametric_var, plot_return_distribution, plot_pnl_vs_var
from src.fixed_income import compute_fixed_income_var, plot_yield_change_distribution, plot_pnl_vs_var as plot_fixed_pnl
from src.portfolio import compute_portfolio_var, plot_correlation_matrix as plot_vcv_corr, plot_individual_distributions, plot_portfolio_pnl_vs_var
from src.monte_carlo import compute_monte_carlo_var, plot_simulated_returns, plot_correlation_matrix as plot_mc_corr, plot_monte_carlo_pnl_vs_var

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
    if st.button("🔢 One Asset (Parametric)"):
        st.session_state.selected_mode = "One Asset (Parametric)"
    if st.button("🪙 One Asset (Fixed Income)"):
        st.session_state.selected_mode = "One Asset (Fixed Income)"

with col2:
    if st.button("🪜 Multiple Assets (Variance-Covariance)"):
        st.session_state.selected_mode = "Multiple Assets (Variance-Covariance)"
    if st.button("🌺 Multiple Assets (Monte Carlo)"):
        st.session_state.selected_mode = "Multiple Assets (Monte Carlo)"

# Get selected mode (persist between reruns)
mode = st.session_state.get("selected_mode", None)

if mode:
    st.markdown("---")
    st.subheader(f"📌 Selected: {mode}")



if mode == "One Asset (Parametric)":
    st.header("🔢 Parametric VaR for One Asset")
    ticker = st.text_input("Enter Asset Ticker (e.g., SPY)", value="SPY")
    position = st.number_input("Position Size ($)", value=1_000_000)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Analysis"):
        results = compute_parametric_var(ticker=ticker, position_size=position, confidence_level=confidence)
        st.write(f"1-Day VaR ({int(confidence*100)}%): ${results['VaR']:,.2f}")
        st.write(f"Volatility: {results['daily_volatility']:.4%}")
        st.write(f"Exceedances: {results['num_exceedances']} ({results['exceedance_pct']:.2f}%)")

        st.pyplot(plot_return_distribution(results['df']))
        st.pyplot(plot_pnl_vs_var(results['df'], results['VaR'], confidence))

elif mode == "One Asset (Fixed Income)":
    st.header("🪙 Fixed Income VaR via PV01")

    tickers = st.multiselect(
        "Select Fixed Income Ticker(s) from FRED (e.g. DGS10 for 10Y Treasury)",
        options=["DGS10", "GS10", "DGS5", "GS5", "DGS2", "DGS30"],
        default=["DGS10"]
    )

    maturity = st.number_input("Maturity (Years)", value=10)
    position_size = st.number_input("Position Size per Asset ($)", value=1_000_000)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Analysis") and tickers:
        results = compute_fixed_income_var(
            tickers=tickers,
            maturity=maturity,
            confidence_level=confidence,
            position_size=position_size
        )

        st.subheader("📊 Results")
        st.write(f"1-Day Portfolio VaR ({int(confidence * 100)}%): ${results['portfolio_var']:,.2f}")
        st.write(f"Exceedances: {results['num_exceedances']} days ({results['exceedance_pct']:.2f}%)")

        # Display individual asset summaries + plots
        for asset in results['individual_assets']:
            st.markdown(f"### 📈 {asset['ticker']}")
            st.write(f"- YTM: {asset['ytm']:.2%}")
            st.write(f"- PV01: {asset['pv01']:.6f}")
            st.write(f"- Individual VaR: ${asset['VaR']:,.2f}")
            st.pyplot(plot_yield_change_distribution(asset['df']))
            st.pyplot(plot_pnl_vs_var(asset['df'], asset['VaR'], confidence))

        # Optional: total portfolio PnL plot (aggregated across tickers)
        st.markdown("### 📊 Portfolio-Level P&L vs VaR")
        st.pyplot(plot_pnl_vs_var(results['pnl_df'], results['portfolio_var'], confidence))
    
    elif not tickers:
        st.warning("Please select at least one fixed income ticker.")


elif mode == "Multiple Assets (Variance-Covariance)":
    st.header("🪜 Portfolio VaR - Variance-Covariance Method")

    normal_assets = st.text_input("Normal Assets (comma-separated)", "GLD,SPY,EURUSD=X").split(',')
    normal_weights_str = st.text_input("Weights for Normal Assets (comma-separated)", "0.3,0.4,0.2")
    normal_weights = list(map(float, normal_weights_str.split(',')))

    fi_count = st.number_input("Number of Fixed Income Assets", value=1)
    fi_assets = []
    for i in range(int(fi_count)):
        ticker = st.text_input(f"FI Ticker {i+1}", f"TLT")
        weight = st.number_input(f"FI Weight {i+1}", value=0.1)
        pv01 = st.number_input(f"PV01 per $1 for {ticker}", value=0.0008, format="%.6f")
        fi_assets.append({'ticker': ticker, 'weight': weight, 'pv01': pv01})

    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Analysis"):
        results = compute_portfolio_var(
            normal_assets=normal_assets,
            normal_weights=normal_weights,
            fixed_income_assets=fi_assets,
            confidence_level=confidence
        )

        st.write(f"Total VaR: ${results['VaR']:,.2f}")
        st.write(f"Normal VaR: ${results['normal_var']:,.2f}")
        st.write(f"Fixed Income VaR: ${results['fixed_income_var']:,.2f}")
        st.write(f"Exceedances: {results['num_exceedances']} ({results['exceedance_pct']:.2f}%)")

        st.pyplot(plot_vcv_corr(results['df']))
        st.pyplot(plot_individual_distributions(results['df']))
        st.pyplot(plot_portfolio_pnl_vs_var(results['pnl_df'], results['VaR'], confidence))

elif mode == "Multiple Assets (Monte Carlo)":
    st.header("🌺 Monte Carlo Portfolio VaR")
    tickers = st.text_input("Tickers (comma-separated)", "GLD,TLT,SPY,EURUSD=X").split(',')
    weights_str = st.text_input("Weights (comma-separated)", "0.25,0.25,0.25,0.25")
    weights = list(map(float, weights_str.split(',')))

    if len(tickers) != len(weights):
        st.error("The number of tickers must match the number of weights.")
        st.stop()

    sims = st.number_input("Number of Simulations", value=10_000)
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Analysis"):
        results = compute_monte_carlo_var(
            tickers=tickers,
            weights=weights,
            num_simulations=sims,
            confidence_level=confidence
        )

        st.write(f"Monte Carlo VaR: ${results['VaR_dollar']:,.2f} ({results['VaR_pct']:.4%})")
        st.write(f"Exceedances: {results['num_exceedances']} ({results['exceedance_pct']:.2f}%)")

        st.pyplot(plot_simulated_returns(results['simulated_returns'], results['VaR_pct'], confidence))
        st.pyplot(plot_mc_corr(results['returns']))
        st.pyplot(plot_monte_carlo_pnl_vs_var(results['pnl_df'], results['VaR_dollar'], confidence))
