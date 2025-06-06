import streamlit as st
from src.parametric import compute_parametric_var, plot_return_distribution, plot_pnl_vs_var
from src.fixed_income import compute_fixed_income_var, plot_yield_change_distribution, plot_pnl_vs_var as plot_fixed_pnl
from src.portfolio import compute_portfolio_var, plot_correlation_matrix as plot_vcv_corr, plot_individual_distributions, plot_portfolio_pnl_vs_var
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
    st.header("📊 Parametric VaR for Multiple Assets")

    st.subheader("⚙️ Configure Parameters")
        tickers_input = st.text_input("Enter Tickers (comma-separated, e.g., AAPL, MSFT, SPY)", value="AAPL, MSFT")
        position = st.number_input("Position Size per Asset ($)", value=1_000_000)
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

    if st.button("Run Multi-Asset Analysis"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        results = compute_parametric_var(tickers, confidence_level=confidence, position_size=position)
        with st.expander("⚙️ Configure Parameters"):
            for res in results:
                if 'error' in res:
                    st.error(f"{res['ticker']}: {res['error']}")
                    continue
    
                st.subheader(f"📈 Results for {res['ticker']}")
                st.write(f"🔹 1-Day VaR ({int(confidence * 100)}%): ${res['VaR']:.2f}")
                st.write(f"🔹 Volatility: {res['daily_volatility']:.4%}")
                st.write(f"🔹 Exceedances: {res['num_exceedances']} ({res['exceedance_pct']:.2f}%)")
    
                st.pyplot(plot_return_distribution(res['df']))
                st.pyplot(plot_pnl_vs_var(res['df'], res['VaR'], confidence))



elif mode == "One Asset (Fixed Income)":
    st.header("💰 Fixed Income VaR using PV01 Approximation")

    st.subheader("📈 Configure Analysis Parameters")
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
