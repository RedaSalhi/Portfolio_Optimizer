# pages/1_Portfolio_Optimizer.py

import streamlit as st
import plotly.graph_objects as go
from optimizer import optimize_portfolio

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")




st.title("📈 Portfolio Optimizer using Markowitz & CAPM")


with st.sidebar:
    st.header("⚙️ User Inputs")
    tickers_str = st.text_input("📃 Tickers (comma-separated)", "AAPL, MSFT, GOOG")

    st.markdown("🎯 **Optimization Target (choose one)**")
    target_option = st.radio("Select Target:", ("None", "Target Return", "Target Volatility"))

    expected_return_val, expected_std_val = None, None
    if target_option == "Target Return":
        expected_return_val = st.slider("Set Target Return", 0.0, 1.0, 0.2, step=0.01)
    elif target_option == "Target Volatility":
        expected_std_val = st.slider("Set Target Volatility", 0.0, 1.0, 0.2, step=0.01)

    include_risk_free = st.checkbox("Include Risk-Free Asset?", value=True)
    use_sp500 = st.checkbox("Use S&P 500 as Market Proxy?", value=True)
    submit = st.button("🚀 Run Optimization")

if submit:
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

    with st.spinner("Optimizing portfolio... Please wait ⏳"):
        try:
            weights, capm, betas, alphas, w, R_target, sigma_target, fig = optimize_portfolio(
                tickers, expected_return_val, expected_std_val, include_risk_free, use_sp500
            )

            st.subheader("📊 Optimal Portfolio Summary")

            st.markdown("#### 🎯 Optimal Weights (Risky Assets Only)")
            fig_weights = go.Figure(data=[go.Pie(
                labels=tickers,
                values=weights,
                hole=0.3,
                textinfo='label+percent',
                hoverinfo='label+percent+value'
            )])
            fig_weights.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_weights, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 💹 CAPM Expected Returns:")
            capm_cols = st.columns(len(tickers))
            for col, t in zip(capm_cols, tickers):
                col.metric(label=t, value=f"{capm[t]*100:.2f} %")

            st.markdown("#### 📈 Betas & 🧾 Alphas:")
            stats_cols = st.columns(len(tickers))
            for col, t in zip(stats_cols, tickers):
                col.markdown(f"**{t}**")
                col.write(f"Beta: `{betas[t]:.4f}`")
                col.write(f"Alpha: `{alphas[t]:.4f}`")

            if w is not None:
                st.markdown("---")
                st.markdown("#### ⚖️ Capital Allocation Breakdown")

                risk_free_weight = 1 - w
                risky_allocations = [w * wt for wt in weights]

                labels = ['Risk-Free'] + tickers
                values = [risk_free_weight] + risky_allocations

                fig_alloc = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    textinfo='label+percent',
                    hoverinfo='label+percent+value'
                )])
                fig_alloc.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

                st.markdown("#### 📌 Portfolio Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Expected Return", f"{R_target:.2%}")
                col2.metric("Expected Volatility", f"{sigma_target:.2%}")

            st.markdown("---")
            st.markdown("#### 🖼️ Efficient Frontier Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
