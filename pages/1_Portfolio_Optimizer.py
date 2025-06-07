# pages/1_Portfolio_Optimizer.py

import streamlit as st
import plotly.graph_objects as go
from optimizer import optimize_portfolio

# Set page configuration
st.set_page_config(page_title="Portfolio Optimizer", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }

        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f4e79;
            margin-top: 1rem;
        }

        .section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.07);
            margin-bottom: 2rem;
        }

        .metric-box {
            text-align: center;
            font-size: 1rem;
            color: #444;
        }
    </style>
""", unsafe_allow_html=True)

# Back navigation
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")

# Title
st.markdown('<div class="main-title">Portfolio Optimizer using Markowitz & CAPM</div>', unsafe_allow_html=True)

# --- Parameters Section ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Optimization Parameters")

tickers_str = st.text_input("Enter Tickers (comma-separated)", "AAPL, MSFT, GOOG")

st.markdown("**Optimization Target**")
target_option = st.radio("Choose a target:", ("None", "Target Return", "Target Volatility"))

expected_return_val, expected_std_val = None, None
if target_option == "Target Return":
    expected_return_val = st.slider("Set Target Return", 0.0, 1.0, 0.2, step=0.01)
elif target_option == "Target Volatility":
    expected_std_val = st.slider("Set Target Volatility", 0.0, 1.0, 0.2, step=0.01)

include_risk_free = st.checkbox(
    "Include Risk-Free Asset? (`^IRX` as proxy)", value=True
)
use_sp500 = st.checkbox(
    "Use S&P 500 as Market Proxy?", value=True
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Run Optimization ---
submit = st.button("Run Optimization")

if submit:
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

    with st.spinner("Optimizing portfolio... Please wait."):
        try:
            weights, capm, betas, alphas, w, R_target, sigma_target, fig = optimize_portfolio(
                tickers, expected_return_val, expected_std_val, include_risk_free, use_sp500
            )

            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📈 Optimal Portfolio Summary")

            # --- Optimal Weights ---
            st.markdown("#### Asset Weights")
            fig_weights = go.Figure(data=[go.Pie(
                labels=tickers,
                values=weights,
                hole=0.3,
                textinfo='label+percent',
                hoverinfo='label+percent+value'
            )])
            fig_weights.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig_weights, use_container_width=True)

            # --- CAPM Metrics ---
            st.markdown("#### CAPM Expected Returns")
            capm_cols = st.columns(len(tickers))
            for col, t in zip(capm_cols, tickers):
                col.metric(label=t, value=f"{capm[t]*100:.2f} %")

            st.markdown("#### Betas & Alphas")
            stats_cols = st.columns(len(tickers))
            for col, t in zip(stats_cols, tickers):
                col.markdown(f"**{t}**")
                col.write(f"Beta: `{betas[t]:.4f}`")
                col.write(f"Alpha: `{alphas[t]:.4f}`")

            # --- Capital Allocation ---
            if w is not None:
                st.markdown("#### Capital Allocation Breakdown")
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
                fig_alloc.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_alloc, use_container_width=True)

                # Portfolio Metrics
                st.markdown("#### Portfolio Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Expected Return", f"{R_target:.2%}")
                col2.metric("Expected Volatility", f"{sigma_target:.2%}")

            st.markdown('</div>', unsafe_allow_html=True)

            # --- Efficient Frontier ---
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📉 Efficient Frontier")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
