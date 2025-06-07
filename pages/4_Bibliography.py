# pages/4_Bibliography.py

import streamlit as st

st.set_page_config(page_title="Bibliography", layout="centered")

# Hide sidebar and Streamlit chrome
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Back to Home
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")

# Page content
st.title("Bibliography")
st.markdown("""
Here are some of the key references and sources used in this project:

### Academic Sources
- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
- Sharpe, W.F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk*. The Journal of Finance, 19(3), 425–442.

### Concepts & Definitions
- Modern Portfolio Theory (MPT)
- Value-at-Risk (VaR) — Parametric, Historical, Monte Carlo
- CAPM — Capital Asset Pricing Model
- Efficient Frontier & Capital Allocation Line

st.markdown("---")
st.subheader("📘 In-Depth Project Paper")

st.markdown("
This project is based on a research paper written by **SALHI Reda** exploring Value at Risk (VaR) estimation across multiple asset classes.

It includes:
- Parametric VaR (Equities, Portfolios)
- PV01-based VaR (Fixed Income)
- Monte Carlo VaR (Diversified Assets)
- Full backtesting, diagnostics, and visualizations

You can download the full paper below:
")



### Data Sources
- Yahoo Finance via `yfinance`
- FRED Economic Data (Federal Reserve)

### Python Libraries
- `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`
- `yfinance`, `statsmodels`, `streamlit`
""")
