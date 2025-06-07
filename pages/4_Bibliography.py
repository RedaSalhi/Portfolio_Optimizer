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

### Data Sources
- Yahoo Finance via `yfinance`
- FRED Economic Data (Federal Reserve)

### Python Libraries
- `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`
- `yfinance`, `statsmodels`, `streamlit`
""")
