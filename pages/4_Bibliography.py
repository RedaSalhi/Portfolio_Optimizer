# pages/4_Bibliography.py

import streamlit as st

# Page config
st.set_page_config(page_title="Bibliography & Research", layout="centered")

# Hide sidebar and Streamlit chrome
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
            border-radius: 12px;
            box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
        }

        .pdf-section {
            text-align: center;
            padding: 1rem;
            background-color: #eef3f8;
            border-radius: 10px;
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Back button
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")

# Title
st.markdown('<div class="main-title">📚 Bibliography & Research Paper</div>', unsafe_allow_html=True)

# Main Bibliography Section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📘 Key References")

st.markdown("""
### Academic Foundations
- Markowitz, H. (1952). *Portfolio Selection*. *Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk*. *Journal of Finance*, 19(3), 425–442.

### Concepts Covered in This App
- Modern Portfolio Theory (MPT)
- Value-at-Risk (VaR): Parametric, Monte Carlo, and Fixed-Income (PV01)
- Capital Asset Pricing Model (CAPM)
- Risk budgeting, efficient frontier, correlation analysis, and exceedance backtesting

### Data Sources
- **Yahoo Finance API** (via `yfinance`) — Equity & ETF price data
- **FRED (Federal Reserve Economic Data)** — U.S. Treasury yields
- **Custom simulation** for Monte Carlo VaR

### Tools & Libraries
- `numpy`, `pandas`, `matplotlib`, `scipy`, `plotly`, `statsmodels`
- `streamlit`, `yfinance`, `pandas_datareader`
""")
st.markdown('</div>', unsafe_allow_html=True)

# PDF Download Section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📄 Download Full Research Document")

st.markdown("""
The following paper, written as part of this project, provides a full breakdown of:
- VaR estimation methods (parametric, PV01, Monte Carlo)
- Empirical analysis of equities, fixed income, and diversified portfolios
- Backtesting results, visualizations, and implementation notes
""")

with open("assets/Value_at_Risk.pdf", "rb") as pdf_file:
    st.download_button(
        label="📥 Download Value_at_Risk.pdf",
        data=pdf_file,
        file_name="assets/Value_at_Risk.pdf",
        mime="application/pdf"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Footer Caption
st.caption("© 2025 | SALHI Reda | Financial Engineering Research")

