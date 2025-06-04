import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
import io

# Core Portfolio Optimization Function
def optimize_portfolio(tickers, expected_return=None, expected_std=None, include_risk_free=False, use_sp500=False):
    data = yf.download(tickers, start="2018-01-01")['Close']
    returns = data.pct_change(fill_method=None).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    irx = yf.download('^IRX', period="5d", interval="1d")['Close'].dropna()
    mean_yield = float(irx.mean())
    rf_rate = mean_yield / 100
    if not include_risk_free:
        rf_rate = 0.0

    num_assets = len(tickers)
    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': []}

    for _ in range(10000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - rf_rate) / port_volatility

        results['Returns'].append(port_return)
        results['Volatility'].append(port_volatility)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)

    results_df = pd.DataFrame(results)
    max_sharpe_idx = results_df['Sharpe'].idxmax()
    opt_weights = results_df.loc[max_sharpe_idx, 'Weights']

    if use_sp500:
        sp500_raw = yf.download('^GSPC', start="2018-01-01")['Close'].pct_change(fill_method=None)
        returns = pd.concat([returns, sp500_raw], axis=1).dropna()
        portfolio_returns = returns['^GSPC']
        returns = returns.drop(columns='^GSPC')
    else:
        portfolio_returns = returns @ opt_weights

    portfolio_returns.name = 'Tangency'
    expected_market_return = portfolio_returns.mean() * 252

    betas, alphas, capm_expected_returns = {}, {}, {}
    for ticker in returns.columns:
        Ri = returns[ticker] - rf_rate
        X = sm.add_constant(portfolio_returns - rf_rate)
        model = sm.OLS(Ri, X).fit()
        alpha = model.params['const']
        beta = model.params['Tangency']
        alphas[ticker] = alpha
        betas[ticker] = beta
        capm_expected_returns[ticker] = rf_rate + beta * (expected_market_return - rf_rate)

    # Plot Efficient Frontier
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(results_df['Volatility'], results_df['Returns'], c=results_df['Sharpe'], cmap='viridis', alpha=0.5)
    plt.colorbar(sc, label='Sharpe Ratio')
    ax.scatter(results_df.loc[max_sharpe_idx, 'Volatility'],
               results_df.loc[max_sharpe_idx, 'Returns'],
               c='red', s=100, label='Max Sharpe Ratio')

    if include_risk_free:
        x = np.linspace(0, results_df['Volatility'].max(), 100)
        y = rf_rate + ((results_df.loc[max_sharpe_idx, 'Returns'] - rf_rate) / results_df.loc[max_sharpe_idx, 'Volatility']) * x
        ax.plot(x, y, color='black', linestyle='--', label='Capital Market Line (CML)')

    ax.set_title('Efficient Frontier' + (' with Risk-Free Asset' if include_risk_free else ''))
    ax.set_xlabel('Volatility (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    ax.grid(True)

    # Capital Allocation
    Rp = results_df.loc[max_sharpe_idx, 'Returns']
    sigmap = results_df.loc[max_sharpe_idx, 'Volatility']
    w, R_target, sigma_target = None, None, None

    if include_risk_free and (expected_return is not None or expected_std is not None):
        if expected_return is not None:
            w = (expected_return - rf_rate) / (Rp - rf_rate)
            sigma_target = w * sigmap
            R_target = expected_return
        elif expected_std is not None:
            w = expected_std / sigmap
            R_target = rf_rate + w * (Rp - rf_rate)
            sigma_target = expected_std

    return opt_weights, capm_expected_returns, betas, alphas, w, R_target, sigma_target, fig

# Streamlit UI
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
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

            # Use columns for better visual layout
            st.markdown("#### 🎯 Optimal Weights (Risky Assets Only)")
            fig_weights, ax_weights = plt.subplots()
            ax_weights.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
            ax_weights.axis('equal')
            st.pyplot(fig_weights)

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
                st.markdown("#### ⚖️ Capital Allocation (Including Risk-Free Asset)")
            
                # Compute weights
                risk_free_weight = 1 - w
                risky_allocations = [w * wt for wt in weights]
            
                labels = ['Risk-Free'] + tickers
                sizes = [risk_free_weight] + risky_allocations
            
                fig_alloc, ax_alloc = plt.subplots()
                ax_alloc.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax_alloc.axis('equal')
                st.pyplot(fig_alloc)
            
                # Portfolio metrics
                st.markdown("#### 📌 Portfolio Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Expected Return", f"{R_target:.2%}")
                col2.metric("Expected Volatility", f"{sigma_target:.2%}")



            st.markdown("---")
            st.markdown("#### 🖼️ Efficient Frontier Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
