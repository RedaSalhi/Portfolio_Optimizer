import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# -------------------------------
# Core Computation Function
# -------------------------------
def compute_parametric_var(ticker="^GSPC", start="2019-01-01", end="2024-12-31", confidence_level=0.95, position_size=1_000_000):
    raw_data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Handle multi-index (OHLCV) or single-column data
    if isinstance(raw_data.columns, pd.MultiIndex):
        close_data = raw_data['Close']
    else:
        if 'Close' in raw_data.columns:
            close_data = raw_data['Close']
        else:
            close_data = raw_data.squeeze()  # fallback for Series



    # Compute returns
    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Simple_Return'] = df['Price'].pct_change()
    df.dropna(inplace=True)

    # Daily volatility
    sigma = df['Log_Return'].std()

    # Z-score and VaR
    z = stats.norm.ppf(1 - confidence_level)
    var_1d = -z * sigma * position_size

    # P&L
    df['PnL'] = df['Simple_Return'] * position_size
    df['VaR_Breach'] = df['PnL'] < -var_1d

    # Summary
    breaches = df['VaR_Breach'].sum()
    total_days = len(df)
    breach_pct = 100 * breaches / total_days

    results = {
        'ticker': ticker,
        'daily_volatility': sigma,
        'VaR': var_1d,
        'z_score': z,
        'num_exceedances': breaches,
        'exceedance_pct': breach_pct,
        'df': df
    }
    return results


# -------------------------------
# Plot: Return Distribution
# -------------------------------
def plot_return_distribution(df, bins=100):
    sigma = df['Log_Return'].std()
    empirical_returns = df['Log_Return']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(empirical_returns, bins=bins, kde=False, stat='density', color='skyblue', label='Empirical Returns', ax=ax)

    x_vals = np.linspace(empirical_returns.min(), empirical_returns.max(), 1000)
    normal_pdf = stats.norm.pdf(x_vals, loc=0, scale=sigma)
    ax.plot(x_vals, normal_pdf, 'r-', lw=2, label='Normal(0, σ²)')

    ax.set_title("Histogram of Log Returns vs Normal(0, σ²)")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    return fig


# -------------------------------
# Plot: PnL vs VaR
# -------------------------------
def plot_pnl_vs_var(df, var_value, confidence_level):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['PnL'], label='Daily P&L', color='blue')
    ax.axhline(-var_value, color='red', linestyle='--', linewidth=2, label=f'-VaR ({int(confidence_level*100)}%)')

    breaches = df[df['VaR_Breach']]
    ax.scatter(breaches.index, breaches['PnL'], color='red', label='VaR Breach', zorder=5)

    ax.set_title("Daily P&L vs Parametric VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True)

    return fig
