import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime, timedelta


# -------------------------------
# Core Computation Function
# -------------------------------
def compute_parametric_var_multi(tickers, confidence_level=0.95, position_size=1_000_000):
    """
    Computes 1-day parametric VaR for one or more equity tickers based on historical returns.
    
    Parameters:
        tickers (list or str): A single ticker or list of tickers.
        confidence_level (float): Confidence level for VaR (e.g., 0.95).
        position_size (float): Notional value of position in USD.
        
    Returns:
        list of dicts: One result dictionary per ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    z = stats.norm.ppf(1 - confidence_level)

    results = []

    for ticker in tickers:
        try:
            raw_data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

            if isinstance(raw_data.columns, pd.MultiIndex):
                close_data = raw_data['Close']
            else:
                close_data = raw_data.get('Close', raw_data.squeeze())

            if isinstance(close_data, pd.Series):
                df = close_data.to_frame(name='Price')
            else:
                df = close_data.rename(columns={close_data.columns[0]: 'Price'})

            # Compute returns
            df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
            df['Simple_Return'] = df['Price'].pct_change()
            df.dropna(inplace=True)

            sigma = df['Log_Return'].std()
            var_1d = -z * sigma * position_size
            df['PnL'] = df['Simple_Return'] * position_size
            df['VaR_Breach'] = df['PnL'] < -var_1d

            breaches = df['VaR_Breach'].sum()
            breach_pct = 100 * breaches / len(df)

            results.append({
                'ticker': ticker,
                'daily_volatility': sigma,
                'VaR': var_1d,
                'z_score': z,
                'num_exceedances': breaches,
                'exceedance_pct': breach_pct,
                'df': df
            })

        except Exception as e:
            results.append({
                'ticker': ticker,
                'error': str(e)
            })

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
