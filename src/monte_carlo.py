import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import datetime, timedelta

# -------------------------------
# Helper: Download data from Yahoo or FRED
# -------------------------------
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)['Close']
        if df.dropna().empty:
            raise ValueError(f"No data from Yahoo for {ticker}")
        print(f"Fetched {ticker} from Yahoo Finance.")
    except Exception as e:
        try:
            df = pdr.DataReader(ticker, 'fred', start, end)
            df = df.squeeze()  # Convert DataFrame to Series if needed
            print(f"Fetched {ticker} from FRED.")
        except Exception as fred_e:
            raise ValueError(f"Could not fetch {ticker} from Yahoo or FRED.\nYahoo error: {e}\nFRED error: {fred_e}")
    return df

# -------------------------------
# 1. Main Monte Carlo Simulation
# -------------------------------
def compute_monte_carlo_var(tickers, weights, portfolio_value=1_000_000, confidence_level=0.95,
                             num_simulations=10_000):
    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    weights = np.array(weights)
    assert len(tickers) == len(weights), "Length of tickers and weights must match."

    # Fetch all data series
    price_data = pd.DataFrame()
    for ticker in tickers:
        series = fetch_data(ticker, start, end)
        price_data[ticker] = series

    price_data = price_data.dropna()
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    cov_matrix = log_returns.cov().values
    cholesky_matrix = np.linalg.cholesky(cov_matrix)

    # Simulate correlated returns
    normal_randoms = norm.ppf(np.random.rand(num_simulations, len(tickers)))
    correlated_shocks = normal_randoms @ cholesky_matrix.T
    simulated_returns = correlated_shocks @ weights

    # Compute VaR
    var_pct = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    var_dollar = var_pct * portfolio_value

    # Historical daily PnL
    historical_portfolio_returns = log_returns @ weights
    simple_returns = np.exp(historical_portfolio_returns) - 1
    pnl_series = simple_returns * portfolio_value
    pnl_df = pd.DataFrame({'PnL': pnl_series}, index=log_returns.index)
    pnl_df['VaR_Breach'] = pnl_df['PnL'] < -var_dollar

    num_exceedances = pnl_df['VaR_Breach'].sum()
    total_days = len(pnl_df)
    exceedance_pct = 100 * num_exceedances / total_days

    return {
        'returns': log_returns,
        'cov_matrix': cov_matrix,
        'VaR_pct': var_pct,
        'VaR_dollar': var_dollar,
        'simulated_returns': simulated_returns,
        'pnl_df': pnl_df,
        'num_exceedances': num_exceedances,
        'exceedance_pct': exceedance_pct
    }

# -------------------------------
# 2. Histogram of Simulated Returns
# -------------------------------
def plot_simulated_returns(simulated_returns, var_pct, confidence_level):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(simulated_returns, bins=50, color='steelblue', edgecolor='black')
    ax.axvline(-var_pct, color='red', linestyle='--', linewidth=2,
               label=f'VaR ({int(confidence_level * 100)}%)')
    ax.set_title("Simulated Portfolio Returns Histogram")
    ax.set_xlabel("Simulated Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    return fig


# -------------------------------
# 3. Correlation Matrix
# -------------------------------
def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Asset Returns")
    plt.tight_layout()
    return fig


# -------------------------------
# 4. Monte Carlo PnL vs VaR Plot
# -------------------------------
def plot_monte_carlo_pnl_vs_var(pnl_df, var_dollar, confidence_level):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pnl_df.index, pnl_df['PnL'], label='Daily P&L', color='blue')
    ax.axhline(-var_dollar, color='red', linestyle='--', linewidth=2,
               label=f'-VaR ({int(confidence_level * 100)}%)')

    breaches = pnl_df[pnl_df['VaR_Breach']]
    ax.scatter(breaches.index, breaches['PnL'], color='red', label='VaR Breach', zorder=5)

    ax.set_title("Portfolio Daily P&L vs Monte Carlo VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True)
    return fig
