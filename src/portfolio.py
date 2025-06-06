import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# -------------------------------
# 1. Main VaR Computation
# -------------------------------
def compute_portfolio_var(normal_assets, normal_weights,
                          fixed_income_assets, portfolio_value=1_000_000,
                          confidence_level=0.95):
    """
    normal_assets: list of tickers for normal assets (log-return based)
    normal_weights: list of weights (must sum with fixed to 1)
    
    fixed_income_assets: list of dicts, each like:
        {'ticker': 'TLT', 'weight': 0.2, 'pv01': 0.0008}
    """
    assert len(normal_assets) == len(normal_weights), "Mismatch in normal asset length."
    total_fixed_weight = sum(asset['weight'] for asset in fixed_income_assets)
    assert np.isclose(sum(normal_weights) + total_fixed_weight, 1.0), "Weights must sum to 1.0"

    # --- Normal assets (log returns) ---
    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    data = yf.download(normal_assets, start=start, end=end)['Close'].dropna()
    returns = np.log(data / data.shift(1)).dropna()

    # --- Fixed income assets (PV01) ---
    pnl_fi = pd.DataFrame(index=returns.index)
    total_fi_var = 0
    z = norm.ppf(1 - confidence_level)

    for asset in fixed_income_assets:
        ticker = asset['ticker']
        weight = asset['weight']
        pv01 = asset['pv01']
        exposure = weight * portfolio_value

        # Download yield-proxy data
        price = yf.download(ticker, start=start, end=end)['Close'].dropna()
        price = price.loc[returns.index]
        yield_change = price.diff() / price.shift(1)
        yield_change = yield_change.loc[returns.index]
        yield_change = 100 * yield_change  # convert to bps

        # Volatility and VaR
        sigma_bps = yield_change.std()
        var = -pv01 * sigma_bps * z * exposure
        total_fi_var += float(var)  # Force scalar


        # Compute P&L
        pnl_fi[ticker] = -pv01 * yield_change * exposure

    # --- Portfolio Return / Volatility ---
    weights = np.array(normal_weights)
    cov_matrix = returns.cov()
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std = np.sqrt(port_var)
    var_normal = -z * port_std * portfolio_value * (1 - total_fixed_weight)

    total_var = float(var_normal + total_fi_var)
                            
    # --- Total PnL ---
    weighted_returns = returns @ weights
    simple_returns = np.exp(weighted_returns) - 1
    pnl_normal = simple_returns * (1 - total_fixed_weight) * portfolio_value
    pnl_total = pnl_normal + pnl_fi.sum(axis=1)
    
    pnl_df = pd.DataFrame({'PnL': pnl_total})
    pnl_df['VaR_Breach'] = pnl_df['PnL'] < -total_var  # this now works safely


    num_exceedances = pnl_df['VaR_Breach'].sum()
    total_days = len(pnl_df)
    exceedance_pct = 100 * num_exceedances / total_days

    return {
        'df': returns,
        'cov_matrix': cov_matrix,
        'VaR': total_var,
        'normal_var': var_normal,
        'fixed_income_var': total_fi_var,
        'pnl_df': pnl_df,
        'daily_volatility': port_std,
        'z_score': z,
        'num_exceedances': num_exceedances,
        'exceedance_pct': exceedance_pct
    }


# -------------------------------
# 2. Correlation Matrix Plot
# -------------------------------
def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Returns")
    plt.tight_layout()
    return fig


# -------------------------------
# 3. Individual Histograms
# -------------------------------
def plot_individual_distributions(df):
    tickers = df.columns.tolist()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    for i, ticker in enumerate(tickers):
        axs[i].hist(df[ticker], bins=50, color='lightblue', edgecolor='black')
        axs[i].set_title(f'{ticker} Daily Returns')
        axs[i].set_xlabel('Log Return')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    return fig


# -------------------------------
# 4. Portfolio P&L vs VaR Plot
# -------------------------------
def plot_portfolio_pnl_vs_var(pnl_df, var_value, confidence_level):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pnl_df.index, pnl_df['PnL'], label='Daily P&L', color='blue')
    ax.axhline(-var_value, color='red', linestyle='--', linewidth=2, label=f'-VaR ({int(confidence_level * 100)}%)')

    breaches = pnl_df[pnl_df['VaR_Breach']]
    ax.scatter(breaches.index, breaches['PnL'], color='red', label='VaR Breach', zorder=5)

    ax.set_title("Portfolio Daily P&L vs Parametric VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
