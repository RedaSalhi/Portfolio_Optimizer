import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# -------------------------------
# 1. Main VaR Computation
# ------------------------------
def compute_portfolio_var(equity_tickers, equity_weights,
                          bond_tickers, bond_weights,
                          confidence_level=0.95,
                          position_size=1_000_000,
                          maturity=10):
    """
    Computes the portfolio VaR using parametric method (variance-covariance approach).
    Handles both equity and fixed income instruments.
    """

    # Normalize weights
    equity_weights = np.array(equity_weights) / np.sum(equity_weights)
    bond_weights = np.array(bond_weights) / np.sum(bond_weights)
    total_weight = np.sum(equity_weights) + np.sum(bond_weights)
    equity_weights *= (1 / total_weight)
    bond_weights *= (1 / total_weight)

    # 1. Get equity data
    equity_results = compute_parametric_var_multi(equity_tickers,
                                                  confidence_level=confidence_level,
                                                  position_size=position_size)

    # 2. Get bond data
    bond_results = compute_fixed_income_var(bond_tickers,
                                            maturity=maturity,
                                            confidence_level=confidence_level,
                                            position_size=position_size)

    # 3. Combine PnL series for portfolio construction
    pnl_series = []
    individual_vars = []
    asset_names = []
    all_df = []

    # Stocks
    for i, res in enumerate(equity_results):
        if 'error' in res:
            continue
        df = res['df'].copy()
        df = df[['PnL']].rename(columns={'PnL': res['ticker']})
        all_df.append(df)
        individual_vars.append(res['VaR'] * equity_weights[i])
        asset_names.append(res['ticker'])

    # Bonds
    for i, res in enumerate(bond_results):
        df = res['df'][['PnL']].rename(columns={'PnL': res['ticker']})
        all_df.append(df)
        individual_vars.append(res['VaR'] * bond_weights[i])
        asset_names.append(res['ticker'])

    # Merge all PnL time series
    combined_df = pd.concat(all_df, axis=1).dropna()
    asset_weights = list(equity_weights) + list(bond_weights)

    # Compute portfolio PnL
    combined_df['Portfolio_PnL'] = combined_df.dot(asset_weights)

    # Portfolio stats
    z = stats.norm.ppf(1 - confidence_level)
    sigma_portfolio = combined_df['Portfolio_PnL'].std()
    var_portfolio = -z * sigma_portfolio

    # Compare with weighted VaRs
    weighted_var_sum = sum(individual_vars)

    # Exceedances
    combined_df['VaR_Breach'] = combined_df['Portfolio_PnL'] < -var_portfolio
    exceedances = combined_df['VaR_Breach'].sum()
    exceedance_pct = 100 * exceedances / len(combined_df)

    results = {
        'var_portfolio': var_portfolio,
        'weighted_var_sum': weighted_var_sum,
        'volatility': sigma_portfolio,
        'exceedances': exceedances,
        'exceedance_pct': exceedance_pct,
        'combined_df': combined_df,
        'asset_names': asset_names,
        'weights': asset_weights
    }

    return results


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
