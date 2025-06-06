import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from src.fixed_income import compute_fixed_income_var
from src.parametric import compute_parametric_var

# -------------------------------
# 1. Main VaR Computation
# ------------------------------
def compute_portfolio_var(equity_tickers, equity_weights,
                               bond_tickers, bond_weights,
                               confidence_level=0.95,
                               position_size=1_000_000,
                               maturity=10):
    """
    Computes portfolio-level parametric VaR using log returns.
    Combines equities and fixed income.
    """

    # Normalize weights
    equity_weights = np.array(equity_weights) / np.sum(equity_weights)
    bond_weights = np.array(bond_weights) / np.sum(bond_weights)
    total_weight = np.sum(equity_weights) + np.sum(bond_weights)
    equity_weights *= (1 / total_weight)
    bond_weights *= (1 / total_weight)
    all_weights = list(equity_weights) + list(bond_weights)

    # 1. Get equity data
    equity_results = compute_parametric_var_multi(equity_tickers,
                                                  confidence_level=confidence_level,
                                                  position_size=position_size)

    # 2. Get bond data
    bond_results = compute_fixed_income_var(bond_tickers,
                                            maturity=maturity,
                                            confidence_level=confidence_level,
                                            position_size=position_size)

    # 3. Combine log return series
    all_log_returns = []
    asset_names = []

    for i, res in enumerate(equity_results):
        if 'error' in res:
            continue
        df = res['df'][['Log_Return']].rename(columns={'Log_Return': res['ticker']})
        all_log_returns.append(df)
        asset_names.append(res['ticker'])

    for i, res in enumerate(bond_results):
        df = res['df'][['Yield_Change_bps']].copy()
        df['Log_Return'] = -res['pv01'] * df['Yield_Change_bps'] / 100 * (1 / position_size)
        df = df[['Log_Return']].rename(columns={'Log_Return': res['ticker']})
        all_log_returns.append(df)
        asset_names.append(res['ticker'])

    # Combine all returns
    return_df = pd.concat(all_log_returns, axis=1).dropna()

    # Compute portfolio log return
    return_df['Portfolio_Log_Return'] = return_df.dot(all_weights)

    # Compute PnL
    return_df['PnL'] = return_df['Portfolio_Log_Return'] * position_size

    # VaR computation
    z = stats.norm.ppf(1 - confidence_level)
    sigma = return_df['Portfolio_Log_Return'].std()
    var = -z * sigma * position_size

    # VaR breaches
    return_df['VaR_Breach'] = return_df['PnL'] < -var

    results = {
        'var_portfolio': var,
        'volatility': sigma,
        'exceedances': return_df['VaR_Breach'].sum(),
        'exceedance_pct': 100 * return_df['VaR_Breach'].mean(),
        'return_df': return_df,
        'asset_names': asset_names,
        'weights': all_weights
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
    n = len(tickers)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axs = axs.ravel()

    plot_count = 0
    for i, ticker in enumerate(tickers):
        series = df[ticker].replace([np.inf, -np.inf], np.nan).dropna()

        if series.empty:
            continue

        axs[plot_count].hist(series, bins=50, color='lightblue', edgecolor='black')
        axs[plot_count].set_title(f'{ticker} Daily Returns')
        axs[plot_count].set_xlabel('Log Return')
        axs[plot_count].set_ylabel('Frequency')
        axs[plot_count].set_xlim(-20, 20) 
        plot_count += 1

    # Hide any unused subplots
    for j in range(plot_count, len(axs)):
        fig.delaxes(axs[j])

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
