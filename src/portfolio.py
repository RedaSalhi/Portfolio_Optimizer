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
    Combines equities and fixed income assets.
    
    Returns:
        dict with portfolio VaR, weighted VaR sum, PnL time series, and diagnostics.
    """

    # -------------------------------
    # 1. Normalize weights
    # -------------------------------
    equity_weights = np.array(equity_weights, dtype=float)
    bond_weights = np.array(bond_weights, dtype=float)

    total_weight = np.sum(equity_weights) + np.sum(bond_weights)
    equity_weights = equity_weights / total_weight
    bond_weights = bond_weights / total_weight
    all_weights = list(equity_weights) + list(bond_weights)

    # -------------------------------
    # 2. Fetch equity and bond data
    # -------------------------------
    equity_results = compute_parametric_var(equity_tickers,
                                                  confidence_level=confidence_level,
                                                  position_size=position_size)

    bond_results = compute_fixed_income_var(bond_tickers,
                                            maturity=maturity,
                                            confidence_level=confidence_level,
                                            position_size=position_size)

    # -------------------------------
    # 3. Collect log returns and individual VaRs
    # -------------------------------
    log_returns_list = []
    individual_vars = []
    asset_names = []

    # Equities
    for i, res in enumerate(equity_results):
        if 'error' in res:
            continue
        df = res['df'][['Log_Return']].rename(columns={'Log_Return': res['ticker']})
        log_returns_list.append(df)
        individual_vars.append(res['VaR'])  # already in $
        asset_names.append(res['ticker'])

    # Bonds
    for i, res in enumerate(bond_results):
        df = res['df'][['Yield_Change_bps']].copy()
        df['Log_Return'] = -res['pv01'] * df['Yield_Change_bps'] / 100 / position_size  # log-return approx
        df = df[['Log_Return']].rename(columns={'Log_Return': res['ticker']})
        log_returns_list.append(df)
        individual_vars.append(res['VaR'])  # already in $
        asset_names.append(res['ticker'])

    # -------------------------------
    # 4. Combine returns
    # -------------------------------
    return_df = pd.concat(log_returns_list, axis=1).dropna()

    # Portfolio log return
    return_df['Portfolio_Log_Return'] = return_df.dot(all_weights)

    # PnL in $
    return_df['PnL'] = return_df['Portfolio_Log_Return'] * position_size

    # -------------------------------
    # 5. Compute Portfolio VaR
    # -------------------------------
    z = stats.norm.ppf(1 - confidence_level)
    sigma = return_df['Portfolio_Log_Return'].std()
    var = -z * sigma * position_size

    # -------------------------------
    # 6. Compute weighted sum of VaRs
    # -------------------------------
    weighted_var_sum = sum(w * v for w, v in zip(all_weights, individual_vars))

    # -------------------------------
    # 7. Exceedance backtest
    # -------------------------------
    return_df['VaR_Breach'] = return_df['PnL'] < -var
    breaches = return_df['VaR_Breach'].sum()
    breach_pct = 100 * breaches / len(return_df)

    # -------------------------------
    # 8. Output
    # -------------------------------
    results = {
        'var_portfolio': var,
        'weighted_var_sum': weighted_var_sum,
        'volatility': sigma,
        'exceedances': breaches,
        'exceedance_pct': breach_pct,
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
