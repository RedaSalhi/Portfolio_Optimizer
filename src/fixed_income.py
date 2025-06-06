import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -------------------------------
# 1. Core Computation Function
# -------------------------------
def compute_fixed_income_var(face=1_000_000, coupon_rate=0.03, maturity=10, confidence_level=0.95, position_size=1_000_000):
    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    
    # Download 10Y US Treasury yield data
    yields = pdr.DataReader('DGS10', 'fred', start, end).dropna()
    yields.rename(columns={'DGS10': 'Yield'}, inplace=True)
    
    # Compute yield changes in basis points
    yields['Yield_Change_bps'] = yields['Yield'].diff() * 100
    yields.dropna(inplace=True)

    # Estimate current YTM from last yield
    latest_ytm = yields['Yield'].iloc[-1] / 100

    # Compute bond price and PV01
    price = bond_price(face=1, coupon_rate=coupon_rate, ytm=latest_ytm, years=maturity)
    bumped_price = bond_price(face=1, coupon_rate=coupon_rate, ytm=latest_ytm + 0.0001, years=maturity)
    pv01 = price - bumped_price

    # Compute rate volatility (bps)
    sigma_r = yields['Yield_Change_bps'].std()
    z = stats.norm.ppf(1 - confidence_level)
    var_1d = -z * pv01 * sigma_r * position_size

    # Simulated P&L
    yields['PnL'] = -pv01 * yields['Yield_Change_bps'] * position_size
    yields['VaR_Breach'] = yields['PnL'] < -var_1d
    num_exceedances = yields['VaR_Breach'].sum()
    total_days = len(yields)
    exceedance_pct = 100 * num_exceedances / total_days

    results = {
        'df': yields,
        'ytm': latest_ytm,
        'bond_price': price,
        'pv01': pv01,
        'daily_volatility': sigma_r,
        'VaR': var_1d,
        'z_score': z,
        'num_exceedances': num_exceedances,
        'exceedance_pct': exceedance_pct
    }
    return results


# -------------------------------
# 2. Plot Histogram of Rate Changes
# -------------------------------
def plot_yield_change_distribution(df):
    sigma_r = df['Yield_Change_bps'].std()
    changes = df['Yield_Change_bps']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(changes, bins=100, kde=False, stat='density', color='skyblue', label='Rates Changes', ax=ax)

    x_vals = np.linspace(changes.min(), changes.max(), 1000)
    normal_pdf = stats.norm.pdf(x_vals, loc=0, scale=sigma_r)
    ax.plot(x_vals, normal_pdf, 'r-', lw=2, label='Normal(0, σ²)')

    ax.set_title("Histogram of Rate Changes vs Normal(0, σ²)")
    ax.set_xlabel("Rate Change (bps)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    return fig


# -------------------------------
# 3. Plot PnL vs VaR with Breaches
# -------------------------------
def plot_pnl_vs_var(df, var_1d, confidence_level):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['PnL'], label='Daily P&L', color='blue')
    ax.axhline(-var_1d, color='red', linestyle='--', linewidth=2, label=f'-VaR ({int(confidence_level*100)}%)')

    breaches = df[df['VaR_Breach']]
    ax.scatter(breaches.index, breaches['PnL'], color='red', label='VaR Breach', zorder=5)

    ax.set_title("Fixed Income P&L vs Parametric VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True)

    return fig


# -------------------------------
# Bond Pricing Helper
# -------------------------------
def bond_price(face, coupon_rate, ytm, years, freq=2):
    periods = int(years * freq)
    coupon = face * coupon_rate / freq
    discount_factors = [(1 + ytm / freq) ** -t for t in range(1, periods + 1)]
    price = sum([coupon * df for df in discount_factors])
    price += face / (1 + ytm / freq) ** periods
    return price
