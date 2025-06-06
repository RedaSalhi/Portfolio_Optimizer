import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta


# -------------------------------
# 1 Bond Pricing Helper
# -------------------------------
def bond_price(face, coupon_rate, ytm, years, freq=2):
    periods = int(years * freq)
    coupon = face * coupon_rate / freq
    discount_factors = [(1 + ytm / freq) ** -t for t in range(1, periods + 1)]
    price = sum([coupon * df for df in discount_factors])
    price += face / (1 + ytm / freq) ** periods
    return price

# -------------------------------
# 1.2 Fixed Income VaR (per bond)
# -------------------------------
def compute_fixed_income_var(tickers,
                             maturity=10,
                             confidence_level=0.95,
                             position_size=1_000_000):
    """
    Computes VaR for each fixed income instrument using PV01 approximation.
    Returns a list of dictionaries with per-ticker results.
    """
    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    z = stats.norm.ppf(1 - confidence_level)

    all_data = []

    for ticker in tickers:
        if ticker.upper().startswith("DGS") or ticker.upper().startswith("GS"):  # FRED pattern
            df = pdr.DataReader(ticker, 'fred', start, end).dropna()
            df = df.rename(columns={ticker: 'Yield'})
        else:
            df = yf.download(ticker, start=start, end=end)[['Close']].dropna()
            df = df.rename(columns={'Close': 'Yield'})

        df['Yield_Change_bps'] = df['Yield'].diff() * 100
        df.dropna(inplace=True)

        latest_yield = df['Yield'].iloc[-1] / 100
        coupon_rate = latest_yield
        ytm = latest_yield

        price = bond_price(face=1, coupon_rate=coupon_rate, ytm=ytm, years=maturity)
        bumped_price = bond_price(face=1, coupon_rate=coupon_rate, ytm=ytm + 0.0001, years=maturity)
        pv01 = price - bumped_price

        sigma_bps = df['Yield_Change_bps'].std()
        var_1d = -z * pv01 * sigma_bps * position_size

        df['PnL'] = -pv01 * df['Yield_Change_bps'] * position_size
        df['VaR_Breach'] = df['PnL'] < -float(var_1d)
        exceedances = df['VaR_Breach'].sum()
        exceedance_pct = 100 * exceedances / len(df)

        all_data.append({
            'ticker': ticker,
            'maturity': maturity,
            'ytm': ytm,
            'pv01': pv01,
            'price': price,
            'vol_bps': sigma_bps,
            'VaR': float(var_1d),
            'exceedances': int(exceedances),
            'exceedance_pct': exceedance_pct,
            'pnl_series': df['PnL'],
            'yield_changes': df['Yield_Change_bps'],
            'df': df  
        })

    return all_data

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
