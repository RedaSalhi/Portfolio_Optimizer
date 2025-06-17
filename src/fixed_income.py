import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st


# -------------------------------
# Enhanced Bond Pricing Functions
# -------------------------------
def bond_price(face, coupon_rate, ytm, years, freq=2):
    """Calculate bond price using discounted cash flow method."""
    periods = int(years * freq)
    coupon = face * coupon_rate / freq
    discount_factors = [(1 + ytm / freq) ** -t for t in range(1, periods + 1)]
    price = sum([coupon * df for df in discount_factors])
    price += face / (1 + ytm / freq) ** periods
    return price


def bond_duration(face, coupon_rate, ytm, years, freq=2):
    """Calculate modified duration of a bond."""
    periods = int(years * freq)
    coupon = face * coupon_rate / freq
    
    # Calculate present values and time-weighted present values
    pv_cash_flows = []
    time_weighted_pv = []
    
    for t in range(1, periods + 1):
        time_period = t / freq
        if t < periods:
            cash_flow = coupon
        else:
            cash_flow = coupon + face
        
        pv = cash_flow / (1 + ytm / freq) ** t
        pv_cash_flows.append(pv)
        time_weighted_pv.append(pv * time_period)
    
    bond_price_calc = sum(pv_cash_flows)
    macaulay_duration = sum(time_weighted_pv) / bond_price_calc
    modified_duration = macaulay_duration / (1 + ytm / freq)
    
    return modified_duration


def bond_convexity(face, coupon_rate, ytm, years, freq=2):
    """Calculate convexity of a bond."""
    periods = int(years * freq)
    coupon = face * coupon_rate / freq
    
    convexity_sum = 0
    bond_price_calc = bond_price(face, coupon_rate, ytm, years, freq)
    
    for t in range(1, periods + 1):
        if t < periods:
            cash_flow = coupon
        else:
            cash_flow = coupon + face
        
        pv = cash_flow / (1 + ytm / freq) ** t
        convexity_sum += pv * t * (t + 1) / (freq ** 2)
    
    convexity = convexity_sum / (bond_price_calc * (1 + ytm / freq) ** 2)
    return convexity

# FIXED: Safe bond calculation functions
def bond_price_safe(face, coupon_rate, ytm, years, freq=2):
    """Safe bond price calculation with error handling."""
    try:
        if ytm <= 0 or years <= 0:
            return face
        
        periods = int(years * freq)
        if periods <= 0:
            return face
            
        coupon = face * coupon_rate / freq
        discount_rate = ytm / freq
        
        if discount_rate <= 0:
            return face + coupon * periods
        
        # Calculate present value of coupons
        pv_coupons = 0
        for t in range(1, periods + 1):
            pv_coupons += coupon / ((1 + discount_rate) ** t)
        
        # Calculate present value of principal
        pv_principal = face / ((1 + discount_rate) ** periods)
        
        return pv_coupons + pv_principal
    except:
        return face  # Return face value as fallback



def bond_duration_safe(face, coupon_rate, ytm, years, freq=2):
    """Safe duration calculation with error handling."""
    try:
        if ytm <= 0 or years <= 0:
            return years * 0.8  # Rough approximation
        
        periods = int(years * freq)
        if periods <= 0:
            return years * 0.8
            
        coupon = face * coupon_rate / freq
        discount_rate = ytm / freq
        
        if discount_rate <= 0:
            return years * 0.8
        
        bond_price_calc = bond_price_safe(face, coupon_rate, ytm, years, freq)
        if bond_price_calc <= 0:
            return years * 0.8
        
        # Calculate weighted average time to cash flows
        weighted_time = 0
        for t in range(1, periods + 1):
            time_period = t / freq
            if t < periods:
                cash_flow = coupon
            else:
                cash_flow = coupon + face
            
            pv = cash_flow / ((1 + discount_rate) ** t)
            weighted_time += pv * time_period
        
        macaulay_duration = weighted_time / bond_price_calc
        modified_duration = macaulay_duration / (1 + discount_rate)
        
        return max(0, min(modified_duration, years))  # Sanity check
    except:
        return years * 0.8  # Fallback approximation


def bond_convexity_safe(face, coupon_rate, ytm, years, freq=2):
    """Safe convexity calculation with error handling."""
    try:
        if ytm <= 0 or years <= 0:
            return years * 0.1  # Rough approximation
        
        periods = int(years * freq)
        if periods <= 0:
            return years * 0.1
            
        coupon = face * coupon_rate / freq
        discount_rate = ytm / freq
        
        if discount_rate <= 0:
            return years * 0.1
        
        bond_price_calc = bond_price_safe(face, coupon_rate, ytm, years, freq)
        if bond_price_calc <= 0:
            return years * 0.1
        
        convexity_sum = 0
        for t in range(1, periods + 1):
            if t < periods:
                cash_flow = coupon
            else:
                cash_flow = coupon + face
            
            pv = cash_flow / ((1 + discount_rate) ** t)
            convexity_sum += pv * t * (t + 1) / (freq ** 2)
        
        convexity = convexity_sum / (bond_price_calc * ((1 + discount_rate) ** 2))
        
        return max(0, convexity)  # Ensure positive
    except:
        return years * 0.1  # Fallback approximation


def calculate_quadratic_pnl_safe(yield_changes_bps, price, duration, convexity, position_size):
    """Safe quadratic P&L calculation with error handling."""
    try:
        yield_changes = yield_changes_bps / 10000  # Convert to decimal
        duration_effect = -duration * yield_changes
        convexity_effect = 0.5 * convexity * (yield_changes ** 2)
        price_change_pct = duration_effect + convexity_effect
        return price_change_pct * price * position_size
    except:
        # Fallback to linear approximation
        return -duration * yield_changes_bps / 10000 * price * position_size


# -------------------------------
# Enhanced Fixed Income VaR Computation
# -------------------------------
def compute_fixed_income_var(tickers,
                             maturity=10,
                             confidence_level=0.95,
                             position_size=1_000_000):
    """
    FIXED: Enhanced fixed income VaR computation with proper error handling.
    """
    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    z = stats.norm.ppf(1 - confidence_level)

    all_data = []

    for ticker in tickers:
        try:
            # Fetch yield data
            if ticker.upper().startswith("DGS") or ticker.upper().startswith("GS"):
                df = pdr.DataReader(ticker, 'fred', start, end).dropna()
                df = df.rename(columns={ticker: 'Yield'})
            else:
                df = yf.download(ticker, start=start, end=end)[['Close']].dropna()
                df = df.rename(columns={'Close': 'Yield'})

            df['Yield_Change_bps'] = df['Yield'].diff() * 100
            df.dropna(inplace=True)

            if df.empty:
                raise ValueError(f"No valid data for {ticker}")

            latest_yield_raw = df['Yield'].iloc[-1]
            latest_yield = float(latest_yield_raw) / 100

            # FIXED: Add safety checks for bond calculations
            if latest_yield <= 0 or latest_yield > 1:  # Sanity check for yield
                latest_yield = max(0.001, min(latest_yield, 0.5))  # Cap between 0.1% and 50%

            coupon_rate = latest_yield
            ytm = latest_yield

            # FIXED: Safe bond calculations with error handling
            try:
                price = bond_price_safe(face=1, coupon_rate=coupon_rate, ytm=ytm, years=maturity)
                duration = bond_duration_safe(face=1, coupon_rate=coupon_rate, ytm=ytm, years=maturity)
                convexity_val = bond_convexity_safe(face=1, coupon_rate=coupon_rate, ytm=ytm, years=maturity)
            except:
                # Fallback to simple calculations
                price = 1.0
                duration = maturity * 0.8  # Rough approximation
                convexity_val = duration * 0.1
            
            # PV01 calculation with safety check
            try:
                bumped_price = bond_price_safe(face=1, coupon_rate=coupon_rate, ytm=ytm + 0.0001, years=maturity)
                pv01 = abs(price - bumped_price)
            except:
                pv01 = duration * price * 0.0001  # Approximation using duration

            # Enhanced VaR calculation
            sigma_bps = df['Yield_Change_bps'].std()
            if sigma_bps <= 0:
                sigma_bps = 10  # Default 10 bps volatility if calculation fails
            
            # Linear VaR (duration only)
            var_linear = abs(z * pv01 * sigma_bps * position_size)
            
            # Quadratic VaR (duration + convexity) with safety check
            try:
                yield_change_var = abs(z * sigma_bps / 10000)
                duration_effect = duration * yield_change_var
                convexity_effect = 0.5 * convexity_val * (yield_change_var ** 2)
                var_quadratic = abs((duration_effect + convexity_effect) * price * position_size)
            except:
                var_quadratic = var_linear * 1.1  # 10% higher than linear as approximation

            # Historical P&L simulation with safety checks
            df['PnL_Linear'] = -pv01 * df['Yield_Change_bps'] * position_size
            df['PnL_Quadratic'] = calculate_quadratic_pnl_safe(df['Yield_Change_bps'], price, duration, convexity_val, position_size)
            
            # VaR breaches
            df['VaR_Breach_Linear'] = df['PnL_Linear'] < -var_linear
            df['VaR_Breach_Quadratic'] = df['PnL_Quadratic'] < -var_quadratic
            
            # FIXED: Ensure compatibility with portfolio calculations
            df['VaR_Breach'] = df['VaR_Breach_Linear']  # For backward compatibility
            df['PnL'] = df['PnL_Linear']  # For backward compatibility
            
            exceedances_linear = df['VaR_Breach_Linear'].sum()
            exceedances_quadratic = df['VaR_Breach_Quadratic'].sum()
            exceedance_pct_linear = 100 * exceedances_linear / len(df) if len(df) > 0 else 0
            exceedance_pct_quadratic = 100 * exceedances_quadratic / len(df) if len(df) > 0 else 0

            # Risk metrics
            yield_volatility_annual = sigma_bps * np.sqrt(252)
            price_volatility = duration * yield_volatility_annual / 100

            all_data.append({
                'ticker': ticker,
                'maturity': maturity,
                'ytm': ytm,
                'coupon_rate': coupon_rate,
                'price': price,
                'duration': duration,
                'convexity': convexity_val,
                'pv01': pv01,
                'vol_bps': sigma_bps,
                'vol_annual_bps': yield_volatility_annual,
                'price_volatility': price_volatility,
                'VaR': var_linear,  # FIXED: For backward compatibility
                'VaR_linear': var_linear,
                'VaR_quadratic': var_quadratic,
                'exceedances': int(exceedances_linear),  # FIXED: For backward compatibility
                'exceedance_pct': exceedance_pct_linear,  # FIXED: For backward compatibility
                'exceedances_linear': int(exceedances_linear),
                'exceedances_quadratic': int(exceedances_quadratic),
                'exceedance_pct_linear': exceedance_pct_linear,
                'exceedance_pct_quadratic': exceedance_pct_quadratic,
                'pnl_series_linear': df['PnL_Linear'],
                'pnl_series_quadratic': df['PnL_Quadratic'],
                'yield_changes': df['Yield_Change_bps'],
                'df': df,
                'confidence_level': confidence_level,
                'position_size': position_size
            })

        except Exception as e:
            all_data.append({
                'ticker': ticker,
                'error': str(e)
            })

    return all_data



def calculate_quadratic_pnl(yield_changes_bps, price, duration, convexity, position_size):
    """Calculate P&L using duration and convexity (quadratic approximation)."""
    yield_changes = yield_changes_bps / 10000  # Convert to decimal
    duration_effect = -duration * yield_changes
    convexity_effect = 0.5 * convexity * (yield_changes ** 2)
    price_change_pct = duration_effect + convexity_effect
    return price_change_pct * price * position_size


# -------------------------------
# Interactive Bond Analytics Dashboard
# -------------------------------
def create_bond_analytics_dashboard(bond_data):
    """
    Create comprehensive bond analytics dashboard.
    
    Parameters:
        bond_data (dict): Bond VaR calculation results
    
    Returns:
        plotly.graph_objects.Figure: Interactive dashboard
    """
    if 'error' in bond_data:
        return None
    
    df = bond_data['df']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Yield Change Distribution',
            'Duration vs Convexity Risk',
            'P&L: Linear vs Quadratic',
            'Risk Metrics Gauge'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "indicator"}]
        ]
    )
    
    # 1. Yield Change Distribution
    yield_changes = df['Yield_Change_bps']
    
    fig.add_trace(
        go.Histogram(
            x=yield_changes,
            nbinsx=50,
            name='Yield Changes',
            opacity=0.7,
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # Add normal distribution overlay
    mu, sigma = yield_changes.mean(), yield_changes.std()
    x_normal = np.linspace(yield_changes.min(), yield_changes.max(), 100)
    y_normal = stats.norm.pdf(x_normal, mu, sigma) * len(yield_changes) * (yield_changes.max() - yield_changes.min()) / 50
    
    fig.add_trace(
        go.Scatter(
            x=x_normal,
            y=y_normal,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # 2. Duration vs Convexity Risk Analysis
    yield_scenarios = np.linspace(-300, 300, 100)  # -3% to +3% yield change
    duration_pnl = []
    convexity_pnl = []
    
    for dy in yield_scenarios:
        dy_decimal = dy / 10000
        duration_effect = -bond_data['duration'] * dy_decimal * bond_data['price'] * bond_data['position_size']
        convexity_effect = 0.5 * bond_data['convexity'] * (dy_decimal ** 2) * bond_data['price'] * bond_data['position_size']
        
        duration_pnl.append(duration_effect)
        convexity_pnl.append(duration_effect + convexity_effect)
    
    fig.add_trace(
        go.Scatter(
            x=yield_scenarios,
            y=duration_pnl,
            mode='lines',
            name='Duration Only',
            line=dict(color='blue', width=2)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=yield_scenarios,
            y=convexity_pnl,
            mode='lines',
            name='Duration + Convexity',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )
    
    # 3. P&L Comparison
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['PnL_Linear'],
            mode='lines',
            name='Linear P&L',
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['PnL_Quadratic'],
            mode='lines',
            name='Quadratic P&L',
            line=dict(color='red', width=1),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add VaR lines
    fig.add_hline(
        y=-bond_data['VaR_linear'],
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Linear VaR: ${bond_data['VaR_linear']:,.0f}",
        row=2, col=1
    )
    
    fig.add_hline(
        y=-bond_data['VaR_quadratic'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Quadratic VaR: ${bond_data['VaR_quadratic']:,.0f}",
        row=2, col=1
    )
    
    # 4. Risk Gauge
    var_pct = abs(bond_data['VaR_quadratic']) / bond_data['position_size'] * 100
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=var_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bond VaR (%)"},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Bond Risk Analytics Dashboard",
        height=800,
        showlegend=True
    )
    
    return fig


# -------------------------------
# Yield Curve Analysis
# -------------------------------
def create_yield_curve_analysis(multiple_bond_data):
    """
    Create yield curve risk analysis across multiple maturities.
    
    Parameters:
        multiple_bond_data (list): List of bond analysis results
    
    Returns:
        plotly.graph_objects.Figure: Yield curve analysis
    """
    if not multiple_bond_data or len(multiple_bond_data) < 2:
        return None
    
    maturities = []
    yields = []
    durations = []
    vars_linear = []
    vars_quadratic = []
    
    for bond_data in multiple_bond_data:
        if 'error' not in bond_data:
            maturities.append(bond_data['maturity'])
            yields.append(bond_data['ytm'] * 100)
            durations.append(bond_data['duration'])
            vars_linear.append(abs(bond_data['VaR_linear']))
            vars_quadratic.append(abs(bond_data['VaR_quadratic']))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Yield Curve',
            'Duration Profile',
            'VaR by Maturity (Linear vs Quadratic)',
            'Risk Concentration'
        )
    )
    
    # 1. Yield Curve
    fig.add_trace(
        go.Scatter(
            x=maturities,
            y=yields,
            mode='lines+markers',
            name='Yield Curve',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # 2. Duration Profile
    fig.add_trace(
        go.Scatter(
            x=maturities,
            y=durations,
            mode='lines+markers',
            name='Duration',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # 3. VaR Comparison
    fig.add_trace(
        go.Scatter(
            x=maturities,
            y=vars_linear,
            mode='lines+markers',
            name='Linear VaR',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=maturities,
            y=vars_quadratic,
            mode='lines+markers',
            name='Quadratic VaR',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    # 4. Risk Concentration Pie Chart
    fig.add_trace(
        go.Pie(
            labels=[f"{m}Y" for m in maturities],
            values=vars_quadratic,
            name="Risk Distribution"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Yield Curve Risk Analysis",
        height=800
    )
    
    return fig


# FIXED: Safe 3D Yield Scenario Analysis
def create_yield_scenario_analysis(bond_data, scenario_range=300):
    """
    FIXED: Create interactive yield scenario analysis with proper error handling.
    """
    if 'error' in bond_data:
        return None
    
    try:
        # Create yield scenario grid with safety checks
        yield_changes = np.linspace(-scenario_range, scenario_range, 25)  # Reduced resolution for performance
        time_points = np.linspace(max(0.1, bond_data['maturity']/10), bond_data['maturity'], 10)  # Reduced resolution
        
        # Calculate P&L surface with error handling
        pnl_surface = np.zeros((len(time_points), len(yield_changes)))
        
        for i, time_to_maturity in enumerate(time_points):
            for j, yield_change_bps in enumerate(yield_changes):
                try:
                    # Safe duration and convexity calculations
                    duration_t = bond_duration_safe(1, bond_data['coupon_rate'], bond_data['ytm'], time_to_maturity)
                    convexity_t = bond_convexity_safe(1, bond_data['coupon_rate'], bond_data['ytm'], time_to_maturity)
                    
                    yield_change_decimal = yield_change_bps / 10000
                    duration_effect = -duration_t * yield_change_decimal
                    convexity_effect = 0.5 * convexity_t * (yield_change_decimal ** 2)
                    
                    pnl_surface[i, j] = (duration_effect + convexity_effect) * bond_data['price'] * bond_data['position_size']
                except:
                    # Use neighboring values or zero
                    pnl_surface[i, j] = 0
        
        # Create 3D surface plot with error handling
        fig = go.Figure(data=[go.Surface(
            z=pnl_surface,
            x=yield_changes,
            y=time_points,
            colorscale='RdYlBu',
            hovertemplate='<b>Yield Change:</b> %{x:.0f} bps<br>' +
                          '<b>Time to Maturity:</b> %{y:.1f} years<br>' +
                          '<b>P&L:</b> $%{z:,.0f}<br>' +
                          '<extra></extra>'
        )])
        
        fig.update_layout(
            title='Bond P&L Scenario Analysis (Duration + Convexity)',
            scene=dict(
                xaxis_title='Yield Change (bps)',
                yaxis_title='Time to Maturity (years)',
                zaxis_title='P&L ($)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating 3D scenario analysis: {str(e)}")
        return None



# -------------------------------
# Legacy Functions (kept for compatibility)
# -------------------------------
def plot_yield_change_distribution(df):
    """Legacy matplotlib version - kept for compatibility."""
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


def plot_pnl_vs_var(df, var_1d, confidence_level):
    """Legacy matplotlib version - kept for compatibility."""
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