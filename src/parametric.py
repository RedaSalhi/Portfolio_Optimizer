import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st


# -------------------------------
# Core Computation Function (Enhanced)
# -------------------------------
def compute_parametric_var(tickers, confidence_level=0.95, position_size=1_000_000):
    """
    Enhanced parametric VaR computation with additional metrics for interactive dashboards.
    
    Parameters:
        tickers (list or str): A single ticker or list of tickers.
        confidence_level (float): Confidence level for VaR (e.g., 0.95).
        position_size (float): Notional value of position in USD.
        
    Returns:
        list of dicts: Enhanced result dictionary per ticker with interactive visualization data.
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

            # Enhanced metrics for interactive dashboards
            rolling_vol = df['Log_Return'].rolling(window=30).std()
            rolling_var = -z * rolling_vol * position_size
            
            # Risk decomposition
            max_loss = df['PnL'].min()
            avg_loss = df['PnL'][df['PnL'] < 0].mean() if (df['PnL'] < 0).any() else 0
            
            # Performance metrics
            sharpe_ratio = df['Log_Return'].mean() / sigma * np.sqrt(252) if sigma > 0 else 0
            max_drawdown = calculate_max_drawdown(df['Price'])
            
            # Distribution fit metrics
            skewness = stats.skew(df['Log_Return'])
            kurtosis = stats.kurtosis(df['Log_Return'])
            jarque_bera_stat, jarque_bera_pval = stats.jarque_bera(df['Log_Return'])

            results.append({
                'ticker': ticker,
                'daily_volatility': sigma,
                'VaR': var_1d,
                'z_score': z,
                'num_exceedances': breaches,
                'exceedance_pct': breach_pct,
                'df': df,
                'rolling_volatility': rolling_vol,
                'rolling_var': rolling_var,
                'max_loss': max_loss,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'jarque_bera_pval': jarque_bera_pval,
                'confidence_level': confidence_level,
                'position_size': position_size
            })

        except Exception as e:
            results.append({
                'ticker': ticker,
                'error': str(e)
            })

    return results


def calculate_max_drawdown(price_series):
    """Calculate maximum drawdown for a price series."""
    peak = price_series.expanding().max()
    drawdown = (price_series - peak) / peak
    return drawdown.min()


# -------------------------------
# Interactive VaR Gauge Dashboard
# -------------------------------
def create_var_gauge(var_value, position_size, confidence_level=0.95):
    """
    Create an interactive VaR gauge with risk zones.
    
    Parameters:
        var_value (float): Current VaR value
        position_size (float): Portfolio size
        confidence_level (float): Confidence level
    
    Returns:
        plotly.graph_objects.Figure: Interactive gauge chart
    """
    # Calculate risk thresholds as percentages of portfolio
    var_pct = abs(var_value) / position_size * 100
    
    # Define risk zones
    safe_threshold = 1.0    # 1% of portfolio
    warning_threshold = 2.5 # 2.5% of portfolio  
    danger_threshold = 5.0  # 5% of portfolio
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=var_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Portfolio VaR ({int(confidence_level * 100)}% Confidence)",
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        number={
            'suffix': "% of Portfolio",
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        delta={
            'reference': safe_threshold,
            'increasing': {'color': '#e74c3c'},
            'decreasing': {'color': '#27ae60'}
        },
        gauge={
            'axis': {
                'range': [None, danger_threshold],
                'tickwidth': 1,
                'tickcolor': "#2c3e50",
                'tickfont': {'color': '#2c3e50', 'size': 12}
            },
            'bar': {'color': "#667eea", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#2c3e50",
            'steps': [
                {'range': [0, safe_threshold], 'color': '#d4edda'},
                {'range': [safe_threshold, warning_threshold], 'color': '#fff3cd'},
                {'range': [warning_threshold, danger_threshold], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "#e74c3c", 'width': 4},
                'thickness': 0.75,
                'value': warning_threshold
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'}
    )
    
    return fig


# -------------------------------
# Interactive Return Distribution with VaR
# -------------------------------
def plot_interactive_return_distribution(df, var_value, confidence_level=0.95):
    """
    FIXED: Create interactive return distribution with proper data validation.
    """
    try:
        if df is None or df.empty:
            st.warning("No data available for return distribution")
            return None
            
        if 'Log_Return' not in df.columns:
            st.warning("Log_Return column not found in data")
            return None
            
        returns = df['Log_Return'].dropna()
        
        if len(returns) == 0:
            st.warning("No valid return data available")
            return None
        
        # Create histogram
        fig = go.Figure()
        
        # Add histogram with error handling
        try:
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=min(50, max(10, len(returns)//20)),  # Adaptive bin count
                name='Actual Returns',
                opacity=0.7,
                marker_color='lightblue',
                histnorm='probability density'
            ))
        except:
            # Fallback to simple histogram
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=20,
                name='Actual Returns',
                opacity=0.7,
                marker_color='lightblue'
            ))
        
        # Fit normal distribution with error handling
        try:
            mu, sigma = returns.mean(), returns.std()
            
            if sigma > 0:
                x_normal = np.linspace(returns.min(), returns.max(), 100)
                y_normal = stats.norm.pdf(x_normal, mu, sigma)
                
                fig.add_trace(go.Scatter(
                    x=x_normal,
                    y=y_normal,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2, dash='dash')
                ))
        except:
            pass  # Skip normal distribution overlay if it fails
        
        # Add VaR line with error handling
        try:
            if 'Price' in df.columns and not df['Price'].empty:
                var_return = var_value / df['Price'].iloc[-1]  # Convert to return space
                fig.add_vline(
                    x=var_return,
                    line_width=3,
                    line_color="red",
                    annotation_text=f"VaR ({int(confidence_level * 100)}%)",
                    annotation_position="top"
                )
        except:
            pass  # Skip VaR line if conversion fails
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Return Distribution Analysis',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='Daily Log Returns',
            yaxis_title='Density',
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating return distribution: {str(e)}")
        return None


# -------------------------------
# Real-Time Risk Dashboard
# -------------------------------
def create_risk_dashboard(results):
    """
    Create comprehensive risk dashboard with multiple metrics.
    
    Parameters:
        results (list): List of VaR calculation results
    
    Returns:
        plotly.graph_objects.Figure: Multi-panel dashboard
    """
    if not results or 'error' in results[0]:
        return None
    
    result = results[0]  # Take first result for single asset
    df = result['df']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rolling Volatility', 'P&L vs VaR Breaches',
            'Return Distribution', 'Risk Metrics'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "indicator"}]]
    )
    
    # 1. Rolling Volatility
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=result['rolling_volatility'] * 100,
            mode='lines',
            name='30-Day Volatility',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 2. P&L vs VaR
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['PnL'],
            mode='lines',
            name='Daily P&L',
            line=dict(color='green', width=1)
        ),
        row=1, col=2
    )
    
    # Add VaR line
    fig.add_hline(
        y=-result['VaR'],
        line_dash="dash",
        line_color="red",
        row=1, col=2
    )
    
    # Mark breaches
    breaches = df[df['VaR_Breach']]
    if not breaches.empty:
        fig.add_trace(
            go.Scatter(
                x=breaches.index,
                y=breaches['PnL'],
                mode='markers',
                name='VaR Breaches',
                marker=dict(color='red', size=8, symbol='x')
            ),
            row=1, col=2
        )
    
    # 3. Return Distribution
    fig.add_trace(
        go.Histogram(
            x=df['Log_Return'],
            nbinsx=30,
            name='Returns',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 4. Risk Indicator
    var_pct = abs(result['VaR']) / result['position_size'] * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=var_pct,
            title={'text': "VaR % of Portfolio"},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2.5], 'color': "yellow"},
                    {'range': [2.5, 5], 'color': "red"}
                ]
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Real-Time Risk Dashboard",
        height=700,
        showlegend=True
    )
    
    return fig


# -------------------------------
# Enhanced PnL vs VaR with Animations
# -------------------------------
def plot_animated_pnl_vs_var(df, var_value, confidence_level):
    """
    Create animated P&L vs VaR plot with interactive features.
    
    Parameters:
        df (DataFrame): Price and return data
        var_value (float): VaR value
        confidence_level (float): Confidence level
    
    Returns:
        plotly.graph_objects.Figure: Animated plot
    """
    fig = go.Figure()
    
    # Add P&L line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['PnL'],
        mode='lines',
        name='Daily P&L',
        line=dict(color='#667eea', width=2),
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>P&L:</b> $%{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add VaR line
    fig.add_hline(
        y=-var_value,
        line_dash="dash",
        line_color="#e74c3c",
        line_width=3,
        annotation_text=f"VaR ({int(confidence_level * 100)}%)",
        annotation_position="bottom right"
    )
    
    # Add zero line
    fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5)
    
    # Mark VaR breaches
    breaches = df[df['VaR_Breach']]
    if not breaches.empty:
        fig.add_trace(go.Scatter(
            x=breaches.index,
            y=breaches['PnL'],
            mode='markers',
            name='VaR Breaches',
            marker=dict(
                color='#e74c3c',
                size=10,
                symbol='x',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>VaR Breach</b><br>' +
                          '<b>Date:</b> %{x}<br>' +
                          '<b>Loss:</b> $%{y:,.0f}<br>' +
                          '<extra></extra>'
        ))
    
    # Add profit/loss fill
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['PnL'],
        fill='tozeroy',
        mode='none',
        name='P&L Fill',
        fillcolor='rgba(102, 126, 234, 0.1)',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Portfolio P&L vs Parametric VaR',
            'x': 0.5,
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title='Date',
        yaxis_title='P&L ($)',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig


# -------------------------------
# Legacy matplotlib functions (kept for compatibility)
# -------------------------------
def plot_return_distribution(df, bins=100):
    """Legacy matplotlib version - kept for compatibility."""
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


def plot_pnl_vs_var(df, var_value, confidence_level):
    """Legacy matplotlib version - kept for compatibility."""
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