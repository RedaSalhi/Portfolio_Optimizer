import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from scipy import stats
from src.fixed_income import compute_fixed_income_var
from src.parametric import compute_parametric_var
import streamlit as st


# -------------------------------
# Enhanced Portfolio VaR Computation
# ------------------------------
def compute_portfolio_var(equity_tickers=None, equity_weights=None,
                          bond_tickers=None, bond_weights=None,
                          confidence_level=0.95,
                          position_size=1_000_000,
                          maturity=10):
    """
    Enhanced portfolio-level parametric VaR with comprehensive risk metrics.
    
    Parameters:
        equity_tickers (list): Equity asset tickers
        equity_weights (list): Equity weights
        bond_tickers (list): Bond instrument tickers
        bond_weights (list): Bond weights
        confidence_level (float): Confidence level
        position_size (float): Portfolio notional value
        maturity (int): Bond maturity for PV01 calculation
    
    Returns:
        dict: Enhanced portfolio VaR results with risk attribution
    """
    equity_tickers = equity_tickers or []
    bond_tickers = bond_tickers or []
    equity_weights = equity_weights or []
    bond_weights = bond_weights or []

    if not equity_tickers and not bond_tickers:
        raise ValueError("At least one asset (equity or bond) must be provided.")

    # Normalize weights
    equity_weights = np.array(equity_weights, dtype=float)
    bond_weights = np.array(bond_weights, dtype=float)
    total_weight = np.sum(equity_weights) + np.sum(bond_weights)

    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero.")

    equity_weights = equity_weights / total_weight if len(equity_weights) > 0 else []
    bond_weights = bond_weights / total_weight if len(bond_weights) > 0 else []
    all_weights = list(equity_weights) + list(bond_weights)

    # Fetch data and compute individual VaRs
    equity_results = compute_parametric_var(equity_tickers, confidence_level, position_size) if equity_tickers else []
    bond_results = compute_fixed_income_var(bond_tickers, maturity, confidence_level, position_size) if bond_tickers else []

    # Process results
    log_returns_list = []
    individual_vars = []
    individual_volatilities = []
    asset_names = []
    component_vars = []

    # Process equity results
    for i, res in enumerate(equity_results):
        if 'error' in res:
            continue
        df = res['df'][['Log_Return']].rename(columns={'Log_Return': res['ticker']})
        log_returns_list.append(df)
        individual_vars.append(res['VaR'])
        individual_volatilities.append(res['daily_volatility'])
        asset_names.append(res['ticker'])
        
        # Component VaR calculation
        weight = equity_weights[i] if i < len(equity_weights) else 0
        component_vars.append(weight * res['VaR'])

    # Process bond results
    for i, res in enumerate(bond_results):
        df = res['df'][['Yield_Change_bps']].copy()
        df['Log_Return'] = -res['pv01'] * df['Yield_Change_bps'] / 100 / position_size
        df = df[['Log_Return']].rename(columns={'Log_Return': res['ticker']})
        log_returns_list.append(df)
        individual_vars.append(res['VaR'])
        individual_volatilities.append(res['vol_bps'] / 10000)  # Convert to decimal
        asset_names.append(res['ticker'])
        
        # Component VaR calculation
        weight = bond_weights[i] if i < len(bond_weights) else 0
        component_vars.append(weight * res['VaR'])

    if len(log_returns_list) == 0:
        raise ValueError("No valid data returned for the given tickers.")

    return_df = pd.concat(log_returns_list, axis=1).dropna()

    # Adjust weights to match surviving assets
    if return_df.shape[1] != len(all_weights):
        valid_cols = return_df.columns.tolist()
        valid_weights = []
        for name in valid_cols:
            if name in asset_names:
                idx = asset_names.index(name)
                valid_weights.append(all_weights[idx])
        all_weights = valid_weights

    # Portfolio calculations
    return_df['Portfolio_Log_Return'] = return_df.dot(all_weights)
    return_df['PnL'] = return_df['Portfolio_Log_Return'] * position_size

    # VaR calculations
    z = stats.norm.ppf(1 - confidence_level)
    sigma = return_df['Portfolio_Log_Return'].std()
    var = -z * sigma * position_size
    weighted_var_sum = sum(w * v for w, v in zip(all_weights, individual_vars))

    # Risk attribution calculations
    correlation_matrix = return_df[asset_names].corr()
    diversification_ratio = var / weighted_var_sum if weighted_var_sum != 0 else 1
    
    # Component and marginal VaR
    cov_matrix = return_df[asset_names].cov()
    portfolio_variance = np.dot(all_weights, np.dot(cov_matrix, all_weights))
    marginal_vars = np.dot(cov_matrix, all_weights) / np.sqrt(portfolio_variance) * z * position_size
    
    # VaR breaches
    return_df['VaR_Breach'] = return_df['PnL'] < -var
    breaches = return_df['VaR_Breach'].sum()
    breach_pct = 100 * breaches / len(return_df)

    # Enhanced metrics
    portfolio_sharpe = return_df['Portfolio_Log_Return'].mean() / sigma * np.sqrt(252) if sigma > 0 else 0
    max_loss = return_df['PnL'].min()
    avg_loss = return_df['PnL'][return_df['PnL'] < 0].mean() if (return_df['PnL'] < 0).any() else 0

    return {
        'var_portfolio': var,
        'weighted_var_sum': weighted_var_sum,
        'volatility': sigma,
        'exceedances': breaches,
        'exceedance_pct': breach_pct,
        'return_df': return_df,
        'asset_names': asset_names,
        'weights': all_weights,
        'individual_vars': individual_vars,
        'individual_volatilities': individual_volatilities,
        'component_vars': component_vars,
        'marginal_vars': marginal_vars.tolist(),
        'correlation_matrix': correlation_matrix,
        'diversification_ratio': diversification_ratio,
        'portfolio_sharpe': portfolio_sharpe,
        'max_loss': max_loss,
        'avg_loss': avg_loss,
        'confidence_level': confidence_level,
        'position_size': position_size
    }


# -------------------------------
# Interactive Portfolio Risk Dashboard
# -------------------------------
def create_portfolio_risk_dashboard(results):
    """
    Create comprehensive interactive portfolio risk dashboard.
    
    Parameters:
        results (dict): Portfolio VaR results
    
    Returns:
        plotly.graph_objects.Figure: Multi-panel dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Portfolio VaR Breakdown',
            'Risk Attribution (Component VaR)',
            'Asset Correlation Heatmap',
            'Rolling Portfolio Volatility',
            'Risk Metrics Gauge',
            'Diversification Analysis'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "indicator"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    asset_names = results['asset_names']
    weights = results['weights']
    individual_vars = results['individual_vars']
    component_vars = results['component_vars']
    
    # 1. Portfolio VaR Breakdown
    fig.add_trace(
        go.Bar(
            x=asset_names,
            y=individual_vars,
            name='Individual VaR',
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Individual VaR: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add portfolio VaR line
    fig.add_hline(
        y=results['var_portfolio'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Portfolio VaR: ${results['var_portfolio']:,.0f}",
        row=1, col=1
    )
    
    # 2. Component VaR (Risk Attribution)
    fig.add_trace(
        go.Bar(
            x=asset_names,
            y=component_vars,
            name='Component VaR',
            marker_color='orange',
            hovertemplate='<b>%{x}</b><br>Component VaR: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Correlation Heatmap
    corr_matrix = results['correlation_matrix']
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            showscale=False,
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Rolling Volatility
    return_df = results['return_df']
    rolling_vol = return_df['Portfolio_Log_Return'].rolling(window=30).std()
    
    fig.add_trace(
        go.Scatter(
            x=return_df.index,
            y=rolling_vol * 100 * np.sqrt(252),  # Annualized %
            mode='lines',
            name='30-Day Rolling Vol',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )
    
    # 5. Risk Gauge
    var_pct = abs(results['var_portfolio']) / results['position_size'] * 100
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=var_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Portfolio VaR (%)"},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ),
        row=3, col=1
    )
    
    # 6. Diversification Analysis
    diversification_ratio = results['diversification_ratio']
    undiversified_risk = results['weighted_var_sum']
    portfolio_risk = results['var_portfolio']
    
    fig.add_trace(
        go.Bar(
            x=['Undiversified Risk', 'Portfolio Risk', 'Risk Reduction'],
            y=[undiversified_risk, portfolio_risk, undiversified_risk - portfolio_risk],
            marker_color=['red', 'blue', 'green'],
            name='Risk Decomposition'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Portfolio Risk Dashboard",
        height=1000,
        showlegend=False
    )
    
    return fig


# -------------------------------
# Risk Attribution Treemap
# -------------------------------
def create_risk_attribution_treemap(results):
    """
    Create interactive treemap showing risk contribution by asset.
    
    Parameters:
        results (dict): Portfolio VaR results
    
    Returns:
        plotly.graph_objects.Figure: Risk attribution treemap
    """
    asset_names = results['asset_names']
    component_vars = [abs(cv) for cv in results['component_vars']]
    weights = [w * 100 for w in results['weights']]  # Convert to percentages
    
    # Create hierarchical data for treemap
    fig = go.Figure(go.Treemap(
        labels=asset_names,
        values=component_vars,
        parents=["Portfolio"] * len(asset_names),
        textinfo="label+value+percent parent",
        texttemplate='<b>%{label}</b><br>VaR: $%{value:,.0f}<br>%{percentParent}',
        hovertemplate='<b>%{label}</b><br>' +
                      'Component VaR: $%{value:,.0f}<br>' +
                      'Weight: %{customdata:.1f}%<br>' +
                      '<extra></extra>',
        customdata=weights,
        marker=dict(
            colorscale='RdYlBu_r',
            cmid=np.mean(component_vars),
            colorbar=dict(title="Component VaR ($)")
        )
    ))
    
    fig.update_layout(
        title="Portfolio Risk Attribution",
        height=500,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    return fig


# -------------------------------
# Interactive Correlation Network
# -------------------------------
def create_correlation_network(results):
    """
    Create interactive network graph showing asset correlations.
    
    Parameters:
        results (dict): Portfolio VaR results
    
    Returns:
        plotly.graph_objects.Figure: Correlation network graph
    """
    correlation_matrix = results['correlation_matrix']
    asset_names = results['asset_names']
    
    if len(asset_names) < 2:
        return None
    
    # Create network layout
    n_assets = len(asset_names)
    angles = np.linspace(0, 2*np.pi, n_assets, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Prepare edge traces for correlations
    edge_traces = []
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = correlation_matrix.iloc[i, j]
            
            # Only show significant correlations
            if abs(corr) > 0.3:
                edge_trace = go.Scatter(
                    x=[x_pos[i], x_pos[j], None],
                    y=[y_pos[i], y_pos[j], None],
                    mode='lines',
                    line=dict(
                        width=abs(corr) * 10,
                        color='red' if corr > 0 else 'blue'
                    ),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
    
    # Node trace
    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=[w * 1000 for w in results['weights']],  # Size by weight
            color=results['individual_vars'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Individual VaR ($)")
        ),
        text=asset_names,
        textposition="middle center",
        hovertemplate='<b>%{text}</b><br>' +
                      'Weight: %{customdata:.1%}<br>' +
                      'VaR: $%{marker.color:,.0f}<br>' +
                      '<extra></extra>',
        customdata=results['weights']
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Asset Correlation Network",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        showlegend=False
    )
    
    return fig


# -------------------------------
# Enhanced Portfolio Performance Chart
# -------------------------------
def plot_enhanced_portfolio_pnl_vs_var(pnl_df, var_value, confidence_level):
    """
    Create enhanced interactive P&L vs VaR visualization.
    
    Parameters:
        pnl_df (DataFrame): Portfolio P&L data
        var_value (float): Portfolio VaR value
        confidence_level (float): Confidence level
    
    Returns:
        plotly.graph_objects.Figure: Interactive P&L chart
    """
    fig = go.Figure()
    
    # Add P&L line with conditional coloring
    colors = ['red' if pnl < 0 else 'green' for pnl in pnl_df['PnL']]
    
    fig.add_trace(go.Scatter(
        x=pnl_df.index,
        y=pnl_df['PnL'],
        mode='lines',
        name='Daily P&L',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
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
        annotation_text=f"VaR ({int(confidence_level * 100)}%): ${var_value:,.0f}",
        annotation_position="bottom right"
    )
    
    # Add zero line
    fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5)
    
    # Mark VaR breaches
    breaches = pnl_df[pnl_df['VaR_Breach']]
    if not breaches.empty:
        fig.add_trace(go.Scatter(
            x=breaches.index,
            y=breaches['PnL'],
            mode='markers',
            name='VaR Breaches',
            marker=dict(
                color='#e74c3c',
                size=12,
                symbol='x',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>VaR Breach</b><br>' +
                          '<b>Date:</b> %{x}<br>' +
                          '<b>Loss:</b> $%{y:,.0f}<br>' +
                          '<extra></extra>'
        ))
    
    # Add statistical annotations
    total_losses = len(pnl_df[pnl_df['PnL'] < 0])
    max_loss = pnl_df['PnL'].min()
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Risk Statistics:</b><br>" +
             f"VaR Breaches: {len(breaches)}<br>" +
             f"Total Loss Days: {total_losses}<br>" +
             f"Max Loss: ${max_loss:,.0f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Portfolio P&L vs VaR Analysis',
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
        )
    )
    
    # Add range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
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
# Legacy Functions (kept for compatibility)
# -------------------------------
def plot_correlation_matrix(df):
    """Legacy matplotlib version - kept for compatibility."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Returns")
    plt.tight_layout()
    return fig


def plot_individual_distributions(df):
    """Legacy matplotlib version - kept for compatibility."""
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
        plot_count += 1

    # Hide any unused subplots
    for j in range(plot_count, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig


def plot_portfolio_pnl_vs_var(pnl_df, var_value, confidence_level):
    """Legacy matplotlib version - kept for compatibility."""
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