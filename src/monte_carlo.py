import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.stats import norm
from datetime import datetime, timedelta
import streamlit as st
import time


# -------------------------------
# Enhanced Monte Carlo Simulation
# -------------------------------
def compute_monte_carlo_var(tickers, weights, portfolio_value=1_000_000, confidence_level=0.95,
                             num_simulations=10_000, time_horizon=1):
    """
    Enhanced Monte Carlo VaR with additional metrics for interactive visualization.
    
    Parameters:
        tickers (list): List of asset tickers
        weights (list): Portfolio weights
        portfolio_value (float): Portfolio value
        confidence_level (float): Confidence level
        num_simulations (int): Number of Monte Carlo simulations
        time_horizon (int): Time horizon in days
    
    Returns:
        dict: Enhanced results with simulation paths and interactive data
    """
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
    if price_data.empty:
        raise ValueError("All price data was dropped due to NaNs. Check your tickers or date range.")

    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    cov_matrix = log_returns.cov().values
    mean_returns = log_returns.mean().values

    # Enhanced simulation with paths
    simulation_results = run_enhanced_simulation(
        mean_returns, cov_matrix, weights, num_simulations, time_horizon
    )
    
    # Calculate VaR
    portfolio_returns = simulation_results['portfolio_returns']
    var_pct = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
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

    # Enhanced metrics for visualization
    percentiles = np.percentile(portfolio_returns, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    
    return {
        'returns': log_returns,
        'cov_matrix': cov_matrix,
        'mean_returns': mean_returns,
        'VaR_pct': var_pct,
        'VaR_dollar': var_dollar,
        'simulated_returns': portfolio_returns,
        'simulation_paths': simulation_results['simulation_paths'],
        'individual_simulations': simulation_results['individual_simulations'],
        'pnl_df': pnl_df,
        'num_exceedances': num_exceedances,
        'exceedance_pct': exceedance_pct,
        'percentiles': percentiles,
        'confidence_level': confidence_level,
        'num_simulations': num_simulations,
        'portfolio_value': portfolio_value,
        'weights': weights,
        'tickers': tickers
    }


def run_enhanced_simulation(mean_returns, cov_matrix, weights, num_simulations, time_horizon):
    """
    Run enhanced Monte Carlo simulation with individual asset paths.
    
    Returns:
        dict: Simulation results including individual paths
    """
    num_assets = len(weights)
    
    # Generate random shocks
    if num_assets == 1:
        std_dev = np.sqrt(cov_matrix[0, 0])
        random_shocks = np.random.normal(0, std_dev, (num_simulations, time_horizon))
        individual_simulations = random_shocks.reshape(num_simulations, time_horizon, 1)
    else:
        # Cholesky decomposition for correlated shocks
        cholesky_matrix = np.linalg.cholesky(cov_matrix)
        normal_randoms = np.random.normal(0, 1, (num_simulations, time_horizon, num_assets))
        individual_simulations = normal_randoms @ cholesky_matrix.T
    
    # Add drift (mean returns)
    for i in range(num_assets):
        individual_simulations[:, :, i] += mean_returns[i]
    
    # Calculate portfolio returns for each simulation
    portfolio_returns = np.zeros(num_simulations)
    simulation_paths = np.zeros((num_simulations, time_horizon))
    
    for sim in range(num_simulations):
        # Portfolio return for this simulation
        sim_portfolio_returns = individual_simulations[sim] @ weights
        simulation_paths[sim] = np.cumsum(sim_portfolio_returns)
        portfolio_returns[sim] = sim_portfolio_returns.sum()  # Total return over time horizon
    
    return {
        'portfolio_returns': portfolio_returns,
        'simulation_paths': simulation_paths,
        'individual_simulations': individual_simulations
    }


def fetch_data(ticker, start, end):
    """Enhanced data fetching with error handling."""
    try:
        df = yf.download(ticker, start=start, end=end)['Close']
        if df.dropna().empty:
            raise ValueError(f"No data from Yahoo for {ticker}")
        print(f"✅ Fetched {ticker} from Yahoo Finance.")
    except Exception as e:
        try:
            df = pdr.DataReader(ticker, 'fred', start, end)
            df = df.squeeze()
            if df.dropna().empty:
                raise ValueError(f"No data from FRED for {ticker}")
            print(f"✅ Fetched {ticker} from FRED.")
        except Exception as fred_e:
            raise ValueError(
                f"❌ Could not fetch {ticker} from Yahoo or FRED.\n"
                f"Yahoo error: {e}\nFRED error: {fred_e}"
            )
    return df


# -------------------------------
# Real-Time Simulation Visualizer
# -------------------------------
def create_realtime_simulation_progress(placeholder, num_simulations, batch_size=1000):
    """
    Create real-time simulation progress with live histogram building.
    
    Parameters:
        placeholder: Streamlit placeholder for updating charts
        num_simulations (int): Total number of simulations
        batch_size (int): Number of simulations per batch
    
    Returns:
        generator: Yields simulation results in batches
    """
    results = []
    
    for batch_start in range(0, num_simulations, batch_size):
        batch_end = min(batch_start + batch_size, num_simulations)
        current_batch = batch_end - batch_start
        
        # Simulate current batch (placeholder for actual simulation)
        batch_results = np.random.normal(0, 0.02, current_batch)
        results.extend(batch_results)
        
        # Update progress
        progress = batch_end / num_simulations
        
        # Create live histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results,
            nbinsx=50,
            name='Simulated Returns',
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Live Monte Carlo Simulation - {batch_end:,}/{num_simulations:,} ({progress:.1%} Complete)',
            xaxis_title='Portfolio Returns',
            yaxis_title='Frequency',
            height=400
        )
        
        placeholder.plotly_chart(fig, use_container_width=True)
        
        yield {
            'progress': progress,
            'current_results': results.copy(),
            'batch_number': batch_end // batch_size
        }
        
        # Small delay for visualization effect
        time.sleep(0.1)


# -------------------------------
# Interactive Monte Carlo Dashboard
# -------------------------------
def create_monte_carlo_dashboard(results):
    """
    FIXED: Create comprehensive Monte Carlo dashboard with proper data validation.
    """
    try:
        if not results or 'simulated_returns' not in results:
            st.warning("No simulation results available")
            return None
        
        # Create subplots with error handling
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Simulation Histogram with VaR',
                'Sample Simulation Paths',
                'Risk Metrics Gauge',
                'Percentile Analysis'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "indicator"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12
        )
        
        simulated_returns = results['simulated_returns']
        var_pct = results['VaR_pct']
        confidence_level = results['confidence_level']
        
        if len(simulated_returns) == 0:
            st.warning("No simulation data available")
            return None
        
        # 1. Histogram with VaR
        try:
            fig.add_trace(
                go.Histogram(
                    x=simulated_returns * 100,  # Convert to percentage
                    nbinsx=min(50, max(10, len(simulated_returns)//100)),
                    name='Simulated Returns',
                    opacity=0.7,
                    marker_color='lightblue',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add VaR line
            fig.add_vline(
                x=var_pct * 100,
                line_width=3,
                line_color="red",
                annotation_text=f"VaR ({int(confidence_level * 100)}%)",
                row=1, col=1
            )
        except Exception as e:
            st.warning(f"Could not create histogram: {str(e)}")
        
        # 2. Sample simulation paths with error handling
        try:
            simulation_paths = results.get('simulation_paths', np.array([]))
            if simulation_paths.size > 0 and len(simulation_paths.shape) == 2:
                # Show first 20 paths for performance
                sample_paths = simulation_paths[:20] * 100  # Convert to percentage
                
                for i in range(min(10, len(sample_paths))):  # Show max 10 paths for clarity
                    fig.add_trace(
                        go.Scatter(
                            y=sample_paths[i],
                            mode='lines',
                            opacity=0.5,
                            line=dict(width=1),
                            showlegend=False,
                            name=f'Path {i+1}',
                            hovertemplate='Step %{x}: %{y:.2f}%<extra></extra>'
                        ),
                        row=1, col=2
                    )
            else:
                # Add placeholder text if no paths available
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text="Simulation paths not available",
                    showarrow=False,
                    row=1, col=2
                )
        except Exception as e:
            st.warning(f"Could not create simulation paths: {str(e)}")
        
        # 3. Risk gauge
        try:
            var_percentage = abs(var_pct) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=var_percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "VaR (% Portfolio)"},
                    gauge={
                        'axis': {'range': [None, min(20, var_percentage * 2)]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2], 'color': "lightgray"},
                            {'range': [2, 5], 'color': "yellow"},
                            {'range': [5, 20], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 5
                        }
                    }
                ),
                row=2, col=1
            )
        except Exception as e:
            st.warning(f"Could not create risk gauge: {str(e)}")
        
        # 4. Percentile analysis
        try:
            percentiles = results.get('percentiles', [])
            if len(percentiles) > 0:
                percentile_labels = ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%']
                
                fig.add_trace(
                    go.Scatter(
                        x=percentile_labels,
                        y=percentiles * 100,
                        mode='lines+markers',
                        name='Return Percentiles',
                        line=dict(color='green', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=2
                )
            else:
                # Calculate basic percentiles if not available
                basic_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = [np.percentile(simulated_returns, p) for p in basic_percentiles]
                percentile_labels = [f'{p}%' for p in basic_percentiles]
                
                fig.add_trace(
                    go.Scatter(
                        x=percentile_labels,
                        y=[p * 100 for p in percentile_values],
                        mode='lines+markers',
                        name='Return Percentiles',
                        line=dict(color='green', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=2
                )
        except Exception as e:
            st.warning(f"Could not create percentile analysis: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title_text="Monte Carlo VaR Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update individual subplot axes
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Percentile", row=2, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)
        
        return fig
    except Exception as e:
        st.error(f"Error creating Monte Carlo dashboard: {str(e)}")
        return None

# -------------------------------
# Interactive 3D Simulation Visualization
# -------------------------------
def create_3d_simulation_visualization(results):
    """
    Create 3D visualization of Monte Carlo simulation results.
    
    Parameters:
        results (dict): Monte Carlo simulation results
    
    Returns:
        plotly.graph_objects.Figure: 3D visualization
    """
    if 'individual_simulations' not in results:
        return None
    
    individual_sims = results['individual_simulations']
    if individual_sims.shape[2] < 2:  # Need at least 2 assets for 3D
        return None
    
    # Take a sample of simulations for performance
    sample_size = min(1000, individual_sims.shape[0])
    sample_indices = np.random.choice(individual_sims.shape[0], sample_size, replace=False)
    
    # Get final returns for first two assets
    asset1_returns = individual_sims[sample_indices, -1, 0] * 100
    asset2_returns = individual_sims[sample_indices, -1, 1] * 100
    portfolio_returns = results['simulated_returns'][sample_indices] * 100
    
    # Color code by portfolio performance
    colors = portfolio_returns
    
    fig = go.Figure(data=go.Scatter3d(
        x=asset1_returns,
        y=asset2_returns,
        z=portfolio_returns,
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            colorscale='RdYlBu_r',
            opacity=0.6,
            colorbar=dict(title="Portfolio Return (%)")
        ),
        hovertemplate=
        f'<b>{results["tickers"][0]}</b>: %{{x:.2f}}%<br>' +
        f'<b>{results["tickers"][1] if len(results["tickers"]) > 1 else "Asset 2"}</b>: %{{y:.2f}}%<br>' +
        '<b>Portfolio</b>: %{z:.2f}%<br>' +
        '<extra></extra>'
    ))
    
    fig.update_layout(
        title='3D Monte Carlo Simulation Results',
        scene=dict(
            xaxis_title=f'{results["tickers"][0]} Return (%)',
            yaxis_title=f'{results["tickers"][1] if len(results["tickers"]) > 1 else "Asset 2"} Return (%)',
            zaxis_title='Portfolio Return (%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600
    )
    
    return fig


# -------------------------------
# Enhanced Correlation Visualization
# -------------------------------
def plot_enhanced_correlation_matrix(df):
    """
    Create enhanced interactive correlation matrix.
    
    Parameters:
        df (DataFrame): Return data
    
    Returns:
        plotly.graph_objects.Figure: Interactive correlation heatmap
    """
    corr_matrix = df.corr()
    
    # Create custom colorscale
    colorscale = [
        [0, '#d73027'],      # Strong negative correlation
        [0.25, '#f46d43'],   # Moderate negative correlation  
        [0.5, '#ffffff'],    # No correlation
        [0.75, '#74add1'],   # Moderate positive correlation
        [1, '#313695']       # Strong positive correlation
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=colorscale,
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Asset Correlation Matrix',
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis_title='Assets',
        yaxis_title='Assets',
        height=500,
        width=500
    )
    
    return fig


# -------------------------------
# Legacy Functions (kept for compatibility)
# -------------------------------
def plot_simulated_returns(simulated_returns, var_pct, confidence_level):
    """Legacy matplotlib version - kept for compatibility."""
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


def plot_correlation_matrix(df):
    """Legacy matplotlib version - kept for compatibility."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Asset Returns")
    plt.tight_layout()
    return fig


def plot_monte_carlo_pnl_vs_var(pnl_df, var_dollar, confidence_level):
    """Legacy matplotlib version - kept for compatibility."""
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