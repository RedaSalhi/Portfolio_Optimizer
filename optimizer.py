# optimizer.py - Enhanced Portfolio Optimization Engine

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Enhanced Portfolio Optimization Class with robust error handling."""
    
    def __init__(self, tickers, lookback_years=3):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
            tickers (list): List of asset tickers
            lookback_years (int): Years of historical data to use
        """
        self.tickers = [ticker.strip().upper() for ticker in tickers]
        self.lookback_years = lookback_years
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.rf_rate = 0.02  # Default risk-free rate
        
    def quick_test_ticker(self, ticker):
        """Quick test for a single ticker."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
            
            if data.empty:
                return False, "No data returned"
                
            # Get price data
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close'].iloc[:, 0]
                else:
                    prices = data['Close'].iloc[:, 0]
            else:
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                else:
                    prices = data['Close']
            
            prices = prices.dropna()
            
            if len(prices) < 50:
                return False, f"Insufficient data: {len(prices)} days"
            
            return True, f"Success: {len(prices)} days of data"
            
        except Exception as e:
            return False, str(e)

    def fetch_data(self):
        """Fetch and validate market data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)
            
            # Download data with error handling
            data = {}
            failed_tickers = []
            
            for ticker in self.tickers:
                try:
                    # Try downloading with different approaches
                    ticker_data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False,
                        threads=False  # Disable threading for stability
                    )
                    
                    # Handle different data structures
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        # Multi-level columns
                        if 'Adj Close' in ticker_data.columns.get_level_values(0):
                            ticker_data = ticker_data['Adj Close'].iloc[:, 0]
                        elif 'Close' in ticker_data.columns.get_level_values(0):
                            ticker_data = ticker_data['Close'].iloc[:, 0]
                        else:
                            ticker_data = ticker_data.iloc[:, -1]  # Last column
                    else:
                        # Single level columns
                        if 'Adj Close' in ticker_data.columns:
                            ticker_data = ticker_data['Adj Close']
                        elif 'Close' in ticker_data.columns:
                            ticker_data = ticker_data['Close']
                        else:
                            ticker_data = ticker_data.iloc[:, -1]
                    
                    # Clean the data
                    ticker_data = ticker_data.dropna()
                    
                    # More lenient data requirement (at least 100 trading days)
                    if len(ticker_data) < 100:
                        failed_tickers.append(ticker)
                        continue
                        
                    data[ticker] = ticker_data
                    
                except Exception as e:
                    failed_tickers.append(ticker)
                    continue
            
            if not data:
                raise ValueError("No valid data could be fetched for any ticker. Please check ticker symbols and internet connection.")
            
            # Create DataFrame and handle missing data
            self.data = pd.DataFrame(data)
            
            # Only drop rows where ALL values are NaN
            self.data = self.data.dropna(how='all')
            
            # Forward fill missing data (common for weekends/holidays)
            self.data = self.data.ffill().dropna()
            
            if len(self.data) < 50:  # More lenient requirement
                raise ValueError("Insufficient data after cleaning (need at least 50 trading days)")
            
            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            
            # Handle edge case where returns might be empty
            if self.returns.empty:
                raise ValueError("Could not calculate returns from price data")
                
            self.mean_returns = self.returns.mean() * 252  # Annualized
            self.cov_matrix = self.returns.cov() * 252     # Annualized
            
            # Ensure covariance matrix is positive definite
            eigenvals = np.linalg.eigvals(self.cov_matrix)
            if np.any(eigenvals <= 0):
                # Add small values to diagonal to ensure positive definiteness
                self.cov_matrix += np.eye(len(self.cov_matrix)) * 1e-8
            
            # Update tickers list to only include successful ones
            self.tickers = list(self.data.columns)
            
            return True, failed_tickers
            
        except Exception as e:
            return False, str(e)
    
    def get_risk_free_rate(self):
        """Get current risk-free rate."""
        try:
            # Try to get 3-month Treasury rate
            treasury = yf.download('^IRX', period='5d', progress=False)
            if not treasury.empty:
                self.rf_rate = treasury['Close'].iloc[-1] / 100
            else:
                self.rf_rate = 0.02  # Default 2%
        except:
            self.rf_rate = 0.02  # Default 2%
        
        return self.rf_rate
    
    def portfolio_stats(self, weights):
        """Calculate portfolio statistics."""
        weights = np.array(weights)
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.rf_rate) / portfolio_std if portfolio_std > 0 else 0
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Objective function for maximum Sharpe ratio."""
        return -self.portfolio_stats(weights)[2]
    
    def portfolio_variance(self, weights):
        """Objective function for minimum variance."""
        return self.portfolio_stats(weights)[1] ** 2
    
    def optimize_portfolio(self, method='max_sharpe', target_return=None, target_volatility=None):
        """
        Optimize portfolio using specified method.
        
        Parameters:
            method (str): 'max_sharpe', 'min_variance', 'target_return', 'target_volatility'
            target_return (float): Target return for constrained optimization
            target_volatility (float): Target volatility for constrained optimization
        
        Returns:
            dict: Optimization results
        """
        num_assets = len(self.tickers)
        args = ()
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1.0/num_assets] * num_assets)
        
        try:
            if method == 'max_sharpe':
                result = minimize(
                    self.negative_sharpe, 
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            
            elif method == 'min_variance':
                result = minimize(
                    self.portfolio_variance,
                    initial_guess,
                    method='SLSQP', 
                    bounds=bounds,
                    constraints=constraints
                )
            
            elif method == 'target_return' and target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: self.portfolio_stats(x)[0] - target_return
                })
                result = minimize(
                    self.portfolio_variance,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            
            elif method == 'target_volatility' and target_volatility is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: self.portfolio_stats(x)[1] - target_volatility
                })
                result = minimize(
                    lambda x: -self.portfolio_stats(x)[0],  # Maximize return
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            if not result.success:
                raise ValueError(f"Optimization failed: {result.message}")
            
            # Calculate final portfolio statistics
            optimal_weights = result.x
            port_return, port_std, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'success': True,
                'weights': optimal_weights,
                'expected_return': port_return,
                'volatility': port_std,
                'sharpe_ratio': sharpe,
                'method': method
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': method
            }
    
    def generate_efficient_frontier(self, num_portfolios=1000):
        """Generate efficient frontier data."""
        if self.returns is None:
            return None
        
        results = {
            'returns': [],
            'volatilities': [],
            'sharpe_ratios': [],
            'weights': []
        }
        
        # Generate target returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        for target_ret in target_returns:
            try:
                opt_result = self.optimize_portfolio('target_return', target_return=target_ret)
                if opt_result['success']:
                    results['returns'].append(opt_result['expected_return'])
                    results['volatilities'].append(opt_result['volatility'])
                    results['sharpe_ratios'].append(opt_result['sharpe_ratio'])
                    results['weights'].append(opt_result['weights'])
            except:
                continue
        
        return results
    
    def calculate_capm_metrics(self):
        """Calculate CAPM metrics using S&P 500 as market proxy."""
        try:
            # Get S&P 500 data
            spy_data = yf.download('^GSPC', 
                                 start=self.data.index[0], 
                                 end=self.data.index[-1], 
                                 progress=False)['Adj Close']
            
            spy_returns = spy_data.pct_change().dropna()
            
            # Align dates
            common_dates = self.returns.index.intersection(spy_returns.index)
            aligned_returns = self.returns.loc[common_dates]
            aligned_spy = spy_returns.loc[common_dates]
            
            capm_metrics = {}
            
            for ticker in self.tickers:
                try:
                    # Calculate beta using regression
                    covariance = np.cov(aligned_returns[ticker], aligned_spy)[0][1]
                    market_variance = np.var(aligned_spy)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                    
                    # Calculate alpha
                    asset_return = aligned_returns[ticker].mean() * 252
                    market_return = aligned_spy.mean() * 252
                    expected_return = self.rf_rate + beta * (market_return - self.rf_rate)
                    alpha = asset_return - expected_return
                    
                    capm_metrics[ticker] = {
                        'beta': beta,
                        'alpha': alpha,
                        'expected_return': expected_return
                    }
                    
                except:
                    capm_metrics[ticker] = {
                        'beta': 1.0,
                        'alpha': 0.0,
                        'expected_return': self.mean_returns[ticker]
                    }
            
            return capm_metrics
            
        except Exception as e:
            return {}


def create_efficient_frontier_plot(optimizer, efficient_frontier_data=None):
    """Create interactive efficient frontier plot with better label positioning."""
    if efficient_frontier_data is None:
        efficient_frontier_data = optimizer.generate_efficient_frontier()
    
    if not efficient_frontier_data or not efficient_frontier_data['returns']:
        return None
    
    fig = go.Figure()
    
    # Plot efficient frontier
    fig.add_trace(go.Scatter(
        x=np.array(efficient_frontier_data['volatilities']) * 100,
        y=np.array(efficient_frontier_data['returns']) * 100,
        mode='markers',
        marker=dict(
            size=6,
            color=efficient_frontier_data['sharpe_ratios'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Efficient Frontier',
        hovertemplate='Return: %{y:.1f}%<br>Risk: %{x:.1f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Add individual assets without text labels to avoid overlap
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    annotations = []
    
    for i, ticker in enumerate(optimizer.tickers):
        asset_return = optimizer.mean_returns[ticker] * 100
        asset_vol = np.sqrt(optimizer.cov_matrix.iloc[i, i]) * 100
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=[asset_vol],
            y=[asset_return],
            mode='markers',
            marker=dict(size=15, color=color, symbol='star', line=dict(width=2, color='white')),
            name=ticker,
            hovertemplate=ticker + '<br>Return: %{y:.1f}%<br>Risk: %{x:.1f}%<extra></extra>',
            showlegend=True
        ))
        
        # Add annotation with smart positioning
        # Offset the annotation based on position to avoid overlap
        x_offset = 0.5 + (i % 3 - 1) * 0.3  # Vary x offset
        y_offset = 0.8 + (i // 3) * 0.05    # Vary y offset for different rows
        
        annotations.append(
            dict(
                x=asset_vol,
                y=asset_return,
                text=ticker,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                ax=20 + (i % 3 - 1) * 15,  # Vary arrow offset
                ay=-20 - (i // 3) * 10,
                font=dict(size=12, color=color),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color,
                borderwidth=1
            )
        )
    
    # Add Capital Allocation Line if risk-free rate is available
    if optimizer.rf_rate > 0 and len(efficient_frontier_data['sharpe_ratios']) > 0:
        try:
            # Find maximum Sharpe ratio portfolio
            max_sharpe_idx = np.argmax(efficient_frontier_data['sharpe_ratios'])
            tangency_vol = efficient_frontier_data['volatilities'][max_sharpe_idx] * 100
            tangency_ret = efficient_frontier_data['returns'][max_sharpe_idx] * 100
            
            # Draw CAL
            cal_x = np.linspace(0, max(tangency_vol * 1.2, 30), 50)
            cal_slope = (tangency_ret - optimizer.rf_rate * 100) / tangency_vol
            cal_y = optimizer.rf_rate * 100 + cal_slope * cal_x
            
            fig.add_trace(go.Scatter(
                x=cal_x,
                y=cal_y,
                mode='lines',
                line=dict(color='orange', width=3, dash='dash'),
                name='Capital Allocation Line',
                hovertemplate='CAL<br>Return: %{y:.1f}%<br>Risk: %{x:.1f}%<extra></extra>',
                showlegend=True
            ))
        except Exception:
            pass  # Skip CAL if calculation fails
    
    fig.update_layout(
        title={
            'text': 'Interactive Efficient Frontier Analysis',
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Expected Return %',
        height=600,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        annotations=annotations
    )
    
    return fig


def create_portfolio_composition_chart(tickers, weights):
    """Create portfolio composition pie chart with improved formatting."""
    # Calculate percentages safely
    weight_percentages = [w * 100 for w in weights]
    
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weight_percentages,
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>',
        marker=dict(
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
            line=dict(color='white', width=2)
        )
    )])
    
    fig.update_layout(
        title={
            'text': 'Optimal Portfolio Composition',
            'x': 0.5,
            'font': {'size': 16}
        },
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig


def create_risk_return_scatter(optimizer, weights=None):
    """Create risk-return scatter plot for individual assets with better positioning."""
    fig = go.Figure()
    
    # Define colors and positions for better visualization
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, ticker in enumerate(optimizer.tickers):
        asset_return = optimizer.mean_returns[ticker] * 100
        asset_vol = np.sqrt(optimizer.cov_matrix.iloc[i, i]) * 100
        weight = weights[i] * 100 if weights is not None else 0
        
        # Calculate marker size based on weight
        marker_size = max(15, weight * 2) if weights is not None else 20
        color = colors[i % len(colors)]
        
        # Create clean hover text
        hover_text = (
            f"<b>{ticker}</b><br>" +
            f"Return: {asset_return:.1f}%<br>" +
            f"Risk: {asset_vol:.1f}%<br>" +
            (f"Weight: {weight:.1f}%<br>" if weights is not None else "") +
            "<extra></extra>"
        )
        
        fig.add_trace(go.Scatter(
            x=[asset_vol],
            y=[asset_return],
            mode='markers+text',
            marker=dict(
                size=marker_size,
                color=color,
                line=dict(width=2, color='black'),
                opacity=0.8
            ),
            text=[ticker],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            name=ticker,
            hovertemplate=hover_text
        ))
    
    fig.update_layout(
        title={
            'text': 'Asset Risk-Return Profile',
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Expected Return %',
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def create_correlation_heatmap(optimizer):
    """Create correlation matrix heatmap."""
    if optimizer.returns is None:
        return None
    
    corr_matrix = optimizer.returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        height=400
    )
    
    return fig


def create_performance_chart(optimizer, weights):
    """Create historical performance chart with improved formatting."""
    if optimizer.data is None or weights is None:
        return None
    
    # Calculate portfolio performance
    portfolio_returns = (optimizer.returns * weights).sum(axis=1)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # Calculate individual asset performance
    individual_cumulative = (1 + optimizer.returns).cumprod()
    
    fig = go.Figure()
    
    # Add portfolio performance
    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative.values,
        mode='lines',
        name='Optimized Portfolio',
        line=dict(color='blue', width=3),
        hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Add individual assets with different colors
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
    
    for i, ticker in enumerate(optimizer.tickers):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=individual_cumulative.index,
            y=individual_cumulative[ticker].values,
            mode='lines',
            name=ticker,
            line=dict(color=color, width=1),
            opacity=0.7,
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Historical Performance Comparison',
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


# Legacy function for backward compatibility
def optimize_portfolio(tickers, expected_return=None, expected_std=None, 
                      include_risk_free=False, use_sp500=False):
    """Legacy function for backward compatibility."""
    try:
        optimizer = PortfolioOptimizer(tickers)
        success, error = optimizer.fetch_data()
        
        if not success:
            raise ValueError(f"Data fetch failed: {error}")
        
        if include_risk_free:
            optimizer.get_risk_free_rate()
        
        # Determine optimization method
        if expected_return is not None:
            result = optimizer.optimize_portfolio('target_return', target_return=expected_return)
        elif expected_std is not None:
            result = optimizer.optimize_portfolio('target_volatility', target_volatility=expected_std)
        else:
            result = optimizer.optimize_portfolio('max_sharpe')
        
        if not result['success']:
            raise ValueError(result['error'])
        
        weights = result['weights']
        
        # CAPM analysis
        capm_metrics = optimizer.calculate_capm_metrics() if use_sp500 else {}
        
        # Create visualization
        fig = create_efficient_frontier_plot(optimizer)
        
        return weights, capm_metrics, {}, {}, None, None, None, fig
        
    except Exception as e:
        # Return default values on error
        n = len(tickers)
        return np.array([1/n] * n), {}, {}, {}, None, None, None, None