# optimizer.py - Advanced Portfolio Optimization Engine

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import streamlit as st
import warnings
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine using Modern Portfolio Theory.
    
    Features:
    - Multiple optimization objectives (Sharpe, variance, target-based)
    - Robust data fetching and validation
    - CAPM analysis with market benchmarking
    - Risk decomposition and attribution
    - Interactive visualizations
    - Performance analytics
    """
    
    def __init__(self, tickers: List[str], lookback_years: int = 3):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
            tickers: List of asset ticker symbols
            lookback_years: Years of historical data for analysis
        """
        self.tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        self.lookback_years = lookback_years
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.rf_rate = 0.02  # Default risk-free rate
        self.market_data = None
        self.market_returns = None
        
    def validate_tickers(self) -> Tuple[List[str], List[str]]:
        """Validate ticker symbols and return valid/invalid lists."""
        # For now, assume all tickers are valid and let fetch_data handle validation
        # This prevents the overly strict validation that was causing issues
        return self.tickers, []
    
    def fetch_data(self) -> Tuple[bool, Optional[str]]:
        """
        Fetch and process market data with comprehensive error handling.
        
        Returns:
            Tuple of (success_flag, error_message)
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)
            
            # Download data for all tickers
            data_dict = {}
            failed_tickers = []
            
            for ticker in self.tickers:
                try:
                    # Use more robust download parameters
                    ticker_data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        show_errors=False,
                        threads=False,
                        group_by='ticker'
                    )
                    
                    if ticker_data.empty:
                        failed_tickers.append(ticker)
                        continue
                    
                    # Handle different data structures more robustly
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        # Multi-index columns
                        price_data = ticker_data.get('Adj Close', ticker_data.get('Close', ticker_data.iloc[:, -1]))
                        if hasattr(price_data, 'iloc'):
                            price_data = price_data.iloc[:, 0] if len(price_data.shape) > 1 else price_data
                    else:
                        # Single level columns
                        price_data = ticker_data.get('Adj Close', ticker_data.get('Close', ticker_data.iloc[:, -1]))
                    
                    # Clean and validate data
                    price_data = pd.Series(price_data).dropna()
                    
                    # More lenient data requirement
                    if len(price_data) < 30:  # Very minimal requirement
                        failed_tickers.append(ticker)
                        continue
                    
                    data_dict[ticker] = price_data
                    
                except Exception as e:
                    failed_tickers.append(ticker)
                    continue
            
            if not data_dict:
                return False, "Failed to fetch data for any ticker. Please check ticker symbols and internet connection."
            
            # Create price DataFrame
            self.data = pd.DataFrame(data_dict)
            
            # Handle missing data more gently
            self.data = self.data.dropna(how='all')
            
            # Forward fill missing values
            self.data = self.data.fillna(method='ffill')
            self.data = self.data.fillna(method='bfill')
            self.data = self.data.dropna(how='all')
            
            if len(self.data) < 20:  # Very minimal requirement
                return False, "Insufficient data after cleaning"
            
            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            
            if self.returns.empty or len(self.returns) < 10:
                return False, "Could not calculate sufficient returns"
            
            # Calculate annualized statistics
            self.mean_returns = self.returns.mean() * 252
            self.cov_matrix = self.returns.cov() * 252
            
            # Ensure positive definite covariance matrix
            try:
                eigenvals = np.linalg.eigvals(self.cov_matrix)
                if np.any(eigenvals <= 0):
                    self.cov_matrix += np.eye(len(self.cov_matrix)) * 1e-6
            except:
                # If eigenvalue calculation fails, add regularization
                self.cov_matrix += np.eye(len(self.cov_matrix)) * 1e-6
            
            # Update tickers to successful ones only
            self.tickers = list(self.data.columns)
            
            # Fetch market data for CAPM analysis
            try:
                self._fetch_market_data(start_date, end_date)
            except:
                pass  # CAPM analysis will be skipped if market data unavailable
            
            error_msg = None
            if failed_tickers:
                error_msg = f"Could not fetch: {', '.join(failed_tickers)}"
            
            return True, error_msg
            
        except Exception as e:
            return False, f"Data fetching error: {str(e)}"
    
    def _fetch_market_data(self, start_date: datetime, end_date: datetime):
        """Fetch market benchmark data for CAPM analysis."""
        try:
            # Try S&P 500 first, then alternatives
            market_symbols = ['^GSPC', 'SPY', '^VTI']
            
            for symbol in market_symbols:
                try:
                    market_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not market_data.empty:
                        if 'Adj Close' in market_data.columns:
                            self.market_data = market_data['Adj Close']
                        else:
                            self.market_data = market_data['Close']
                        
                        self.market_returns = self.market_data.pct_change().dropna()
                        break
                except:
                    continue
        except:
            pass  # CAPM analysis will be skipped if market data unavailable
    
    def get_risk_free_rate(self) -> float:
        """Fetch current risk-free rate from Treasury data."""
        try:
            # Try different Treasury instruments
            treasury_symbols = ['^IRX', '^TNX', 'DGS3MO', 'DGS10']
            
            for symbol in treasury_symbols:
                try:
                    treasury_data = yf.download(symbol, period='5d', progress=False)
                    if not treasury_data.empty:
                        latest_rate = treasury_data['Close'].iloc[-1]
                        if 0 < latest_rate < 100:  # Reasonable rate check
                            self.rf_rate = latest_rate / 100
                            return self.rf_rate
                except:
                    continue
            
            # Fallback to default
            self.rf_rate = 0.02
            return self.rf_rate
            
        except:
            self.rf_rate = 0.02
            return self.rf_rate
    
    def portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        weights = np.array(weights)
        
        # Basic metrics
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.rf_rate) / portfolio_std if portfolio_std > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile((self.returns * weights).sum(axis=1), 5)
        var_99 = np.percentile((self.returns * weights).sum(axis=1), 1)
        
        # Diversification metrics
        concentration = np.sum(weights ** 2)  # Herfindahl index
        diversification_ratio = 1 / concentration
        
        # Maximum drawdown
        portfolio_cumret = ((self.returns * weights).sum(axis=1) + 1).cumprod()
        running_max = portfolio_cumret.expanding().max()
        drawdown = (portfolio_cumret - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'concentration': concentration,
            'diversification_ratio': diversification_ratio,
            'max_drawdown': max_drawdown
        }
    
    def objective_sharpe(self, weights: np.ndarray) -> float:
        """Objective function for maximum Sharpe ratio optimization."""
        metrics = self.portfolio_metrics(weights)
        return -metrics['sharpe_ratio']
    
    def objective_variance(self, weights: np.ndarray) -> float:
        """Objective function for minimum variance optimization."""
        return self.portfolio_metrics(weights)['variance']
    
    def constraint_return(self, weights: np.ndarray, target_return: float) -> float:
        """Constraint function for target return."""
        return self.portfolio_metrics(weights)['expected_return'] - target_return
    
    def constraint_volatility(self, weights: np.ndarray, target_vol: float) -> float:
        """Constraint function for target volatility."""
        return self.portfolio_metrics(weights)['volatility'] - target_vol
    
    def optimize_portfolio(self, 
                         method: str = 'max_sharpe',
                         target_return: Optional[float] = None,
                         target_volatility: Optional[float] = None,
                         min_weight: float = 0.0,
                         max_weight: float = 1.0) -> Dict:
        """
        Optimize portfolio using specified method with enhanced constraints.
        
        Parameters:
            method: Optimization method ('max_sharpe', 'min_variance', 'target_return', 'target_volatility')
            target_return: Target return for constrained optimization
            target_volatility: Target volatility for constrained optimization
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        
        Returns:
            Dictionary containing optimization results
        """
        if self.returns is None:
            return {'success': False, 'error': 'No data available'}
        
        num_assets = len(self.tickers)
        
        # Initial guess - equal weights
        initial_weights = np.array([1.0/num_assets] * num_assets)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        try:
            if method == 'max_sharpe':
                result = minimize(
                    self.objective_sharpe,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
            
            elif method == 'min_variance':
                result = minimize(
                    self.objective_variance,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
            
            elif method == 'target_return' and target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: self.constraint_return(x, target_return)
                })
                result = minimize(
                    self.objective_variance,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
            
            elif method == 'target_volatility' and target_volatility is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: self.constraint_volatility(x, target_volatility)
                })
                result = minimize(
                    lambda x: -self.portfolio_metrics(x)['expected_return'],
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
            
            else:
                return {'success': False, 'error': f'Invalid optimization method: {method}'}
            
            if not result.success:
                return {'success': False, 'error': f'Optimization failed: {result.message}'}
            
            # Calculate final metrics
            optimal_weights = result.x
            metrics = self.portfolio_metrics(optimal_weights)
            
            # Risk attribution
            marginal_contrib = np.dot(self.cov_matrix, optimal_weights) / metrics['volatility']
            component_contrib = optimal_weights * marginal_contrib
            
            return {
                'success': True,
                'method': method,
                'weights': optimal_weights,
                'expected_return': metrics['expected_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'var_95': metrics['var_95'],
                'var_99': metrics['var_99'],
                'max_drawdown': metrics['max_drawdown'],
                'diversification_ratio': metrics['diversification_ratio'],
                'risk_contribution': component_contrib,
                'marginal_contribution': marginal_contrib
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Optimization error: {str(e)}'}
    
    def generate_efficient_frontier(self, num_points: int = 100) -> Optional[Dict]:
        """Generate efficient frontier data points."""
        if self.returns is None:
            return None
        
        # Define return range
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_points)
        
        frontier_data = {
            'returns': [],
            'volatilities': [],
            'sharpe_ratios': [],
            'weights': []
        }
        
        for target_ret in target_returns:
            result = self.optimize_portfolio('target_return', target_return=target_ret)
            if result['success']:
                frontier_data['returns'].append(result['expected_return'])
                frontier_data['volatilities'].append(result['volatility'])
                frontier_data['sharpe_ratios'].append(result['sharpe_ratio'])
                frontier_data['weights'].append(result['weights'])
        
        return frontier_data if frontier_data['returns'] else None
    
    def calculate_capm_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate CAPM metrics for each asset."""
        if self.market_returns is None or self.returns is None:
            return {}
        
        capm_metrics = {}
        
        # Align returns data
        common_dates = self.returns.index.intersection(self.market_returns.index)
        if len(common_dates) < 50:  # Minimum data requirement
            return {}
        
        aligned_returns = self.returns.loc[common_dates]
        aligned_market = self.market_returns.loc[common_dates]
        
        market_excess = aligned_market - self.rf_rate/252
        
        for ticker in self.tickers:
            try:
                asset_returns = aligned_returns[ticker]
                asset_excess = asset_returns - self.rf_rate/252
                
                # Calculate beta using regression
                covariance = np.cov(asset_excess, market_excess)[0, 1]
                market_variance = np.var(market_excess)
                beta = covariance / market_variance if market_variance > 0 else 1.0
                
                # Calculate alpha
                market_return = aligned_market.mean() * 252
                expected_return = self.rf_rate + beta * (market_return - self.rf_rate)
                actual_return = asset_returns.mean() * 252
                alpha = actual_return - expected_return
                
                # R-squared
                correlation = np.corrcoef(asset_excess, market_excess)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
                
                capm_metrics[ticker] = {
                    'beta': beta,
                    'alpha': alpha,
                    'expected_return': expected_return,
                    'r_squared': r_squared,
                    'systematic_risk': beta * np.std(market_excess) * np.sqrt(252),
                    'idiosyncratic_risk': np.sqrt(np.var(asset_excess) - (beta ** 2) * np.var(market_excess)) * np.sqrt(252)
                }
                
            except Exception:
                # Default values if calculation fails
                capm_metrics[ticker] = {
                    'beta': 1.0,
                    'alpha': 0.0,
                    'expected_return': self.mean_returns[ticker],
                    'r_squared': 0.5,
                    'systematic_risk': 0.0,
                    'idiosyncratic_risk': 0.0
                }
        
        return capm_metrics


# Advanced Visualization Functions

def create_efficient_frontier_plot(optimizer: PortfolioOptimizer, 
                                 optimal_portfolio: Optional[Dict] = None) -> Optional[go.Figure]:
    """Create interactive efficient frontier visualization."""
    frontier_data = optimizer.generate_efficient_frontier()
    if not frontier_data:
        return None
    
    fig = go.Figure()
    
    # Plot efficient frontier
    fig.add_trace(go.Scatter(
        x=np.array(frontier_data['volatilities']) * 100,
        y=np.array(frontier_data['returns']) * 100,
        mode='markers',
        marker=dict(
            size=8,
            color=frontier_data['sharpe_ratios'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio", x=1.02)
        ),
        name='Efficient Frontier',
        hovertemplate='<b>Efficient Portfolio</b><br>' +
                      'Return: %{y:.2f}%<br>' +
                      'Risk: %{x:.2f}%<br>' +
                      'Sharpe: %{marker.color:.3f}<extra></extra>'
    ))
    
    # Plot individual assets
    asset_colors = px.colors.qualitative.Set1
    for i, ticker in enumerate(optimizer.tickers):
        asset_return = optimizer.mean_returns[ticker] * 100
        asset_vol = np.sqrt(optimizer.cov_matrix.iloc[i, i]) * 100
        color = asset_colors[i % len(asset_colors)]
        
        fig.add_trace(go.Scatter(
            x=[asset_vol],
            y=[asset_return],
            mode='markers+text',
            marker=dict(size=15, color=color, symbol='star'),
            text=[ticker],
            textposition="top center",
            name=ticker,
            hovertemplate=f'<b>{ticker}</b><br>' +
                          'Return: %{y:.2f}%<br>' +
                          'Risk: %{x:.2f}%<extra></extra>'
        ))
    
    # Highlight optimal portfolio if provided
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['volatility'] * 100],
            y=[optimal_portfolio['expected_return'] * 100],
            mode='markers',
            marker=dict(size=20, color='red', symbol='diamond'),
            name='Optimal Portfolio',
            hovertemplate='<b>Optimal Portfolio</b><br>' +
                          'Return: %{y:.2f}%<br>' +
                          'Risk: %{x:.2f}%<br>' +
                          f'Sharpe: {optimal_portfolio["sharpe_ratio"]:.3f}<extra></extra>'
        ))
    
    # Add Capital Allocation Line
    if optimal_portfolio and optimizer.rf_rate > 0:
        tangency_vol = optimal_portfolio['volatility'] * 100
        tangency_ret = optimal_portfolio['expected_return'] * 100
        rf_ret = optimizer.rf_rate * 100
        
        cal_x = np.linspace(0, tangency_vol * 1.5, 50)
        cal_slope = (tangency_ret - rf_ret) / tangency_vol
        cal_y = rf_ret + cal_slope * cal_x
        
        fig.add_trace(go.Scatter(
            x=cal_x,
            y=cal_y,
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name='Capital Allocation Line',
            hovertemplate='<b>Capital Allocation Line</b><br>' +
                          'Return: %{y:.2f}%<br>' +
                          'Risk: %{x:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier Analysis',
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Expected Return %',
        height=600,
        hovermode='closest',
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def create_portfolio_composition_chart(tickers: List[str], weights: np.ndarray) -> go.Figure:
    """Create portfolio composition visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Portfolio Allocation', 'Asset Weights'),
        column_widths=[0.6, 0.4]
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=tickers,
            values=weights * 100,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=weights * 100,
            text=[f'{w:.1f}%' for w in weights * 100],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig


def create_risk_return_analysis(optimizer: PortfolioOptimizer, 
                               weights: Optional[np.ndarray] = None) -> go.Figure:
    """Create comprehensive risk-return analysis visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk-Return Scatter', 'Correlation Matrix', 
                       'Risk Decomposition', 'Asset Statistics'),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    # Risk-Return Scatter
    for i, ticker in enumerate(optimizer.tickers):
        ret = optimizer.mean_returns[ticker] * 100
        vol = np.sqrt(optimizer.cov_matrix.iloc[i, i]) * 100
        size = weights[i] * 100 if weights is not None else 10
        
        fig.add_trace(
            go.Scatter(
                x=[vol], y=[ret],
                mode='markers+text',
                marker=dict(size=max(15, size)),
                text=[ticker],
                textposition="middle center",
                name=ticker,
                hovertemplate=f'<b>{ticker}</b><br>Return: %{{y:.2f}}%<br>Risk: %{{x:.2f}}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Correlation Matrix
    corr_matrix = optimizer.returns.corr()
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
        row=1, col=2
    )
    
    # Risk Decomposition (if weights provided)
    if weights is not None:
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(optimizer.cov_matrix, weights)))
        marginal_contrib = np.dot(optimizer.cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        fig.add_trace(
            go.Bar(
                x=optimizer.tickers,
                y=risk_contrib * 100,
                name='Risk Contribution',
                hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Asset Statistics Table
    stats_data = []
    for i, ticker in enumerate(optimizer.tickers):
        ret = optimizer.mean_returns[ticker]
        vol = np.sqrt(optimizer.cov_matrix.iloc[i, i])
        sharpe = (ret - optimizer.rf_rate) / vol
        weight = weights[i] if weights is not None else 0
        
        stats_data.append([
            ticker,
            f'{ret:.2%}',
            f'{vol:.2%}',
            f'{sharpe:.3f}',
            f'{weight:.2%}' if weights is not None else 'N/A'
        ])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Asset', 'Return', 'Risk', 'Sharpe', 'Weight']),
            cells=dict(values=list(zip(*stats_data)))
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig


def create_performance_analytics(optimizer: PortfolioOptimizer, 
                                weights: np.ndarray) -> go.Figure:
    """Create performance analytics dashboard."""
    portfolio_returns = (optimizer.returns * weights).sum(axis=1)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # Calculate individual asset performance
    individual_cumulative = (1 + optimizer.returns).cumprod()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Performance', 'Rolling Volatility',
                       'Drawdown Analysis', 'Return Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cumulative Performance
    fig.add_trace(
        go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    for ticker in optimizer.tickers:
        fig.add_trace(
            go.Scatter(
                x=individual_cumulative.index,
                y=individual_cumulative[ticker].values,
                mode='lines',
                name=ticker,
                opacity=0.6,
                hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Rolling Volatility
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode='lines',
            name='30-Day Rolling Volatility',
            line=dict(color='red'),
            hovertemplate='<b>Rolling Volatility</b><br>Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Drawdown Analysis
    running_max = portfolio_cumulative.expanding().max()
    drawdown = (portfolio_cumulative - running_max) / running_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red'),
            hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Return Distribution
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=50,
            name='Daily Returns',
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    return fig


def create_capm_analysis_chart(capm_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create CAPM analysis visualization."""
    if not capm_metrics:
        return None
    
    tickers = list(capm_metrics.keys())
    betas = [capm_metrics[t]['beta'] for t in tickers]
    alphas = [capm_metrics[t]['alpha'] * 100 for t in tickers]  # Convert to percentage
    r_squared = [capm_metrics[t]['r_squared'] for t in tickers]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Beta Analysis', 'Alpha Analysis', 
                       'Systematic vs Idiosyncratic Risk', 'CAPM Summary'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "table"}]]
    )
    
    # Beta Analysis
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=betas,
            name='Beta',
            text=[f'{b:.2f}' for b in betas],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Alpha Analysis
    colors = ['green' if a > 0 else 'red' for a in alphas]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=alphas,
            name='Alpha',
            marker_color=colors,
            text=[f'{a:.2f}%' for a in alphas],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Risk Decomposition
    sys_risk = [capm_metrics[t]['systematic_risk'] * 100 for t in tickers]
    idio_risk = [capm_metrics[t]['idiosyncratic_risk'] * 100 for t in tickers]
    
    fig.add_trace(
        go.Scatter(
            x=sys_risk,
            y=idio_risk,
            mode='markers+text',
            text=tickers,
            textposition="top center",
            marker=dict(size=15, color=r_squared, colorscale='Viridis', showscale=True),
            name='Risk Decomposition',
            hovertemplate='<b>%{text}</b><br>' +
                          'Systematic Risk: %{x:.2f}%<br>' +
                          'Idiosyncratic Risk: %{y:.2f}%<br>' +
                          'R²: %{marker.color:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # CAPM Summary Table
    table_data = []
    for ticker in tickers:
        table_data.append([
            ticker,
            f'{capm_metrics[ticker]["beta"]:.3f}',
            f'{capm_metrics[ticker]["alpha"]:.2%}',
            f'{capm_metrics[ticker]["expected_return"]:.2%}',
            f'{capm_metrics[ticker]["r_squared"]:.3f}'
        ])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Asset', 'Beta', 'Alpha', 'Expected Return', 'R²']),
            cells=dict(values=list(zip(*table_data)))
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig