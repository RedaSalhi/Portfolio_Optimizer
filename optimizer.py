# optimizer.py - Enhanced Interactive Portfolio Optimization Engine

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# ENHANCED PORTFOLIO OPTIMIZATION ENGINE
# ==============================================================================

def optimize_portfolio_advanced(tickers, expected_return=None, expected_std=None, 
                               include_risk_free=False, use_sp500=False, 
                               optimization_method='max_sharpe', constraints=None,
                               num_portfolios=25000):
    """
    Advanced portfolio optimization with multiple methods and interactive analytics.
    
    Parameters:
        tickers (list): Asset tickers
        expected_return (float): Target return constraint
        expected_std (float): Target volatility constraint  
        include_risk_free (bool): Include risk-free asset
        use_sp500 (bool): Use S&P 500 as market proxy
        optimization_method (str): 'max_sharpe', 'min_variance', 'max_return', 'efficient_frontier'
        constraints (dict): Additional constraints for optimization
        num_portfolios (int): Number of random portfolios for efficient frontier
    
    Returns:
        dict: Comprehensive optimization results with interactive data
    """
    end = datetime.today().date()
    start = end - timedelta(days=5 * 365)
    
    if constraints is None:
        constraints = {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1  # weights sum to 1
        }
    
    # Enhanced data fetching with error handling
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if data.empty:
            raise ValueError("No data retrieved for the given tickers")
        
        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        data = data.dropna()
        if len(data) < 252:  # Need at least 1 year of data
            st.warning(f"Limited data available: {len(data)} days. Results may be less reliable.")
        
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")
    
    # Calculate returns and risk metrics
    returns = data.pct_change(fill_method=None).dropna()
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252     # Annualized
    
    # Check if covariance matrix is positive definite
    if not np.all(np.linalg.eigvals(cov_matrix) > 0):
        # Add small diagonal element to ensure positive definiteness
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
    
    # Risk-free rate
    rf_rate = get_risk_free_rate() if include_risk_free else 0.0
    
    num_assets = len(tickers)
    
    try:
        # Enhanced optimization results
        optimization_results = {
            'tickers': tickers,
            'returns_data': returns,
            'mean_returns': mean_returns,
            'cov_matrix': cov_matrix,
            'rf_rate': rf_rate,
            'num_assets': num_assets,
            'data_points': len(data),
            'start_date': start,
            'end_date': end
        }
        
        # Generate efficient frontier
        efficient_frontier_data = generate_efficient_frontier(
            mean_returns, cov_matrix, rf_rate, num_portfolios, constraints
        )
        optimization_results.update(efficient_frontier_data)
        
        # Initial guess for optimization
        initial_weights = np.array([1.0/num_assets] * num_assets)
        
        # Bounds for weights (0 to 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Optimize based on method
        if optimization_method == 'max_sharpe':
            optimal_weights = optimize_sharpe_ratio(mean_returns, cov_matrix, rf_rate, constraints, bounds, initial_weights)
        elif optimization_method == 'min_variance':
            optimal_weights = optimize_minimum_variance(mean_returns, cov_matrix, constraints, bounds, initial_weights)
        elif optimization_method == 'max_return':
            optimal_weights = optimize_maximum_return(mean_returns, cov_matrix, constraints, bounds, initial_weights)
        elif optimization_method == 'target_return':
            if expected_return is None:
                expected_return = mean_returns.mean()
            optimal_weights = optimize_target_return(mean_returns, cov_matrix, expected_return, constraints, bounds, initial_weights)
        elif optimization_method == 'target_volatility':
            if expected_std is None:
                expected_std = np.sqrt(np.diag(cov_matrix)).mean()
            optimal_weights = optimize_target_volatility(mean_returns, cov_matrix, expected_std, constraints, bounds, initial_weights)
        else:
            optimal_weights = optimize_sharpe_ratio(mean_returns, cov_matrix, rf_rate, constraints, bounds, initial_weights)
        
        # Validate optimization results
        if optimal_weights is None or not np.any(optimal_weights):
            raise ValueError("Optimization failed to find valid weights")
            
        # Normalize weights to ensure they sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        optimization_results['optimal_weights'] = optimal_weights
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(
            optimal_weights, mean_returns, cov_matrix, rf_rate
        )
        optimization_results.update(portfolio_metrics)
        
        # CAPM Analysis
        if use_sp500:
            capm_results = perform_capm_analysis(returns, optimal_weights, rf_rate)
            optimization_results.update(capm_results)
        
        # Risk attribution analysis
        risk_attribution = calculate_risk_attribution(optimal_weights, cov_matrix, mean_returns)
        optimization_results.update(risk_attribution)
        
        # Performance attribution
        performance_attribution = calculate_performance_attribution(
            returns, optimal_weights, mean_returns
        )
        optimization_results.update(performance_attribution)
        
        # Capital allocation line (if risk-free asset included)
        if include_risk_free and expected_return is not None:
            cal_results = calculate_capital_allocation(
                portfolio_metrics, rf_rate, expected_return, expected_std
            )
            optimization_results.update(cal_results)
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return None
    
    return optimization_results


def get_risk_free_rate():
    """Get current risk-free rate from 3-month Treasury."""
    try:
        irx = yf.download('^IRX', period="5d", interval="1d", progress=False)['Close'].dropna()
        if not irx.empty:
            return float(irx.mean()) / 100
    except:
        pass
    return 0.02  # Default 2% if fetch fails


def generate_efficient_frontier(mean_returns, cov_matrix, rf_rate, num_portfolios=25000, constraints=None):
    """Generate efficient frontier data with enhanced analytics."""
    num_assets = len(mean_returns)
    results = {
        'portfolio_returns': [],
        'portfolio_volatilities': [],
        'portfolio_sharpe_ratios': [],
        'portfolio_weights': [],
        'portfolio_betas': [],
        'portfolio_tracking_errors': []
    }
    
    # Generate random portfolios
    np.random.seed(42)  # For reproducibility
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Apply basic constraints if provided
        if constraints and 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            if np.any(weights > max_weight):
                continue
        
        if constraints and 'min_weight' in constraints:
            min_weight = constraints['min_weight']
            if np.any(weights < min_weight):
                continue
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
        else:
            continue
        
        # Calculate portfolio beta (vs equal-weighted market proxy)
        market_weights = np.ones(num_assets) / num_assets
        market_variance = np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
        portfolio_market_cov = np.dot(weights.T, np.dot(cov_matrix, market_weights))
        portfolio_beta = portfolio_market_cov / market_variance if market_variance > 0 else 1.0
        
        # Calculate tracking error vs market
        weight_diff = weights - market_weights
        tracking_error = np.sqrt(np.dot(weight_diff.T, np.dot(cov_matrix, weight_diff)))
        
        # Store results
        results['portfolio_returns'].append(portfolio_return)
        results['portfolio_volatilities'].append(portfolio_volatility)
        results['portfolio_sharpe_ratios'].append(sharpe_ratio)
        results['portfolio_weights'].append(weights.copy())
        results['portfolio_betas'].append(portfolio_beta)
        results['portfolio_tracking_errors'].append(tracking_error)
    
    # Convert to numpy arrays
    for key in ['portfolio_returns', 'portfolio_volatilities', 'portfolio_sharpe_ratios', 
                'portfolio_betas', 'portfolio_tracking_errors']:
        results[key] = np.array(results[key])
    
    # Find key portfolios
    if len(results['portfolio_sharpe_ratios']) > 0:
        max_sharpe_idx = np.argmax(results['portfolio_sharpe_ratios'])
        min_variance_idx = np.argmin(results['portfolio_volatilities'])
        max_return_idx = np.argmax(results['portfolio_returns'])
        
        results['max_sharpe_portfolio'] = {
            'weights': results['portfolio_weights'][max_sharpe_idx],
            'return': results['portfolio_returns'][max_sharpe_idx],
            'volatility': results['portfolio_volatilities'][max_sharpe_idx],
            'sharpe': results['portfolio_sharpe_ratios'][max_sharpe_idx]
        }
        
        results['min_variance_portfolio'] = {
            'weights': results['portfolio_weights'][min_variance_idx],
            'return': results['portfolio_returns'][min_variance_idx],
            'volatility': results['portfolio_volatilities'][min_variance_idx],
            'sharpe': results['portfolio_sharpe_ratios'][min_variance_idx]
        }
        
        results['max_return_portfolio'] = {
            'weights': results['portfolio_weights'][max_return_idx],
            'return': results['portfolio_returns'][max_return_idx],
            'volatility': results['portfolio_volatilities'][max_return_idx],
            'sharpe': results['portfolio_sharpe_ratios'][max_return_idx]
        }
    
    return results


def optimize_sharpe_ratio(mean_returns, cov_matrix, rf_rate, constraints, bounds, initial_weights):
    try:
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_std == 0:
                return 0
            return -(portfolio_return - rf_rate) / portfolio_std

        result = minimize(negative_sharpe_ratio, initial_weights,
                        constraints=constraints,
                        bounds=bounds,
                        method='SLSQP')
        
        if not result.success:
            st.warning(f"Optimization warning: {result.message}")
        
        return result.x
    except Exception as e:
        st.error(f"Error in Sharpe optimization: {str(e)}")
        return initial_weights


def optimize_minimum_variance(mean_returns, cov_matrix, constraints, bounds, initial_weights):
    try:
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        result = minimize(portfolio_volatility, initial_weights,
                        constraints=constraints,
                        bounds=bounds,
                        method='SLSQP')
        
        if not result.success:
            st.warning(f"Optimization warning: {result.message}")
        
        return result.x
    except Exception as e:
        st.error(f"Error in minimum variance optimization: {str(e)}")
        return initial_weights


def optimize_maximum_return(mean_returns, cov_matrix, constraints, bounds, initial_weights):
    try:
        def negative_return(weights):
            return -np.sum(mean_returns * weights)

        result = minimize(negative_return, initial_weights,
                        constraints=constraints,
                        bounds=bounds,
                        method='SLSQP')
        
        if not result.success:
            st.warning(f"Optimization warning: {result.message}")
        
        return result.x
    except Exception as e:
        st.error(f"Error in maximum return optimization: {str(e)}")
        return initial_weights


def optimize_target_return(mean_returns, cov_matrix, target_return, constraints, bounds, initial_weights):
    try:
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Add target return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda x: np.sum(mean_returns * x) - target_return
        }
        
        constraints_list = []
        if isinstance(constraints, dict):
            constraints_list = [constraints]
        elif isinstance(constraints, (list, tuple)):
            constraints_list = list(constraints)
        constraints_list.append(return_constraint)

        result = minimize(portfolio_volatility, initial_weights,
                        constraints=constraints_list,
                        bounds=bounds,
                        method='SLSQP')
        
        if not result.success:
            st.warning(f"Optimization warning: {result.message}")
        
        return result.x
    except Exception as e:
        st.error(f"Error in target return optimization: {str(e)}")
        return initial_weights


def optimize_target_volatility(mean_returns, cov_matrix, target_vol, constraints, bounds, initial_weights):
    try:
        def negative_return(weights):
            return -np.sum(mean_returns * weights)
        
        def volatility_constraint(weights):
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_std - target_vol
        
        # Add volatility constraint
        vol_constraint = {
            'type': 'eq',
            'fun': volatility_constraint
        }
        
        constraints_list = []
        if isinstance(constraints, dict):
            constraints_list = [constraints]
        elif isinstance(constraints, (list, tuple)):
            constraints_list = list(constraints)
        constraints_list.append(vol_constraint)

        result = minimize(negative_return, initial_weights,
                        constraints=constraints_list,
                        bounds=bounds,
                        method='SLSQP')
        
        if not result.success:
            st.warning(f"Optimization warning: {result.message}")
        
        return result.x
    except Exception as e:
        st.error(f"Error in target volatility optimization: {str(e)}")
        return initial_weights


def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, rf_rate):
    """Calculate comprehensive portfolio metrics."""
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    if portfolio_volatility > 0:
        sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
    else:
        sharpe_ratio = 0
    
    # Additional risk metrics
    var_95 = stats.norm.ppf(0.05, portfolio_return/252, portfolio_volatility/np.sqrt(252)) * 252
    cvar_95 = portfolio_return - portfolio_volatility * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
    
    # Diversification metrics
    naive_risk = np.sqrt(np.mean(np.diag(cov_matrix)))
    diversification_ratio = (np.dot(weights, np.sqrt(np.diag(cov_matrix)))) / portfolio_volatility
    
    return {
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'diversification_ratio': diversification_ratio
    }


def perform_capm_analysis(returns, weights, rf_rate):
    """Enhanced CAPM analysis with factor decomposition."""
    try:
        # Download S&P 500 data for market proxy
        end = datetime.today().date()
        start = end - timedelta(days=5 * 365)
        sp500_data = yf.download('^GSPC', start=start, end=end, progress=False)['Close']
        sp500_returns = sp500_data.pct_change().dropna()
        
        # Align dates
        common_dates = returns.index.intersection(sp500_returns.index)
        returns_aligned = returns.loc[common_dates]
        sp500_aligned = sp500_returns.loc[common_dates]
        
        if len(common_dates) == 0:
            return {'capm_error': 'No overlapping dates with market data'}
        
        # Calculate portfolio returns
        portfolio_returns = returns_aligned @ weights
        
        # CAPM regression for portfolio
        excess_portfolio = portfolio_returns - rf_rate/252
        excess_market = sp500_aligned - rf_rate/252
        
        # Portfolio beta and alpha
        X = sm.add_constant(excess_market)
        model = sm.OLS(excess_portfolio, X).fit()
        portfolio_alpha = model.params['const'] * 252  # Annualized
        portfolio_beta = model.params[sp500_aligned.name] if sp500_aligned.name in model.params else model.params.iloc[1]
        portfolio_r_squared = model.rsquared
        
        # Individual asset CAPM metrics
        capm_metrics = {}
        for i, ticker in enumerate(returns_aligned.columns):
            try:
                asset_excess = returns_aligned[ticker] - rf_rate/252
                asset_model = sm.OLS(asset_excess, X).fit()
                
                capm_metrics[ticker] = {
                    'alpha': asset_model.params['const'] * 252,
                    'beta': asset_model.params.iloc[1],
                    'r_squared': asset_model.rsquared,
                    'expected_return': rf_rate + asset_model.params.iloc[1] * (sp500_aligned.mean() * 252 - rf_rate)
                }
            except:
                capm_metrics[ticker] = {
                    'alpha': 0,
                    'beta': 1,
                    'r_squared': 0,
                    'expected_return': returns_aligned[ticker].mean() * 252
                }
        
        return {
            'portfolio_alpha': portfolio_alpha,
            'portfolio_beta': portfolio_beta,
            'portfolio_r_squared': portfolio_r_squared,
            'asset_capm_metrics': capm_metrics,
            'market_return': sp500_aligned.mean() * 252
        }
    
    except Exception as e:
        return {'capm_error': str(e)}


def calculate_risk_attribution(weights, cov_matrix, mean_returns):
    """Calculate risk attribution and decomposition."""
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal contribution to risk
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
    
    # Component contribution to risk
    component_contrib = weights * marginal_contrib
    
    # Percentage contribution to risk
    percent_contrib = component_contrib / portfolio_volatility
    
    return {
        'marginal_risk_contrib': marginal_contrib,
        'component_risk_contrib': component_contrib,
        'percent_risk_contrib': percent_contrib,
        'risk_concentration': np.sum(percent_contrib**2)  # Herfindahl index
    }


def calculate_performance_attribution(returns, weights, mean_returns):
    """Calculate performance attribution analysis."""
    portfolio_returns = returns @ weights
    
    # Performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Rolling metrics
    rolling_sharpe = portfolio_returns.rolling(window=252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    
    rolling_volatility = portfolio_returns.rolling(window=60).std() * np.sqrt(252)
    
    # Drawdown analysis
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'max_drawdown': max_drawdown,
        'rolling_sharpe': rolling_sharpe,
        'rolling_volatility': rolling_volatility,
        'portfolio_returns_series': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'drawdown_series': drawdown
    }


def calculate_capital_allocation(portfolio_metrics, rf_rate, target_return=None, target_volatility=None):
    """Calculate capital allocation line parameters."""
    portfolio_return = portfolio_metrics['portfolio_return']
    portfolio_volatility = portfolio_metrics['portfolio_volatility']
    
    if target_return is not None:
        # Calculate weight in risky portfolio
        w_risky = (target_return - rf_rate) / (portfolio_return - rf_rate)
        target_vol = w_risky * portfolio_volatility
        
        return {
            'risky_weight': w_risky,
            'risk_free_weight': 1 - w_risky,
            'target_return': target_return,
            'target_volatility': target_vol
        }
    
    elif target_volatility is not None:
        # Calculate weight in risky portfolio
        w_risky = target_volatility / portfolio_volatility
        target_ret = rf_rate + w_risky * (portfolio_return - rf_rate)
        
        return {
            'risky_weight': w_risky,
            'risk_free_weight': 1 - w_risky,
            'target_return': target_ret,
            'target_volatility': target_volatility
        }
    
    return {}


# ==============================================================================
# INTERACTIVE VISUALIZATION FUNCTIONS
# ==============================================================================

def create_interactive_efficient_frontier(optimization_results, highlight_portfolios=True):
    """Create interactive efficient frontier with multiple overlays."""
    fig = go.Figure()
    
    # Main efficient frontier scatter
    fig.add_trace(go.Scatter(
        x=optimization_results['portfolio_volatilities'] * 100,
        y=optimization_results['portfolio_returns'] * 100,
        mode='markers',
        marker=dict(
            size=4,
            color=optimization_results['portfolio_sharpe_ratios'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            opacity=0.6
        ),
        text=[f"Return: {r:.1%}<br>Risk: {v:.1%}<br>Sharpe: {s:.2f}" 
              for r, v, s in zip(optimization_results['portfolio_returns'],
                                optimization_results['portfolio_volatilities'],
                                optimization_results['portfolio_sharpe_ratios'])],
        hovertemplate='<b>Portfolio</b><br>%{text}<extra></extra>',
        name='Efficient Frontier'
    ))
    
    if highlight_portfolios:
        # Highlight special portfolios
        special_portfolios = [
            ('Max Sharpe', optimization_results.get('max_sharpe_portfolio'), 'red', '★'),
            ('Min Variance', optimization_results.get('min_variance_portfolio'), 'blue', '◆'),
            ('Max Return', optimization_results.get('max_return_portfolio'), 'green', '▲')
        ]
        
        for name, portfolio, color, symbol in special_portfolios:
            if portfolio:
                fig.add_trace(go.Scatter(
                    x=[portfolio['volatility'] * 100],
                    y=[portfolio['return'] * 100],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol=symbol),
                    name=name,
                    hovertemplate=f'<b>{name} Portfolio</b><br>' +
                                  f'Return: {portfolio["return"]:.1%}<br>' +
                                  f'Risk: {portfolio["volatility"]:.1%}<br>' +
                                  f'Sharpe: {portfolio["sharpe"]:.2f}<extra></extra>'
                ))
    
    # Capital allocation line (if risk-free rate available)
    if optimization_results.get('rf_rate', 0) > 0:
        max_sharpe = optimization_results.get('max_sharpe_portfolio')
        if max_sharpe:
            # Draw CAL from risk-free rate to tangency portfolio
            rf_rate = optimization_results['rf_rate']
            tangency_vol = max_sharpe['volatility']
            tangency_ret = max_sharpe['return']
            
            cal_x = np.linspace(0, tangency_vol * 1.5, 100)
            cal_y = rf_rate + (tangency_ret - rf_rate) / tangency_vol * cal_x
            
            fig.add_trace(go.Scatter(
                x=cal_x * 100,
                y=cal_y * 100,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Capital Allocation Line',
                hovertemplate='<b>CAL</b><br>Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title='Interactive Efficient Frontier',
        xaxis_title='Risk (Volatility) %',
        yaxis_title='Expected Return %',
        height=600,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_3d_risk_return_time_surface(optimization_results):
    """Create 3D surface showing risk-return relationship over time."""
    try:
        returns_data = optimization_results['returns_data']
        weights = optimization_results['optimal_weights']
        
        # Calculate rolling portfolio metrics
        window = 60  # 60-day rolling window
        portfolio_returns = returns_data @ weights
        
        dates = []
        rolling_returns = []
        rolling_volatilities = []
        
        for i in range(window, len(portfolio_returns)):
            end_date = portfolio_returns.index[i]
            period_returns = portfolio_returns.iloc[i-window:i]
            
            dates.append(end_date)
            rolling_returns.append(period_returns.mean() * 252)  # Annualized
            rolling_volatilities.append(period_returns.std() * np.sqrt(252))  # Annualized
        
        # Create 3D surface
        fig = go.Figure(data=[go.Scatter3d(
            x=rolling_volatilities,
            y=rolling_returns,
            z=list(range(len(dates))),
            mode='markers+lines',
            marker=dict(
                size=3,
                color=rolling_returns,
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Return")
            ),
            line=dict(color='blue', width=2),
            text=[f"Date: {d.strftime('%Y-%m-%d')}<br>Return: {r:.1%}<br>Risk: {v:.1%}" 
                  for d, r, v in zip(dates, rolling_returns, rolling_volatilities)],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Risk-Return Evolution Over Time',
            scene=dict(
                xaxis_title='Volatility',
                yaxis_title='Return', 
                zaxis_title='Time Period',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create 3D visualization: {str(e)}")
        return None


def create_portfolio_composition_sunburst(tickers, weights):
    """Create interactive sunburst chart for portfolio composition."""
    # Create hierarchical data for sunburst
    # Simple version: just show weights
    fig = go.Figure(go.Sunburst(
        labels=tickers + ['Portfolio'],
        parents=['Portfolio'] * len(tickers) + [''],
        values=list(weights) + [1.0],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>',
        maxdepth=2,
    ))
    
    fig.update_layout(
        title="Portfolio Composition",
        height=500
    )
    
    return fig


def create_risk_attribution_dashboard(optimization_results):
    """Create comprehensive risk attribution dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Risk Contribution by Asset',
            'Return vs Risk Contribution', 
            'Correlation Heatmap',
            'Diversification Metrics'
        ],
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "indicator"}]
        ]
    )
    
    tickers = optimization_results['tickers']
    weights = optimization_results['optimal_weights']
    risk_contrib = optimization_results.get('percent_risk_contrib', weights)
    mean_returns = optimization_results['mean_returns']
    cov_matrix = optimization_results['cov_matrix']
    
    # 1. Risk contribution bar chart
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=risk_contrib * 100,
            name='Risk Contribution %',
            marker_color='lightcoral'
        ),
        row=1, col=1
    )
    
    # 2. Return vs Risk scatter
    fig.add_trace(
        go.Scatter(
            x=risk_contrib * 100,
            y=mean_returns * 100,
            mode='markers+text',
            text=tickers,
            textposition="top center",
            marker=dict(size=weights * 1000, color='lightblue'),
            name='Assets'
        ),
        row=1, col=2
    )
    
    # 3. Correlation heatmap
    corr_matrix = cov_matrix / np.outer(np.sqrt(np.diag(cov_matrix)), np.sqrt(np.diag(cov_matrix)))
    
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix,
            x=tickers,
            y=tickers,
            colorscale='RdBu',
            zmid=0,
            showscale=False
        ),
        row=2, col=1
    )
    
    # 4. Diversification indicator
    diversification_ratio = optimization_results.get('diversification_ratio', 1.0)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=diversification_ratio,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diversification Ratio"},
            gauge={
                'axis': {'range': [None, 2]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 1.5], 'color': "yellow"},
                    {'range': [1.5, 2], 'color': "green"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Portfolio Risk Attribution Dashboard",
        height=700,
        showlegend=False
    )
    
    return fig


def create_performance_attribution_dashboard(optimization_results):
    """Create performance attribution dashboard with time series analysis."""
    if 'portfolio_returns_series' not in optimization_results:
        return None
    
    portfolio_returns = optimization_results['portfolio_returns_series']
    cumulative_returns = optimization_results['cumulative_returns']
    drawdown_series = optimization_results['drawdown_series']
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Cumulative Portfolio Performance',
            'Rolling Sharpe Ratio',
            'Drawdown Analysis'
        ],
        vertical_spacing=0.08
    )
    
    # 1. Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=(cumulative_returns - 1) * 100,
            mode='lines',
            name='Cumulative Return %',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 2. Rolling Sharpe ratio
    rolling_sharpe = optimization_results.get('rolling_sharpe')
    if rolling_sharpe is not None:
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # 3. Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown %',
            line=dict(color='red', width=2),
            fillcolor='rgba(255,0,0,0.3)'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title_text="Portfolio Performance Attribution",
        height=800,
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    
    return fig


# ==============================================================================
# LEGACY FUNCTION (Maintained for backward compatibility)
# ==============================================================================

def optimize_portfolio(tickers, expected_return=None, expected_std=None, include_risk_free=False, use_sp500=False):
    """
    Legacy function maintained for backward compatibility.
    Calls the enhanced version with default parameters.
    """
    try:
        results = optimize_portfolio_advanced(
            tickers=tickers,
            expected_return=expected_return,
            expected_std=expected_std,
            include_risk_free=include_risk_free,
            use_sp500=use_sp500,
            optimization_method='max_sharpe'
        )
        
        # Extract legacy format results
        weights = results['optimal_weights']
        capm = results.get('asset_capm_metrics', {})
        betas = {ticker: capm.get(ticker, {}).get('beta', 1.0) for ticker in tickers}
        alphas = {ticker: capm.get(ticker, {}).get('alpha', 0.0) for ticker in tickers}
        
        # Create matplotlib figure for backward compatibility
        fig = create_matplotlib_efficient_frontier(results)
        
        # Capital allocation results
        w = results.get('risky_weight')
        R_target = results.get('target_return')
        sigma_target = results.get('target_volatility')
        
        return weights, capm, betas, alphas, w, R_target, sigma_target, fig
        
    except Exception as e:
        # Fallback to simple equal weights if optimization fails
        n = len(tickers)
        weights = np.array([1/n] * n)
        capm = {ticker: {'expected_return': 0.1, 'beta': 1.0, 'alpha': 0.0} for ticker in tickers}
        betas = {ticker: 1.0 for ticker in tickers}
        alphas = {ticker: 0.0 for ticker in tickers}
        
        # Simple matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Optimization failed: {str(e)}', ha='center', va='center')
        ax.set_title('Portfolio Optimization Error')
        
        return weights, capm, betas, alphas, None, None, None, fig


def create_matplotlib_efficient_frontier(results):
    """Create matplotlib version for legacy compatibility."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot efficient frontier
    scatter = ax.scatter(
        results['portfolio_volatilities'],
        results['portfolio_returns'],
        c=results['portfolio_sharpe_ratios'],
        cmap='viridis',
        alpha=0.6
    )
    
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Highlight optimal portfolio
    max_sharpe = results.get('max_sharpe_portfolio')
    if max_sharpe:
        ax.scatter(
            max_sharpe['volatility'],
            max_sharpe['return'],
            color='red',
            s=100,
            marker='*',
            label='Max Sharpe Portfolio'
        )
    
    # Add capital allocation line if applicable
    if results.get('rf_rate', 0) > 0 and max_sharpe:
        rf_rate = results['rf_rate']
        x = np.linspace(0, max_sharpe['volatility'] * 1.5, 100)
        y = rf_rate + (max_sharpe['return'] - rf_rate) / max_sharpe['volatility'] * x
        ax.plot(x, y, 'r--', label='Capital Allocation Line')
    
    ax.set_xlabel('Volatility (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    ax.grid(True)
    
    return fig