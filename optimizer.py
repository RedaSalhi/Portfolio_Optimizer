# optimizer.py - Enhanced Portfolio Optimization Engine
"""
Enhanced Portfolio Optimization Engine using Modern Portfolio Theory with CAPM analysis
and advanced visualization capabilities.
"""

import warnings
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Enhanced Portfolio Optimization Engine using Modern Portfolio Theory.
    
    Features:
    - Efficient frontier generation
    - Multiple optimization methods (max Sharpe, min variance, target return/volatility)
    - Risk-free asset integration with Capital Allocation Line
    - CAPM analysis and metrics
    - Comprehensive risk analytics
    - Advanced visualization
    """
    
    def __init__(self, tickers: List[str], lookback_years: int = 3):
        """Initialize the portfolio optimizer."""
        self.tickers = self._clean_tickers(tickers)
        self.lookback_years = max(1, min(lookback_years, 10))
        
        # Data storage
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.market_data = None
        self.market_returns = None
        
        # Risk-free rate and tracking
        self.rf_rate = 0.02
        self.failed_tickers = []
        self.data_quality_info = {}

    # ================== DATA CLEANING AND VALIDATION ==================
    
    def _clean_tickers(self, tickers: List[str]) -> List[str]:
        """Clean and validate ticker format."""
        cleaned = []
        for ticker in tickers:
            if ticker and isinstance(ticker, str):
                clean_ticker = re.sub(r'[^A-Za-z0-9\.\-\^]', '', ticker.strip().upper())
                if clean_ticker:
                    cleaned.append(clean_ticker)
        return list(set(cleaned))
    
    def validate_tickers(self) -> Tuple[List[str], List[str]]:
        """Validate ticker symbols with comprehensive checks."""
        valid_tickers = []
        invalid_tickers = []
        
        valid_patterns = [
            r'^[A-Z]{1,5}$',
            r'^[A-Z]{1,4}\.[A-Z]{1,3}$',
            r'^\^[A-Z0-9]{1,6}$',
        ]
        
        for ticker in self.tickers:
            is_valid = any(re.match(pattern, ticker) for pattern in valid_patterns)
            
            if len(ticker) > 6 or ticker.isdigit():
                is_valid = False
                
            if is_valid:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        
        return valid_tickers, invalid_tickers
    
    def quick_test_ticker(self, ticker: str) -> Tuple[bool, str]:
        """Quick test for individual ticker validation."""
        try:
            success, message, data = self.simple_ticker_test(ticker)
            
            if not success:
                return False, message
            
            price_series = self._extract_price_series(data, ticker)
            
            if price_series is None:
                return False, f"Data fetched but could not extract price series for {ticker}"
            
            if not isinstance(price_series, pd.Series):
                return False, f"Extracted data is not a Series for {ticker}: {type(price_series)}"
            
            if len(price_series) < 3:
                return False, f"Insufficient data points for {ticker}: {len(price_series)}"
            
            return True, f"✅ Valid ticker: {ticker} ({len(price_series)} data points)"
                
        except Exception as e:
            return False, f"Error testing {ticker}: {str(e)}"
    
    def simple_ticker_test(self, ticker: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Very simple ticker test to debug yfinance issues."""
        try:
            data = yf.download(ticker, period='1mo', progress=False)
            
            if data.empty:
                return False, f"No data returned for {ticker}", None
            
            return True, f"Success: {ticker} returned {data.shape[0]} rows, {data.shape[1]} columns", data
            
        except Exception as e:
            return False, f"Error: {str(e)}", None

    # ================== DATA FETCHING AND PROCESSING ==================
    
    def _extract_price_series(self, data: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
        """Extract price series from yfinance data."""
        try:
            if data.empty or not isinstance(data, pd.DataFrame):
                return None
                
            # Handle single ticker download (most common case)
            for price_col in ['Adj Close', 'Close']:
                if price_col in data.columns:
                    price_series = data[price_col].dropna()
                    if isinstance(price_series, pd.Series) and len(price_series) > 0:
                        return price_series
            
            # Handle multi-index columns (when downloading multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                for price_col in ['Adj Close', 'Close']:
                    if (price_col, ticker) in data.columns:
                        price_series = data[(price_col, ticker)].dropna()
                        if isinstance(price_series, pd.Series) and len(price_series) > 0:
                            return price_series
                    
                    # Try flattened approach
                    try:
                        if any(col[0] == price_col for col in data.columns):
                            price_data = data.xs(price_col, level=0, axis=1)
                            if not price_data.empty and hasattr(price_data, 'iloc'):
                                price_series = price_data.iloc[:, 0].dropna()
                                if isinstance(price_series, pd.Series) and len(price_series) > 0:
                                    return price_series
                    except:
                        continue
            
            # Try to find any price-like column
            for col in data.columns:
                if 'close' in str(col).lower() and not isinstance(col, tuple):
                    price_series = data[col].dropna()
                    if isinstance(price_series, pd.Series) and len(price_series) > 0:
                        return price_series
            
            # Last resort - try first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_series = data[numeric_cols[0]].dropna()
                if isinstance(price_series, pd.Series) and len(price_series) > 0:
                    return price_series
            
            return None
            
        except Exception as e:
            st.warning(f"Error extracting price for {ticker}: {str(e)}")
            return None
    
    def _fetch_data_simple(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
        """Simple data fetching method with robust validation."""
        try:
            methods = [
                lambda: yf.download(ticker, start=start_date, end=end_date, progress=False),
                lambda: yf.download(ticker, period='2y', progress=False),
                lambda: yf.download(ticker, period='1y', progress=False)
            ]
            
            for i, method in enumerate(methods):
                try:
                    data = method()
                    if not data.empty:
                        price_series = self._extract_price_series(data, ticker)
                        if isinstance(price_series, pd.Series) and len(price_series) >= 50:
                            st.info(f"✅ Method {i+1} worked for {ticker}")
                            return price_series
                except Exception as e:
                    st.warning(f"Method {i+1} failed for {ticker}: {str(e)}")
                    continue
                    
            return None
            
        except Exception as e:
            st.error(f"All simple fetch methods failed for {ticker}: {str(e)}")
            return None
    
    def fetch_data(self) -> Tuple[bool, Optional[str]]:
        """Fetch and process market data."""
        try:
            if not self.tickers:
                return False, "No valid tickers provided"
            
            if len(self.tickers) < 2:
                return False, "Portfolio optimization requires at least 2 assets"
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years + 30)
            
            st.info(f"Fetching {self.lookback_years} years of data for {len(self.tickers)} assets...")
            
            successful_data = {}
            self.failed_tickers = []
            self.data_quality_info = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(self.tickers):
                try:
                    status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(self.tickers)})")
                    progress_bar.progress((i + 1) / len(self.tickers))
                    
                    # Primary fetch method
                    ticker_data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    price_series = None
                    
                    if not ticker_data.empty:
                        price_series = self._extract_price_series(ticker_data, ticker)
                        
                        if price_series is not None:
                            st.info(f"✅ Extracted {len(price_series)} price points for {ticker}")
                        else:
                            st.warning(f"⚠️ Could not extract price series from data for {ticker}")
                    else:
                        st.warning(f"⚠️ Empty data returned for {ticker}")
                    
                    # Fallback method if needed
                    if price_series is None or len(price_series) < 50:
                        st.info(f"🔄 Trying alternative method for {ticker}...")
                        price_series = self._fetch_data_simple(ticker, start_date, end_date)
                        
                        if price_series is not None:
                            st.info(f"✅ Alternative method succeeded for {ticker}")
                    
                    # Final validation
                    if price_series is None:
                        st.error(f"❌ All methods failed for {ticker}")
                        self.failed_tickers.append(ticker)
                        continue
                        
                    if len(price_series) < 50:
                        st.error(f"❌ Insufficient data for {ticker}: only {len(price_series)} points")
                        self.failed_tickers.append(ticker)
                        continue
                    
                    if not isinstance(price_series, pd.Series):
                        st.error(f"❌ Invalid data type for {ticker}: {type(price_series)}")
                        self.failed_tickers.append(ticker)
                        continue
                    
                    # Data quality assessment
                    quality_info = self._assess_data_quality(price_series, ticker)
                    self.data_quality_info[ticker] = quality_info
                    
                    successful_data[ticker] = price_series
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    st.error(f"Error fetching {ticker}: {str(e)}")
                    self.failed_tickers.append(ticker)
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if len(successful_data) < 2:
                return False, f"Could not fetch sufficient data. Failed tickers: {', '.join(self.failed_tickers)}"
            
            # Create DataFrame and process
            self.data = pd.DataFrame(successful_data)
            
            if not self._clean_and_align_data():
                return False, "Data cleaning failed"
            
            self._calculate_statistics()
            self.tickers = list(self.data.columns)
            self._fetch_market_data_enhanced(start_date, end_date)
            
            error_msg = None
            if self.failed_tickers:
                error_msg = f"Could not fetch data for: {', '.join(self.failed_tickers)}"
            
            return True, error_msg
            
        except Exception as e:
            return False, f"Critical error in data fetching: {str(e)}"

    # ================== DATA PROCESSING AND STATISTICS ==================
    
    def _assess_data_quality(self, price_series: pd.Series, ticker: str) -> Dict:
        """Assess data quality for a price series."""
        try:
            total_points = len(price_series)
            null_count = price_series.isnull().sum()
            zero_count = (price_series == 0).sum()
            
            price_changes = price_series.pct_change().dropna()
            extreme_changes = (abs(price_changes) > 0.5).sum()
            
            date_range = price_series.index.max() - price_series.index.min()
            expected_days = self.lookback_years * 365
            coverage = date_range.days / expected_days
            
            return {
                'total_points': total_points,
                'null_count': null_count,
                'zero_count': zero_count,
                'extreme_changes': extreme_changes,
                'coverage': coverage,
                'quality_score': max(0, 1 - (null_count + zero_count + extreme_changes) / total_points)
            }
        except Exception:
            return {'quality_score': 0}
    
    def _clean_and_align_data(self) -> bool:
        """Clean and align data across all assets."""
        try:
            if self.data.empty:
                return False
            
            self.data = self.data.dropna(how='all')
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            self.data = self.data.dropna()
            
            min_required_days = max(100, self.lookback_years * 50)
            if len(self.data) < min_required_days:
                return False
            
            # Remove low quality assets
            low_quality_assets = []
            for asset in self.data.columns:
                if asset in self.data_quality_info:
                    if self.data_quality_info[asset]['quality_score'] < 0.7:
                        low_quality_assets.append(asset)
            
            if low_quality_assets:
                self.data = self.data.drop(columns=low_quality_assets)
                st.warning(f"Removed low quality assets: {', '.join(low_quality_assets)}")
            
            return len(self.data.columns) >= 2
            
        except Exception:
            return False
    
    def _calculate_statistics(self):
        """Calculate returns and statistical measures."""
        try:
            self.returns = self.data.pct_change().dropna()
            
            # Remove extreme outliers
            for col in self.returns.columns:
                mean = self.returns[col].mean()
                std = self.returns[col].std()
                outlier_mask = abs(self.returns[col] - mean) > 3 * std
                self.returns.loc[outlier_mask, col] = np.nan
            
            self.returns = self.returns.fillna(method='ffill').fillna(method='bfill')
            
            trading_days = 252
            self.mean_returns = self.returns.mean() * trading_days
            self.cov_matrix = self.returns.cov() * trading_days
            
            self._fix_covariance_matrix()
            
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
            raise
    
    def _fix_covariance_matrix(self):
        """Ensure covariance matrix is positive definite."""
        try:
            eigenvals = np.linalg.eigvals(self.cov_matrix)
            
            if np.any(eigenvals <= 0):
                min_eigenval = np.min(eigenvals)
                regularization = abs(min_eigenval) + 1e-6
                self.cov_matrix += np.eye(len(self.cov_matrix)) * regularization
                st.info("Applied covariance matrix regularization for numerical stability")
                
        except Exception:
            self.cov_matrix += np.eye(len(self.cov_matrix)) * 1e-6

    # ================== MARKET DATA AND RISK-FREE RATE ==================
    
    def _fetch_market_data_enhanced(self, start_date: datetime, end_date: datetime):
        """Enhanced market data fetching for CAPM analysis."""
        market_symbols = ['^GSPC', 'SPY', '^IXIC', 'VTI', '^DJI']
        
        for symbol in market_symbols:
            try:
                market_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not market_data.empty:
                    price_series = self._extract_price_series(market_data, symbol)
                    if price_series is not None and len(price_series) > 100:
                        common_dates = self.data.index.intersection(price_series.index)
                        if len(common_dates) > 100:
                            self.market_data = price_series.loc[common_dates]
                            self.market_returns = self.market_data.pct_change().dropna()
                            st.info(f"Using {symbol} as market benchmark")
                            break
            except Exception:
                continue
    
    # 1. FIXED RISK-FREE RATE FETCHING
    def get_risk_free_rate(self) -> float:
        """Enhanced risk-free rate fetching with better error handling."""
        # Treasury symbols in order of preference
        treasury_symbols = ['^IRX', '^TNX', '^FVX', 'DGS3MO', 'DGS10']
        
        for symbol in treasury_symbols:
            try:
                st.info(f"Attempting to fetch risk-free rate from {symbol}...")
                
                # Try different time periods
                for period in ['5d', '1mo', '3mo']:
                    try:
                        treasury_data = yf.download(symbol, period=period, progress=False)
                        
                        if not treasury_data.empty:
                            # Handle different column structures
                            if 'Close' in treasury_data.columns:
                                rates = treasury_data['Close'].dropna()
                            elif 'Adj Close' in treasury_data.columns:
                                rates = treasury_data['Adj Close'].dropna()
                            else:
                                # Handle potential multi-index columns
                                if isinstance(treasury_data.columns, pd.MultiIndex):
                                    for col in treasury_data.columns:
                                        if 'close' in str(col).lower():
                                            rates = treasury_data[col].dropna()
                                            break
                                else:
                                    rates = treasury_data.iloc[:, -1].dropna()  # Last column
                            
                            if len(rates) > 0:
                                latest_rate = rates.iloc[-1]
                                
                                # Validate rate (should be between 0 and 20% for most scenarios)
                                if 0 < latest_rate < 20:
                                    self.rf_rate = latest_rate / 100
                                    st.success(f"✅ Successfully fetched risk-free rate: {self.rf_rate:.3%} from {symbol}")
                                    return self.rf_rate
                                
                                st.warning(f"Rate {latest_rate} from {symbol} seems invalid")
                            
                    except Exception as e:
                        st.warning(f"Failed to fetch {symbol} with period {period}: {str(e)}")
                        continue
                        
            except Exception as e:
                st.warning(f"Error with {symbol}: {str(e)}")
                continue
        
        # If all fails, try FRED API approach (alternative data source)
        try:
            import pandas_datareader as pdr
            st.info("Trying FRED data source...")
            
            # Try 3-month Treasury rate from FRED
            end = datetime.now()
            start = end - timedelta(days=30)
            
            fred_rate = pdr.get_data_fred('DGS3MO', start, end).dropna()
            if not fred_rate.empty:
                latest_rate = fred_rate.iloc[-1, 0]
                if 0 < latest_rate < 20:
                    self.rf_rate = latest_rate / 100
                    st.success(f"✅ Successfully fetched risk-free rate from FRED: {self.rf_rate:.3%}")
                    return self.rf_rate
                    
        except Exception as e:
            st.warning(f"FRED approach failed: {str(e)}")
        
        # Final fallback - use reasonable default based on current economic conditions
        self.rf_rate = 0.045  # 4.5% - reasonable default for 2024-2025
        st.warning(f"⚠️ Using fallback risk-free rate: {self.rf_rate:.3%}")
        return self.rf_rate

    # ================== PORTFOLIO METRICS AND CALCULATIONS ==================
    
    def portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        try:
            weights = np.array(weights)
            
            if len(weights) != len(self.mean_returns):
                raise ValueError("Weight vector length mismatch")
            
            if abs(np.sum(weights) - 1.0) > 1e-6:
                weights = weights / np.sum(weights)
            
            portfolio_return = np.dot(self.mean_returns, weights)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            portfolio_std = np.sqrt(max(0, portfolio_variance))
            
            sharpe_ratio = (portfolio_return - self.rf_rate) / portfolio_std if portfolio_std > 1e-10 else 0
            
            portfolio_returns = (self.returns * weights).sum(axis=1)
            
            try:
                var_95 = np.percentile(portfolio_returns, 5)
                var_99 = np.percentile(portfolio_returns, 1)
                cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            except Exception:
                var_95 = var_99 = cvar_95 = 0
            
            concentration = np.sum(weights ** 2)
            diversification_ratio = 1 / concentration if concentration > 0 else 1
            
            try:
                portfolio_cumret = (portfolio_returns + 1).cumprod()
                running_max = portfolio_cumret.expanding().max()
                drawdown = (portfolio_cumret - running_max) / running_max
                max_drawdown = drawdown.min()
            except Exception:
                max_drawdown = 0
            
            try:
                negative_returns = portfolio_returns[portfolio_returns < 0]
                downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)
                sortino_ratio = (portfolio_return - self.rf_rate) / downside_deviation if downside_deviation > 0 else 0
            except Exception:
                sortino_ratio = 0
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'variance': portfolio_variance,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'concentration': concentration,
                'diversification_ratio': diversification_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'expected_return': 0, 'volatility': 0, 'variance': 0, 'sharpe_ratio': 0,
                'sortino_ratio': 0, 'var_95': 0, 'var_99': 0, 'cvar_95': 0,
                'concentration': 1, 'diversification_ratio': 1, 'max_drawdown': 0
            }

    # ================== OPTIMIZATION OBJECTIVES AND CONSTRAINTS ==================
    
    def objective_sharpe(self, weights: np.ndarray) -> float:
        """Objective function for maximum Sharpe ratio optimization."""
        try:
            metrics = self.portfolio_metrics(weights)
            return -metrics['sharpe_ratio']
        except Exception:
            return 1e6
    
    def objective_variance(self, weights: np.ndarray) -> float:
        """Objective function for minimum variance optimization."""
        try:
            return self.portfolio_metrics(weights)['variance']
        except Exception:
            return 1e6
    
    def constraint_return(self, weights: np.ndarray, target_return: float) -> float:
        """Constraint function for target return."""
        try:
            return self.portfolio_metrics(weights)['expected_return'] - target_return
        except Exception:
            return 1e6
    
    def constraint_volatility(self, weights: np.ndarray, target_vol: float) -> float:
        """Constraint function for target volatility."""
        try:
            return self.portfolio_metrics(weights)['volatility'] - target_vol
        except Exception:
            return 1e6

    # ================== PORTFOLIO OPTIMIZATION ==================
    
    def optimize_portfolio(self, 
                         method: str = 'max_sharpe',
                         target_return: Optional[float] = None,
                         target_volatility: Optional[float] = None,
                         min_weight: float = 0.0,
                         max_weight: float = 1.0,
                         max_iterations: int = 1000,
                         include_risk_free: bool = False) -> Dict:
        """Portfolio optimization with option to include risk-free asset."""
        
        # If including risk-free asset, use the enhanced method
        if include_risk_free:
            return self.optimize_portfolio_with_risk_free_fixed(
                method, target_return, target_volatility, min_weight, max_weight, max_iterations
            )
        
        # Original method for risky assets only
        if self.returns is None or self.mean_returns is None:
            return {'success': False, 'error': 'No data available for optimization'}
        
        try:
            num_assets = len(self.tickers)
            
            min_weight = max(0, min(min_weight, 0.9))
            max_weight = max(min_weight + 0.01, min(max_weight, 1.0))
            
            if min_weight * num_assets > 1:
                min_weight = 0.8 / num_assets
            
            starting_points = [
                np.array([1.0/num_assets] * num_assets),
                np.random.dirichlet(np.ones(num_assets)),
                np.random.dirichlet(np.ones(num_assets) * 2)
            ]
            
            best_result = None
            best_objective = float('inf')
            
            for initial_weights in starting_points:
                try:
                    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
                    
                    if method == 'max_sharpe':
                        result = minimize(
                            self.objective_sharpe, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': max_iterations, 'ftol': 1e-9}
                        )
                        current_objective = self.objective_sharpe(result.x)
                        
                    elif method == 'min_variance':
                        result = minimize(
                            self.objective_variance, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': max_iterations, 'ftol': 1e-9}
                        )
                        current_objective = self.objective_variance(result.x)
                        
                    elif method == 'target_return' and target_return is not None:
                        min_possible = self.mean_returns.min()
                        max_possible = self.mean_returns.max()
                        target_return = max(min_possible, min(target_return, max_possible))
                        
                        constraints.append({
                            'type': 'eq',
                            'fun': lambda x: self.constraint_return(x, target_return)
                        })
                        result = minimize(
                            self.objective_variance, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': max_iterations}
                        )
                        current_objective = self.objective_variance(result.x)
                        
                    elif method == 'target_volatility' and target_volatility is not None:
                        min_possible_vol = np.sqrt(self.cov_matrix.values.min())
                        max_possible_vol = np.sqrt(self.cov_matrix.values.max())
                        target_volatility = max(min_possible_vol, min(target_volatility, max_possible_vol))
                        
                        constraints.append({
                            'type': 'eq',
                            'fun': lambda x: self.constraint_volatility(x, target_volatility)
                        })
                        result = minimize(
                            lambda x: -self.portfolio_metrics(x)['expected_return'],
                            initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': max_iterations}
                        )
                        current_objective = -self.portfolio_metrics(result.x)['expected_return']
                        
                    else:
                        return {'success': False, 'error': f'Invalid optimization method: {method}'}
                    
                    if result.success and current_objective < best_objective:
                        best_result = result
                        best_objective = current_objective
                        
                except Exception as e:
                    continue
            
            if best_result is None or not best_result.success:
                return {'success': False, 'error': 'Optimization failed to converge'}
            
            optimal_weights = best_result.x
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            metrics = self.portfolio_metrics(optimal_weights)
            
            portfolio_vol = metrics['volatility']
            if portfolio_vol > 0:
                marginal_contrib = np.dot(self.cov_matrix, optimal_weights) / portfolio_vol
                component_contrib = optimal_weights * marginal_contrib
            else:
                marginal_contrib = np.zeros(len(optimal_weights))
                component_contrib = np.zeros(len(optimal_weights))
            
            return {
                'success': True,
                'method': method,
                'weights': optimal_weights,
                'expected_return': metrics['expected_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'var_95': metrics['var_95'],
                'var_99': metrics['var_99'],
                'cvar_95': metrics['cvar_95'],
                'max_drawdown': metrics['max_drawdown'],
                'diversification_ratio': metrics['diversification_ratio'],
                'concentration': metrics['concentration'],
                'risk_contribution': component_contrib,
                'marginal_contribution': marginal_contrib,
                'optimization_details': {
                    'converged': best_result.success,
                    'iterations': best_result.nit if hasattr(best_result, 'nit') else 0,
                    'message': best_result.message if hasattr(best_result, 'message') else 'Completed'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Optimization error: {str(e)}'}
    
    # 2. FIXED TARGET OPTIMIZATION WITH PROPER RISK-FREE INTEGRATION
    def optimize_portfolio_with_risk_free_fixed(self, 
                                            method: str = 'max_sharpe',
                                            target_return: Optional[float] = None,
                                            target_volatility: Optional[float] = None,
                                            min_weight: float = 0.0,
                                            max_weight: float = 1.0,
                                            max_iterations: int = 1000) -> Dict:
        """FIXED: Portfolio optimization with proper risk-free asset integration."""
        if self.returns is None or self.mean_returns is None:
            return {'success': False, 'error': 'No data available for optimization'}
        
        try:
            # Step 1: Always find the tangency portfolio first (optimal risky portfolio)
            st.info("🎯 Finding optimal risky portfolio (tangency)...")
            tangency_result = self.optimize_portfolio(
                method='max_sharpe',
                min_weight=min_weight,
                max_weight=max_weight,
                max_iterations=max_iterations,
                include_risk_free=False  # Important: risky assets only
            )
            
            if not tangency_result['success']:
                return tangency_result
            
            tangency_weights = tangency_result['weights']
            tangency_return = tangency_result['expected_return']
            tangency_volatility = tangency_result['volatility']
            tangency_sharpe = tangency_result['sharpe_ratio']
            
            st.success(f"✅ Tangency portfolio: Return={tangency_return:.2%}, Vol={tangency_volatility:.2%}, Sharpe={tangency_sharpe:.3f}")
            
            # Step 2: Determine allocation based on method and constraints
            rf_weight = 0.0
            risky_weight = 1.0
            
            if method == 'max_sharpe':
                # Pure tangency portfolio gives maximum Sharpe ratio
                rf_weight = 0.0
                risky_weight = 1.0
                st.info("📊 Using pure tangency portfolio (max Sharpe)")
                
            elif method == 'target_return' and target_return is not None:
                st.info(f"🎯 Optimizing for target return: {target_return:.2%}")
                
                if target_return <= self.rf_rate:
                    # Target below risk-free rate: invest only in risk-free asset
                    rf_weight = 1.0
                    risky_weight = 0.0
                    st.info("💰 Target return below risk-free rate - using 100% risk-free asset")
                    
                elif target_return >= tangency_return:
                    # Target above tangency portfolio: use leverage (with limits)
                    if tangency_return <= self.rf_rate:
                        st.error("❌ Cannot achieve target return - tangency portfolio return too low")
                        return {'success': False, 'error': 'Target return unachievable with available assets'}
                    
                    leverage_ratio = (target_return - self.rf_rate) / (tangency_return - self.rf_rate)
                    max_leverage = 2.0  # Allow up to 200% in risky assets
                    
                    if leverage_ratio > max_leverage:
                        leverage_ratio = max_leverage
                        actual_return = self.rf_rate + leverage_ratio * (tangency_return - self.rf_rate)
                        st.warning(f"⚠️ Target return requires excessive leverage. Limited to {max_leverage:.0%} risky assets.")
                        st.warning(f"Achievable return with max leverage: {actual_return:.2%}")
                    
                    rf_weight = 1.0 - leverage_ratio
                    risky_weight = leverage_ratio
                    st.info(f"📈 Using leverage: {risky_weight:.1%} risky assets, {rf_weight:.1%} risk-free (borrowed if negative)")
                    
                else:
                    # Target between risk-free rate and tangency portfolio
                    risky_weight = (target_return - self.rf_rate) / (tangency_return - self.rf_rate)
                    rf_weight = 1.0 - risky_weight
                    st.info(f"⚖️ Mixing assets: {risky_weight:.1%} risky, {rf_weight:.1%} risk-free")
                    
            elif method == 'target_volatility' and target_volatility is not None:
                st.info(f"📊 Optimizing for target volatility: {target_volatility:.2%}")
                
                if target_volatility <= 1e-6:
                    # Target volatility near zero: use only risk-free asset
                    rf_weight = 1.0
                    risky_weight = 0.0
                    st.info("🛡️ Target volatility near zero - using 100% risk-free asset")
                    
                elif target_volatility >= tangency_volatility:
                    # Target volatility above tangency portfolio: use leverage
                    leverage_ratio = target_volatility / tangency_volatility
                    max_leverage = 2.0
                    
                    if leverage_ratio > max_leverage:
                        leverage_ratio = max_leverage
                        actual_vol = leverage_ratio * tangency_volatility
                        st.warning(f"⚠️ Target volatility requires excessive leverage. Limited to {max_leverage:.0%} risky assets.")
                        st.warning(f"Achievable volatility with max leverage: {actual_vol:.2%}")
                    
                    rf_weight = 1.0 - leverage_ratio
                    risky_weight = leverage_ratio
                    st.info(f"📈 Using leverage: {risky_weight:.1%} risky assets")
                    
                else:
                    # Target volatility between zero and tangency portfolio
                    risky_weight = target_volatility / tangency_volatility
                    rf_weight = 1.0 - risky_weight
                    st.info(f"⚖️ Mixing assets: {risky_weight:.1%} risky, {rf_weight:.1%} risk-free")
                    
            elif method == 'min_variance':
                # For minimum variance, compare risk-free asset vs minimum variance portfolio
                min_var_result = self.optimize_portfolio(
                    method='min_variance',
                    min_weight=min_weight,
                    max_weight=max_weight,
                    max_iterations=max_iterations,
                    include_risk_free=False
                )
                
                if min_var_result['success']:
                    min_var_volatility = min_var_result['volatility']
                    
                    if min_var_volatility > 0.001:  # If min var portfolio has significant risk
                        # Use conservative allocation
                        risky_weight = 0.3
                        rf_weight = 0.7
                        tangency_weights = min_var_result['weights']  # Use min var weights instead
                        st.info("🛡️ Using conservative allocation for minimum variance")
                    else:
                        # Min var portfolio already very low risk
                        risky_weight = 1.0
                        rf_weight = 0.0
                        tangency_weights = min_var_result['weights']
                else:
                    st.warning("Could not find minimum variance portfolio, using tangency")
            
            else:
                return {'success': False, 'error': f'Invalid optimization method: {method}'}
            
            # Step 3: Calculate final portfolio weights and metrics
            final_weights = tangency_weights * risky_weight
            
            # Calculate final portfolio metrics
            if risky_weight > 0:
                final_return = self.rf_rate * rf_weight + tangency_return * risky_weight
                final_volatility = abs(risky_weight) * tangency_volatility
            else:
                final_return = self.rf_rate
                final_volatility = 0.0
            
            final_sharpe = (final_return - self.rf_rate) / final_volatility if final_volatility > 0 else float('inf')
            
            # Get additional metrics from the risky portion
            if risky_weight > 0:
                risky_metrics = self.portfolio_metrics(tangency_weights)
            else:
                risky_metrics = {
                    'sortino_ratio': 0, 'var_95': 0, 'var_99': 0, 'cvar_95': 0,
                    'max_drawdown': 0, 'diversification_ratio': 1, 'concentration': 0,
                }
            
            # Create comprehensive result
            result = {
                'success': True,
                'method': method,
                'weights': final_weights,
                'rf_weight': rf_weight,
                'risky_weight': risky_weight,
                'tangency_weights': tangency_weights,
                'expected_return': final_return,
                'volatility': final_volatility,
                'sharpe_ratio': final_sharpe,
                'tangency_return': tangency_return,
                'tangency_volatility': tangency_volatility,
                'tangency_sharpe': tangency_sharpe,
                'sortino_ratio': risky_metrics['sortino_ratio'] * risky_weight if risky_weight > 0 else 0,
                'var_95': risky_metrics['var_95'] * risky_weight if risky_weight > 0 else 0,
                'var_99': risky_metrics['var_99'] * risky_weight if risky_weight > 0 else 0,
                'cvar_95': risky_metrics['cvar_95'] * risky_weight if risky_weight > 0 else 0,
                'max_drawdown': risky_metrics['max_drawdown'] * risky_weight if risky_weight > 0 else 0,
                'diversification_ratio': risky_metrics['diversification_ratio'],
                'concentration': risky_metrics['concentration'],
                'risk_contribution': tangency_result.get('risk_contribution', np.zeros(len(final_weights))) * risky_weight,
                'marginal_contribution': tangency_result.get('marginal_contribution', np.zeros(len(final_weights))) * risky_weight,
                'optimization_details': {
                    'converged': True,
                    'rf_rate': self.rf_rate,
                    'capital_allocation_line': True,
                    'leverage_used': risky_weight > 1.0,
                    'message': f'Optimal allocation: {rf_weight:.1%} risk-free, {risky_weight:.1%} risky portfolio'
                }
            }
            
            # Validation
            if method == 'target_return' and target_return is not None:
                achieved_return = result['expected_return']
                if abs(achieved_return - target_return) > 0.001:  # Allow small tolerance
                    if target_return > self.rf_rate and target_return < tangency_return:
                        st.warning(f"⚠️ Target return {target_return:.2%} vs achieved {achieved_return:.2%}")
            
            if method == 'target_volatility' and target_volatility is not None:
                achieved_vol = result['volatility']
                if abs(achieved_vol - target_volatility) > 0.001:
                    if target_volatility > 0 and target_volatility < tangency_volatility:
                        st.warning(f"⚠️ Target volatility {target_volatility:.2%} vs achieved {achieved_vol:.2%}")
            
            st.success(f"🎉 Final portfolio: Return={final_return:.2%}, Vol={final_volatility:.2%}, Sharpe={final_sharpe:.3f}")
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Optimization with risk-free asset failed: {str(e)}'}

    # ================== EFFICIENT FRONTIER ==================
    
    def generate_efficient_frontier(self, num_points: int = 50) -> Optional[Dict]:
        """Generate efficient frontier with enhanced error handling."""
        if self.returns is None:
            return None
        
        try:
            min_ret = self.mean_returns.min()
            max_ret = self.mean_returns.max()
            
            ret_range = max_ret - min_ret
            min_ret -= ret_range * 0.1
            max_ret += ret_range * 0.1
            
            target_returns = np.linspace(min_ret, max_ret, num_points)
            
            frontier_data = {
                'returns': [],
                'volatilities': [],
                'sharpe_ratios': [],
                'weights': []
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            successful_optimizations = 0
            
            for i, target_ret in enumerate(target_returns):
                try:
                    status_text.text(f"Generating efficient frontier point {i+1}/{num_points}")
                    progress_bar.progress((i + 1) / num_points)
                    
                    result = self.optimize_portfolio(
                        'target_return', 
                        target_return=target_ret,
                        max_iterations=500
                    )
                    
                    if result['success']:
                        frontier_data['returns'].append(result['expected_return'])
                        frontier_data['volatilities'].append(result['volatility'])
                        frontier_data['sharpe_ratios'].append(result['sharpe_ratio'])
                        frontier_data['weights'].append(result['weights'])
                        successful_optimizations += 1
                        
                except Exception:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if successful_optimizations < 5:
                st.warning("Could not generate sufficient efficient frontier points")
                return None
            
            st.success(f"Generated efficient frontier with {successful_optimizations} points")
            return frontier_data
            
        except Exception as e:
            st.error(f"Error generating efficient frontier: {str(e)}")
            return None

    # ================== CAPM ANALYSIS ==================
    
    def calculate_capm_metrics(self) -> Dict[str, Dict[str, float]]:
        """Enhanced CAPM metrics calculation."""
        if self.market_returns is None or self.returns is None:
            return {}
        
        try:
            capm_metrics = {}
            
            common_dates = self.returns.index.intersection(self.market_returns.index)
            if len(common_dates) < 50:
                return {}
            
            aligned_returns = self.returns.loc[common_dates]
            aligned_market = self.market_returns.loc[common_dates]
            
            market_excess = aligned_market - self.rf_rate/252
            market_return_annual = aligned_market.mean() * 252
            market_volatility = aligned_market.std() * np.sqrt(252)
            
            for ticker in self.tickers:
                try:
                    asset_returns = aligned_returns[ticker]
                    asset_excess = asset_returns - self.rf_rate/252
                    
                    covariance = np.cov(asset_excess, market_excess)[0, 1]
                    market_variance = np.var(market_excess)
                    
                    if market_variance > 0:
                        beta = covariance / market_variance
                    else:
                        beta = 1.0
                    
                    expected_return = self.rf_rate + beta * (market_return_annual - self.rf_rate)
                    actual_return = asset_returns.mean() * 252
                    alpha = actual_return - expected_return
                    
                    correlation_matrix = np.corrcoef(asset_excess.dropna(), market_excess.dropna())
                    if correlation_matrix.shape == (2, 2):
                        correlation = correlation_matrix[0, 1]
                        r_squared = correlation ** 2
                    else:
                        correlation = 0
                        r_squared = 0
                    
                    asset_volatility = asset_returns.std() * np.sqrt(252)
                    systematic_risk = abs(beta) * market_volatility
                    total_risk_squared = asset_volatility ** 2
                    systematic_risk_squared = systematic_risk ** 2
                    idiosyncratic_risk = np.sqrt(max(0, total_risk_squared - systematic_risk_squared))
                    
                    capm_metrics[ticker] = {
                        'beta': beta,
                        'alpha': alpha,
                        'expected_return': expected_return,
                        'actual_return': actual_return,
                        'r_squared': r_squared,
                        'correlation': correlation,
                        'systematic_risk': systematic_risk,
                        'idiosyncratic_risk': idiosyncratic_risk,
                        'total_risk': asset_volatility
                    }
                    
                except Exception as e:
                    capm_metrics[ticker] = {
                        'beta': 1.0, 'alpha': 0.0, 'expected_return': self.mean_returns[ticker],
                        'actual_return': self.mean_returns[ticker], 'r_squared': 0.5,
                        'correlation': 0.0, 'systematic_risk': 0.0, 'idiosyncratic_risk': 0.0,
                        'total_risk': 0.0
                    }
            
            return capm_metrics
            
        except Exception as e:
            st.error(f"Error calculating CAPM metrics: {str(e)}")
            return {}

    # ================== DEBUG AND UTILITY FUNCTIONS ==================
    
    def get_debug_info(self) -> Dict:
        """Get debug information about the optimizer state."""
        debug_info = {
            'tickers_requested': self.tickers,
            'failed_tickers': self.failed_tickers,
            'successful_tickers': list(self.data.columns) if self.data is not None else [],
            'data_points': len(self.data) if self.data is not None else 0,
            'data_date_range': {
                'start': self.data.index.min().strftime('%Y-%m-%d') if self.data is not None else None,
                'end': self.data.index.max().strftime('%Y-%m-%d') if self.data is not None else None
            } if self.data is not None else None,
            'risk_free_rate': self.rf_rate,
            'market_data_available': self.market_data is not None,
            'data_quality': self.data_quality_info
        }
        return debug_info


# ================== VISUALIZATION FUNCTIONS ==================

# 3. FIXED EFFICIENT FRONTIER PLOT WITH CORRECT CAL REPRESENTATION
def create_efficient_frontier_plot_fixed(optimizer: PortfolioOptimizer, 
                                        optimal_portfolio: Optional[Dict] = None,
                                        include_risk_free: bool = True) -> Optional[go.Figure]:
    """FIXED: Create efficient frontier with proper CAL and portfolio positioning."""
    try:
        with st.spinner("Generating efficient frontier..."):
            frontier_data = optimizer.generate_efficient_frontier()
            
        if not frontier_data or len(frontier_data['returns']) < 5:
            st.warning("Could not generate sufficient frontier data")
            return None
        
        fig = go.Figure()
        
        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=np.array(frontier_data['volatilities']) * 100,
            y=np.array(frontier_data['returns']) * 100,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=frontier_data['sharpe_ratios'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio", x=1.02)
            ),
            line=dict(width=2, color='rgba(102, 126, 234, 0.8)'),
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
                x=[asset_vol], y=[asset_return],
                mode='markers+text',
                marker=dict(size=15, color=color, symbol='star', line=dict(width=2, color='white')),
                text=[ticker], textposition="top center",
                textfont=dict(size=12, color='white'),
                name=ticker,
                hovertemplate=f'<b>{ticker}</b><br>Return: %{{y:.2f}}%<br>Risk: %{{x:.2f}}%<extra></extra>'
            ))
        
        # Add risk-free asset point
        rf_return = optimizer.rf_rate * 100
        fig.add_trace(go.Scatter(
            x=[0], y=[rf_return],
            mode='markers+text',
            marker=dict(size=20, color='gold', symbol='circle', line=dict(width=3, color='black')),
            text=['Risk-Free'], textposition="top center",
            textfont=dict(size=12, color='black', family='Arial Black'),
            name='Risk-Free Asset',
            hovertemplate=f'<b>Risk-Free Asset</b><br>Return: {rf_return:.2f}%<br>Risk: 0.0%<extra></extra>'
        ))
        
        # Find tangency portfolio and draw CAL
        if frontier_data['sharpe_ratios']:
            max_sharpe_idx = np.argmax(frontier_data['sharpe_ratios'])
            tangency_vol = frontier_data['volatilities'][max_sharpe_idx] * 100
            tangency_ret = frontier_data['returns'][max_sharpe_idx] * 100
            tangency_sharpe = frontier_data['sharpe_ratios'][max_sharpe_idx]
            
            # Mark tangency portfolio
            fig.add_trace(go.Scatter(
                x=[tangency_vol], y=[tangency_ret],
                mode='markers+text',
                marker=dict(size=25, color='red', symbol='diamond', line=dict(width=3, color='white')),
                text=['Tangency'], textposition="top center",
                textfont=dict(size=12, color='black', family='Arial Black'),
                name='Tangency Portfolio',
                hovertemplate=f'<b>Tangency Portfolio</b><br>Return: {tangency_ret:.2f}%<br>Risk: {tangency_vol:.2f}%<br>Sharpe: {tangency_sharpe:.3f}<extra></extra>'
            ))
            
            # Draw Capital Allocation Line (CAL) - extended in both directions
            max_vol_display = max(60, tangency_vol * 2.5)
            cal_x = np.linspace(0, max_vol_display, 100)
            cal_slope = (tangency_ret - rf_return) / tangency_vol if tangency_vol > 0 else 0
            cal_y = rf_return + cal_slope * cal_x
            
            fig.add_trace(go.Scatter(
                x=cal_x, y=cal_y, mode='lines',
                line=dict(color='orange', width=4, dash='dash'),
                name='Capital Allocation Line',
                hovertemplate='<b>Capital Allocation Line</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<br>Sharpe: ' + f'{tangency_sharpe:.3f}<extra></extra>'
            ))
        
        # Mark optimal portfolio if provided
        if optimal_portfolio and optimal_portfolio.get('success'):
            # Always show the actual portfolio position
            your_vol = optimal_portfolio['volatility'] * 100
            your_ret = optimal_portfolio['expected_return'] * 100
            your_sharpe = optimal_portfolio['sharpe_ratio']
            
            # Check if this involves risk-free asset
            rf_weight = optimal_portfolio.get('rf_weight', 0)
            risky_weight = optimal_portfolio.get('risky_weight', 1)
            
            # Color and label based on allocation
            if rf_weight > 0.01:  # Significant risk-free allocation
                marker_color = 'lime'
                portfolio_label = f'Your Portfolio (CAL)'
                hover_text = f'<b>Your Optimal Portfolio</b><br>Return: {your_ret:.2f}%<br>Risk: {your_vol:.2f}%<br>Sharpe: {your_sharpe:.3f}<br>Risk-Free: {rf_weight:.1%}<br>Risky: {risky_weight:.1%}<extra></extra>'
            elif risky_weight > 1.01:  # Leverage used
                marker_color = 'purple'
                portfolio_label = f'Your Portfolio (Leveraged)'
                hover_text = f'<b>Your Leveraged Portfolio</b><br>Return: {your_ret:.2f}%<br>Risk: {your_vol:.2f}%<br>Sharpe: {your_sharpe:.3f}<br>Leverage: {risky_weight:.1%}<extra></extra>'
            else:  # Pure risky portfolio
                marker_color = 'blue'
                portfolio_label = f'Your Portfolio (Risky Only)'
                hover_text = f'<b>Your Portfolio</b><br>Return: {your_ret:.2f}%<br>Risk: {your_vol:.2f}%<br>Sharpe: {your_sharpe:.3f}<extra></extra>'
            
            fig.add_trace(go.Scatter(
                x=[your_vol], y=[your_ret],
                mode='markers+text',
                marker=dict(size=30, color=marker_color, symbol='star', line=dict(width=3, color='black')),
                text=['Your Portfolio'], textposition="bottom center",
                textfont=dict(size=12, color='black', family='Arial Black'),
                name=portfolio_label,
                hovertemplate=hover_text
            ))
            
            # If using risk-free asset, draw line from risk-free to your portfolio
            if abs(rf_weight) > 0.01:
                fig.add_trace(go.Scatter(
                    x=[0, your_vol], y=[rf_return, your_ret],
                    mode='lines',
                    line=dict(color=marker_color, width=6, dash='solid'),
                    name='Your CAL Position',
                    opacity=0.7,
                    hovertemplate='<b>Your Position on CAL</b><extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title={'text': 'Efficient Frontier with Capital Allocation Line', 'x': 0.5, 'font': {'size': 20, 'color': '#2c3e50'}},
            xaxis_title='Risk (Volatility %)', yaxis_title='Expected Return %',
            height=600, hovermode='closest', legend=dict(x=0.02, y=0.98),
            template='plotly_white', showlegend=True,
            xaxis=dict(range=[0, max(60, max(frontier_data['volatilities']) * 120)]),
            yaxis=dict(range=[min(0, rf_return * 0.5), max(frontier_data['returns']) * 120])
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating efficient frontier plot: {str(e)}")
        return None



def create_portfolio_composition_chart(tickers: List[str], 
                                     weights: np.ndarray, 
                                     rf_weight: Optional[float] = None) -> go.Figure:
    """Enhanced portfolio composition visualization including risk-free asset."""
    try:
        # Prepare data including risk-free asset if present
        labels = []
        values = []
        colors = px.colors.qualitative.Set3
        
        # Add risk-free asset if significant
        if rf_weight is not None and rf_weight > 0.001:
            labels.append('Risk-Free Asset')
            values.append(rf_weight * 100)
        
        # Add risky assets
        for i, ticker in enumerate(tickers):
            if weights[i] > 0.001:  # Only show significant weights
                labels.append(ticker)
                values.append(weights[i] * 100)
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=('Portfolio Allocation', 'Asset Weights'),
            column_widths=[0.6, 0.4]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=labels, values=values,
                textinfo='label+percent', textposition='auto',
                marker=dict(colors=colors[:len(labels)], line=dict(color='white', width=2)),
                hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>',
                textfont=dict(size=12)
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=labels, y=values,
                text=[f'{v:.1f}%' for v in values],
                textposition='auto',
                marker=dict(color=colors[:len(labels)], line=dict(color='white', width=1)),
                hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, template='plotly_white')
        return fig
        
    except Exception as e:
        st.error(f"Error creating composition chart: {str(e)}")
        return go.Figure()


def create_risk_return_analysis(optimizer: PortfolioOptimizer, 
                               weights: Optional[np.ndarray] = None) -> go.Figure:
    """Enhanced risk-return analysis visualization."""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk-Return Scatter', 'Correlation Matrix', 
                           'Risk Decomposition', 'Asset Statistics'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        colors = px.colors.qualitative.Set1
        for i, ticker in enumerate(optimizer.tickers):
            ret = optimizer.mean_returns[ticker] * 100
            vol = np.sqrt(optimizer.cov_matrix.iloc[i, i]) * 100
            size = weights[i] * 100 if weights is not None else 15
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=[vol], y=[ret], mode='markers+text',
                    marker=dict(size=max(15, size), color=color, line=dict(width=2, color='white')),
                    text=[ticker], textposition="middle center",
                    textfont=dict(color='white', size=10), name=ticker,
                    hovertemplate=f'<b>{ticker}</b><br>Return: %{{y:.2f}}%<br>Risk: %{{x:.2f}}%<extra></extra>'
                ),
                row=1, col=1
            )
        
        corr_matrix = optimizer.returns.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmid=0,
                text=np.round(corr_matrix.values, 2), texttemplate='%{text}',
                textfont=dict(size=10), showscale=True,
                colorbar=dict(title="Correlation", x=0.48),
                hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        if weights is not None:
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(optimizer.cov_matrix, weights)))
            if portfolio_vol > 0:
                marginal_contrib = np.dot(optimizer.cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                fig.add_trace(
                    go.Bar(
                        x=optimizer.tickers, y=risk_contrib * 100,
                        name='Risk Contribution', marker=dict(color=colors),
                        text=[f'{rc:.1f}%' for rc in risk_contrib * 100], textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        stats_data = []
        for i, ticker in enumerate(optimizer.tickers):
            ret = optimizer.mean_returns[ticker]
            vol = np.sqrt(optimizer.cov_matrix.iloc[i, i])
            sharpe = (ret - optimizer.rf_rate) / vol if vol > 0 else 0
            weight = weights[i] if weights is not None else 0
            
            stats_data.append([
                ticker, f'{ret:.2%}', f'{vol:.2%}', f'{sharpe:.3f}',
                f'{weight:.2%}' if weights is not None else 'N/A'
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Asset', 'Return', 'Risk', 'Sharpe', 'Weight'],
                    fill_color='lightblue', font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*stats_data)),
                    fill_color='lightgray', font=dict(size=10)
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        return fig
        
    except Exception as e:
        st.error(f"Error creating risk-return analysis: {str(e)}")
        return go.Figure()


def create_performance_analytics(optimizer: PortfolioOptimizer, 
                                weights: np.ndarray) -> go.Figure:
    """Enhanced performance analytics dashboard."""
    try:
        portfolio_returns = (optimizer.returns * weights).sum(axis=1)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        individual_cumulative = (1 + optimizer.returns).cumprod()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Performance', 'Rolling Volatility (30-Day)',
                           'Drawdown Analysis', 'Return Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_cumulative.index, y=portfolio_cumulative.values,
                mode='lines', name='Portfolio', line=dict(color='blue', width=4),
                hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        colors = px.colors.qualitative.Set1
        for i, ticker in enumerate(optimizer.tickers):
            fig.add_trace(
                go.Scatter(
                    x=individual_cumulative.index, y=individual_cumulative[ticker].values,
                    mode='lines', name=ticker, line=dict(color=colors[i % len(colors)]),
                    opacity=0.6,
                    hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index, y=rolling_vol.values,
                mode='lines', name='30-Day Rolling Volatility',
                line=dict(color='red', width=2), fill='tonexty',
                hovertemplate='<b>Rolling Volatility</b><br>Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown.values,
                mode='lines', fill='tozeroy', name='Drawdown',
                line=dict(color='red', width=2), fillcolor='rgba(255, 0, 0, 0.3)',
                hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=portfolio_returns * 100, nbinsx=50,
                name='Daily Returns', marker=dict(color='blue', opacity=0.7),
                hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, template='plotly_white')
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance analytics: {str(e)}")
        return go.Figure()


def create_capm_analysis_chart(capm_metrics: Dict[str, Dict[str, float]]) -> Optional[go.Figure]:
    """Enhanced CAPM analysis visualization."""
    if not capm_metrics:
        return None
    
    try:
        tickers = list(capm_metrics.keys())
        betas = [capm_metrics[t]['beta'] for t in tickers]
        alphas = [capm_metrics[t]['alpha'] * 100 for t in tickers]
        r_squared = [capm_metrics[t]['r_squared'] for t in tickers]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Beta Analysis', 'Alpha Analysis', 
                           'Systematic vs Idiosyncratic Risk', 'CAPM Summary'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        colors = px.colors.qualitative.Set1
        fig.add_trace(
            go.Bar(
                x=tickers, y=betas, name='Beta', marker=dict(color=colors),
                text=[f'{b:.2f}' for b in betas], textposition='auto',
                hovertemplate='<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        alpha_colors = ['green' if a > 0 else 'red' for a in alphas]
        fig.add_trace(
            go.Bar(
                x=tickers, y=alphas, name='Alpha', marker=dict(color=alpha_colors),
                text=[f'{a:.2f}%' for a in alphas], textposition='auto',
                hovertemplate='<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        sys_risk = [capm_metrics[t]['systematic_risk'] * 100 for t in tickers]
        idio_risk = [capm_metrics[t]['idiosyncratic_risk'] * 100 for t in tickers]
        
        fig.add_trace(
            go.Scatter(
                x=sys_risk, y=idio_risk, mode='markers+text',
                text=tickers, textposition="top center",
                marker=dict(
                    size=[max(10, r*30) for r in r_squared], 
                    color=r_squared, colorscale='Viridis', showscale=True,
                    colorbar=dict(title="R²", x=0.48)
                ),
                name='Risk Decomposition',
                hovertemplate='<b>%{text}</b><br>Systematic Risk: %{x:.2f}%<br>Idiosyncratic Risk: %{y:.2f}%<br>R²: %{marker.color:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
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
                header=dict(
                    values=['Asset', 'Beta', 'Alpha', 'Expected Return', 'R²'],
                    fill_color='lightblue', font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lightgray', font=dict(size=10)
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        return fig
        
    except Exception as e:
        st.error(f"Error creating CAPM analysis chart: {str(e)}")
        return None


# ================== USAGE EXAMPLE ==================

if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(tickers, lookback_years=3)
    
    # Fetch data
    success, error = optimizer.fetch_data()
    
    if success:
        # Optimize portfolio
        result = optimizer.optimize_portfolio(
            method='max_sharpe',
            include_risk_free=True
        )
        
        if result['success']:
            print("Optimization successful!")
            print(f"Expected Return: {result['expected_return']:.2%}")
            print(f"Volatility: {result['volatility']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            
            # Generate visualizations
            frontier_plot = create_efficient_frontier_plot_fixed(optimizer, result, include_risk_free=True)
            composition_chart = create_portfolio_composition_chart(
                optimizer.tickers, 
                result['weights'], 
                result.get('rf_weight')
            )
            
        else:
            print(f"Optimization failed: {result['error']}")
    else:
        print(f"Data fetching failed: {error}")

create_efficient_frontier_plot = create_efficient_frontier_plot_fixed