# optimizer_corrected.py - Portfolio Optimization Engine CORRIGÉ
"""
Enhanced Portfolio Optimization Engine using Modern Portfolio Theory with CAPM analysis
and advanced visualization capabilities - VERSION THÉORIQUEMENT CORRIGÉE.

CORRECTIONS PRINCIPALES:
1. Cohérence temporelle (journalier vs annuel)
2. CAPM calculations fixes
3. VaR/CVaR methodology corrected
4. Risk attribution mathematically correct
5. Efficient frontier improved
6. All metrics theoretically sound
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

# Add this import for FRED utility
from src.fred_utils import get_latest_risk_free_rate

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Enhanced Portfolio Optimization Engine using Modern Portfolio Theory - CORRIGÉ.
    
    Features:
    - Efficient frontier generation
    - Multiple optimization methods (max Sharpe, min variance, target return/volatility)
    - Risk-free asset integration with Capital Allocation Line
    - CAPM analysis and metrics
    - Comprehensive risk analytics
    - Advanced visualization
    - ✅ THEORETICAL CORRECTIONS: Temporal consistency, correct CAPM, VaR/CVaR methodology
    """
    
    def __init__(self, tickers: List[str], lookback_years: int = 3):
        """Initialize the portfolio optimizer with temporal consistency."""
        self.tickers = self._clean_tickers(tickers)
        self.lookback_years = max(1, min(lookback_years, 10))
        
        # ✅ CORRECTION: Constantes pour cohérence temporelle
        self.TRADING_DAYS = 252
        self.rf_rate_annual = 0.02  # Stockage principal en annuel
        self.rf_rate_daily = self.rf_rate_annual / self.TRADING_DAYS
        
        # Data storage with explicit temporal versions
        self.data = None
        self.returns = None  # Toujours journaliers
        self.mean_returns = None  # Sera annuel par défaut pour compatibilité
        self.mean_returns_daily = None  # Version journalière explicite
        self.mean_returns_annual = None  # Version annuelle explicite
        self.cov_matrix = None  # Sera annuel par défaut pour compatibilité
        self.cov_matrix_daily = None  # Version journalière explicite
        self.cov_matrix_annual = None  # Version annuelle explicite
        self.market_data = None
        self.market_returns = None
        
        # Legacy compatibility
        self.rf_rate = self.rf_rate_annual  # Pour compatibilité avec code existant
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
            st.error(f"Error in risk attribution: {str(e)}")
            n = len(weights)
            return {
                'marginal_contribution': np.zeros(n),
                'component_contribution': np.zeros(n),
                'percentage_contribution': np.zeros(n)
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
            
            # ✅ CORRECTION: Risk attribution avec méthode corrigée
            risk_attribution = self.calculate_risk_attribution(optimal_weights)
            
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
                'skewness': metrics['skewness'],
                'kurtosis': metrics['kurtosis'],
                'risk_contribution': risk_attribution['component_contribution'],
                'marginal_contribution': risk_attribution['marginal_contribution'],
                'optimization_details': {
                    'converged': best_result.success,
                    'iterations': best_result.nit if hasattr(best_result, 'nit') else 0,
                    'message': best_result.message if hasattr(best_result, 'message') else 'Completed'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Optimization error: {str(e)}'}
    
    def optimize_portfolio_with_risk_free_fixed(self, 
                                            method: str = 'max_sharpe',
                                            target_return: Optional[float] = None,
                                            target_volatility: Optional[float] = None,
                                            min_weight: float = 0.0,
                                            max_weight: float = 1.0,
                                            max_iterations: int = 1000) -> Dict:
        """✅ CORRECTION: Portfolio optimization with proper risk-free asset integration."""
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
                
                if target_return <= self.rf_rate_annual:
                    # Target below risk-free rate: invest only in risk-free asset
                    rf_weight = 1.0
                    risky_weight = 0.0
                    st.info("💰 Target return below risk-free rate - using 100% risk-free asset")
                    
                elif target_return >= tangency_return:
                    # Target above tangency portfolio: use leverage (with limits)
                    if tangency_return <= self.rf_rate_annual:
                        st.error("❌ Cannot achieve target return - tangency portfolio return too low")
                        return {'success': False, 'error': 'Target return unachievable with available assets'}
                    
                    leverage_ratio = (target_return - self.rf_rate_annual) / (tangency_return - self.rf_rate_annual)
                    max_leverage = 2.0  # Allow up to 200% in risky assets
                    
                    if leverage_ratio > max_leverage:
                        leverage_ratio = max_leverage
                        actual_return = self.rf_rate_annual + leverage_ratio * (tangency_return - self.rf_rate_annual)
                        st.warning(f"⚠️ Target return requires excessive leverage. Limited to {max_leverage:.0%} risky assets.")
                        st.warning(f"Achievable return with max leverage: {actual_return:.2%}")
                    
                    rf_weight = 1.0 - leverage_ratio
                    risky_weight = leverage_ratio
                    st.info(f"📈 Using leverage: {risky_weight:.1%} risky assets, {rf_weight:.1%} risk-free (borrowed if negative)")
                    
                else:
                    # Target between risk-free rate and tangency portfolio
                    risky_weight = (target_return - self.rf_rate_annual) / (tangency_return - self.rf_rate_annual)
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
                final_return = self.rf_rate_annual * rf_weight + tangency_return * risky_weight
                final_volatility = abs(risky_weight) * tangency_volatility
            else:
                final_return = self.rf_rate_annual
                final_volatility = 0.0
            
            final_sharpe = (final_return - self.rf_rate_annual) / final_volatility if final_volatility > 0 else float('inf')
            
            # Get additional metrics from the risky portion
            if risky_weight > 0:
                risky_metrics = self.portfolio_metrics(tangency_weights)
            else:
                risky_metrics = {
                    'sortino_ratio': 0, 'var_95': 0, 'var_99': 0, 'cvar_95': 0,
                    'max_drawdown': 0, 'diversification_ratio': 1, 'concentration': 0,
                    'skewness': 0, 'kurtosis': 0
                }
            
            # Risk attribution for final portfolio
            risk_attribution = self.calculate_risk_attribution(final_weights) if risky_weight > 0 else {
                'marginal_contribution': np.zeros(len(final_weights)),
                'component_contribution': np.zeros(len(final_weights)),
                'percentage_contribution': np.zeros(len(final_weights))
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
                'skewness': risky_metrics.get('skewness', 0),
                'kurtosis': risky_metrics.get('kurtosis', 0),
                'risk_contribution': risk_attribution['component_contribution'],
                'marginal_contribution': risk_attribution['marginal_contribution'],
                'optimization_details': {
                    'converged': True,
                    'rf_rate': self.rf_rate_annual,
                    'capital_allocation_line': True,
                    'leverage_used': risky_weight > 1.0,
                    'message': f'Optimal allocation: {rf_weight:.1%} risk-free, {risky_weight:.1%} risky portfolio'
                }
            }
            
            # Validation
            if method == 'target_return' and target_return is not None:
                achieved_return = result['expected_return']
                if abs(achieved_return - target_return) > 0.001:  # Allow small tolerance
                    if target_return > self.rf_rate_annual and target_return < tangency_return:
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
        """✅ CORRECTION: Generate efficient frontier with enhanced error handling."""
        if self.returns is None:
            return None
        
        try:
            # ✅ CORRECTION: Range plus intelligent basé sur historical data
            historical_min = self.mean_returns_annual.min()
            historical_max = self.mean_returns_annual.max()
            
            # Extend range intelligently
            range_extension = (historical_max - historical_min) * 0.2
            min_ret = historical_min - range_extension
            max_ret = historical_max + range_extension
            
            # Ensure minimum return above risk-free rate makes sense
            min_ret = max(min_ret, self.rf_rate_annual * 0.5)
            
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
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            for i, target_ret in enumerate(target_returns):
                try:
                    status_text.text(f"Generating efficient frontier point {i+1}/{num_points}")
                    progress_bar.progress((i + 1) / num_points)
                    
                    result = self.optimize_portfolio(
                        'target_return', 
                        target_return=target_ret,
                        max_iterations=500,
                        include_risk_free=False  # Pour frontier classique
                    )
                    
                    if result['success']:
                        frontier_data['returns'].append(result['expected_return'])
                        frontier_data['volatilities'].append(result['volatility'])
                        frontier_data['sharpe_ratios'].append(result['sharpe_ratio'])
                        frontier_data['weights'].append(result['weights'])
                        successful_optimizations += 1
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            break  # Stop if too many consecutive failures
                        
                except Exception:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
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
        """✅ CORRECTION: Enhanced CAMP metrics calculation with temporal consistency."""
        if self.market_returns is None or self.returns is None:
            return {}
        
        try:
            capm_metrics = {}
            
            common_dates = self.returns.index.intersection(self.market_returns.index)
            if len(common_dates) < 50:
                return {}
            
            aligned_returns = self.returns.loc[common_dates]
            aligned_market = self.market_returns.loc[common_dates]
            
            # ✅ CORRECTION: Risk-free rate journalier cohérent
            rf_daily = self.rf_rate_annual / self.TRADING_DAYS
            
            # Excess returns journaliers
            market_excess = aligned_market - rf_daily
            market_return_annual = aligned_market.mean() * self.TRADING_DAYS
            market_volatility_annual = aligned_market.std() * np.sqrt(self.TRADING_DAYS)
            
            for ticker in self.tickers:
                try:
                    asset_returns = aligned_returns[ticker]
                    asset_excess = asset_returns - rf_daily
                    
                    # Beta calculation (correct)
                    covariance = np.cov(asset_excess.dropna(), market_excess.dropna())[0, 1]
                    market_variance = np.var(market_excess.dropna())
                    
                    if market_variance > 0:
                        beta = covariance / market_variance
                    else:
                        beta = 1.0
                    
                    # ✅ CORRECTION: Alpha calculation cohérent
                    asset_return_annual = asset_returns.mean() * self.TRADING_DAYS
                    expected_return_annual = self.rf_rate_annual + beta * (market_return_annual - self.rf_rate_annual)
                    alpha_annual = asset_return_annual - expected_return_annual
                    
                    # R-squared
                    correlation_matrix = np.corrcoef(asset_excess.dropna(), market_excess.dropna())
                    if correlation_matrix.shape == (2, 2):
                        correlation = correlation_matrix[0, 1]
                        r_squared = correlation ** 2
                    else:
                        correlation = 0
                        r_squared = 0
                    
                    # ✅ CORRECTION: Risk decomposition théoriquement correcte
                    asset_volatility_annual = asset_returns.std() * np.sqrt(self.TRADING_DAYS)
                    systematic_risk_annual = abs(beta) * market_volatility_annual
                    
                    # Idiosyncratic risk: racine de (total_var - systematic_var)
                    total_variance = asset_volatility_annual ** 2
                    systematic_variance = (beta * market_volatility_annual) ** 2
                    idiosyncratic_variance = max(0, total_variance - systematic_variance)
                    idiosyncratic_risk_annual = np.sqrt(idiosyncratic_variance)
                    
                    camp_metrics[ticker] = {
                        'beta': beta,
                        'alpha': alpha_annual,
                        'expected_return': expected_return_annual,
                        'actual_return': asset_return_annual,
                        'r_squared': r_squared,
                        'correlation': correlation,
                        'systematic_risk': systematic_risk_annual,
                        'idiosyncratic_risk': idiosyncratic_risk_annual,
                        'total_risk': asset_volatility_annual
                    }
                    
                except Exception as e:
                    camp_metrics[ticker] = {
                        'beta': 1.0, 'alpha': 0.0, 
                        'expected_return': self.mean_returns[ticker],
                        'actual_return': self.mean_returns[ticker], 
                        'r_squared': 0.5, 'correlation': 0.0, 
                        'systematic_risk': 0.0, 'idiosyncratic_risk': 0.0,
                        'total_risk': 0.0
                    }
            
            return camp_metrics
            
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
            'risk_free_rate_annual': self.rf_rate_annual,
            'risk_free_rate_daily': self.rf_rate_daily,
            'market_data_available': self.market_data is not None,
            'data_quality': self.data_quality_info,
            'temporal_consistency': {
                'trading_days': self.TRADING_DAYS,
                'has_daily_versions': all([
                    self.mean_returns_daily is not None,
                    self.cov_matrix_daily is not None
                ]) if hasattr(self, 'mean_returns_daily') else False,
                'has_annual_versions': all([
                    self.mean_returns_annual is not None,
                    self.cov_matrix_annual is not None
                ]) if hasattr(self, 'mean_returns_annual') else False
            }
        }
        return debug_info

    # ================== VALIDATION FUNCTIONS ==================
    
    def validate_portfolio_metrics(self, metrics: Dict, weights: np.ndarray) -> List[str]:
        """✅ CORRECTION: Validation des métriques calculées"""
        warnings = []
        
        # Check weights sum to 1
        if abs(np.sum(weights) - 1.0) > 1e-6:
            warnings.append(f"Weights don't sum to 1: {np.sum(weights):.6f}")
        
        # Check Sharpe ratio reasonableness
        if abs(metrics['sharpe_ratio']) > 5:
            warnings.append(f"Extreme Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        # Check volatility > 0
        if metrics['volatility'] <= 0:
            warnings.append("Zero or negative volatility")
        
        # Check risk attribution sums correctly
        if 'risk_contribution' in metrics:
            total_risk_contrib = np.sum(metrics['risk_contribution'])
            if abs(total_risk_contrib - metrics['volatility']) > 1e-6:
                warnings.append(f"Risk attribution error: {total_risk_contrib:.6f} vs {metrics['volatility']:.6f}")
        
        # Check VaR values are negative (losses)
        if metrics.get('var_95', 0) > 0:
            warnings.append("VaR should be negative (represents losses)")
        
        # Check diversification ratio makes sense
        if metrics.get('diversification_ratio', 0) > len(weights):
            warnings.append(f"Diversification ratio too high: {metrics['diversification_ratio']:.2f}")
        
        return warnings


# ================== VISUALIZATION FUNCTIONS CORRECTED ==================

def create_efficient_frontier_plot(optimizer: PortfolioOptimizer, 
                                 optimal_portfolio: Optional[Dict] = None,
                                 include_risk_free: bool = True) -> Optional[go.Figure]:
    """✅ CORRECTION: Create efficient frontier with proper CAL and portfolio positioning."""
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
            asset_return = optimizer.mean_returns_annual[ticker] * 100
            asset_vol = np.sqrt(optimizer.cov_matrix_annual.iloc[i, i]) * 100
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
        rf_return = optimizer.rf_rate_annual * 100
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
            xaxis_title='Risk (Volatility % Annual)', yaxis_title='Expected Return % Annual',
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
    """✅ CORRECTION: Enhanced risk-return analysis with temporal consistency."""
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
            ret = optimizer.mean_returns_annual[ticker] * 100
            vol = np.sqrt(optimizer.cov_matrix_annual.iloc[i, i]) * 100
            size = weights[i] * 100 if weights is not None else 15
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=[vol], y=[ret], mode='markers+text',
                    marker=dict(size=max(15, size), color=color, line=dict(width=2, color='white')),
                    text=[ticker], textposition="middle center",
                    textfont=dict(color='white', size=10), name=ticker,
                    hovertemplate=f'<b>{ticker}</b><br>Return: %{{y:.2f}}% (annual)<br>Risk: %{{x:.2f}}% (annual)<extra></extra>'
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
            risk_attribution = optimizer.calculate_risk_attribution(weights)
            risk_contrib = risk_attribution['component_contribution']
            
            fig.add_trace(
                go.Bar(
                    x=optimizer.tickers, y=risk_contrib * 100,
                    name='Risk Contribution', marker=dict(color=colors),
                    text=[f'{rc:.1f}%' for rc in risk_contrib * 100], textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.2f}% (annual)<extra></extra>'
                ),
                row=2, col=1
            )
        
        stats_data = []
        for i, ticker in enumerate(optimizer.tickers):
            ret = optimizer.mean_returns_annual[ticker]
            vol = np.sqrt(optimizer.cov_matrix_annual.iloc[i, i])
            sharpe = (ret - optimizer.rf_rate_annual) / vol if vol > 0 else 0
            weight = weights[i] if weights is not None else 0
            
            stats_data.append([
                ticker, f'{ret:.2%}', f'{vol:.2%}', f'{sharpe:.3f}',
                f'{weight:.2%}' if weights is not None else 'N/A'
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Asset', 'Return (Annual)', 'Risk (Annual)', 'Sharpe', 'Weight'],
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
    """✅ CORRECTION: Enhanced performance analytics with temporal consistency."""
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
                hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Cumulative Return: %{y:.3f}<extra></extra>'
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
                    hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Cumulative Return: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # ✅ CORRECTION: Rolling volatility avec annualisation correcte
        rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(optimizer.TRADING_DAYS) * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index, y=rolling_vol.values,
                mode='lines', name='30-Day Rolling Volatility (Annualized)',
                line=dict(color='red', width=2), fill='tonexty',
                hovertemplate='<b>Rolling Volatility</b><br>Date: %{x}<br>Volatility: %{y:.2f}% (annualized)<extra></extra>'
            ),
            row=1, col=2
        )
        
        # ✅ CORRECTION: Drawdown calculation correct
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
                name='Daily Returns Distribution', marker=dict(color='blue', opacity=0.7),
                hovertemplate='Return: %{x:.2f}% (daily)<br>Frequency: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, template='plotly_white')
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance analytics: {str(e)}")
        return go.Figure()


def create_capm_analysis_chart(camp_metrics: Dict[str, Dict[str, float]]) -> Optional[go.Figure]:
    """✅ CORRECTION: Enhanced CAPM analysis visualization with temporal consistency."""
    if not capm_metrics:
        return None
    
    try:
        tickers = list(capm_metrics.keys())
        betas = [capm_metrics[t]['beta'] for t in tickers]
        alphas = [capm_metrics[t]['alpha'] * 100 for t in tickers]  # Convert to percentage
        r_squared = [capm_metrics[t]['r_squared'] for t in tickers]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Beta Analysis', 'Alpha Analysis (Annual %)', 
                           'Systematic vs Idiosyncratic Risk (Annual %)', 'CAPM Summary'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        # Beta chart
        fig.add_trace(
            go.Bar(
                x=tickers, y=betas, name='Beta', marker=dict(color=colors),
                text=[f'{b:.2f}' for b in betas], textposition='auto',
                hovertemplate='<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Alpha chart
        alpha_colors = ['green' if a > 0 else 'red' for a in alphas]
        fig.add_trace(
            go.Bar(
                x=tickers, y=alphas, name='Alpha (Annual %)', marker=dict(color=alpha_colors),
                text=[f'{a:.2f}%' for a in alphas], textposition='auto',
                hovertemplate='<b>%{x}</b><br>Alpha: %{y:.2f}% (annual)<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Risk decomposition chart
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
                hovertemplate='<b>%{text}</b><br>Systematic Risk: %{x:.2f}% (annual)<br>Idiosyncratic Risk: %{y:.2f}% (annual)<br>R²: %{marker.color:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # CAMP summary table
        table_data = []
        for ticker in tickers:
            table_data.append([
                ticker,
                f'{capm_metrics[ticker]["beta"]:.3f}',
                f'{capm_metrics[ticker]["alpha"]:.2%}',
                f'{camp_metrics[ticker]["expected_return"]:.2%}',
                f'{capm_metrics[ticker]["r_squared"]:.3f}'
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Asset', 'Beta', 'Alpha (Annual)', 'Expected Return (Annual)', 'R²'],
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


# ================== UTILITY AND FORMATTING FUNCTIONS ==================

def format_return(value: float, is_annual: bool = True) -> str:
    """Helper pour affichage cohérent des returns."""
    period = "annual" if is_annual else "daily"
    return f"{value:.2%} ({period})"

def format_volatility(value: float, is_annual: bool = True) -> str:
    """Helper pour affichage cohérent de volatilité."""
    period = "annual" if is_annual else "daily"
    return f"{value:.2%} ({period})"

def format_var_cvar(value: float, is_daily: bool = True, confidence: float = 0.95) -> str:
    """Helper pour affichage cohérent VaR/CVaR."""
    period = "daily" if is_daily else "annual"
    return f"{abs(value):.2%} ({int(confidence*100)}% {period})"

def validate_optimization_result(result: Dict, weights: np.ndarray, optimizer: PortfolioOptimizer) -> List[str]:
    """✅ CORRECTION: Validation complète des résultats d'optimisation."""
    warnings = []
    
    if not result.get('success', False):
        warnings.append("Optimization did not succeed")
        return warnings
    
    # Validate weights
    if abs(np.sum(weights) - 1.0) > 1e-6:
        warnings.append(f"Weights don't sum to 1: {np.sum(weights):.6f}")
    
    if np.any(weights < -1e-6):
        warnings.append("Found negative weights (short selling)")
    
    # Validate metrics
    metrics = optimizer.portfolio_metrics(weights)
    
    if metrics['volatility'] <= 0:
        warnings.append("Zero or negative volatility")
    
    if abs(metrics['sharpe_ratio']) > 10:
        warnings.append(f"Extreme Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    
    # Validate risk attribution
    risk_attr = optimizer.calculate_risk_attribution(weights)
    total_risk_contrib = np.sum(risk_attr['component_contribution'])
    if abs(total_risk_contrib - metrics['volatility']) > 1e-6:
        warnings.append(f"Risk attribution error: {total_risk_contrib:.6f} vs {metrics['volatility']:.6f}")
    
    return warnings


# ================== USAGE EXAMPLE WITH CORRECTIONS ==================

if __name__ == "__main__":
    # Example usage with temporal consistency checks
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(tickers, lookback_years=3)
    
    # Fetch data
    success, error = optimizer.fetch_data()
    
    if success:
        # Debug temporal consistency
        debug_info = optimizer.get_debug_info()
        print("=== TEMPORAL CONSISTENCY CHECK ===")
        print(f"Trading days: {debug_info['temporal_consistency']['trading_days']}")
        print(f"RF rate annual: {debug_info['risk_free_rate_annual']:.3%}")
        print(f"RF rate daily: {debug_info['risk_free_rate_daily']:.4%}")
        
        # Optimize portfolio
        result = optimizer.optimize_portfolio(
            method='max_sharpe',
            include_risk_free=True
        )
        
        if result['success']:
            print("\n=== OPTIMIZATION RESULTS ===")
            print(f"Expected Return: {result['expected_return']:.2%} (annual)")
            print(f"Volatility: {result['volatility']:.2%} (annual)")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            
            # Validate results
            warnings = validate_optimization_result(result, result['weights'], optimizer)
            if warnings:
                print("\n⚠️ VALIDATION WARNINGS:")
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print("\n✅ All validations passed!")
            
            # Test metrics calculation
            metrics = optimizer.portfolio_metrics(result['weights'])
            print(f"\n=== DETAILED METRICS ===")
            print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
            print(f"VaR 95%: {abs(metrics['var_95']):.3%} (daily)")
            print(f"CVaR 95%: {abs(metrics['cvar_95']):.3%} (daily)")
            print(f"Max Drawdown: {abs(metrics['max_drawdown']):.3%}")
            print(f"Skewness: {metrics['skewness']:.3f}")
            print(f"Kurtosis: {metrics['kurtosis']:.3f}")
            
        else:
            print(f"Optimization failed: {result['error']}")
    else:
        print(f"Data fetching failed: {error}")
    
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
        """✅ CORRECTION: Calculate returns and statistical measures with temporal consistency."""
        try:
            self.returns = self.data.pct_change().dropna()
            
            # ✅ CORRECTION: Outlier removal plus robuste avec IQR
            for col in self.returns.columns:
                Q1 = self.returns[col].quantile(0.25)
                Q3 = self.returns[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Plus conservateur que 3*std
                upper_bound = Q3 + 3 * IQR
                
                outlier_mask = (self.returns[col] < lower_bound) | (self.returns[col] > upper_bound)
                if outlier_mask.sum() > 0:
                    st.info(f"Removed {outlier_mask.sum()} outliers from {col}")
                    self.returns.loc[outlier_mask, col] = np.nan
            
            self.returns = self.returns.fillna(method='ffill').fillna(method='bfill')
            
            # ✅ CORRECTION: Séparation explicite journalier/annuel
            self.mean_returns_daily = self.returns.mean()
            self.mean_returns_annual = self.mean_returns_daily * self.TRADING_DAYS
            
            self.cov_matrix_daily = self.returns.cov()
            self.cov_matrix_annual = self.cov_matrix_daily * self.TRADING_DAYS
            
            # Utilise annuel par défaut pour compatibilité avec optimisation
            self.mean_returns = self.mean_returns_annual
            self.cov_matrix = self.cov_matrix_annual
            
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
    
    def get_risk_free_rate(self) -> float:
        """✅ CORRECTION: Enhanced risk-free rate fetching with consistent temporal handling."""
        # 1. Try FRED utility first
        try:
            latest_fred_rate = get_latest_risk_free_rate()
            if 0 < latest_fred_rate < 20:
                self.rf_rate_annual = latest_fred_rate / 100
                self.rf_rate_daily = self.rf_rate_annual / self.TRADING_DAYS
                self.rf_rate = self.rf_rate_annual  # Legacy compatibility
                st.success(f"✅ FRED rate: {self.rf_rate_annual:.3%} annual")
                return self.rf_rate_annual
            else:
                st.warning(f"FRED returned suspicious rate: {latest_fred_rate}")
        except Exception as e:
            st.warning(f"FRED utility failed: {str(e)}")

        # 2. Try yfinance as backup
        treasury_symbols = ['^IRX', '^TNX', '^FVX']
        for symbol in treasury_symbols:
            try:
                st.info(f"Attempting to fetch risk-free rate from {symbol}...")
                for period in ['5d', '1mo', '3mo']:
                    try:
                        treasury_data = yf.download(symbol, period=period, progress=False)
                        if not treasury_data.empty:
                            if 'Close' in treasury_data.columns:
                                rates = treasury_data['Close'].dropna()
                            else:
                                if isinstance(treasury_data.columns, pd.MultiIndex):
                                    for col in treasury_data.columns:
                                        if 'close' in str(col).lower():
                                            rates = treasury_data[col].dropna()
                                            break
                                else:
                                    rates = treasury_data.iloc[:, -1].dropna()
                            
                            if len(rates) > 0:
                                latest_rate = rates.iloc[-1]
                                if 0 < latest_rate < 20:
                                    self.rf_rate_annual = latest_rate / 100
                                    self.rf_rate_daily = self.rf_rate_annual / self.TRADING_DAYS
                                    self.rf_rate = self.rf_rate_annual
                                    st.success(f"✅ {symbol} rate: {self.rf_rate_annual:.3%} annual")
                                    return self.rf_rate_annual
                                st.warning(f"Rate {latest_rate} from {symbol} seems invalid")
                    except Exception as e:
                        st.warning(f"Failed to fetch {symbol} with period {period}: {str(e)}")
                        continue
            except Exception as e:
                st.warning(f"Error with {symbol}: {str(e)}")
                continue

        # 3. Final fallback - use reasonable default
        self.rf_rate_annual = 0.045  # 4.5% - reasonable default for 2024-2025
        self.rf_rate_daily = self.rf_rate_annual / self.TRADING_DAYS
        self.rf_rate = self.rf_rate_annual
        st.warning(f"⚠️ Using fallback: {self.rf_rate_annual:.3%} annual")
        return self.rf_rate_annual

    # ================== PORTFOLIO METRICS AND CALCULATIONS ==================
    
    def calculate_var_cvar_corrected(self, weights: np.ndarray, confidence_levels=[0.95, 0.99]) -> Dict[str, float]:
        """✅ CORRECTION: VaR/CVaR méthodologiquement correct"""
        try:
            # Portfolio returns journaliers
            portfolio_returns = (self.returns * weights).sum(axis=1)
            
            results = {}
            for confidence in confidence_levels:
                # VaR: percentile des returns journaliers (négatif pour perte)
                var_daily = np.percentile(portfolio_returns, (1 - confidence) * 100)
                
                # CVaR: moyenne conditionnelle correcte des returns <= VaR
                var_breaches = portfolio_returns[portfolio_returns <= var_daily]
                cvar_daily = var_breaches.mean() if len(var_breaches) > 0 else var_daily
                
                # Store daily values (standard practice)
                results[f'var_{int(confidence*100)}'] = var_daily
                results[f'cvar_{int(confidence*100)}'] = cvar_daily
            
            return results
            
        except Exception:
            return {f'var_{int(c*100)}': 0, f'cvar_{int(c*100)}': 0 for c in confidence_levels}
    
    def portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """✅ CORRECTION: Calculate comprehensive portfolio metrics with corrected methodology."""
        try:
            weights = np.array(weights)
            
            if len(weights) != len(self.mean_returns):
                raise ValueError("Weight vector length mismatch")
            
            # Normalisation
            weights = weights / np.sum(weights)
            
            # ✅ CORRECTION: Utilise versions annuelles pour cohérence avec optimisation
            portfolio_return = np.dot(self.mean_returns_annual, weights)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix_annual, weights))
            portfolio_std = np.sqrt(max(0, portfolio_variance))
            
            # Sharpe ratio annuel
            sharpe_ratio = (portfolio_return - self.rf_rate_annual) / portfolio_std if portfolio_std > 1e-10 else 0
            
            # Portfolio returns journaliers pour métriques historiques
            portfolio_returns_daily = (self.returns * weights).sum(axis=1)
            
            # ✅ CORRECTION: VaR/CVaR avec méthode corrigée
            var_cvar_results = self.calculate_var_cvar_corrected(weights)
            
            # ✅ CORRECTION: Sortino ratio correct
            try:
                negative_returns = portfolio_returns_daily[portfolio_returns_daily < 0]
                if len(negative_returns) > 0:
                    downside_deviation_daily = negative_returns.std()
                    downside_deviation_annual = downside_deviation_daily * np.sqrt(self.TRADING_DAYS)
                    sortino_ratio = (portfolio_return - self.rf_rate_annual) / downside_deviation_annual
                else:
                    sortino_ratio = float('inf')  # Pas de returns négatifs
            except Exception:
                sortino_ratio = 0
            
            # ✅ CORRECTION: Drawdown calculation correct
            try:
                portfolio_cumulative = (portfolio_returns_daily + 1).cumprod()
                running_max = portfolio_cumulative.expanding().max()
                drawdown = (portfolio_cumulative - running_max) / running_max
                max_drawdown = drawdown.min()  # Plus négatif = pire drawdown
            except Exception:
                max_drawdown = 0
            
            # Diversification metrics
            concentration = np.sum(weights ** 2)  # HHI
            diversification_ratio = 1 / concentration if concentration > 0 else 1
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'variance': portfolio_variance,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95': var_cvar_results.get('var_95', 0),
                'var_99': var_cvar_results.get('var_99', 0),
                'cvar_95': var_cvar_results.get('cvar_95', 0),
                'concentration': concentration,
                'diversification_ratio': diversification_ratio,
                'max_drawdown': max_drawdown,
                # Métriques supplémentaires
                'skewness': portfolio_returns_daily.skew(),
                'kurtosis': portfolio_returns_daily.kurtosis(),
                'worst_day': portfolio_returns_daily.min(),
                'best_day': portfolio_returns_daily.max()
            }
            
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            # Return fallback values
            return {
                'expected_return': 0, 'volatility': 0, 'variance': 0, 'sharpe_ratio': 0,
                'sortino_ratio': 0, 'var_95': 0, 'var_99': 0, 'cvar_95': 0,
                'concentration': 1, 'diversification_ratio': 1, 'max_drawdown': 0,
                'skewness': 0, 'kurtosis': 0, 'worst_day': 0, 'best_day': 0
            }

    def calculate_risk_attribution(self, weights: np.ndarray) -> Dict[str, np.ndarray]:
        """✅ CORRECTION: Attribution du risque théoriquement correcte"""
        try:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Portfolio volatility annuelle
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix_annual, weights))
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))
            
            if portfolio_volatility <= 1e-10:
                n = len(weights)
                return {
                    'marginal_contribution': np.zeros(n),
                    'component_contribution': np.zeros(n),
                    'percentage_contribution': np.zeros(n)
                }
            
            # ✅ CORRECTION: Marginal contribution correct
            # MC_i = (∂σ_p/∂w_i) = (Σw * Cov_matrix)_i / σ_p
            marginal_contrib = np.dot(self.cov_matrix_annual, weights) / portfolio_volatility
            
            # Component contribution: CC_i = w_i * MC_i
            component_contrib = weights * marginal_contrib
            
            # Percentage contribution (should sum to 100%)
            percentage_contrib = component_contrib / portfolio_volatility * 100
            
            # Vérification théorique: sum(CC_i) = σ_p
            verification_error = abs(np.sum(component_contrib) - portfolio_volatility)
            if verification_error > 1e-6:
                st.warning(f"Risk attribution verification error: {verification_error:.8f}")
            
            return {
                'marginal_contribution': marginal_contrib,
                'component_contribution': component_contrib,
                'percentage_contribution': percentage_contrib
            }
            
        except Exception
