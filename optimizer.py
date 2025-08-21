"""
optimizer.py — Core portfolio optimization engine used by pages/1_Portfolio_Optimizer.py

This module provides:
- PortfolioOptimizer: data ingestion, statistics, optimizers, risk metrics, CAPM, utilities
- Visualization helpers (Plotly):
    * create_efficient_frontier_plot
    * create_portfolio_composition_chart
    * create_risk_return_analysis
    * create_performance_analytics
    * create_capm_analysis_chart
- validate_optimization_result: post-optimization sanity checks

Author: QuantRisk Analytics
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("yfinance is required. Install with `pip install yfinance`. Original error: %s" % e)

try:
    from scipy import optimize as sco  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("scipy is required. Install with `pip install scipy`. Original error: %s" % e)

TRADING_DAYS = 252
EPS = 1e-10


# ------------------------------- Utility helpers ------------------------------- #

def _annualize_return(daily_ret: pd.Series | pd.DataFrame) -> pd.Series:
    """Annualize mean daily return.
    For a DataFrame, returns a Series for each column.
    """
    mu_d = daily_ret.mean()
    return (1 + mu_d) ** TRADING_DAYS - 1


def _annualize_cov(cov_daily: pd.DataFrame) -> pd.DataFrame:
    return cov_daily * TRADING_DAYS


def _to_numpy(x: Iterable[float]) -> np.ndarray:
    arr = np.array(list(x), dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _ridge_cov(cov: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
    cov = cov.copy()
    diag = np.eye(cov.shape[0]) * ridge
    cov.values[:] = cov.values + diag
    return cov


# ------------------------------- Core class ------------------------------- #

class PortfolioOptimizer:
    """End-to-end mean-variance optimizer with data, risk metrics & CAPM.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance tickers.
    lookback_years : int
        Number of trailing calendar years of daily history to load.
    """

    def __init__(self, tickers: List[str], lookback_years: int = 3) -> None:
        self.original_tickers: List[str] = [t.strip().upper() for t in tickers if str(t).strip()]
        self.tickers: List[str] = self.original_tickers.copy()  # will shrink to successful downloads
        self.lookback_years = int(lookback_years)

        # Data containers
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.mean_returns_annual: Optional[pd.Series] = None
        self.cov_matrix_annual: Optional[pd.DataFrame] = None

        self.failed_tickers: List[str] = []
        self.data_quality_info: Dict[str, Dict[str, float | int]] = {}

        # Risk-free cache
        self._rf_cache_annual: Optional[float] = None

        # Debug/meta
        self._start_date: Optional[pd.Timestamp] = None
        self._end_date: Optional[pd.Timestamp] = None

    # ----------------------------- Validation & data ---------------------------- #
    def validate_tickers(self) -> Tuple[List[str], List[str]]:
        """Basic validation on ticker formatting (alnum, .-^ allowed)."""
        import re
        valid, invalid = [], []
        pat = re.compile(r"^[A-Z0-9\-\.^]+$")
        for t in self.original_tickers:
            (valid if pat.match(t) else invalid).append(t)
        return valid, invalid

    def _compute_date_range(self) -> Tuple[str, str]:
        today = pd.Timestamp.today().normalize()
        start = today - pd.DateOffset(years=self.lookback_years) - pd.Timedelta(days=5)
        # 5-day buffer to ensure enough data around holidays
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    def fetch_data(self) -> Tuple[bool, Optional[str]]:
        """Download Adjusted Close from Yahoo Finance and build return matrix.

        Returns
        -------
        (success, error_info)
            success=True if at least two tickers return valid data.
            error_info is a message listing failures if any.
        """
        if len(self.original_tickers) < 2:
            return False, "Need at least two tickers."

        start, end = self._compute_date_range()

        # Download in one shot for alignment
        try:
            data = yf.download(self.original_tickers, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as e:  # pragma: no cover
            return False, f"Download failed: {e}"

        # Extract Adjusted Close robustly across shapes (Series/DataFrame/MultiIndex)
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                prices = data['Close'].copy()
            else:
                prices = data.copy()
        elif isinstance(data, pd.Series):
            # Convert to DataFrame to keep consistent downstream operations
            prices = data.to_frame(name=self.original_tickers[0])
        else:
            return False, "Unexpected data format from yfinance."

        # Normalize columns to list of tickers that actually downloaded
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        prices = prices.dropna(how='all')

        # Identify successes/failures
        available = [c for c in prices.columns if prices[c].dropna().shape[0] > TRADING_DAYS]
        self.failed_tickers = [t for t in self.original_tickers if t not in available]
        self.tickers = available

        if len(self.tickers) < 2:
            return False, ", ".join(sorted(set(self.failed_tickers))) or "No usable data."

        # Clean & align
        prices = prices[self.tickers].ffill().bfill()
        self.prices = prices
        self.returns = prices.pct_change().dropna(how='all')

        # Drop any column with too many NaNs after pct_change
        valid_cols = [c for c in self.returns.columns if self.returns[c].dropna().shape[0] > TRADING_DAYS // 2]
        self.returns = self.returns[valid_cols].dropna()
        self.tickers = list(self.returns.columns)

        if len(self.tickers) < 2:
            return False, "Insufficient overlapping history between tickers."

        # Stats
        cov_daily = self.returns.cov().fillna(0.0)
        cov_daily = _ridge_cov(cov_daily)  # numeric stability
        self.cov_matrix_annual = _annualize_cov(cov_daily)
        self.mean_returns_annual = _annualize_return(self.returns)

        self._start_date = self.returns.index.min()
        self._end_date = self.returns.index.max()

        # Data quality report
        self.data_quality_info = {}
        max_points = int(self.returns.shape[0])
        for t in self.tickers:
            col = self.returns[t]
            pts = int(col.dropna().shape[0])
            # Simple quality proxy: availability * non-zero variance
            var = float(col.var())
            quality = (pts / max(max_points, 1)) * (1.0 if var > 0 else 0.6)
            self.data_quality_info[t] = {
                "total_points": pts,
                "quality_score": float(quality),
            }

        return True, ", ".join(sorted(set(self.failed_tickers))) if self.failed_tickers else None

    # ----------------------------- Risk-free rate ----------------------------- #
    def get_risk_free_rate(self) -> float:
        """Fetch a robust proxy for the annualized risk-free rate.
        Strategy (fallbacks):
            1) ^IRX (13-week T-bill) latest close (% -> /100)
            2) ^TNX (10Y) latest close (% -> /100)
            3) Fallback 1.5%
        """
        if self._rf_cache_annual is not None:
            return self._rf_cache_annual

        tickers_order = ["^IRX", "^TNX"]
        for t in tickers_order:
            try:
                df = yf.download(t, period="6mo", interval="1d", auto_adjust=False, progress=False)
                if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                    last = float(df["Close"].dropna().iloc[-1])
                    # Yahoo quotes these in percentage points
                    rate = last / 100.0
                    if rate >= 0:
                        self._rf_cache_annual = float(rate)
                        return self._rf_cache_annual
            except Exception:
                pass
        # Fallback
        self._rf_cache_annual = 0.015
        return self._rf_cache_annual

    # ----------------------------- Portfolio maths ---------------------------- #
    def portfolio_metrics(self, weights: np.ndarray, rf_annual: Optional[float] = None) -> Dict[str, float]:
        """Compute standard portfolio metrics from weights.
        Returns expected_return (annual), volatility (annual), Sharpe (using rf if provided else 0).
        """
        w = _to_numpy(weights)
        mu = self.mean_returns_annual[self.tickers].values  # type: ignore
        cov = self.cov_matrix_annual.loc[self.tickers, self.tickers].values  # type: ignore

        exp_ret = float(w @ mu)
        vol = float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))
        rf = 0.0 if rf_annual is None else float(rf_annual)
        sharpe = (exp_ret - rf) / (vol + EPS)
        return {"expected_return": exp_ret, "volatility": vol, "sharpe_ratio": sharpe}

    def _risk_contributions(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (risk_contrib, marginal_contrib) as fractions that sum to 1 for risk_contrib."""
        w = _to_numpy(weights)
        cov = self.cov_matrix_annual.loc[self.tickers, self.tickers].values  # type: ignore
        port_var = max(w @ cov @ w, EPS)
        sigma = math.sqrt(port_var)
        # Marginal contribution: Σw / σ
        mrc = (cov @ w) / (sigma + EPS)
        # Total risk contribution: w * (Σw) / σ
        trc = (w * (cov @ w)) / (sigma + EPS)
        frac = trc / (sigma + EPS)
        frac = np.maximum(frac, 0.0)
        if frac.sum() > 0:
            frac = frac / frac.sum()
        return frac, mrc

    def _objective_sharpe(self, weights: np.ndarray, rf: float) -> float:
        m = self.portfolio_metrics(weights, rf)
        return -m["sharpe_ratio"]  # maximize Sharpe -> minimize negative

    def _objective_variance(self, weights: np.ndarray) -> float:
        w = _to_numpy(weights)
        cov = self.cov_matrix_annual.loc[self.tickers, self.tickers].values  # type: ignore
        return float(w @ cov @ w)

    def _constraints_sum1(self):
        return ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def _bounds(self, min_w: float, max_w: float) -> Tuple[Tuple[float, float], ...]:
        return tuple((min_w, max_w) for _ in self.tickers)

    def _solve_weights(
        self,
        x0: Optional[np.ndarray],
        objective,
        bounds,
        constraints,
    ) -> sco.OptimizeResult:
        if x0 is None:
            x0 = np.ones(len(self.tickers)) / len(self.tickers)
        return sco.minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )

    # ----------------------------- Optimizers ----------------------------- #
    def _tangency_weights(self, rf: float, min_w: float, max_w: float) -> Tuple[np.ndarray, sco.OptimizeResult]:
        cons = self._constraints_sum1()
        bnds = self._bounds(min_w, max_w)
        res = self._solve_weights(
            None,
            lambda w: self._objective_sharpe(w, rf),
            bnds,
            cons,
        )
        w = np.clip(res.x, min_w, max_w)
        # Re-normalize to sum to 1 subject to bounds feasibility
        if abs(w.sum()) > EPS:
            w = w / w.sum()
        return w, res

    def _min_variance_weights(self, min_w: float, max_w: float) -> Tuple[np.ndarray, sco.OptimizeResult]:
        cons = self._constraints_sum1()
        bnds = self._bounds(min_w, max_w)
        res = self._solve_weights(None, self._objective_variance, bnds, cons)
        w = np.clip(res.x, min_w, max_w)
        if abs(w.sum()) > EPS:
            w = w / w.sum()
        return w, res

    def _target_return_weights(self, target_ret: float, min_w: float, max_w: float) -> Tuple[np.ndarray, sco.OptimizeResult]:
        cons = (
            {'type': 'eq', 'fun': lambda w, tr=target_ret: (w @ self.mean_returns_annual[self.tickers].values) - tr},  # type: ignore
            *self._constraints_sum1(),
        )
        bnds = self._bounds(min_w, max_w)
        res = self._solve_weights(None, self._objective_variance, bnds, cons)
        w = np.clip(res.x, min_w, max_w)
        if abs(w.sum()) > EPS:
            w = w / w.sum()
        return w, res

    def _target_vol_weights(self, target_vol: float, min_w: float, max_w: float) -> Tuple[np.ndarray, sco.OptimizeResult]:
        # Penalize squared deviation from target volatility while maximizing expected return
        mu = self.mean_returns_annual[self.tickers].values  # type: ignore
        cov = self.cov_matrix_annual.loc[self.tickers, self.tickers].values  # type: ignore

        def obj(w: np.ndarray) -> float:
            w = _to_numpy(w)
            vol = math.sqrt(max(w @ cov @ w, 0.0))
            penalty = (vol - target_vol) ** 2
            return - (w @ mu) + 50.0 * penalty  # trade-off weight

        cons = self._constraints_sum1()
        bnds = self._bounds(min_w, max_w)
        res = self._solve_weights(None, obj, bnds, cons)
        w = np.clip(res.x, min_w, max_w)
        if abs(w.sum()) > EPS:
            w = w / w.sum()
        return w, res

    def optimize_portfolio(
        self,
        method: str = "max_sharpe",
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        include_risk_free: bool = True,
    ) -> Dict[str, object]:
        """Run the selected optimization and compute a rich result dict."""
        if self.mean_returns_annual is None or self.cov_matrix_annual is None:
            raise RuntimeError("Data not loaded. Call fetch_data() first.")

        rf = self.get_risk_free_rate() if include_risk_free else 0.0

        details: Dict[str, object] = {}

        if method == "max_sharpe":
            w_tan, res = self._tangency_weights(rf, min_weight, max_weight)
            details.update({"iterations": getattr(res, "nit", None), "converged": bool(res.success)})
            risky_weight = 1.0
            rf_weight = 0.0
            final_w = w_tan
        elif method == "min_variance":
            w_min, res = self._min_variance_weights(min_weight, max_weight)
            details.update({"iterations": getattr(res, "nit", None), "converged": bool(res.success)})
            risky_weight = 1.0
            rf_weight = 0.0
            final_w = w_min
        elif method == "target_return":
            if include_risk_free:
                # Best practice on CAL: scale tangency portfolio to hit target
                w_tan, res = self._tangency_weights(rf, min_weight, max_weight)
                m = self.portfolio_metrics(w_tan, rf)
                mu_tan = m["expected_return"]
                if target_return is None:
                    target_return = mu_tan
                risky_weight = (target_return - rf) / max(mu_tan - rf, EPS)
                rf_weight = 1.0 - risky_weight
                final_w = np.clip(w_tan * risky_weight, min_weight, max_weight)
                details.update({"iterations": getattr(res, "nit", None), "converged": bool(res.success)})
            else:
                if target_return is None:
                    target_return = float(self.mean_returns_annual.mean())
                w_tr, res = self._target_return_weights(target_return, min_weight, max_weight)
                risky_weight = 1.0
                rf_weight = 0.0
                final_w = w_tr
                details.update({"iterations": getattr(res, "nit", None), "converged": bool(res.success)})
        elif method == "target_volatility":
            if include_risk_free:
                w_tan, res = self._tangency_weights(rf, min_weight, max_weight)
                m = self.portfolio_metrics(w_tan, rf)
                vol_tan = m["volatility"]
                if target_volatility is None:
                    target_volatility = vol_tan
                risky_weight = target_volatility / max(vol_tan, EPS)
                rf_weight = 1.0 - risky_weight
                final_w = np.clip(w_tan * risky_weight, min_weight, max_weight)
                details.update({"iterations": getattr(res, "nit", None), "converged": bool(res.success)})
            else:
                if target_volatility is None:
                    target_volatility = float(np.sqrt(np.diag(self.cov_matrix_annual)).mean())  # type: ignore
                w_tv, res = self._target_vol_weights(target_volatility, min_weight, max_weight)
                risky_weight = 1.0
                rf_weight = 0.0
                final_w = w_tv
                details.update({"iterations": getattr(res, "nit", None), "converged": bool(res.success)})
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize (within bounds) and compute metrics
        final_w = np.clip(final_w, min_weight, max_weight)
        if abs(final_w.sum()) > EPS:
            final_w = final_w / final_w.sum()

        pm = self.portfolio_metrics(final_w, rf)
        rc, mrc = self._risk_contributions(final_w)

        # Risk metrics from realized history (daily)
        port_daily = (self.returns[self.tickers] * final_w).sum(axis=1)  # type: ignore
        # VaR/CVaR (historical)
        var_95 = float(np.percentile(port_daily, 5))
        var_99 = float(np.percentile(port_daily, 1))
        cvar_95 = float(port_daily[port_daily <= var_95].mean()) if (port_daily <= var_95).any() else float(var_95)

        # Max drawdown
        wealth = (1 + port_daily).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1
        max_drawdown = float(dd.min())

        # Sortino
        downside = port_daily.clip(upper=0)
        downside_dev = float(downside.std()) * math.sqrt(TRADING_DAYS)
        sortino = (pm["expected_return"] - rf) / (downside_dev + EPS)

        # Diversification & concentration (HHI)
        hhi = float((final_w ** 2).sum())
        effective_n = 1.0 / max(hhi, EPS)

        leverage_used = risky_weight > 1.0 - 1e-6
        details.update({"leverage_used": bool(leverage_used)})

        # Tangency stats (even if not used)
        w_tan, _ = self._tangency_weights(rf, min_weight, max_weight)
        m_tan = self.portfolio_metrics(w_tan, rf)

        result: Dict[str, object] = {
            "success": True,
            "method": method,
            "weights": final_w,
            "expected_return": pm["expected_return"],
            "volatility": pm["volatility"],
            "sharpe_ratio": pm["sharpe_ratio"],
            "risk_contribution": rc.tolist(),
            "marginal_contribution": mrc.tolist(),
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino,
            "diversification_ratio": effective_n,
            "concentration": hhi,
            "optimization_details": details,
            # CAL fields
            "rf_weight": float(rf_weight),
            "risky_weight": float(risky_weight),
            "tangency_return": float(m_tan["expected_return"]),
            "tangency_volatility": float(m_tan["volatility"]),
            "tangency_sharpe": float(m_tan["sharpe_ratio"]),
        }
        return result

    # ----------------------------- Diagnostics ----------------------------- #
    def get_debug_info(self) -> Dict[str, object]:
        start = self._start_date or (pd.Timestamp.today() - pd.DateOffset(years=self.lookback_years))
        end = self._end_date or pd.Timestamp.today()
        return {
            "temporal_consistency": {
                "start": str(pd.Timestamp(start).date()),
                "end": str(pd.Timestamp(end).date()),
                "trading_days": int(self.returns.shape[0]) if self.returns is not None else 0,
            },
            "universe": {
                "tickers": self.tickers,
                "failed": self.failed_tickers,
            },
        }

    # ----------------------------- CAPM metrics ----------------------------- #
    def calculate_capm_metrics(self, market_ticker: str = "^GSPC") -> Optional[Dict[str, Dict[str, float]]]:
        if self.returns is None or len(self.tickers) < 1:
            return None
        try:
            mkt = yf.download(market_ticker, period=f"{max(self.lookback_years,1)}y", interval="1d", auto_adjust=True, progress=False)
            if not isinstance(mkt, (pd.Series, pd.DataFrame)) or len(mkt) == 0:
                return None

            # Extract a single adjusted close/close price series
            if isinstance(mkt, pd.Series):
                price_series = mkt
            else:
                if isinstance(mkt.columns, pd.MultiIndex):
                    # Prefer 'Close' if present at last level, otherwise try 'Adj Close'
                    last_level = mkt.columns.get_level_values(-1)
                    if "Close" in last_level:
                        price = mkt.xs("Close", axis=1, level=-1)
                    elif "Adj Close" in last_level:
                        price = mkt.xs("Adj Close", axis=1, level=-1)
                    else:
                        price = mkt.select_dtypes(include=[np.number])
                else:
                    if "Close" in mkt.columns:
                        price = mkt["Close"]
                    elif "Adj Close" in mkt.columns:
                        price = mkt["Adj Close"]
                    else:
                        price = mkt.select_dtypes(include=[np.number])

                # Squeeze to Series if single column remains
                price_series = price.iloc[:, 0] if isinstance(price, pd.DataFrame) and price.shape[1] >= 1 else price

            mkt_ret = price_series.pct_change().dropna()
        except Exception:
            return None

        # Align with portfolio window
        ret = self.returns[self.tickers].copy()
        # Ensure market return is a 1-col DataFrame with name _MKT_
        mkt_ret_df = mkt_ret.to_frame(name="_MKT_") if isinstance(mkt_ret, pd.Series) else mkt_ret.rename(columns={mkt_ret.columns[0]: "_MKT_"})
        df = pd.concat([ret, mkt_ret_df], axis=1).dropna()
        if df.shape[0] < TRADING_DAYS // 2:
            return None

        rf_annual = self.get_risk_free_rate()
        rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

        metrics: Dict[str, Dict[str, float]] = {}
        mu_m_annual = _annualize_return(df["_MKT_"])
        var_m = float(df["_MKT_"].var())
        for t in self.tickers:
            x = df["_MKT_"]
            y = df[t]
            cov = float(np.cov(x, y)[0, 1])
            beta = cov / max(var_m, EPS)
            corr = float(np.corrcoef(x, y)[0, 1])
            alpha_daily = (y.mean() - rf_daily) - beta * (x.mean() - rf_daily)
            alpha_annual = (1 + alpha_daily) ** TRADING_DAYS - 1
            exp_ret_capm = rf_annual + beta * (mu_m_annual - rf_annual)
            actual_annual = _annualize_return(y)
            r2 = corr ** 2
            # Risk decomposition (approx): systematic share of variance
            sys_risk = abs(beta) * y.std() * corr
            metrics[t] = {
                "beta": float(beta),
                "alpha": float(alpha_annual),
                "expected_return": float(exp_ret_capm),
                "actual_return": float(actual_annual),
                "r_squared": float(r2),
                "correlation": float(corr),
                "systematic_risk": float(sys_risk),
                "total_risk": float(y.std()),
            }
        return metrics


# ----------------------------- Plotly charts ----------------------------- #

def create_portfolio_composition_chart(tickers: List[str], weights: np.ndarray, rf_weight: Optional[float] = None) -> go.Figure:
    w = _to_numpy(weights)
    labels = list(tickers)
    vals = w.copy()

    if rf_weight is not None and abs(rf_weight) > 1e-6:
        if rf_weight >= 0:
            labels.append("RISK-FREE")
            vals = np.append(vals, rf_weight)
        else:
            labels.append("LEVERAGE")
            vals = np.append(vals, abs(rf_weight))
    # Normalize to 1 for pie
    s = vals.sum()
    if s <= 0:
        vals = np.ones_like(vals) / len(vals)
    else:
        vals = vals / s

    fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0.45)])
    fig.update_layout(title="Portfolio Allocation", legend_title="Assets", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _efficient_frontier_points(opt: PortfolioOptimizer, n_points: int = 50, min_w: float = 0.0, max_w: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les points de la frontière efficiente en optimisant pour différents niveaux de rendement cible.
    Utilise TOUS les tickers dans l'optimisation avec contraintes de poids.
    """
    if opt.mean_returns_annual is None or opt.cov_matrix_annual is None:
        return np.array([]), np.array([])
    
    mu = opt.mean_returns_annual[opt.tickers].values
    n_assets = len(opt.tickers)
    
    # Calcul des rendements min et max possibles avec les contraintes de poids
    # Pour le minimum: portfolio de variance minimale
    try:
        w_min_var, _ = opt._min_variance_weights(min_w, max_w)
        min_return = float(w_min_var @ mu)
    except:
        min_return = float(mu.min())
    
    # Pour le maximum: tous les poids sur l'actif avec le plus haut rendement (dans les limites)
    max_return_idx = np.argmax(mu)
    w_max = np.full(n_assets, min_w)
    remaining_weight = 1.0 - min_w * n_assets
    w_max[max_return_idx] = min(max_w, min_w + remaining_weight)
    # Redistribuer le poids restant si nécessaire
    if w_max.sum() < 1.0:
        deficit = 1.0 - w_max.sum()
        for i in range(n_assets):
            if i != max_return_idx and w_max[i] < max_w:
                add_weight = min(deficit, max_w - w_max[i])
                w_max[i] += add_weight
                deficit -= add_weight
                if deficit <= 1e-10:
                    break
    max_return = float(w_max @ mu)
    
    # Génération des rendements cibles
    target_returns = np.linspace(min_return, max_return, n_points)
    
    vols = []
    rets = []
    
    for target_ret in target_returns:
        try:
            # Optimisation de variance minimale pour un rendement cible donné
            w_target, res = opt._target_return_weights(target_ret, min_w, max_w)
            
            if res.success:
                # Vérification que les contraintes sont respectées
                if (w_target >= min_w - 1e-6).all() and (w_target <= max_w + 1e-6).all() and abs(w_target.sum() - 1.0) < 1e-6:
                    metrics = opt.portfolio_metrics(w_target)
                    vols.append(metrics["volatility"])
                    rets.append(metrics["expected_return"])
        except Exception:
            continue
    
    return np.array(vols), np.array(rets)


def _generate_random_portfolios(opt: PortfolioOptimizer, n_portfolios: int = 1000, min_w: float = 0.0, max_w: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère des portfolios aléatoires pour visualiser l'espace risque-rendement.
    Retourne (volatilités, rendements, ratios de Sharpe).
    """
    if opt.mean_returns_annual is None or opt.cov_matrix_annual is None:
        return np.array([]), np.array([]), np.array([])
    
    n_assets = len(opt.tickers)
    rf = opt.get_risk_free_rate()
    
    vols = []
    rets = []
    sharpes = []
    
    for _ in range(n_portfolios):
        # Génération de poids aléatoires respectant les contraintes
        weights = np.random.uniform(min_w, max_w, n_assets)
        # Normalisation pour que la somme = 1
        weights = weights / weights.sum()
        
        # Vérification des contraintes après normalisation
        if (weights >= min_w - 1e-6).all() and (weights <= max_w + 1e-6).all():
            try:
                metrics = opt.portfolio_metrics(weights, rf)
                vols.append(metrics["volatility"])
                rets.append(metrics["expected_return"])
                sharpes.append(metrics["sharpe_ratio"])
            except Exception:
                continue
    
    return np.array(vols), np.array(rets), np.array(sharpes)


def create_efficient_frontier_plot(opt: PortfolioOptimizer, n_points: int = 50, min_w: float = 0.0, max_w: float = 1.0, show_random: bool = True, n_random: int = 1000) -> go.Figure:
    """Create an efficient frontier plot with optional random portfolios overlay."""
    # Generate efficient frontier points
    vols, rets = _efficient_frontier_points(opt, n_points, min_w, max_w)
    
    # Create the main figure
    fig = go.Figure()
    
    # Add efficient frontier
    if len(vols) > 0 and len(rets) > 0:
        fig.add_trace(go.Scatter(
            x=vols, y=rets,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'),
            hovertemplate='Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
    
    # Add random portfolios if requested
    if show_random and n_random > 0:
        rand_vols, rand_rets, rand_sharpes = _generate_random_portfolios(opt, n_random, min_w, max_w)
        if len(rand_vols) > 0:
            # Color by Sharpe ratio
            colors = rand_sharpes
            fig.add_trace(go.Scatter(
                x=rand_vols, y=rand_rets,
                mode='markers',
                name='Random Portfolios',
                marker=dict(
                    size=4,
                    color=colors,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.6
                ),
                hovertemplate='Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
    
    # Add key portfolio points
    try:
        # Min variance portfolio
        w_min, _ = opt._min_variance_weights(min_w, max_w)
        m_min = opt.portfolio_metrics(w_min)
        fig.add_trace(go.Scatter(
            x=[m_min["volatility"]], y=[m_min["expected_return"]],
            mode='markers',
            name='Min Variance',
            marker=dict(size=12, color='red', symbol='diamond'),
            hovertemplate='Min Variance<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        # Tangency portfolio
        rf = opt.get_risk_free_rate()
        w_tan, _ = opt._tangency_weights(rf, min_w, max_w)
        m_tan = opt.portfolio_metrics(w_tan, rf)
        fig.add_trace(go.Scatter(
            x=[m_tan["volatility"]], y=[m_tan["expected_return"]],
            mode='markers',
            name='Tangency',
            marker=dict(size=12, color='green', symbol='star'),
            hovertemplate='Tangency<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        # Risk-free rate point
        fig.add_trace(go.Scatter(
            x=[0], y=[rf],
            mode='markers',
            name='Risk-Free',
            marker=dict(size=10, color='black', symbol='circle'),
            hovertemplate='Risk-Free<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        # Capital Allocation Line
        if m_tan["volatility"] > 0:
            cal_vols = np.linspace(0, m_tan["volatility"] * 1.5, 100)
            cal_rets = rf + (m_tan["expected_return"] - rf) / m_tan["volatility"] * cal_vols
            fig.add_trace(go.Scatter(
                x=cal_vols, y=cal_rets,
                mode='lines',
                name='Capital Allocation Line',
                line=dict(color='purple', width=2, dash='dash'),
                hovertemplate='CAL<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
    except Exception:
        pass  # Skip if optimization fails
    
    # Update layout
    fig.update_layout(
        title="Efficient Frontier & Capital Allocation Line",
        xaxis_title="Portfolio Volatility (Annualized)",
        yaxis_title="Expected Return (Annualized)",
        hovermode='closest',
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%')
    )
    
    return fig


def create_risk_return_analysis(opt: PortfolioOptimizer, weights: np.ndarray, rf_weight: Optional[float] = None) -> go.Figure:
    """Create a comprehensive risk-return analysis chart."""
    w = _to_numpy(weights)
    
    # Get portfolio metrics
    rf = opt.get_risk_free_rate()
    pm = opt.portfolio_metrics(w, rf)
    rc, mrc = opt._risk_contributions(w)
    
    # Create subplots
    fig = go.Figure()
    
    # Risk contribution chart
    fig.add_trace(go.Bar(
        x=opt.tickers,
        y=rc,
        name='Risk Contribution',
        marker_color='lightblue',
        hovertemplate='%{x}<br>Risk Contribution: %{y:.2%}<extra></extra>'
    ))
    
    # Marginal risk contribution
    fig.add_trace(go.Scatter(
        x=opt.tickers,
        y=mrc,
        mode='lines+markers',
        name='Marginal Risk Contribution',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='red'),
        yaxis='y2',
        hovertemplate='%{x}<br>Marginal Risk: %{y:.2%}<extra></extra>'
    ))
    
    # Update layout for dual y-axis
    fig.update_layout(
        title="Risk Contribution Analysis",
        xaxis_title="Assets",
        yaxis=dict(
            title="Risk Contribution",
            tickformat='.1%',
            side='left'
        ),
        yaxis2=dict(
            title="Marginal Risk Contribution",
            tickformat='.1%',
            side='right',
            overlaying='y'
        ),
        barmode='group',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


def create_performance_analytics(opt: PortfolioOptimizer, weights: np.ndarray, rf_weight: Optional[float] = None) -> go.Figure:
    """Create performance analytics including drawdown and rolling metrics."""
    w = _to_numpy(weights)
    
    # Calculate portfolio returns
    port_daily = (opt.returns[opt.tickers] * w).sum(axis=1)
    
    # Calculate cumulative wealth and drawdown
    wealth = (1 + port_daily).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth / peak - 1) * 100
    
    # Rolling metrics (30-day rolling)
    window = min(30, len(port_daily) // 4)
    if window > 5:
        rolling_vol = port_daily.rolling(window).std() * np.sqrt(252) * 100
        rolling_sharpe = (port_daily.rolling(window).mean() * 252) / (rolling_vol / 100 + EPS)
    else:
        rolling_vol = pd.Series(index=port_daily.index, dtype=float)
        rolling_sharpe = pd.Series(index=port_daily.index, dtype=float)
    
    # Create subplots
    fig = go.Figure()
    
    # Wealth evolution
    fig.add_trace(go.Scatter(
        x=wealth.index,
        y=wealth,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2),
        yaxis='y',
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown (%)',
        line=dict(color='red', width=1),
        yaxis='y2',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
    ))
    
    # Rolling volatility
    if not rolling_vol.empty:
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='Rolling Volatility (%)',
            line=dict(color='green', width=1, dash='dot'),
            yaxis='y3',
            hovertemplate='Date: %{x}<br>Vol: %{y:.1f}%<extra></extra>'
        ))
    
    # Rolling Sharpe
    if not rolling_sharpe.empty:
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='purple', width=1, dash='dot'),
            yaxis='y4',
            hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout for multiple y-axes
    fig.update_layout(
        title="Portfolio Performance Analytics",
        xaxis_title="Date",
        yaxis=dict(title="Portfolio Value", side='left'),
        yaxis2=dict(title="Drawdown (%)", side='right', overlaying='y'),
        yaxis3=dict(title="Rolling Vol (%)", side='right', anchor='x', position=0.95),
        yaxis4=dict(title="Rolling Sharpe", side='right', anchor='x', position=0.9),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified'
    )
    
    return fig


def create_capm_analysis_chart(opt: PortfolioOptimizer, capm_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create CAPM analysis chart showing beta vs return and alpha."""
    if not capm_metrics:
        # Return empty figure if no CAPM data
        fig = go.Figure()
        fig.update_layout(
            title="CAPM Analysis - No Data Available",
            xaxis_title="Beta",
            yaxis_title="Return (%)",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
    
    # Extract data
    tickers = list(capm_metrics.keys())
    betas = [capm_metrics[t]["beta"] for t in tickers]
    returns = [capm_metrics[t]["actual_return"] * 100 for t in tickers]  # Convert to percentage
    alphas = [capm_metrics[t]["alpha"] * 100 for t in tickers]  # Convert to percentage
    r_squared = [capm_metrics[t]["r_squared"] for t in tickers]
    
    # Create subplots
    fig = go.Figure()
    
    # Beta vs Return scatter
    fig.add_trace(go.Scatter(
        x=betas,
        y=returns,
        mode='markers',
        name='Assets',
        marker=dict(
            size=[r * 20 + 5 for r in r_squared],  # Size by R²
            color=alphas,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Alpha (%)"),
            opacity=0.7
        ),
        text=tickers,
        hovertemplate='<b>%{text}</b><br>Beta: %{x:.2f}<br>Return: %{y:.1f}%<br>Alpha: %{marker.color:.1f}%<br>R²: %{marker.size:.2f}<extra></extra>'
    ))
    
    # Market line (CAPM line)
    rf = opt.get_risk_free_rate() * 100  # Convert to percentage
    market_return = np.mean(returns)  # Approximate market return
    
    # Add CAPM line
    beta_range = np.linspace(min(betas) - 0.2, max(betas) + 0.2, 100)
    capm_line = rf + (market_return - rf) * beta_range
    
    fig.add_trace(go.Scatter(
        x=beta_range,
        y=capm_line,
        mode='lines',
        name='CAPM Line',
        line=dict(color='black', width=2, dash='dash'),
        hovertemplate='Beta: %{x:.2f}<br>CAPM Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Risk-free rate point
    fig.add_trace(go.Scatter(
        x=[0], y=[rf],
        mode='markers',
        name='Risk-Free Rate',
        marker=dict(size=12, color='black', symbol='diamond'),
        hovertemplate='Risk-Free<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="CAPM Analysis: Beta vs Return",
        xaxis_title="Beta (Systematic Risk)",
        yaxis_title="Annual Return (%)",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(tickformat='.1f'),
        xaxis=dict(tickformat='.2f')
    )
    
    return fig


# ----------------------------- Validation helper ----------------------------- #

def validate_optimization_result(result: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Post-optimization sanity checks.
    
    Returns
    -------
    (is_valid, list_of_warnings)
        is_valid=True if all critical checks pass.
        warnings is a list of non-critical issues.
    """
    warnings_list = []
    
    # Check required fields
    required_fields = ["weights", "expected_return", "volatility", "sharpe_ratio"]
    for field in required_fields:
        if field not in result:
            return False, [f"Missing required field: {field}"]
    
    # Check weights sum to 1
    weights = result["weights"]
    if isinstance(weights, (list, np.ndarray)):
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            warnings_list.append(f"Weights sum to {weight_sum:.6f}, not 1.0")
    
    # Check for extreme values
    if result["volatility"] <= 0:
        return False, ["Volatility must be positive"]
    
    if result["volatility"] > 2.0:  # >200% annual volatility
        warnings_list.append("Extremely high volatility detected")
    
    if abs(result["expected_return"]) > 1.0:  # >100% annual return
        warnings_list.append("Extremely high return detected")
    
    # Check optimization convergence
    if "optimization_details" in result:
        details = result["optimization_details"]
        if isinstance(details, dict) and "converged" in details:
            if not details["converged"]:
                warnings_list.append("Optimization did not converge")
    
    # Check for leverage
    if "leverage_used" in result and result["leverage_used"]:
        warnings_list.append("Portfolio uses leverage (weights > 100%)")
    
    return True, warnings_list
