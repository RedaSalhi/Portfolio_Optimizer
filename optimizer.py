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
            if isinstance(mkt, pd.DataFrame) and mkt.shape[0] > 0:
                mkt_ret = mkt["Close"].pct_change().dropna()
            else:
                return None
        except Exception:
            return None

        # Align with portfolio window
        ret = self.returns[self.tickers].copy()
        # Avoid Series.rename with a string (can be interpreted as index mapper in some pandas versions)
        mkt_ret_df = mkt_ret.to_frame(name="_MKT_")
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
    mu = opt.mean_returns_annual[opt.tickers].values  # type: ignore
    mu_min, mu_max = float(mu.min()), float(mu.max())
    targets = np.linspace(mu_min, mu_max, n_points)
    vols: List[float] = []
    rets: List[float] = []
    for tr in targets:
        try:
            w, _ = opt._target_return_weights(tr, min_w, max_w)
            m = opt.portfolio_metrics(w)
            vols.append(m["volatility"])
            rets.append(m["expected_return"])
        except Exception:
            continue
    return np.array(vols), np.array(rets)


def create_efficient_frontier_plot(opt: PortfolioOptimizer, result: Dict[str, object], include_risk_free: bool = True, frontier_points: int = 50) -> Optional[go.Figure]:
    if opt.mean_returns_annual is None or opt.cov_matrix_annual is None:
        return None

    try:
        vols, rets = _efficient_frontier_points(opt, max(25, int(frontier_points)))
        fig = go.Figure()
        if len(vols) > 0:
            fig.add_trace(go.Scatter(x=vols, y=rets, mode='lines', name='Efficient Frontier'))

        # Equal-weight and optimized portfolio
        eq_w = np.ones(len(opt.tickers)) / len(opt.tickers)
        pm_eq = opt.portfolio_metrics(eq_w, opt.get_risk_free_rate() if include_risk_free else 0.0)
        fig.add_trace(go.Scatter(x=[pm_eq["volatility"]], y=[pm_eq["expected_return"]], mode='markers', name='Equal Weight', marker=dict(size=10)))

        fig.add_trace(go.Scatter(x=[result["volatility"]], y=[result["expected_return"]], mode='markers', name='Optimized', marker=dict(size=12)))

        # Tangency & CAL
        if include_risk_free:
            rf = opt.get_risk_free_rate()
            w_tan, _ = opt._tangency_weights(rf, 0.0, 1.0)
            m_tan = opt.portfolio_metrics(w_tan, rf)
            fig.add_trace(go.Scatter(x=[m_tan["volatility"]], y=[m_tan["expected_return"]], mode='markers', name='Tangency', marker=dict(size=11)))
            # CAL line
            x = np.linspace(0, float(max(vols.max() if len(vols) else m_tan["volatility"], result["volatility"]) * 1.3), 50)
            slope = (m_tan["expected_return"] - rf) / max(m_tan["volatility"], EPS)
            y = rf + slope * x
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Capital Allocation Line', line=dict(dash='dash')))

        fig.update_layout(
            title="Efficient Frontier & Portfolio Position",
            xaxis_title="Volatility (Annual)",
            yaxis_title="Expected Return (Annual)",
            margin=dict(l=10, r=10, t=40, b=10),
            legend_title="",
        )
        return fig
    except Exception:
        return None


def create_risk_return_analysis(opt: PortfolioOptimizer, weights: np.ndarray) -> go.Figure:
    mu = opt.mean_returns_annual[opt.tickers].values  # type: ignore
    sig = np.sqrt(np.diag(opt.cov_matrix_annual.loc[opt.tickers, opt.tickers].values))  # type: ignore
    w = _to_numpy(weights)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig, y=mu, mode='markers', text=opt.tickers, name='Assets', marker=dict(size=10)))

    pm = opt.portfolio_metrics(w, opt.get_risk_free_rate())
    fig.add_trace(go.Scatter(x=[pm["volatility"]], y=[pm["expected_return"]], mode='markers', name='Portfolio', marker=dict(size=14)))

    fig.update_layout(
        title='Risk vs Return (Assets & Portfolio)',
        xaxis_title='Volatility (Annual)',
        yaxis_title='Expected Return (Annual)',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def create_performance_analytics(opt: PortfolioOptimizer, weights: np.ndarray) -> go.Figure:
    w = _to_numpy(weights)
    port_daily = (opt.returns[opt.tickers] * w).sum(axis=1)  # type: ignore
    eq_daily = (opt.returns[opt.tickers] * (np.ones_like(w) / len(w))).sum(axis=1)  # type: ignore

    wealth_port = (1 + port_daily).cumprod()
    wealth_eq = (1 + eq_daily).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wealth_port.index, y=wealth_port.values, name='Portfolio Equity', mode='lines'))
    fig.add_trace(go.Scatter(x=wealth_eq.index, y=wealth_eq.values, name='Equal-Weight Equity', mode='lines'))

    fig.update_layout(
        title='Performance (Cumulative Growth of $1)',
        xaxis_title='Date',
        yaxis_title='Wealth',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def create_capm_analysis_chart(capm_metrics: Optional[Dict[str, Dict[str, float]]]) -> Optional[go.Figure]:
    if not capm_metrics:
        return None
    # Scatter: Beta (x) vs Alpha (y)
    betas = [capm_metrics[t]["beta"] for t in capm_metrics]
    alphas = [capm_metrics[t]["alpha"] for t in capm_metrics]
    labels = list(capm_metrics.keys())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=alphas, mode='markers+text', text=labels, textposition='top center', name='Assets'))
    fig.add_hline(y=0, line=dict(dash='dot'))
    fig.add_vline(x=1, line=dict(dash='dot'))
    fig.update_layout(title='CAPM: Alpha vs Beta', xaxis_title='Beta', yaxis_title='Alpha (Annual)', margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ----------------------------- Validation helper ----------------------------- #

def validate_optimization_result(result: Dict[str, object], weights: Iterable[float], optimizer: PortfolioOptimizer) -> List[str]:
    """Return a list of human-readable warnings about potential issues.
    Never raises – returns [] when all looks OK.
    """
    warnings_list: List[str] = []

    try:
        w = _to_numpy(weights)
        if np.isnan(w).any():
            warnings_list.append("Weights contain NaN values.")
        if (w < -1e-6).any():
            warnings_list.append("Negative weights detected (shorting). Ensure this is intended or raise min_weight.")
        s = float(w.sum())
        if abs(s - 1.0) > 1e-3:
            warnings_list.append(f"Weights do not sum to 1 (sum={s:.4f}).")

        # Basic metric sanity
        for k in ("expected_return", "volatility", "sharpe_ratio"):
            v = float(result.get(k, np.nan))
            if not np.isfinite(v):
                warnings_list.append(f"Metric {k} is not finite.")
        if float(result.get("volatility", 0)) <= 0:
            warnings_list.append("Portfolio volatility is non-positive (check data quality).")

        # Check consistency with inputs
        if optimizer.mean_returns_annual is not None:
            if len(w) != len(optimizer.tickers):
                warnings_list.append("Weights length does not match number of tickers.")

        # Drawdown outliers
        mdd = abs(float(result.get("max_drawdown", 0)))
        if mdd > 0.6:
            warnings_list.append("Max drawdown exceeds 60% – very high historical drawdown.")

        # Concentration (HHI)
        hhi = float(result.get("concentration", 0))
        if hhi > 0.4:
            warnings_list.append("Portfolio highly concentrated (HHI > 0.40). Consider adding constraints.")

    except Exception as e:  # pragma: no cover
        warnings_list.append(f"Validation encountered an error: {e}")

    return warnings_list
