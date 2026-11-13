"""
Position sizing and portfolio combination.

PRIMARY method (produces all headline/reported numbers): deterministic,
three-step inverse-volatility / equal-risk-budget combination --
  1. per-leg vol targeting:      w_leg,t = (target_leg_vol / realized_vol_leg,t) * signal_leg,t
  2. portfolio-level vol scale:  scale_t = target_portfolio_vol / vol(current weights applied
                                            to trailing 60d leg returns)   [no full-sample vol -- see below]
  3. 2x gross exposure cap:      if sum(|w|) > 2.0: w *= 2.0 / sum(|w|)

No optimizer anywhere in this path -- one deterministic scale factor and one
deterministic min()-style cap, by design (Occam's razor / "no large
optimisation").

APPENDIX method (combine_portfolio_risk_parity_appendix): correlation-
adjusted equal-risk-contribution (ERC) weights via a small numerical solve
on the trailing 60-day covariance matrix. This uses scipy.optimize and is
INTENTIONALLY separated from the primary path -- it is exploratory, shown
only in the notebook's appendix, and must never feed the headline metrics
or report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily % returns from adjusted-close prices."""
    return prices.pct_change()


def annualized_realized_vol(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling `window`-day annualized stdev of daily returns, per column.
    Uses only trailing/past data at each date (pandas rolling is backward-
    looking by construction) -- no look-ahead.
    """
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def get_weekly_rebalance_dates(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """One rebalance date per calendar week: the last trading day of each
    week present in the daily index (handles holidays/short weeks cleanly).
    """
    s = pd.Series(daily_index, index=daily_index)
    weekly_last = s.groupby(pd.Grouper(freq="W-FRI")).last()
    return pd.DatetimeIndex(weekly_last.dropna().values)


def leg_vol_target_weights(signal: pd.DataFrame, leg_vol: pd.DataFrame,
                            target_leg_vol: float) -> pd.DataFrame:
    """Step 1: w_leg,t = (target_leg_vol / realized_vol_leg,t) * signal_leg,t."""
    return (target_leg_vol / leg_vol) * signal


def _portfolio_vol_from_current_weights(weights_at_rebal: pd.DataFrame,
                                          leg_returns: pd.DataFrame,
                                          rebalance_dates: pd.DatetimeIndex,
                                          window: int = 60) -> pd.Series:
    """At each rebalance date t, apply that date's CURRENT weights (already
    known at t, no future info) statically to the trailing `window` days of
    ACTUAL leg returns ending at t, then take the annualized stdev of the
    resulting hypothetical portfolio-return series. This is the "trailing
    60-day realized vol of the current-weight portfolio" used to compute
    the deterministic, proportional portfolio-level vol scale factor.

    Deliberately NOT a full-sample scale factor: computing this from
    whole-history vol would use future information at every historical
    rebalance and constitute look-ahead bias. Using only the trailing
    window ending at t avoids this.
    """
    idx = leg_returns.index
    vol_est = pd.Series(index=rebalance_dates, dtype=float)
    for t in rebalance_dates:
        pos = idx.searchsorted(t, side="right")  # rows with date <= t
        if pos < window:
            vol_est.loc[t] = np.nan
            continue
        window_returns = leg_returns.iloc[pos - window:pos]
        w = weights_at_rebal.loc[t]
        hypothetical_port_ret = window_returns.mul(w, axis=1).sum(axis=1)
        vol_est.loc[t] = hypothetical_port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol_est


def apply_gross_cap(weights: pd.DataFrame, cap: float = 2.0) -> dict:
    """Deterministic proportional scale-down: if sum(|w_leg|) > cap, scale
    ALL legs down by cap / sum(|w_leg|). One min()-style operation, no
    optimization. Returns capped weights plus diagnostics (pre/post gross
    exposure and a cap-binding boolean per rebalance).
    """
    gross_pre = weights.abs().sum(axis=1)
    scale = (cap / gross_pre).clip(upper=1.0)  # only ever scales DOWN
    capped = weights.mul(scale, axis=0)
    gross_post = capped.abs().sum(axis=1)
    cap_binding = gross_pre > cap
    return {
        "weights": capped,
        "gross_pre_cap": gross_pre,
        "gross_post_cap": gross_post,
        "cap_binding": cap_binding,
    }


def combine_portfolio_primary(signal: pd.DataFrame, prices: pd.DataFrame,
                               rebalance_dates: pd.DatetimeIndex,
                               target_portfolio_vol: float = 0.10,
                               target_leg_vol: float | None = None,
                               vol_window: int = 60,
                               gross_cap: float = 2.0) -> dict:
    """PRIMARY combination method. Returns a dict of rebalance-date-indexed
    DataFrames/Series: weights (final, post-cap), weights_precap, leg_vol,
    portfolio_vol_estimate, gross_pre_cap, gross_post_cap, cap_binding.
    """
    legs = list(prices.columns)
    if target_leg_vol is None:
        target_leg_vol = target_portfolio_vol / len(legs)

    rets = daily_returns(prices)
    leg_vol_daily = annualized_realized_vol(rets, window=vol_window)

    signal_r = signal.loc[rebalance_dates]
    leg_vol_r = leg_vol_daily.loc[rebalance_dates]

    step1_weights = leg_vol_target_weights(signal_r, leg_vol_r, target_leg_vol)

    port_vol_est = _portfolio_vol_from_current_weights(
        step1_weights, rets, rebalance_dates, window=vol_window
    )
    scale = target_portfolio_vol / port_vol_est
    step2_weights = step1_weights.mul(scale, axis=0)

    cap_result = apply_gross_cap(step2_weights, cap=gross_cap)

    return {
        "weights": cap_result["weights"],
        "weights_precap": step2_weights,
        "leg_vol": leg_vol_r,
        "portfolio_vol_estimate": port_vol_est,
        "portfolio_scale": scale,
        "gross_pre_cap": cap_result["gross_pre_cap"],
        "gross_post_cap": cap_result["gross_post_cap"],
        "cap_binding": cap_result["cap_binding"],
    }


# ---------------------------------------------------------------------------
# APPENDIX (exploratory, notebook appendix only) -- correlation-adjusted ERC
# ---------------------------------------------------------------------------

def _erc_weights_from_cov(cov: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Solve for equal-risk-contribution magnitudes y >= 0, sum(y) = 1,
    given fixed directional signs from the TSMOM signal. Small 3-asset
    numerical solve via scipy -- fine here because this function is only
    ever called from the exploratory appendix path, never the primary
    reported path.
    """
    n = len(signs)

    def risk_contributions(y):
        w = signs * y
        port_var = w @ cov @ w
        marginal = cov @ w
        return w * marginal / np.sqrt(max(port_var, 1e-16))

    def objective(y):
        rc = risk_contributions(y)
        return float(np.sum((rc - rc.mean()) ** 2))

    x0 = np.full(n, 1.0 / n)
    bounds = [(1e-6, 5.0)] * n
    constraints = [{"type": "eq", "fun": lambda y: np.sum(y) - 1.0}]
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    y = result.x if result.success else x0
    return y


def combine_portfolio_risk_parity_appendix(signal: pd.DataFrame, prices: pd.DataFrame,
                                            rebalance_dates: pd.DatetimeIndex,
                                            target_portfolio_vol: float = 0.10,
                                            vol_window: int = 60,
                                            gross_cap: float = 2.0) -> dict:
    """EXPLORATORY / NOT USED FOR PRIMARY RESULTS.

    Correlation-adjusted risk parity using the trailing 60-day covariance
    matrix (equal risk contribution across legs, given fixed directional
    signs from the same TSMOM signal). Supplementary comparison only --
    must never feed headline metrics or the report.
    """
    legs = list(prices.columns)
    n = len(legs)
    rets = daily_returns(prices)
    idx = rets.index

    signal_r = signal.loc[rebalance_dates]
    raw_weights = pd.DataFrame(index=rebalance_dates, columns=legs, dtype=float)

    for t in rebalance_dates:
        pos = idx.searchsorted(t, side="right")
        if pos < vol_window:
            raw_weights.loc[t] = np.nan
            continue
        window_returns = rets.iloc[pos - vol_window:pos]
        cov = window_returns.cov().values * TRADING_DAYS_PER_YEAR
        signs = signal_r.loc[t].values
        if np.any(np.isnan(signs)):
            raw_weights.loc[t] = np.nan
            continue
        y = _erc_weights_from_cov(cov, signs)
        raw_weights.loc[t] = signs * y

    port_vol_est = _portfolio_vol_from_current_weights(raw_weights, rets, rebalance_dates, window=vol_window)
    scale = target_portfolio_vol / port_vol_est
    scaled_weights = raw_weights.mul(scale, axis=0)

    cap_result = apply_gross_cap(scaled_weights, cap=gross_cap)

    return {
        "weights": cap_result["weights"],
        "weights_precap": scaled_weights,
        "portfolio_vol_estimate": port_vol_est,
        "gross_pre_cap": cap_result["gross_pre_cap"],
        "gross_post_cap": cap_result["gross_post_cap"],
        "cap_binding": cap_result["cap_binding"],
    }
