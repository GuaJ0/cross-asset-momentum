"""
Performance metrics: annualized return/vol/Sharpe/Sortino, max drawdown,
Calmar, hit rate, trade/holding-period stats, turnover, costs, exposure --
computed gross, net-of-transaction-cost, and net-all-in (transaction cost +
expense ratio), for any date-sliced period.

Note on "average holding period": because this strategy is continuously
sized (vol-scaled weekly, not discrete entries/exits), holding period is
defined as the number of trading days between consecutive SIGN FLIPS of
each leg's underlying TSMOM signal -- NOT the weekly resizing cadence,
which would trivially always equal one week and add no information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Core return-series stats
# ---------------------------------------------------------------------------

def cumulative_return(daily_ret: pd.Series) -> pd.Series:
    return (1.0 + daily_ret.fillna(0.0)).cumprod()


def drawdown_series(daily_ret: pd.Series) -> pd.Series:
    cum = cumulative_return(daily_ret)
    return cum / cum.cummax() - 1.0


def annualized_return(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    n = len(daily_ret)
    if n == 0:
        return np.nan
    growth = (1.0 + daily_ret).prod()
    return growth ** (TRADING_DAYS_PER_YEAR / n) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    return daily_ret.dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_ret: pd.Series) -> float:
    # Risk-free rate assumed 0 -- this is a long-short, vol-targeted book
    # without a modeled cash/financing leg (see borrow-cost disclosure in
    # the memo/source notes), so raw (not excess-of-cash) returns are used.
    vol = annualized_vol(daily_ret)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualized_return(daily_ret) / vol


def downside_deviation(daily_ret: pd.Series, target: float = 0.0) -> float:
    """Annualized downside deviation vs. `target` (0 here, matching Sharpe's
    zero risk-free-rate assumption above): sqrt(mean(min(r - target, 0)^2))
    over ALL periods (not just negative ones) -- the standard Sortino & van
    der Meer definition, so up-days contribute zero rather than being
    dropped from the sample.
    """
    daily_ret = daily_ret.dropna()
    downside = np.minimum(daily_ret - target, 0.0)
    return np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_ret: pd.Series) -> float:
    # Same zero risk-free-rate assumption as sharpe_ratio above, but
    # penalizes only downside volatility -- useful alongside Sharpe here
    # since this strategy's return distribution isn't guaranteed symmetric
    # (vol targeting + a hard gross cap can skew realized up/down moves).
    dd = downside_deviation(daily_ret)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return annualized_return(daily_ret) / dd


def max_drawdown(daily_ret: pd.Series) -> float:
    dd = drawdown_series(daily_ret)
    return dd.min()


def calmar_ratio(daily_ret: pd.Series) -> float:
    mdd = max_drawdown(daily_ret)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(daily_ret) / abs(mdd)


def hit_rate(daily_ret: pd.Series) -> float:
    nonzero = daily_ret[daily_ret != 0].dropna()
    if len(nonzero) == 0:
        return np.nan
    return (nonzero > 0).mean()


def return_stats(daily_ret: pd.Series) -> dict:
    return {
        "annual_return": annualized_return(daily_ret),
        "annual_vol": annualized_vol(daily_ret),
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_drawdown": max_drawdown(daily_ret),
        "calmar": calmar_ratio(daily_ret),
        "hit_rate": hit_rate(daily_ret),
    }


# ---------------------------------------------------------------------------
# Holding period (signal sign-flip based, not rebalance cadence)
# ---------------------------------------------------------------------------

def signal_holding_periods(signal_leg: pd.Series) -> pd.DataFrame:
    """Trading-day durations between consecutive sign flips of one leg's
    signal. One row per completed flip-to-flip interval.
    """
    s = signal_leg.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["start", "end", "duration_days"])
    changed = s != s.shift(1)
    changed.iloc[0] = True
    change_dates = s.index[changed]
    rows = []
    pos_lookup = {d: i for i, d in enumerate(s.index)}
    for i in range(1, len(change_dates)):
        start, end = change_dates[i - 1], change_dates[i]
        rows.append({"start": start, "end": end, "duration_days": pos_lookup[end] - pos_lookup[start]})
    return pd.DataFrame(rows)


def _duration_start_mask(durations: pd.DataFrame, period_start=None, period_end=None,
                          dates_index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Selects flip-intervals whose start date falls in the requested
    period -- either a contiguous [period_start, period_end) range (Dev /
    Full OOS) or an explicit, possibly discontiguous, set of dates
    (OOS High-Vol / Low-Vol regime subsets, which interleave day by day).
    """
    if dates_index is not None:
        return durations["start"].isin(dates_index)
    return (durations["start"] >= period_start) & (durations["start"] < period_end)


def avg_holding_period_by_leg(signal: pd.DataFrame, period_start=None, period_end=None,
                               dates_index: pd.DatetimeIndex | None = None) -> dict:
    """Average holding period (trading days) per leg, using only flip
    intervals that START within the requested period/date-set."""
    out = {}
    for leg in signal.columns:
        durations = signal_holding_periods(signal[leg])
        if len(durations) == 0:
            out[leg] = np.nan
            continue
        mask = _duration_start_mask(durations, period_start, period_end, dates_index)
        sub = durations.loc[mask, "duration_days"]
        out[leg] = sub.mean() if len(sub) else np.nan
    values = [v for v in out.values() if not np.isnan(v)]
    out["average"] = float(np.mean(values)) if values else np.nan
    return out


def num_signal_flips_by_leg(signal: pd.DataFrame, period_start=None, period_end=None,
                             dates_index: pd.DatetimeIndex | None = None) -> dict:
    out = {}
    for leg in signal.columns:
        durations = signal_holding_periods(signal[leg])
        if len(durations) == 0:
            out[leg] = 0
            continue
        mask = _duration_start_mask(durations, period_start, period_end, dates_index)
        out[leg] = int(mask.sum())
    return out


# ---------------------------------------------------------------------------
# Regime split (within OOS only): rolling 60d realized vol of the strategy's
# own gross daily return, median split
# ---------------------------------------------------------------------------

def regime_masks(gross_return: pd.Series, oos_start, oos_end, window: int = 60) -> dict:
    oos_ret = gross_return[(gross_return.index >= oos_start) & (gross_return.index < oos_end)]
    rolling_vol = oos_ret.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    median_vol = rolling_vol.median()
    high_vol_mask = rolling_vol >= median_vol
    low_vol_mask = rolling_vol < median_vol
    return {"rolling_vol": rolling_vol, "median_vol": median_vol,
            "high_vol_dates": rolling_vol.index[high_vol_mask.fillna(False)],
            "low_vol_dates": rolling_vol.index[low_vol_mask.fillna(False)]}


# ---------------------------------------------------------------------------
# Period-level full metrics bundle
# ---------------------------------------------------------------------------

def _slice(s: pd.Series, start=None, end=None, dates_index: pd.DatetimeIndex | None = None) -> pd.Series:
    if dates_index is not None:
        return s[s.index.isin(dates_index)]
    return s[(s.index >= start) & (s.index < end)]


def compute_period_metrics(bt: dict, signal: pd.DataFrame, period_start=None, period_end=None,
                            notional: float = 1_000_000.0,
                            dates_index: pd.DatetimeIndex | None = None) -> dict:
    """Full metrics bundle for one period, using a backtest result dict
    (as returned by backtest.run_backtest / run_benchmark).

    Pass either a contiguous (period_start, period_end) range (Dev / Full
    OOS) or an explicit `dates_index` (OOS High-Vol / Low-Vol regime
    subsets, which interleave day-by-day rather than forming a contiguous
    block).
    """
    gross = _slice(bt["gross_return"], period_start, period_end, dates_index)
    net_tx = _slice(bt["net_tx_return"], period_start, period_end, dates_index)
    net_all_in = _slice(bt["net_all_in_return"], period_start, period_end, dates_index)

    metrics = {
        "gross": return_stats(gross),
        "net_tx": return_stats(net_tx),
        "net_all_in": return_stats(net_all_in),
        "n_days": int(len(gross)),
    }

    if "weekly_weights" in bt:  # strategy result, not benchmark
        weekly_idx = bt["weekly_weights"].index
        if dates_index is not None:
            period_weekly_mask = weekly_idx.isin(dates_index)
        else:
            period_weekly_mask = (weekly_idx >= period_start) & (weekly_idx < period_end)
        n_rebal = int(period_weekly_mask.sum())

        turnover_period = bt["turnover"][period_weekly_mask]
        tx_cost_period = bt["tx_cost_frac"][period_weekly_mask]
        expense_period = _slice(bt["expense_drag"], period_start, period_end, dates_index)

        gross_pre = bt["gross_pre_cap"][period_weekly_mask]
        gross_post = bt["gross_post_cap"][period_weekly_mask]
        cap_bind = bt["cap_binding"][period_weekly_mask]

        total_tx_cost_frac = tx_cost_period.sum()
        total_expense_frac = expense_period.sum()

        metrics.update({
            "n_rebalances": n_rebal,
            "avg_holding_period_days": avg_holding_period_by_leg(signal, period_start, period_end, dates_index),
            "num_signal_flips": num_signal_flips_by_leg(signal, period_start, period_end, dates_index),
            "turnover_avg_weekly": float(turnover_period.mean()) if n_rebal else np.nan,
            "turnover_total": float(turnover_period.sum()) if n_rebal else np.nan,
            "total_tx_cost_bps_of_notional": total_tx_cost_frac * 1e4,
            "total_tx_cost_dollars": total_tx_cost_frac * notional,
            "total_expense_bps_of_notional": total_expense_frac * 1e4,
            "total_expense_dollars": total_expense_frac * notional,
            "total_all_in_cost_dollars": (total_tx_cost_frac + total_expense_frac) * notional,
            "avg_gross_exposure_precap": float(gross_pre.mean()) if n_rebal else np.nan,
            "avg_gross_exposure_postcap": float(gross_post.mean()) if n_rebal else np.nan,
            "pct_rebalances_cap_binds": float(cap_bind.mean()) if n_rebal else np.nan,
        })

    return metrics


def metrics_table_to_frame(metrics: dict) -> pd.DataFrame:
    """Flatten gross/net_tx/net_all_in return_stats into a small comparison
    DataFrame for display."""
    rows = {}
    for bucket in ["gross", "net_tx", "net_all_in"]:
        rows[bucket] = metrics[bucket]
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Cost sensitivity (0.5x / 1x / 1.5x / 2x base cost assumptions), OOS only
# ---------------------------------------------------------------------------

def cost_sensitivity_table(run_backtest_fn, prices, signal, oos_start, oos_end,
                            base_tx_cost_bps: dict, multipliers=(0.5, 1.0, 1.5, 2.0),
                            **backtest_kwargs) -> pd.DataFrame:
    """Re-run the OOS backtest at each cost multiplier (fixed stress-test
    grid, not a search) and report net Sharpe / Sortino / annualized return /
    Calmar (net-all-in) degradation as costs rise.
    """
    rows = []
    for m in multipliers:
        bt = run_backtest_fn(prices, signal, tx_cost_bps=base_tx_cost_bps,
                              cost_multiplier=m, **backtest_kwargs)
        net = _slice(bt["net_all_in_return"], oos_start, oos_end)
        stats = return_stats(net)
        row = {"cost_multiplier": m}
        for leg, bps in base_tx_cost_bps.items():
            row[f"{leg.lower()}_bps"] = bps * m
        row.update({
            "net_annual_return": stats["annual_return"],
            "net_sharpe": stats["sharpe"],
            "net_sortino": stats["sortino"],
            "net_calmar": stats["calmar"],
        })
        rows.append(row)
    return pd.DataFrame(rows)
