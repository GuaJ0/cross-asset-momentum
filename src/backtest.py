"""
Vectorized backtest loop: weekly rebalance, one-day lookahead shift, cost
application (transaction costs + ETF expense-ratio drag), gross/net/all-in
return series.

Cost assumptions (one-way, applied on every position CHANGE at each weekly
rebalance, proportional to |change in weight|):
    GLD:              7 bps  (commodity ETF bracket)
    IEF:               3 bps  (liquid rates-ETF bracket)
    EURUSD/GBPUSD/USDJPY:  2 bps each (typical institutional round-trip
             spread on the three most liquid G10 spot pairs is well under
             this; 2bps one-way is a conservative, round-number estimate
             applied uniformly across all three FX legs, not a fitted or
             per-pair-tuned number)
Annualized ETF expense ratios (charged daily, proportional to |weight|,
applied to both long and short legs as a simplifying assumption -- true
short-sellers typically don't bear a fund's expense ratio directly, but the
goal here is to account for expense ratios as a drag without carving out
shorts, and charging it symmetrically is the conservative/simpler choice):
    GLD: 0.40%/yr   IEF: 0.15%/yr   EURUSD/GBPUSD/USDJPY: 0%/yr each (spot
    FX has no fund wrapper or expense ratio; the excluded cost that matters
    for a real FX position -- swap/rollover points -- is a financing cost,
    grouped with the other financing-cost exclusions below, not an expense
    ratio)

Borrow/leverage-financing costs are DELIBERATELY EXCLUDED from these cost
lines (see reports/source_note.md for the quantitative disclosure of
assumed borrow fees and gross-exposure financing drag) -- this is a
disclosed simplification, not an oversight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import portfolio as pf

TX_COST_BPS = {"GLD": 7.0, "IEF": 3.0, "EURUSD": 2.0, "GBPUSD": 2.0, "USDJPY": 2.0}
EXPENSE_RATIOS = {"GLD": 0.0040, "IEF": 0.0015, "EURUSD": 0.0, "GBPUSD": 0.0, "USDJPY": 0.0}
TRADING_DAYS_PER_YEAR = 252


def _weekly_weights_to_daily(weekly_weights: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill rebalance-date weights onto the daily calendar: the
    weight decided at rebalance date r is held constant until the next
    rebalance date.
    """
    return weekly_weights.reindex(daily_index).ffill()


def _weekly_costs_to_daily(weekly_cost: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Place each rebalance's one-time transaction cost on its own
    rebalance day only (NOT forward-filled -- it's a one-off event cost).
    """
    return weekly_cost.reindex(daily_index, fill_value=0.0)


def run_backtest(prices: pd.DataFrame, signal: pd.DataFrame,
                  target_portfolio_vol: float = 0.10,
                  target_leg_vol: float | None = None,
                  vol_window: int = 60,
                  gross_cap: float = 2.0,
                  tx_cost_bps: dict | None = None,
                  expense_ratios: dict | None = None,
                  cost_multiplier: float = 1.0,
                  combo_method: str = "primary") -> dict:
    """Run the weekly-rebalanced, vol-targeted, cost-aware backtest.

    combo_method: "primary" (deterministic inverse-vol, headline numbers)
        or "risk_parity_appendix" (exploratory ERC comparison -- notebook
        appendix only, never used for headline metrics).
    cost_multiplier scales tx_cost_bps uniformly (used for the
    0.5x/1x/1.5x/2x cost-sensitivity stress test -- not a search, a fixed
    stress-test grid over the same unchanged strategy).

    Returns a dict of daily return series (gross, net of transaction costs
    only, and net all-in i.e. transaction costs + expense ratios) plus
    rebalance-level diagnostics (gross exposure pre/post cap, cap-binding
    flags, turnover) needed for the metrics table.
    """
    tx_cost_bps = dict(TX_COST_BPS if tx_cost_bps is None else tx_cost_bps)
    tx_cost_bps = {k: v * cost_multiplier for k, v in tx_cost_bps.items()}
    expense_ratios = dict(EXPENSE_RATIOS if expense_ratios is None else expense_ratios)

    legs = list(prices.columns)
    daily_index = prices.index
    rebalance_dates = pf.get_weekly_rebalance_dates(daily_index)

    if combo_method == "primary":
        combo = pf.combine_portfolio_primary(
            signal, prices, rebalance_dates,
            target_portfolio_vol=target_portfolio_vol,
            target_leg_vol=target_leg_vol,
            vol_window=vol_window, gross_cap=gross_cap,
        )
    elif combo_method == "risk_parity_appendix":
        combo = pf.combine_portfolio_risk_parity_appendix(
            signal, prices, rebalance_dates,
            target_portfolio_vol=target_portfolio_vol,
            vol_window=vol_window, gross_cap=gross_cap,
        )
    else:
        raise ValueError(f"Unknown combo_method: {combo_method!r}")

    weekly_weights = combo["weights"].dropna(how="all")

    # turnover / transaction cost at each rebalance, proportional to |Δw_leg|
    weight_change = weekly_weights.diff()
    weight_change.iloc[0] = weekly_weights.iloc[0]  # entering from flat
    turnover = weight_change.abs().sum(axis=1)
    tx_cost_frac = pd.Series(0.0, index=weekly_weights.index)
    for leg in legs:
        tx_cost_frac += (tx_cost_bps[leg] / 1e4) * weight_change[leg].abs()

    daily_weights = _weekly_weights_to_daily(weekly_weights, daily_index)
    daily_tx_cost = _weekly_costs_to_daily(tx_cost_frac, daily_index)

    # ---- shift to avoid lookahead: weights/costs decided using data
    # through day t are applied to the RETURN realized on day t+1, i.e.
    # they take effect one trading day after the information they're
    # based on becomes available. ----
    applied_weights = daily_weights.shift(1)
    applied_tx_cost = daily_tx_cost.shift(1).fillna(0.0)

    leg_returns = pf.daily_returns(prices)
    gross_return = (applied_weights * leg_returns).sum(axis=1)

    expense_drag = pd.Series(0.0, index=daily_index)
    for leg in legs:
        expense_drag += applied_weights[leg].abs().fillna(0.0) * (expense_ratios[leg] / TRADING_DAYS_PER_YEAR)

    net_tx_return = gross_return - applied_tx_cost
    net_all_in_return = gross_return - applied_tx_cost - expense_drag

    valid = applied_weights.notna().any(axis=1)

    # gross_pre_cap/gross_post_cap/cap_binding come from combo, which is
    # computed over ALL rebalance_dates including the pre-warm-up weeks
    # where the signal is still NaN (weights NaN -> abs().sum() silently
    # reads as 0, not NaN). Restrict these diagnostics to the same valid
    # rebalance set as weekly_weights so period slicing/masking downstream
    # can't accidentally include phantom "flat" warm-up weeks.
    valid_rebal_idx = weekly_weights.index

    return {
        "gross_return": gross_return[valid],
        "net_tx_return": net_tx_return[valid],
        "net_all_in_return": net_all_in_return[valid],
        "applied_weights": applied_weights[valid],
        "weekly_weights": weekly_weights,
        "turnover": turnover,
        "tx_cost_frac": tx_cost_frac,
        "gross_pre_cap": combo["gross_pre_cap"].loc[valid_rebal_idx],
        "gross_post_cap": combo["gross_post_cap"].loc[valid_rebal_idx],
        "cap_binding": combo["cap_binding"].loc[valid_rebal_idx],
        "rebalance_dates": rebalance_dates,
        "expense_drag": expense_drag[valid],
    }


def run_benchmark(prices: pd.DataFrame, tx_cost_bps: dict | None = None,
                   expense_ratios: dict | None = None) -> dict:
    """Static buy-and-hold basket: fixed equal weight per leg (1/len(legs)),
    no signal, no vol targeting, held with an annual drift-reset back to
    equal weight (small rebalancing turnover once a year, costed the same
    way as the strategy for a fair net-of-cost comparison).
    """
    tx_cost_bps = dict(TX_COST_BPS if tx_cost_bps is None else tx_cost_bps)
    expense_ratios = dict(EXPENSE_RATIOS if expense_ratios is None else expense_ratios)

    legs = list(prices.columns)
    daily_index = prices.index
    leg_returns = pf.daily_returns(prices)

    target = pd.Series({leg: 1.0 / len(legs) for leg in legs})
    weights = pd.DataFrame(index=daily_index, columns=legs, dtype=float)
    current = target.copy()
    reset_dates = []
    last_year = None
    for i, t in enumerate(daily_index):
        if i == 0:
            current = target.copy()
            reset_dates.append(t)
        elif t.year != last_year:
            current = target.copy()  # annual drift-reset
            reset_dates.append(t)
        else:
            growth = (1.0 + leg_returns.loc[t].fillna(0.0))
            drifted = current * growth
            current = drifted / drifted.sum()
        weights.loc[t] = current
        last_year = t.year

    weight_change = weights.diff()
    weight_change.iloc[0] = weights.iloc[0]
    tx_cost_daily = pd.Series(0.0, index=daily_index)
    for leg in legs:
        tx_cost_daily += (tx_cost_bps[leg] / 1e4) * weight_change[leg].abs().fillna(0.0)

    applied_weights = weights.shift(1)
    applied_tx_cost = tx_cost_daily.shift(1).fillna(0.0)

    gross_return = (applied_weights * leg_returns).sum(axis=1)
    expense_drag = pd.Series(0.0, index=daily_index)
    for leg in legs:
        expense_drag += applied_weights[leg].abs().fillna(0.0) * (expense_ratios[leg] / TRADING_DAYS_PER_YEAR)

    net_tx_return = gross_return - applied_tx_cost
    net_all_in_return = gross_return - applied_tx_cost - expense_drag

    valid = applied_weights.notna().any(axis=1)
    return {
        "gross_return": gross_return[valid],
        "net_tx_return": net_tx_return[valid],
        "net_all_in_return": net_all_in_return[valid],
        "applied_weights": applied_weights[valid],
        "expense_drag": expense_drag[valid],
    }
