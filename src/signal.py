"""
Time-series momentum (TSMOM) signal construction.

signal_t = sign(P_t / P_{t-lookback} - 1)

Following Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum",
Journal of Financial Economics -- trailing return sign as the trend signal.
Primary lookback is 252 trading days (~12 months). 189 (~9mo) and 126
(~6mo) trading-day lookbacks are computed as sensitivity variants only,
reported side by side for robustness -- never used to select a "best"
lookback (no parameter search).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_LOOKBACK = 252
SENSITIVITY_LOOKBACKS = {"6mo": 126, "9mo": 189, "12mo": 252}


def trailing_return(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """P_t / P_{t-lookback} - 1, per column (per leg)."""
    return prices / prices.shift(lookback) - 1.0


def tsmom_signal(prices: pd.DataFrame, lookback: int = PRIMARY_LOOKBACK) -> pd.DataFrame:
    """sign(trailing_return), per leg. NaN during the lookback warm-up
    period (insufficient history) rather than defaulting to 0/flat, so
    downstream code must explicitly decide how to treat warm-up (backtest.py
    drops rows with any NaN signal).
    """
    ret = trailing_return(prices, lookback)
    sig = np.sign(ret)
    sig[ret.isna()] = np.nan
    return sig


def all_lookback_signals(prices: pd.DataFrame,
                          lookbacks: dict = SENSITIVITY_LOOKBACKS) -> dict[str, pd.DataFrame]:
    """Signal computed at each sensitivity lookback, for side-by-side
    reporting only (parameter-sensitivity requirement, reported not
    selected -- see the notebook's Robustness section).
    """
    return {label: tsmom_signal(prices, lb) for label, lb in lookbacks.items()}
