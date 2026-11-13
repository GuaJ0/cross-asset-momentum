"""
Data loading, cleaning, and validation for the GLD/IEF/EURUSD/GBPUSD/USDJPY
TSMOM strategy.

Design note: Gold (GLD) and the govt bond leg (IEF) are single-source,
continuous yfinance series -- no adapter needed, no splice. The three FX legs
can be built two ways:
  * If vendor spot-FX CSVs are present in `data/vendor_fx/` (the original
    research run used a vendor pack whose history stops 2016-12-30), each leg
    is the vendor history before 2017-01-01 spliced with yfinance from
    2017-01-01 onward via `load_fx_spliced`. This is a plain concatenation,
    not a return-rebasing splice, because both sides are real spot-rate
    quotes in the same units.
  * Otherwise each FX leg is loaded from yfinance alone, which covers the
    whole GLD-bounded window (Nov 2004 onward). The vendor files are not
    redistributed in this repo, so this is the default for anyone re-running it.

Quote convention note: EURUSD and GBPUSD are quoted foreign-currency-as-base
(USD per 1 EUR / GBP), matching real FX market convention for those pairs.
USDJPY is quoted USD-as-base (JPY per 1 USD) -- this is not an inconsistency,
it's the real market convention for JPY (unlike EUR/GBP). A "+1" TSMOM signal
therefore means "long EUR"/"long GBP" on those two legs but "long USD" on the
JPY leg -- sign() only cares about each leg's own trailing-return direction,
so this doesn't affect the mechanics, but it's worth knowing when reading the
position chart. See reports/source_note.md and the notebook's Introduction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
PROVIDED_FX_DIR = ROOT / "data" / "vendor_fx"

FX_SPLICE_DATE = pd.Timestamp("2017-01-01")

# leg name -> (provided CSV filename, yfinance ticker / raw CSV filename)
FX_LEGS = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
}


# ---------------------------------------------------------------------------
# Per-source readers -- each returns the same standardized schema
# ---------------------------------------------------------------------------

def _read_yfinance_raw_csv(path: Path) -> pd.DataFrame:
    """Read a raw yfinance CSV (as written by scripts/download_data.py).

    Uses the 'Adj Close' column as AdjClose. Adjusted close is used
    throughout (not raw Close) because dividends/distributions are
    material for the bond ETF (IEF) and harmless-but-consistent to include
    for GLD -- see reports/source_note.md.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.rename(columns={"Adj Close": "AdjClose"})
    return df[STANDARD_COLUMNS].sort_index()


def _read_provided_fx_csv(path: Path) -> pd.DataFrame:
    """Read a provided spot-FX CSV: Date, Open, High, Low, Close only.

    No Volume/AdjClose in the source -- FX has no dividend adjustment, so
    AdjClose is just Close, and Volume is set to NaN for schema consistency.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    expected = {"Open", "High", "Low", "Close"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected FX columns {missing}")
    df["AdjClose"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df[STANDARD_COLUMNS].sort_index()


# ---------------------------------------------------------------------------
# FX adapter: provided spot CSV (through 2016-12-30) + yfinance (2017-present)
# ---------------------------------------------------------------------------

def load_fx_spliced(provided_path: Path, yfinance_path: Path,
                     splice_date: pd.Timestamp = FX_SPLICE_DATE,
                     max_boundary_gap: float = 0.03) -> pd.DataFrame:
    """Concatenate a provided spot-FX series with its yfinance continuation.

    provided_path covers rows before `splice_date`; yfinance_path covers
    `splice_date` onward. Both are real spot quotes in the same units, so
    this is a plain join (no return-rebasing needed, unlike a back-adjusted
    futures splice). A boundary sanity check guards against a silent vendor
    discontinuity: if the nearest quotes on either side of the splice
    disagree by more than `max_boundary_gap`, something is wrong with one of
    the sources (real spot FX does not gap like that day-to-day) and this
    refuses to proceed silently.
    """
    provided = _read_provided_fx_csv(provided_path)
    provided = provided[provided.index < splice_date]

    live = _read_yfinance_raw_csv(yfinance_path)
    live = live[live.index >= splice_date]

    if len(provided) == 0 or len(live) == 0:
        raise ValueError(
            f"{provided_path.stem} splice produced an empty half: provided={len(provided)} rows, "
            f"live={len(live)} rows (splice_date={splice_date.date()})"
        )

    boundary_before = provided["AdjClose"].iloc[-1]
    boundary_after = live["AdjClose"].iloc[0]
    pct_gap = abs(boundary_after - boundary_before) / boundary_before
    if pct_gap > max_boundary_gap:
        raise ValueError(
            f"{provided_path.stem} splice boundary gap too large: provided closed at "
            f"{boundary_before:.4f} ({provided.index[-1].date()}), live opens at "
            f"{boundary_after:.4f} ({live.index[0].date()}) -- a {pct_gap:.1%} gap. "
            "Refusing to splice silently."
        )

    combined = pd.concat([provided, live]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def load_leg(ticker: str, raw_dir: Path) -> pd.DataFrame:
    """Load a plain single-source yfinance leg (GLD, IEF)."""
    return _read_yfinance_raw_csv(Path(raw_dir) / f"{ticker}.csv")


# ---------------------------------------------------------------------------
# Validation (no filling -- report only)
# ---------------------------------------------------------------------------

def validate_prices(df: pd.DataFrame, name: str) -> dict:
    """Check duplicates, missing/NaN observations, non-positive or stale
    prices. Never fills or interpolates -- reports issues for disclosure.
    """
    issues = {}
    issues["n_rows"] = int(len(df))
    issues["date_range"] = (str(df.index.min().date()), str(df.index.max().date()))
    issues["n_duplicate_dates"] = int(df.index.duplicated().sum())
    issues["n_nan_adjclose"] = int(df["AdjClose"].isna().sum())
    issues["n_non_positive_adjclose"] = int((df["AdjClose"] <= 0).sum())
    # stale price = AdjClose unchanged for >= 5 consecutive trading sessions
    unchanged = df["AdjClose"].diff().eq(0)
    stale_streak = unchanged.groupby((~unchanged).cumsum()).cumsum()
    issues["n_stale_5plus_days"] = int((stale_streak >= 4).sum())
    if issues["n_duplicate_dates"] or issues["n_nan_adjclose"] or issues["n_non_positive_adjclose"]:
        print(f"[validate_prices:{name}] WARNING: {issues}")
    return issues


def align_legs(legs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Align all legs on the INTERSECTION of trading dates. Dates present
    in some legs but not others are dropped (not forward-filled), since
    doing otherwise would fabricate prices on days a given instrument did
    not actually trade. Returns the aligned legs plus a report of how many
    dates were dropped per leg.
    """
    common_index = None
    for df in legs.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = common_index.sort_values()

    report = {"common_dates": int(len(common_index))}
    aligned = {}
    for name, df in legs.items():
        dropped = df.index.difference(common_index)
        report[f"{name}_dropped_dates"] = int(len(dropped))
        report[f"{name}_original_dates"] = int(len(df))
        aligned[name] = df.loc[common_index]
    return aligned, report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_combined_prices(raw_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load, validate, and align GLD/IEF/EURUSD/GBPUSD/USDJPY. Returns a
    DataFrame of AdjClose columns aligned on the intersection of trading
    dates, plus a diagnostics dict covering validation and alignment
    (dropped-date counts etc.).
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else ROOT / "data" / "raw"

    legs = {
        "GLD": load_leg("GLD", raw_dir),
        "IEF": load_leg("IEF", raw_dir),
    }
    for leg_name, filename in FX_LEGS.items():
        vendor_path = PROVIDED_FX_DIR / f"{filename}.csv"
        if vendor_path.exists():
            legs[leg_name] = load_fx_spliced(
                provided_path=vendor_path,
                yfinance_path=raw_dir / f"{filename}.csv",
            )
        else:
            # No vendor history available: use yfinance for the full window.
            legs[leg_name] = load_leg(filename, raw_dir)

    diagnostics = {"validation": {name: validate_prices(df, name) for name, df in legs.items()}}

    aligned, align_report = align_legs(legs)
    diagnostics["alignment"] = align_report

    combined = pd.DataFrame({name: df["AdjClose"] for name, df in aligned.items()})
    combined.index.name = "Date"
    diagnostics["combined_range"] = (str(combined.index.min().date()), str(combined.index.max().date()))
    diagnostics["combined_n_dates"] = int(len(combined))
    return combined, diagnostics


def clean_and_cache(raw_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Build combined prices and cache to data/cleaned/combined_prices.parquet
    plus a JSON diagnostics log alongside it, so the notebook can reproduce
    offline.
    """
    combined, diagnostics = build_combined_prices(raw_dir=raw_dir)
    out_dir = ROOT / "data" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "combined_prices.parquet"
    combined.to_parquet(parquet_path)
    log_path = out_dir / "cleaning_log.json"
    with open(log_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Wrote {parquet_path.relative_to(ROOT)} ({len(combined)} rows) and {log_path.relative_to(ROOT)}")
    return combined, diagnostics


def load_cached_prices() -> pd.DataFrame:
    """Load the cached, cleaned combined-price parquet (no internet needed).
    This is what the notebook's main path uses.
    """
    path = ROOT / "data" / "cleaned" / "combined_prices.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    clean_and_cache()
