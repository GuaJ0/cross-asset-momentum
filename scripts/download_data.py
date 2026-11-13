"""
Live data pull for the GLD / IEF / EURUSD / GBPUSD / USDJPY strategy.

Run this manually and separately from the notebook:
    python scripts/download_data.py

Fetches raw daily OHLCV (unadjusted Close + Adj Close) for GLD and IEF, plus
EURUSD=X, GBPUSD=X, USDJPY=X (spot FX), from Yahoo Finance via yfinance, from
2004-01-01 (covers GLD's Nov 2004 IPO) through the latest available session,
and writes one CSV per ticker into data/raw/.

All five legs are saved in full. If vendor spot-FX CSVs are placed in
data/vendor_fx/, src/data_pipeline.py uses them for pre-2017 FX history and
splices yfinance on from 2017-01-01 (this is how the original research run was
built); otherwise the yfinance FX history is used for the whole window.

This script requires internet access. The notebook itself never calls this
script or yfinance directly -- it reads the cleaned, cached Parquet file
produced by src/data_pipeline.py so the deliverable reproduces offline.
"""
import sys
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
START_DATE = "2004-01-01"
TICKERS = ["GLD", "IEF", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]
RAW_DIR = ROOT / "data" / "raw"


def download_ticker(ticker: str) -> "pd.DataFrame":
    df = yf.download(
        ticker,
        start=START_DATE,
        end=None,
        auto_adjust=False,  # keep raw Close AND Adj Close separate/explicit
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"Downloading {ticker} from yfinance (start={START_DATE}) ...")
        df = download_ticker(ticker)
        print(f"  {ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
        out_name = f"{ticker[:-2]}.csv" if ticker.endswith("=X") else f"{ticker}.csv"
        out_path = RAW_DIR / out_name
        df.to_csv(out_path)
        print(f"  wrote {out_path.relative_to(ROOT)}")

    print("\nDone. Next: run `python -m src.data_pipeline` to produce "
          "data/cleaned/combined_prices.parquet.")


if __name__ == "__main__":
    sys.exit(main())
