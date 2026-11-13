# Source Note

## Data sources

| Leg | Instrument | Source | Field | Frequency |
|---|---|---|---|---|
| Gold | GLD (SPDR Gold Shares) | Yahoo Finance via `yfinance` | Adjusted close | Daily |
| Govt bonds | IEF (iShares 7-10 Year Treasury Bond ETF) | Yahoo Finance via `yfinance` | Adjusted close | Daily |
| G10 FX | EUR/USD, GBP/USD, USD/JPY spot | Vendor spot history to 2016-12-30, then `yfinance` (`EURUSD=X`, `GBPUSD=X`, `USDJPY=X`) from 2017 | Close | Daily |

- **Download date for this run:** 18 to 19 August 2026, through the latest available session.
- **Cleaned cache:** `data/cleaned/combined_prices.parquet`, built by `src/data_pipeline.py`. The parquet is not committed because part of its FX history comes from vendor files that aren't mine to redistribute. Run `scripts/download_data.py` and `python -m src.data_pipeline` to rebuild it.
- **Without the vendor files**, the pipeline uses `yfinance` for the whole FX history. Results may differ slightly from the committed notebook.

## Why ETFs instead of continuous futures for gold and bonds

The natural instruments for this strategy are gold and Treasury futures. I had back-adjusted continuous-futures exports for both, but did not use them, for two reasons:

1. **History stopped in 2016.** That cannot support a 10-year out-of-sample window running to the present.
2. **Back-adjusted prices are not on a real-dollar scale.** A difference-back-adjusted gold series sat at roughly 2 to 2.5x the true spot price in 2004 and 2015, because roll adjustments accumulate. Extending it with live data would need return-rebasing, not a simple join. The signal's direction would survive back-adjustment, but volatility estimates from it would be increasingly understated further back in history.

GLD and IEF avoid both problems, with continuous single-source history from 2004.

**What this costs:** ETFs are not futures. They have no roll or carry dynamics, they charge expense ratios instead of embedding financing in the price, and shorting them requires borrowing shares. This is a real conceptual mismatch, disclosed in the notebook's takeaways.

## The FX splice

The vendor FX files and Yahoo Finance both quote real spot rates in the same units, so extending the vendor history is a plain concatenation:

- Vendor data supplies every date before 2017-01-01.
- `yfinance` supplies every date from 2017-01-01 onward.
- `load_fx_spliced` refuses to proceed if the two sides differ by more than 3% at the join.

Boundary check for this run:

| Leg | Vendor close (2016-12-30) | yfinance (next session, 2017) | Gap |
|---|---|---|---|
| EURUSD | 1.0515 | 1.0527 | 0.1% |
| GBPUSD | 1.2334 | 1.2349 | 0.1% |
| USDJPY | 116.87 | 116.79 | 0.1% |

All three are consistent with normal daily FX moves. It is still a seam between two data sources, and the notebook says so.

**Quote convention.** EUR/USD and GBP/USD are quoted as USD per EUR or GBP. USD/JPY is quoted as JPY per USD, which is the real market convention. A +1 signal therefore means long EUR or GBP, but long USD on the JPY leg.

**Diversification caveat.** All three FX legs are quoted against USD, so a broad dollar move can push them in the same direction. The notebook measures this: mean absolute correlation among the FX legs is about 0.38 (EUR/USD and GBP/USD about 0.64), against about 0.15 between the FX legs and GLD/IEF.

## Calendar alignment

GLD and IEF trade on the US equity calendar, while FX trades on most weekdays including some US holidays. All five legs are aligned on the dates where every leg traded, and nothing is forward-filled. For this run that dropped 4 dates from GLD, 226 from IEF (before GLD's November 2004 launch), 1,715 from EUR/USD and 3,782 each from GBP/USD and USD/JPY (mostly history before 2004). The full log is in `data/cleaned/cleaning_log.json`.
