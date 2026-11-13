"""
Shared matplotlib chart builders used by the notebook. Colors follow a
validated colorblind-safe categorical order (fixed hue per series, never
cycled).
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from . import metrics as mx

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
AXIS = "#c3c2b7"
TEXT = "#0b0b0b"
MUTED = "#898781"

# Fixed categorical order (validated colorblind-safe, 5 slots: worst adjacent
# CVD ΔE 9.1, worst adjacent normal-vision ΔE 19.6) -- never reassigned or
# cycled per leg.
GLD_COLOR, IEF_COLOR, EURUSD_COLOR, GBPUSD_COLOR, USDJPY_COLOR = BLUE, ORANGE, AQUA, YELLOW, MAGENTA
STRATEGY_COLOR, BENCHMARK_COLOR = BLUE, ORANGE

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.grid": False,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def show_fig(fig) -> None:
    """Render `fig` to PNG bytes and display it in the notebook output.

    Needed because this module forces the Agg backend (so chart code also
    runs headless, e.g. via `jupyter nbconvert --execute`), and `plt.show()`
    under Agg is a no-op -- it has no display to show to. Rendering directly
    via `savefig` and `IPython.display.Image` sidesteps the backend/display
    -hook question entirely and works the same in a script, a headless
    nbconvert run, or an interactive notebook.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    display(Image(data=buf.read()))


def plot_cumulative_vs_benchmark(strategy_ret: pd.Series, benchmark_ret: pd.Series,
                                  title: str = "Cumulative net performance vs. benchmark"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    (mx.cumulative_return(strategy_ret) - 1).mul(100).plot(ax=ax, color=STRATEGY_COLOR, linewidth=1.6, label="Strategy (net, all-in)")
    (mx.cumulative_return(benchmark_ret) - 1).mul(100).plot(ax=ax, color=BENCHMARK_COLOR, linewidth=1.6, label="Benchmark (equal-weight buy-and-hold, net)")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_drawdown(strategy_ret: pd.Series, benchmark_ret: pd.Series, title: str = "Drawdown"):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    (mx.drawdown_series(strategy_ret) * 100).plot(ax=ax, color=STRATEGY_COLOR, linewidth=1.4, label="Strategy")
    (mx.drawdown_series(benchmark_ret) * 100).plot(ax=ax, color=BENCHMARK_COLOR, linewidth=1.4, label="Benchmark")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    return fig


def plot_rolling_sharpe_and_return(strategy_ret: pd.Series, window: int = 252,
                                    title: str = "Rolling 12-month return & Sharpe"):
    roll_ret = strategy_ret.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True) * 100
    roll_sharpe = (strategy_ret.rolling(window).mean() / strategy_ret.rolling(window).std()) * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    roll_ret.plot(ax=axes[0], color=STRATEGY_COLOR, linewidth=1.4)
    axes[0].axhline(0, color=AXIS, linewidth=0.8)
    axes[0].set_ylabel("Rolling 12m return (%)")
    axes[0].set_title(title)

    roll_sharpe.plot(ax=axes[1], color=STRATEGY_COLOR, linewidth=1.4)
    axes[1].axhline(0, color=AXIS, linewidth=0.8)
    axes[1].set_ylabel("Rolling 12m Sharpe")
    fig.tight_layout()
    return fig


def plot_positions(applied_weights: pd.DataFrame, title: str = "Position (applied weight) by leg over time"):
    """Small multiples: one panel per leg, sharing an x-axis and a common
    y-axis range. At 5 overlapping signed series, one shared-axes line plot
    is unreadable (constant criss-crossing between long and short); small
    multiples remove the need to color-match lines entirely, since each
    panel is self-labeled and only ever shows one series.
    """
    colors = {"GLD": GLD_COLOR, "IEF": IEF_COLOR, "EURUSD": EURUSD_COLOR,
              "GBPUSD": GBPUSD_COLOR, "USDJPY": USDJPY_COLOR}
    legs = list(applied_weights.columns)
    n = len(legs)

    fig, axes = plt.subplots(n, 1, figsize=(9, 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    y_max = applied_weights.abs().max().max() * 1.1
    for ax, leg in zip(axes, legs):
        ax.plot(applied_weights.index, applied_weights[leg], color=colors.get(leg, MUTED), linewidth=1.0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_ylim(-y_max, y_max)
        ax.set_ylabel(leg, rotation=0, ha="right", va="center", color=TEXT)

    axes[0].set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)
    return fig


def plot_regime_bar(high_vol_stats: dict, low_vol_stats: dict,
                     title: str = "OOS regime comparison: High-Vol vs Low-Vol (net, all-in)"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = ["High-Vol", "Low-Vol"]

    sharpes = [high_vol_stats["net_all_in"]["sharpe"], low_vol_stats["net_all_in"]["sharpe"]]
    axes[0].bar(labels, sharpes, color=[STRATEGY_COLOR, BENCHMARK_COLOR], width=0.5)
    axes[0].axhline(0, color=AXIS, linewidth=0.8)
    axes[0].set_title("Net Sharpe")

    vols = [high_vol_stats["net_all_in"]["annual_vol"] * 100, low_vol_stats["net_all_in"]["annual_vol"] * 100]
    axes[1].bar(labels, vols, color=[STRATEGY_COLOR, BENCHMARK_COLOR], width=0.5)
    axes[1].set_title("Annualized Vol (%)")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
