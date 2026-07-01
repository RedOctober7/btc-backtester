# BTC 4h Backtesting Engine

A correctness-first, event-driven backtesting engine for BTC 4h candles.
No lookahead bias. Honest fees. No backtesting libraries.

## Why event-driven, not vectorized?

Vectorized approaches (computing signals across the whole array at once) are faster
but make it easy to accidentally peek at future bars. This engine iterates bar-by-bar:
on bar `t`, the strategy literally cannot see bar `t+1` because the `Context` object
handed to it only contains rows `0..t`. Vectorizing for speed is a deliberate later
optimization once correctness is proven.

## Execution model

> **Signal on bar `t` -> fill at bar `t+1`'s OPEN price.**

A signal generated at bar `t`'s close is *queued* as a pending order and executed
at the very next bar's opening price. This reflects reality: you can't trade on a
candle until it's closed. This is the "next-bar open" fill rule.

## Setup (Windows PowerShell 5.1)

Commands must be run **one per line** — PowerShell 5.1 does not support `&&` chaining.

```powershell
# Navigate into the project directory first
cd btc-backtester

# Option A: uv (faster, recommended if installed)
uv venv .venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

# Option B: standard venv
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run a backtest

```powershell
# Default: BTCUSDT, 3-year range, SMA(50,200) with 8% stop, 10k USDT
python run.py

# Custom range and parameters
python run.py --start 2021-01-01 --end 2024-01-01 --fast 20 --slow 100 --stop 0.05

# Skip chart generation
python run.py --no-plot

# Overlay swing points, Fibonacci levels, and trendlines on the price panel
python run.py --start 2026-04-01 --show-levels

# Update the README's canonical chart (backtest_results_full.png)
python run.py --update-readme-chart
```

## Run tests

```powershell
pytest tests/ -v
```

## Defining a new strategy

Create a class in `strategies/` that inherits from `Strategy`:

```python
from strategies.base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def init(self, data: pd.DataFrame) -> None:
        # Called ONCE. Receives the full OHLCV DataFrame.
        # ONLY use this to precompute strictly backward-looking indicators.
        # Never use centered windows, forward fills, or any future-leaking transform.
        self.sma = data["close"].rolling(20).mean()

    def on_bar(self, context) -> None:
        # Called once per bar. context is windowed to the current bar only.
        i = context.current_idx
        val = self.sma.iloc[i]
        if pd.isna(val):
            return  # warmup guard -- always check for NaN explicitly
        if context.close > val and not context.position.has_position:
            context.buy()
        elif context.close < val and context.position.has_position:
            context.close_position(reason="crossover")
```

Then pass it to `run_backtest()`.

## Technical-analysis overlay (visualization only)

`analysis/levels.py` is a standalone module (depends only on pandas + numpy,
imports nothing from the engine) that marks chart structure. It exposes three
functions, each returning plain dataclasses — no chart objects — so `run.py`
and a future UI can consume them identically:

- `find_swing_points(df, lookback=5)` — fractal swing highs/lows.
- `compute_fib_levels(df, swing_points)` — Fibonacci retracement grid.
- `fit_trendline(df, swing_points, side)` — OLS support/resistance line.

Enable the overlay with `python run.py --show-levels` (or `plot_results(...,
show_levels=True)` from code).

### What the overlay draws

- **Swing points** — small gold ▼ markers at highs, purple ▲ markers at lows.
- **Fibonacci levels** — amber dashed horizontal lines at their true prices,
  decoded by a compact legend key in the top-left corner. The labels live in the
  key, not inline next to each line, so they stay readable even when the swing
  range is narrow. When the swing band is narrower than **8.5% of the visible
  y-axis range** the five lines would merge into one indistinguishable smear, so
  they are replaced by a single translucent shaded band (low..high); the legend
  key still lists all five exact levels. (Measured: a trending window's band is
  ~13.8% of its y-range → five lines; a choppy window's ~5.4% → shaded band.)
- **Trendlines** — teal for support, coral for resistance, tagged with their R²
  on the chart; low-confidence fits (R² < 0.5) drawn faint and dashed, not
  hidden. Labels are clamped and vertically de-collided so none clips at either
  edge or overlaps another, even when all four (two sides × two modes) land near
  the same corner.

### These are algorithmic approximations, not ground truth

Each function implements **one** standard, defensible method — not "the"
correct one. Two competent chartists routinely disagree on where a swing sits,
which swing pair anchors a Fibonacci grid, and how a trendline should be drawn.
This is the same honest framing as the [strategy regime-dependence](#strategy-regime-dependence)
note below: the output is a *reproducible, documented approximation*, useful for
visualization, not an authoritative read of the market. Every result carries
metadata (method, parameters, analyzed range) so two runs with different
settings are never silently conflated.

The specific, documented choices:

- **Swing points** — a bar is a swing high if its `high` is the strict, unique
  maximum of a symmetric `2*lookback+1` window (inverse for lows). The first and
  last `lookback` bars can never qualify (they lack bars on one side); too-short
  series (`len < lookback*2+1`) return empty rather than erroring.
- **Fibonacci** — anchored on the **most recent** swing high and **most recent**
  swing low (regardless of order, as long as both exist). Levels: 23.6%, 38.2%,
  50%, 61.8%, 78.6%. A degenerate (near-flat) range returns `None` instead of
  five identical prices.
- **Trendline** — OLS through swing points of one type. Fit quality is reported
  honestly: **R² below 0.5 is still returned but tagged `low_confidence=True`**
  (drawn faint + dashed, not hidden). Fewer than `min_points` (default 3) swings
  returns `None`. Point selection has two modes (`n_points`, below).

### Two lookback defaults, on purpose

The swing `lookback` has **two intentionally different defaults** for two
different jobs — this is a deliberate design choice, not an inconsistency:

| Default | Where | Why |
|---|---|---|
| `lookback=5` | `find_swing_points()` | Detection default — maximum sensitivity, surfaces every minor pivot for a script/UI that wants them. |
| `lookback=8` | chart overlay (`plot_results(show_levels=True)`) | Rendering default — fewer, more *significant* swings, which produces a legible Fibonacci grid and a clear high- vs low-confidence trendline contrast. At lookback 5 the "most recent high + most recent low" pair is often two adjacent minor pivots, collapsing the Fib grid into a thin band. |

### Recent-N vs. whole-window trendlines

`fit_trendline(..., n_points=...)` answers two genuinely different questions, and
the default is unchanged:

| `n_points` | Mode | Question | Projection |
|---|---|---|---|
| int `N` (default `5`) | recent | "Is there a tradeable trend **right now**?" — fit the most recent N swings. | extends 20% of its span past the last swing (a short forward stub shows where the current trend points next) |
| `"all"` | whole-window | "Was this entire period trending or choppy **overall**?" — fit every swing of that type across the range. | **none** — the line characterizes the past, not a forward breakout, so it stops at the last swing (extrapolating a multi-month slope forward is meaningless and visually absurd) |

These can point in **opposite directions** — which is the whole point. On the
2018-08 → 2019-04 BTC window (choppy overall; the SMA strategy lost money there),
the support fits diverge sharply:

| | recent-5 | all-swings |
|---|---|---|
| Support | R²=0.91, slope **+0.95/bar** (rising) | R²=0.71, slope **−2.83/bar** (falling) |
| Resistance | R²=0.50, slope **+1.01/bar** (rising) | R²=0.78, slope **−2.99/bar** (falling) |

Both recent-5 lines slope **up** — they caught the clean late-March 2019 bounce.
Both all-swings lines slope **down** — they caught the 2018 bear decline. Recent-N
is a *local* signal; "all" is a *global* characterization. (Note the all-swings
fits aren't low-confidence chop — this window was a net decline with a choppy
bottom, not pure sideways, and the regression honestly reports that.) On a chart,
pass `trendline_n_points=(5, "all")` to draw both per side; "all"-mode lines are
dotted and labeled `(all swings)` vs `(recent-5)`.

### ⚠️ Lookahead boundary — visualization only

Swing detection is **centered**: a swing at bar `i` is only confirmed once
`lookback` more bars print to its right. That is fine for charting historical
data, but it means **these functions must NOT be called inside a strategy's
`on_bar()` trading logic** — doing so would leak bars `i+1..i+lookback` into the
decision at bar `i`, violating the engine's lookahead firewall. A causal
(right-edge-only) variant would have to be written first; that adaptation is
deliberately out of scope here.

## Sample results — SMA(50,200), 8% stop, 10k USDT

> **Representative run: 2022-01-01 to 2026-06-25 (4.5 years, 31 trades)**

| Metric | Value |
|---|---|
| Total Return | +61.42% |
| CAGR | 11.27% |
| Annualized Volatility | 25.03% |
| Sharpe Ratio | 0.55 |
| Max Drawdown | -41.75% |
| Max Drawdown Duration | 734 days |
| Win Rate | 38.71% |
| Avg Win / Avg Loss | +$1,784 / -$803 |
| Profit Factor | 1.40 |
| Exposure | 35.1% |

The chart uses a TradingView-style dark theme with three panels: BTC 4h price (top — candlesticks for ranges up to 90 days, a price line for longer ranges), equity curve in USDT (middle), and underwater drawdown as a percentage (bottom, red fill). Short-range runs show entry/exit triangle markers at actual fill prices; multi-year runs switch to line mode to avoid overplotting.

![Backtest chart](backtest_results_full.png)

### Why earlier runs showed stronger numbers

A 2022-01-01 to 2024-06-01 subset of the same strategy produced +111% total return,
Sharpe 1.24, and max drawdown -20.7% — substantially better on every metric. That
window is not representative, and should not be quoted as "the" performance number.

What happened: the strategy caught the 2023-2024 BTC bull run almost perfectly. It sat
out most of the 2022 bear market (flat from ~$47k down to ~$16k), then re-entered in
January 2023 and rode two large trending moves — a +25% trade in Jan-Feb 2023 and a
+59% trade from September 2023 to January 2024 (+$5,979), followed immediately by a
+42% trade in the February-March 2024 rally (+$6,734). Those two trades alone account
for the bulk of the shorter window's gains.

The 2024-2026 period reversed much of this. After BTC's post-ATH oscillation began,
the strategy accumulated 14 additional trades with a negative net: repeated stop-outs
in the $60k-$110k range as price moved in both directions without sustaining a trend.
This is the whipsaw regime described in the limitations section. Equity peaked near
$24k in early 2024 and ground back to ~$16k by mid-2026.

The full 4.5-year run is harder to look at but more honest: it includes both the regime
the strategy is built for (trending) and the one it handles poorly (choppy). A Sharpe
of 0.55 and profit factor of 1.40 over that full period mean the strategy is still
positive-expectancy, but not a free lunch.

### Strategy performance across regimes

Same strategy, identical parameters, three different market conditions:

| Period | Regime | Total Return | Sharpe | Max Drawdown | Profit Factor | Trades |
|---|---|---|---|---|---|---|
| 2022-01-01 to 2024-06-01 | Trending | +111.0% | 1.24 | -20.7% | 3.11 | 17 |
| 2022-01-01 to 2026-06-25 | Mixed | +61.4% | 0.55 | -41.75% | 1.40 | 31 |
| 2018-08-01 to 2019-04-01 | Choppy/sideways | -4.87% | -0.23 | -11.47% | 0.42 | 4 |

The only variable across these three runs is the market regime — parameters, fees, and
logic are identical. A profit factor range of 0.42 to 3.11 on the same rules illustrates
the core lesson of trend-following: regime awareness matters more than parameter tuning,
because no parameter set makes a crossover strategy profitable in a directionless market.
This is precisely why walk-forward testing and regime detection exist — and why they are
correctly out of scope for this v1.

## Walk-Forward Analysis

A single backtest optimizes parameters on the same data it then measures — the result is an in-sample fit, not a forward estimate. Walk-forward analysis breaks that circularity: optimize parameters on an in-sample (IS) window, test the winning parameters on the very next out-of-sample (OOS) window the optimizer never saw, roll the window forward, and repeat. Stitching the OOS segments together produces one continuous equity curve built entirely from data the optimizer was blind to. Walk-Forward Efficiency (WFE) is the ratio of annualized OOS return to annualized IS return; it measures how much of the IS edge survives contact with unseen data rather than evaporating as overfit.

### How to run

```powershell
python walk_forward.py
```

Parameters, grid, IS/OOS window sizes, and the holdout split date are all set inside the `__main__` block of `walk_forward.py` at the repo root.

### How to read WFE

| WFE | Interpretation |
|---|---|
| >= 60% | Robust — OOS captures most of the IS edge |
| 40–60% | Marginal — real signal present, IS optimizer somewhat optimistic |
| < 40% | Overfit — IS edge does not carry to OOS |

### MACrossover results (SMA 20/200, 8% stop)

The grid search was narrowed through three iterations (27 combos → 6 → 2) after observing WFE of 17% on the full grid. With `slow_period` fixed to 200, WFE stabilized at 53%.

**Dev period walk-forward OOS (2022-01 to 2026-02, 44 folds):**

| Metric | Value |
|---|---|
| Stitched OOS return | +83.6% |
| CAGR | +18.3% |
| Sharpe | 0.81 |
| Max Drawdown | -25.0% |
| WFE | 53% (marginal) |
| Param churn | fast 23%, slow 0%, stop 0% (stable) |

**True holdout (2026-03-01 to 2026-06-25, never seen by any fold):**

| Metric | Value |
|---|---|
| Return | -5.3% |
| CAGR | -15.7% |
| Sharpe | -0.57 |
| Max Drawdown | -12.2% |
| Trades | 3 |

The holdout result is negative. This does not validate or invalidate the strategy — 3 trades over 4 months is too small a sample to draw a statistical conclusion either way. The loss is consistent with the choppy/sideways BTC regime that has persisted since the post-ATH oscillation began in late 2024, which matches the known whipsaw pattern documented in the regime-dependence section. That is a plausible explanation, not a proven one.

**These numbers are a development-period diagnostic, not a live-trading signal.** WFE of 53% indicates the IS optimizer is partially overfitting; the negative holdout coincides with an unfavorable regime; and no position-sizing, risk management, or regime filter has been applied. Do not interpret any result in this section as evidence the strategy is ready for live trading.

## Correctness guarantees

1. **No lookahead**: `Context._data = data.iloc[:t+1]` -- future rows are absent, not just hidden.
2. **Next-bar fill**: orders queue on bar `t`, fill at bar `t+1`'s open.
3. **Fees on every fill**: 0.1% per side by default; configurable via `--fee`.
4. **Slippage configurable**: `--slippage` in basis points; default 0.
5. **Warmup enforced**: strategy guards NaN explicitly; engine does not short-circuit.
6. **Force-close**: final bar position is closed at that bar's close, tagged `end_of_data`.
7. **Mark-to-market**: equity is recorded at every bar's close, not just at trade exits.

## V1 limitations (known, intentional)

- Long-or-flat only. No shorting, leverage, or margin.
- Single asset. No portfolio.
- No parameter optimization beyond the walk-forward grid search in `walk_forward.py`.
- No intrabar stop (stop evaluates at close, exits at next open -- see `ma_crossover.py`).
- No web UI, no live trading.

### Strategy regime-dependence

The bundled SMA crossover is a trend-following strategy. It performs well in sustained
directional moves (strong bull or bear trends) because it catches the move early and
rides it. It struggles in sideways, choppy, or mean-reverting markets: price oscillates
around the moving averages without committing to a direction, producing rapid crossover
signals that enter and exit at nearly identical prices while each side pays fees. This
pattern of small repeated losses is called "whipsaw." The 8% drawdown stop limits
individual loss size but does not prevent whipsaw — it only caps how far a single bad
position can run. If you see a backtest period with many short trades (< 50 bars) and
near-zero or negative returns, that is likely a sideways regime, not a strategy bug.
