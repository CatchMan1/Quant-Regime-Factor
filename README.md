# Regime Factor — Market Regime Detection for CSI 1000 Index Futures

A research framework for **unsupervised market-regime detection** on intraday
(index-future) data, together with a **leakage-free backtesting and validation
stack**. The project compares two regime models — a **Gaussian Hidden Markov
Model (HMM)** and a **Statistical Jump Model (JM)** — and validates them with
**Combinatorial Purged Cross-Validation (CPCV)**, the scheme introduced by
López de Prado to generate multiple, statistically independent backtest paths
from a single time series.

The goal is not a single "best" equity curve, but a reproducible pipeline that
quantifies how stable regime-based signals are across folds and how robust the
resulting allocation is under realistic transaction costs.

---

## Table of Contents

- [Motivation](#motivation)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Methodology](#methodology)
  - [Feature Engineering](#feature-engineering)
  - [Regime Models](#regime-models)
  - [State-to-Position Mapping & Backtest](#state-to-position-mapping--backtest)
  - [Regime Forecasting with XGBoost](#regime-forecasting-with-xgboost)
- [Combinatorial Purged Cross-Validation](#combinatorial-purged-cross-validation)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Outputs](#results--outputs)
- [Design Notes](#design-notes)
- [Disclaimer](#disclaimer)

---

## Motivation

Classical factor / momentum strategies assume a single stationary data-generating
process. In practice, intraday index-futures returns switch between distinct
volatility and liquidity regimes. Two questions drive this project:

1. **Can latent regimes be recovered in an unsupervised way** from a compact set
   of price/volume/microstructure features, without hand-labelling market states?
2. **Do regime-conditioned signals survive out-of-sample** once training/testing
   overlap is properly purged and the evaluation is repeated over many
   combinatorial folds?

The first question is addressed with HMM and Jump Model state inference; the
second with a CPCV harness that produces $\varphi = \binom{N-1}{k-1}$ complete
backtest paths and enforces strict train/test separation.

---

## Repository Structure

```
Regime-factor/
├── data/                          # input / intermediate data (NOT tracked by git)
│   ├── kline_im.csv               # 15-min OHLCV + microstructure features for IM futures
│   ├── im_model.csv               # combined signal output (factors + positions + PnL)
│   └── factor_values_norm_im.csv  # large normalized factor matrix + regime labels
└── market-regime/
    ├── cpcv.py                    # CPCV framework + HMM / Jump Model helpers
    ├── HMM-regime.ipynb           # Gaussian-HMM pipeline (end to end)
    └── JM-regime.ipynb            # Statistical Jump Model pipeline (end to end)
```

---

## Data

All datasets live under `data/` and **are excluded from version control** (see
[`.gitignore`](.gitignore)); they must be supplied locally to run the notebooks.

| File | Description |
| --- | --- |
| `kline_im.csv` | 15-minute bars for the IM (CSI 1000) index future, ~14.2k rows spanning **2022-07-22 → 2026-03-09**. Columns: `open, high, low, close, vol, amount, twap, vwap, rsrs, top_amount`, successive-run statistics (`successive_increase_amount / count`, …), amplitude statistics (`amplitude_max / mean / std`), order-flow ratios (`increase_kline_ratio`, …), `vwap_ema_1/2`, the bar return `pct`, and `slippage`. |
| `im_model.csv` | Downstream combination layer: 11 fitted alpha factors (`factor1…factor11`), the aggregate position `pos`, the benchmark `close`, and net-value column `net`, plus the regime overlay positions `hmm_pos1`, `hmm_pos2`. |
| `factor_values_norm_im.csv` | A wide (~819-column) normalized factor matrix produced by an upstream alpha-mining process, joined with the regime labels exported by the notebooks (`pct`, `state`, `state_0_prob…state_3_prob`). This is the largest artifact (≈195 MB). |

> **Data is intentionally not shipped.** Point `kline_im.csv` at any intraday
> OHLCV file with a `DateTime` column and the same field names to reproduce the
> pipeline.

---

## Methodology

### Feature Engineering

Both notebooks build features from the raw bars, then standardize with a
**rolling z-score of window 336 bars** (≈ one week of 15-min bars) so that every
observation is scaled on information available *only up to that time* — a key
guard against look-ahead bias; the first 336 rows are dropped as warm-up.

**HMM feature set (5 features, price/volume focus):**

| Feature | Definition |
| --- | --- |
| `vol_log_ret_1p` | 1-bar log change in volume |
| `vol_log_ret_5p` | 5-bar log change in volume |
| `close_log_ret_1p` | 1-bar log return (short-term momentum) |
| `close_log_ret_5p` | 5-bar log return (medium-term momentum) |
| `intraday_range_log` | $\log(\text{high}) - \log(\text{low})$, intraday volatility |

**Jump Model feature set (12 features, adds risk/tail measures):**

| Feature | Definition |
| --- | --- |
| `ewma_ret` | EWMA of 1-bar return (halflife = 120) |
| `dd20 / dd60 / dd120` | **Downside deviation** over EWMA halflifes 20 / 60 / 120, i.e. $\sqrt{\text{EWMA}_{hl}(\min(r,0)^2)}$ |
| `dd20_60`, `dd60_120`, `dd20_120` | Term-structure spreads between downside deviations |
| `f1`, `f2` | 1-bar and 5-bar log volume changes |
| `f4` | 5-bar log return |
| `f5` | Intraday log high–low range |

The train/test boundary is fixed at **2024-09-30 15:00**; the out-of-sample
period runs from **2024-10-01** onwards.

### Regime Models

**1. Gaussian Hidden Markov Model** — `hmmlearn.hmm.GaussianHMM`

```python
GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, tol=0.01, random_state=42)
```

Four hidden states, full covariance, forward-backward inference. Four states were
selected empirically as the best trade-off between regime granularity and
parameter stability. States are re-labelled **by their in-sample realized
return** so that state ids are comparable across fits.

**2. Statistical Jump Model (SJM)** — implemented from scratch in
[cpcv.py](file:///d:/Job-project/Regime-factor/market-regime/cpcv.py)

The JM fits $K$ state centroids $\theta_k$ and a state path $s_t$ by minimizing
within-state squared error **plus a penalty $\lambda$ for every state change**:

$$\min_{\{\theta_k\},\, s_{1:T}} \sum_{t=1}^{T} \lVert y_t - \theta_{s_t} \rVert_2^2 + \lambda \sum_{t=2}^{T} \mathbb{1}[s_t \neq s_{t-1}]$$

- **`viterbi_jump`** — dynamic-programming solver for the optimal path given
  centroids, with a jump penalty embedded in the transition cost.
- **`fit_sjm`** — coordinate descent: alternate (A) Viterbi path update, (B)
  centroid update as the within-state mean, initialized with KMeans, iterating
  until the path is unchanged.
- **`predict_test_set`** — out-of-sample inference that warm-starts the DP cost
  from the last training day, so regime continuity is preserved across the
  train/test boundary.
- **`SJMRegimeModel`** — a scikit-learn-style `fit` / `predict` wrapper so the
  model can be dropped into the CPCV harness.

Hyper-parameters used: `K = 4`, `λ = 150`, `max_iter = 1000`.

### State-to-Position Mapping & Backtest

Regimes are converted into positions and evaluated with a simple,
cost-aware backtest:

- **Net value:** `net_values = 1 + cumsum(pos * ret - |Δpos| * fee)`, where `ret`
  is the next-bar return `pct` and `fee = 0.0005` (5 bps per unit turnover).
  The signal at $t$ is traded against the return at $t+1$ to avoid look-ahead.
- **Position mapping:** directional maps such as `{3: +1, 2: −1, 1: +1, 0: −1}`
  (full long/short) and a half-size variant `{1: +0.5, 0: −0.5}`, chosen from the
  in-sample, per-state contribution.
- **Metrics:** annualized Sharpe ratio ($\sqrt{252}$ scaling on daily-aggregated
  returns) and **maximum drawdown** on the cumulative net-value curve, both
  marked with the market (buy-and-hold) benchmark for reference.
- **Per-state diagnostics:** individual state net-value and per-bar return
  curves isolate which regimes carry the edge.

### Regime Forecasting with XGBoost

`JM-regime.ipynb` extends the pipeline by treating the inferred regime as a
**supervised target**: `target_regime = state.shift(-1)` (predict tomorrow's
regime from today's features). An `XGBoost` multi-class classifier
(`multi:softprob`, `n_estimators=100`, `max_depth=3`, `lr=0.05`) is trained on
20+ engineered features, and the predicted class probabilities are exported for
downstream probability-weighted allocation. The notebook reports test-set
accuracy and a full per-class classification report.

---

## Combinatorial Purged Cross-Validation

Standard walk-forward CV yields a single backtest path, which makes it hard to
tell skill from luck. [cpcv.py](file:///d:/Job-project/Regime-factor/market-regime/cpcv.py)
implements **CPCV**: the series is cut into $N$ non-overlapping, time-ordered
groups; every split holds out $k$ groups as test and trains on the rest,
yielding

$$\varphi = \binom{N-1}{k-1}$$

**complete** backtest paths. With the configuration used here ($N = 6$, $k = 2$),
each split trains on 4 groups and holds out 2, producing $\varphi = 5$ paths that
each cover the whole sample exactly once.

Key implementation properties:

- **Model-agnostic** — any object exposing `fit(X)` / `fit(X, y)` and
  `predict(X)` works (HMM, SJM, XGBoost, …). Supervised models are supported via
  `target_col`.
- **Purging & embargo** — for every contiguous training block, samples whose
  label horizon overlaps a test interval (or that fall within `purge_ticks` of
  it) are purged, and samples within `embargo_ticks` after a test interval are
  embargoed, all measured in ticks.
- **No cross-gap leakage** — standardization statistics are computed **per
  contiguous block**, and rolling windows are reset at block boundaries, so no
  window ever spans a purged gap or a test region.
- **Explicit leakage assertions** — `_assert_no_leakage` verifies group
  membership, purge/embargo correctness, NaN-free features and correct scaling
  statistics on every split.
- **Path assignment** — a backtracking search assigns splits to paths so that
  every group appears exactly once in every path (an internal invariant is
  asserted per path).
- **State ranking** — for unsupervised models, `rank_states=True` re-orders raw
  labels by in-sample realized return so that state `0` is always the worst
  regime, making state ids consistent across folds.

Common options (all forwarded to `run_cpcv`): `n_groups`, `k_test_groups`,
`window_size`, `purge_ticks`, `embargo_ticks`, `label_horizon_ticks`,
`min_train_samples`, `standardize` (`rolling` / `global` / `none`),
`return_col`, `rank_states`, `target_col`, `group_col`.

---

## Installation

Python 3.11 is used in the notebooks. Install the dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter hmmlearn xgboost
```

| Package | Role |
| --- | --- |
| `numpy`, `pandas`, `scipy` | core numerics / data handling |
| `scikit-learn` | KMeans initialization, metrics |
| `hmmlearn` | Gaussian HMM |
| `xgboost` | supervised regime forecasting |
| `matplotlib` | backtest visualization |
| `jupyter` | running the notebooks |

Place the CSV files in `data/` (see [Data](#data)); they are not tracked by git.

---

## Usage

**Notebooks** — run from inside `market-regime/` so that `import cpcv` resolves:

```bash
cd market-regime
jupyter notebook HMM-regime.ipynb   # Gaussian-HMM pipeline
jupyter notebook JM-regime.ipynb    # Jump-Model + XGBoost pipeline
```

**CPCV as a library** — one-call helper:

```python
from cpcv import run_cpcv, SJMRegimeModel

result = run_cpcv(
    df,                                  # DataFrame with time + feature + return columns
    feature_cols=['ewma_ret', 'dd20', 'f1', 'f2', 'f4', 'f5'],
    model_factory=lambda: SJMRegimeModel(n_states=4, lam=150, max_iter=1000),
    time_col='DateTime',
    prediction_col='state',
    n_groups=6, k_test_groups=2,
    window_size=200,
    purge_ticks=0, embargo_ticks=0,
    return_col='pct',                    # rank states by in-sample return
)

backtest_paths = result['backtest_paths']   # {path_id: DataFrame}
print(result['phi'])                         # == 5
```

Or drive the class directly for finer control:

```python
from cpcv import CPCVGrouped

cpcv = CPCVGrouped(n_groups=6, k_test_groups=2, window_size=200,
                   purge_ticks=0, embargo_ticks=0, return_col='pct')
result = cpcv.run(df, feature_cols, model_factory, 'DateTime', 'state')
paths = cpcv.build_backtest_paths(result)
```

---

## Results & Outputs

The notebooks produce, in `data/`:

- **`im_model.csv`** — regime-aware positions (`hmm_pos1`, `hmm_pos2`) appended
  to the factor-combination layer, ready for the portfolio backtest.
- **`factor_values_norm_im.csv`** — the normalized factor matrix joined with the
  regime label and the four state posterior probabilities (`state_0_prob` …
  `state_3_prob`), enabling probability-weighted downstream allocation.

Plots generated by each notebook:

- Per-state net-value and return curves (in-sample and out-of-sample) to
  identify which regimes carry the edge.
- Market-vs-strategy equity curves with shaded drawdown regions, annotated
  max-drawdown points and the annualized Sharpe in the title.
- Per-path CPCV state-contribution and return plots for all $\varphi$ paths, to
  inspect cross-fold stability.

---

## Design Notes

- **Look-ahead discipline:** next-bar returns, rolling (not global)
  standardization, and per-block purging are used throughout; the CPCV harness
  asserts these invariants rather than trusting them.
- **Reproducibility:** fixed random seeds (`42`) for HMM, KMeans initialization.
- **Separation of concerns:** `cpcv.py` contains **no feature engineering** —
  the caller prepares the DataFrame, keeping the validation framework reusable
  and testable. HMM/JM helpers are bundled only for the example notebooks and
  are not required by the CPCV class itself.
- **Scalability:** CPCV is model-agnostic by design, so the same harness can
  benchmark HMM, Jump Model, XGBoost or any other regime model on identical
  folds.

---

## Disclaimer

This repository is a personal research project for educational and portfolio
purposes only. It is **not investment advice**, and no warranty is made
regarding the accuracy or profitability of the strategies shown. Past
performance does not guarantee future results.
