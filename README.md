# Regime Factor — Market Regime Detection for CSI 1000 Index Futures

A research framework for **unsupervised market-regime detection** on intraday
(index-future) data, together with a **leakage-free backtesting and validation
stack**. The project compares two regime models — a **Gaussian Hidden Markov
Model (HMM)** and a **Statistical Jump Model (JM)** — and validates them with
**Combinatorial Purged Cross-Validation (CPCV)**, the scheme introduced by
López de Prado to generate multiple complete backtest paths
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
    ├── cpcv.py                    # Jump Model numerical functions (supports independent segments)
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
[cpcv.py](market-regime/cpcv.py)

The JM fits $K$ state centroids $\theta_k$ and a state path $s_t$ by minimizing
within-state squared error **plus a penalty $\lambda$ for every state change**:

$$\min_{\{\theta_k\},\, s_{1:T}} \sum_{t=1}^{T} \lVert y_t - \theta_{s_t} \rVert_2^2 + \lambda \sum_{t=2}^{T} \mathbb{1}[s_t \neq s_{t-1}]$$

- **`viterbi_jump`** — dynamic-programming solver for the optimal path given
  centroids, with a jump penalty embedded in the transition cost.
- **`fit_sjm`** — coordinate descent: alternate (A) Viterbi path update, (B)
  centroid update as the within-state mean, initialized with KMeans, iterating
  until the path is unchanged.
- **`predict_test_set`** — causal, per-bar inference. Ordinary train/test
  calls can pass training costs for a warm start; CPCV omits this context and
  restarts costs for each independent test segment.
- All three numerical functions accept `lengths` to identify independent
  sequences while retaining a shared set of state centers.

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

CPCV is written as seven sequential Notebook steps: section **4** of
`HMM-regime.ipynb` and section **5** of `JM-regime.ipynb`. Run section 1 first
to prepare `df_min`, then run the CPCV cells in order; the intervening model,
backtest and data-export sections are not required.

1. Prepare raw features, existing next-bar simple returns (`pct`) and actual
   return-end timestamps. Drop invalid features and the row without a known
   next-bar timestamp, retaining original row positions to identify gaps.
2. Divide the ordered observations into `N` contiguous groups.
3. Enumerate all `C(N,k)` test-group combinations and remove training events
   overlapping test event intervals, plus optional purge and embargo buffers.
4. Inspect the first fold. Fit one global z-score on each fold's training
   rows and apply the same transform to its training and test data.
5. Train and predict each fold, respecting independent sequence boundaries.
   Rank states using training-only cumulative gross returns. HMM test states
   use causal forward filtering; JM test costs restart after each gap.
6. Save `data/cpcv_hmm_signals.csv` or `data/cpcv_jm_signals.csv`, then read the
   signals back and assign each group's successive predictions to paths.
7. Evaluate every path with explicit positions, turnover fees, additive net
   values, fixed-capital drawdown and daily-aggregated Sharpe ratios.

With `N=6`, `k=2`, this gives **15 model fits** and
`phi = C(N-1,k-1) = 5` complete paths. Each path contains every eligible row
exactly once, and each `(split, group)` prediction is used once. Path assembly
uses per-group occurrence counts, without classes or backtracking. Different
test groups from one split can enter different paths, so path curves differ
from the previous implementation. No `N % k == 0` restriction is required.

`pct` already represents the current row's next-bar return; multiply it by
the current row's position without another shift. Purging uses closed
`[row, next row]` event intervals even when extra purge/embargo buffers are zero.
Training and test scaling now use the same training-only statistics, replacing
the previous rolling-training/global-test mismatch. Additive drawdown is the
peak-to-trough net-value difference as a fraction of initial capital.

CPCV uses training groups on both sides of held-out periods and measures
cross-period robustness. It is distinct from a historical-only walk-forward
simulation, and its paths are not statistically independent. Model parameters,
features, fees and isolation buffers are directly editable in the input cell.
The small `cpcv.py` file contains only JM numerical helpers; HMM is imported
directly from `hmmlearn`.

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

**CPCV** — after preparing features in section 1, run the seven CPCV steps
in the relevant Notebook. Inspect `cpcv_group_summary`, `cpcv_split_summary`,
`cpcv_train_norm`, `cpcv_state_maps`, `backtest_paths` and `cpcv_metrics` as
ordinary DataFrames. The old `run_cpcv` / `CPCVGrouped` API has been removed.

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

- **CPCV time boundaries:** return-event purging, training-only scaling and
  independent sequence lengths are explicit in the Notebook cells. Test HMM
  filtering and JM inference use observations only through the current time.
- **Reproducibility:** fixed random seeds (`42`) for HMM, KMeans initialization.
- **Visible calculations:** CPCV grouping, standardization, state ranking,
  signal export and evaluation are written directly in the Notebooks.
  `cpcv.py` retains only the JM dynamic-programming and fitting functions.

---

## Disclaimer

This repository is a personal research project for educational and portfolio
purposes only. It is **not investment advice**, and no warranty is made
regarding the accuracy or profitability of the strategies shown. Past
performance does not guarantee future results.
