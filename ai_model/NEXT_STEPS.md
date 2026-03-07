# Next Steps: Building a Lightweight Transformer Prediction Model for PSE Trade Signals

This guide walks you through the full journey from raw PSE market data to a
production-ready, lightweight Transformer that predicts **BUY / HOLD / SELL**
signals on 1-minute OHLCV data for BDO, SM, ALI, JFC, AC, and TEL.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Data Collection & Quality Checks](#2-data-collection--quality-checks)
3. [Feature Engineering](#3-feature-engineering)
4. [Label Design & Class Imbalance](#4-label-design--class-imbalance)
5. [Training the Transformer](#5-training-the-transformer)
6. [Evaluating Model Performance](#6-evaluating-model-performance)
7. [Integrating with the Trading Agent](#7-integrating-with-the-trading-agent)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Deployment & Monitoring](#9-deployment--monitoring)
10. [Roadmap & Recommended Reading](#10-roadmap--recommended-reading)

---

## 1. Prerequisites & Environment Setup

### Install ML dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` already includes `torch`, `scikit-learn`, and all
existing pipeline dependencies.  If you want GPU acceleration, replace the
`torch` line with the appropriate CUDA wheel from
<https://pytorch.org/get-started/locally/>.

### Verify the installation

```python
import torch
from ai_model import TransformerSignalModel, ModelConfig

cfg = ModelConfig()
model = TransformerSignalModel(cfg)
print(f"Parameters: {model.count_parameters():,}")  # ~130 k with default config
```

---

## 2. Data Collection & Quality Checks

### Collect historical 1-minute OHLCV data

The existing `data_pipeline.py` fetches up to **7 days** of 1-minute data
from Yahoo Finance.  For robust model training you need **at least 30 days**
of data, which requires a premium data source or local storage.

**Recommended sources:**

| Source | Notes |
|--------|-------|
| [Yahoo Finance](https://finance.yahoo.com) | Free, 7-day limit for 1m data |
| [PSEi Data Portal](https://www.pse.com.ph) | Official PSE historical data |
| [Polygon.io](https://polygon.io) | Paid, minute-level global data |
| Manual CSV collection | Run `main.py` daily and append to a master CSV |

**Action:** Accumulate data by scheduling `python main.py` daily and appending
to `data/processed/market_data_processed.csv`.

### Quality checks to perform

- [ ] Check for missing candles (market halts, thin trading hours).
- [ ] Remove pre-market / after-market rows outside PSE hours (09:30–15:30 PHT).
- [ ] Verify volume > 0 for all rows (zero-volume bars are suspicious).
- [ ] Align timestamps across tickers (all 6 stocks should share the same timestamps).
- [ ] Look for price outliers (spikes > 5σ from the rolling mean).

---

## 3. Feature Engineering

The `ai_model/features.py` module computes the following **13 input features**
out of the box:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `Open` | Candle open price (z-score normalised per ticker) |
| 2 | `High` | Candle high price |
| 3 | `Low` | Candle low price |
| 4 | `Close` | Candle close price |
| 5 | `Volume` | Trade volume |
| 6 | `EMA_9` | Fast exponential moving average (9-period) |
| 7 | `EMA_21` | Slow exponential moving average (21-period) |
| 8 | `RSI` | Relative Strength Index (14-period) |
| 9 | `BB_upper` | Bollinger Band upper (20-period, 2σ) |
| 10 | `BB_middle` | Bollinger Band middle (SMA-20) |
| 11 | `BB_lower` | Bollinger Band lower |
| 12 | `Returns` | 1-period percentage return |
| 13 | `Volatility` | Rolling annualised volatility (20-period) |

### Recommended additional features to explore

- **VWAP** (Volume-Weighted Average Price) — a key intraday level
- **Order book imbalance** — bid/ask volume ratio (requires L2 data)
- **Time-of-day encoding** — `sin(2π·minute / 390)` and `cos(...)` to capture
  intraday seasonality
- **Day-of-week encoding** — `sin(2π·weekday / 5)` etc.
- **Cross-ticker correlation features** — e.g., BDO's z-score vs. SM's z-score
  (sector co-movement)
- **Lagged returns** — t-1, t-5, t-15 lagged returns as explicit features

To add features, extend the `add_indicators()` function in `indicators.py`
and add the new column names to `ModelConfig.feature_columns`.

---

## 4. Label Design & Class Imbalance

### Current label scheme

The `features.py` module labels each candle by its **look-ahead return**:

```
future_return = (Close[t + horizon] - Close[t]) / Close[t]

BUY  (0)  if future_return ≥ +threshold   (default: +0.2%)
HOLD (1)  if |future_return| < threshold
SELL (2)  if future_return ≤ -threshold   (default: -0.2%)
```

**Default parameters:** `label_horizon = 5` candles, `label_threshold = 0.002`.

### Tuning label parameters

- **Increase `label_horizon`** (e.g., 10–15) to capture longer moves — but the
  model becomes less responsive to micro-structure.
- **Increase `label_threshold`** (e.g., 0.003–0.005) to reduce HOLD dominance
  and focus only on stronger moves.
- **Consider transaction costs** when calibrating the threshold.  PSE typical
  commission is ~0.25–0.50%, so a 0.2% threshold may already be below
  break-even.  Use at least 0.5% to capture net-profitable moves.

### Handling class imbalance

On 1-minute PSE data, HOLD typically represents ~80–90% of all labels.
The dataset module already computes **inverse-frequency class weights**
(see `dataset.compute_class_weights()`), which are passed to
`CrossEntropyLoss(weight=...)` during training.

Additional strategies:
- [ ] Try **focal loss** (down-weights well-classified easy examples).
- [ ] Try **under-sampling** HOLD candles during dataset construction.
- [ ] Track **class-conditional metrics** (precision/recall per class), not
      just overall accuracy.

---

## 5. Training the Transformer

### Quick start

Once the processed CSV has at least a few thousand rows per ticker:

```bash
python -m ai_model.train
```

Override settings without changing code:

```bash
python -m ai_model.train --epochs 100 --lr 5e-4 --seq-len 60 --batch-size 128
```

Or customise via `ModelConfig` in Python:

```python
from ai_model.config import ModelConfig
from ai_model.train import train

cfg = ModelConfig(
    seq_len=60,
    label_horizon=10,
    label_threshold=0.003,
    d_model=128,
    nhead=4,
    num_encoder_layers=3,
    dim_feedforward=256,
    num_epochs=100,
    learning_rate=5e-4,
)
model = train(cfg=cfg)
```

### Training tips

- **Start small.** Begin with default config (~130 k params) and small data.
  Scale up only after the pipeline works end-to-end.
- **Watch validation loss, not training loss.** Overfitting is the primary
  risk on small financial datasets.
- **Use early stopping.** The default `patience=10` stops training if
  validation loss stops improving for 10 consecutive epochs.
- **Monitor class-conditional val accuracy.** The weighted loss means overall
  accuracy can look good while BUY/SELL recall remains near zero.

---

## 6. Evaluating Model Performance

### Run evaluation on the test set

```python
import torch
from ai_model.config import ModelConfig
from ai_model.dataset import build_dataloaders
from ai_model.model import TransformerSignalModel
from ai_model.evaluate import evaluate_model, print_report

cfg = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader, _ = build_dataloaders(cfg=cfg)

checkpoint = torch.load(cfg.best_model_path, map_location=device, weights_only=False)
model = TransformerSignalModel(cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

report = evaluate_model(model, test_loader, cfg, device)
print_report(report)
```

### Key metrics to track

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| BUY F1 | > 0.45 | 0.30–0.45 | < 0.30 |
| SELL F1 | > 0.45 | 0.30–0.45 | < 0.30 |
| Sharpe Ratio | > 1.0 | 0.5–1.0 | < 0.5 |
| Max Drawdown | < 5% | 5–15% | > 15% |
| Win Rate | > 55% | 50–55% | < 50% |

> **Important:** A model that predicts HOLD 100% of the time achieves ~85%
> accuracy on imbalanced data.  Always report per-class F1 scores alongside
> overall accuracy.

### Backtesting with the full pipeline

After integration (see Section 7), run the standard backtest:

```bash
python main.py --backtest
```

This executes the full pipeline with `TransformerStrategy` and generates a
trade log in `reports/trade_log.csv`.

---

## 7. Integrating with the Trading Agent

The `TransformerStrategy` class in `ai_model/predict.py` is a drop-in
replacement for any rule-based strategy:

```python
import pandas as pd
from ai_model.predict import TransformerStrategy
from trading_agent import TradingAgent
from portfolio import Portfolio
import data_pipeline as dp
import indicators

# 1. Fetch and process data
raw_df = dp.fetch_all_tickers()
processed_df = indicators.add_indicators(raw_df)

# 2. Create strategy (auto-loads trained checkpoint)
strategy = TransformerStrategy(confidence_threshold=0.55)
strategy.set_dataframe(processed_df)   # required: give strategy access to history

# 3. Run agent
portfolio = Portfolio()
agent = TradingAgent(portfolio=portfolio, strategy=strategy)
signals_df = agent.run(processed_df)

portfolio.print_summary()
```

You can also register it in `STRATEGY_REGISTRY` in `trading_agent.py`:

```python
from ai_model.predict import TransformerStrategy
STRATEGY_REGISTRY["Transformer"] = TransformerStrategy()
```

---

## 8. Hyperparameter Tuning

### Suggested grid to search

| Parameter | Values to try |
|-----------|--------------|
| `seq_len` | 15, 30, 60 |
| `label_horizon` | 3, 5, 10, 15 |
| `label_threshold` | 0.002, 0.003, 0.005 |
| `d_model` | 32, 64, 128 |
| `nhead` | 2, 4, 8 |
| `num_encoder_layers` | 1, 2, 3 |
| `dropout` | 0.05, 0.1, 0.2 |
| `learning_rate` | 1e-4, 5e-4, 1e-3 |

### Tools

- **Manual grid search** — loop over a `ModelConfig` grid, save per-run
  metrics to a CSV.
- **Optuna** (`pip install optuna`) — Bayesian optimisation, 3–5× more
  efficient than grid search.
- **Ray Tune** (`pip install ray[tune]`) — distributed hyperparameter search,
  useful if you have GPU access.

---

## 9. Deployment & Monitoring

### Live signal generation

```bash
python main.py --live          # runs the trading loop every minute
```

With `TransformerStrategy` registered, the same loop uses the Transformer
model instead of rule-based signals.

### Model drift monitoring

Financial data distributions shift over time.  Plan to:

- [ ] Re-train the model weekly or monthly on the most recent 60–90 days.
- [ ] Monitor **validation loss on the latest data** as a drift indicator.
- [ ] Implement a **shadow mode**: log Transformer signals alongside
      rule-based signals without executing trades, and compare performance
      before switching.

### Lightweight inference checklist

- [ ] Export model to **TorchScript** (`torch.jit.script(model)`) for faster
      inference and portable deployment.
- [ ] Use `torch.inference_mode()` (more efficient than `torch.no_grad()`
      in production).
- [ ] Benchmark inference latency — target < 50 ms per prediction on CPU
      with the default architecture.

---

## 10. Roadmap & Recommended Reading

### Short-term (1–2 weeks)

- [x] Create `ai_model/` package with Transformer modules *(done)*
- [ ] Accumulate ≥ 30 days of 1-minute PSE data
- [ ] Run first end-to-end training with default config
- [ ] Validate pipeline with the full backtest

### Medium-term (1–2 months)

- [ ] Add VWAP and time-of-day features
- [ ] Implement focal loss for class imbalance
- [ ] Run hyperparameter search with Optuna
- [ ] Compare Transformer vs. LSTM vs. rule-based strategies
- [ ] Paper-trade with live signals for at least 2 weeks

### Long-term (3–6 months)

- [ ] Explore multi-task learning (predict direction + volatility jointly)
- [ ] Add order-book features (requires L2 data source)
- [ ] Investigate reinforcement learning agent (PPO / SAC) as the
      signal-generation layer
- [ ] Build ensemble: Transformer + rule-based signals → meta-learner

### Recommended reading

| Resource | Why |
|----------|-----|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | Original Transformer paper |
| [Temporal Fusion Transformers (Lim et al., 2020)](https://arxiv.org/abs/1912.09363) | Designed for multi-horizon time-series forecasting |
| [Trading with Transformers (various)](https://paperswithcode.com/task/stock-market-prediction) | Applied financial ML survey |
| [Advances in Financial ML (López de Prado, 2018)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | Labelling, backtesting, and meta-labelling |
| [PyTorch Documentation](https://pytorch.org/docs/stable/) | TransformerEncoderLayer, DataLoader API |

---

*Last updated: 2026-03-07 — PSE Trading Bot AI Model Package*
