# btcore

A local Python backtesting CLI for daily and minute bars.

## Quick start

```powershell
$env:PYTHONPATH='src'
python -m btcore.cli init
python -m btcore.cli fetch --symbols 000001.XSHE --freq 1m --start 2026-01-05 --end 2026-01-05
python -m btcore.cli run --strategy strategies/moving_average.py --symbols 000001.XSHE --freq 1m --start 2026-01-05 --end 2026-01-05
python -m btcore.cli run --strategy strategies/moving_average_5m.py --symbols 000001.XSHE --freq 5m --start 2026-01-05 --end 2026-01-05
python -m btcore.cli report --run latest
python -m btcore.cli patch-strategy --strategy strategies/moving_average.py --dry-run
python -m btcore.cli patch-strategy --strategy strategies/moving_average.py --set fast=8 --dry-run
```

For real JQData integration, set `JQDATA_USERNAME` and `JQDATA_PASSWORD`, then install and configure the official `jqdatasdk` package in your local environment.

## HMM market state

The repository also includes `hmm_market_state`, a lightweight Gaussian HMM toolkit for labeling market regimes from OHLCV-style data.

```python
import pandas as pd
from hmm_market_state import FeatureConfig, HMMMarketRegimePipeline

data = pd.read_csv("your_market_data.csv", parse_dates=["date"]).set_index("date")

pipeline = HMMMarketRegimePipeline(
    feature_config=FeatureConfig(vol_window=20, use_volume=True),
    model_kwargs={"n_states": 3, "n_iter": 50, "random_state": 42},
)

labels = pipeline.fit_transform(data)
print(labels[["state", "regime", "position"]].tail())
```

`fit_transform` trains on the full input before labeling that same input, so use it for in-sample regime research. For backtests, use walk-forward detection; it trains on each rolling training window and predicts test-window states with forward-only filtered probabilities.

```python
from hmm_market_state import FeatureConfig, walk_forward_regime_detection

result = walk_forward_regime_detection(
    data,
    feature_config=FeatureConfig(vol_window=20, use_volume=True),
    train_window=756,
    test_window=63,
    step=63,
    model_kwargs={"n_states": 3, "n_iter": 50, "random_state": 42},
)

print(result.predictions.tail())
```
