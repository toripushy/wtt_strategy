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
