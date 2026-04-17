from pathlib import Path

from btcore.engine import run
from btcore.models import BacktestConfig


def test_next_open_fill_uses_next_bar(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("JQDATA_USERNAME", raising=False)
    monkeypatch.delenv("JQDATA_PASSWORD", raising=False)
    strategy = tmp_path / "strategies" / "buy_once.py"
    strategy.parent.mkdir()
    strategy.write_text(
        """class Strategy:
    frequency = "1m"
    def __init__(self):
        self.done = False
    def on_bar(self, ctx, data):
        if not self.done:
            ctx.order_value("000001.XSHE", 1000)
            self.done = True
""",
        encoding="utf-8",
    )
    config = BacktestConfig(symbols=("000001.XSHE",), frequency="1m", start="2026-01-05", end="2026-01-05", cache_dir=str(tmp_path / "cache"))

    result = run(config, strategy)

    assert result.trades
    assert result.trades[0].timestamp == "2026-01-05T09:31:00"


def test_smoke_5m_from_1m(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("JQDATA_USERNAME", raising=False)
    monkeypatch.delenv("JQDATA_PASSWORD", raising=False)
    strategy = tmp_path / "strategies" / "ma.py"
    strategy.parent.mkdir()
    strategy.write_text(Path("strategies/moving_average.py").read_text(encoding="utf-8").replace('frequency = "1m"', 'frequency = "5m"'), encoding="utf-8")
    config = BacktestConfig(symbols=("000001.XSHE",), frequency="5m", start="2026-01-05", end="2026-01-05", cache_dir=str(tmp_path / "cache"))

    result = run(config, strategy)

    assert result.equity_curve
    assert set(result.metrics) == {"total_return", "max_drawdown", "sharpe"}
