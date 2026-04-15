from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

FREQUENCIES = {"1d", "1m", "5m"}


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_factor: float = 1.0


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.001
    slippage_bps: float = 1.0
    min_commission: float = 5.0


@dataclass(frozen=True)
class BacktestConfig:
    symbols: tuple[str, ...]
    frequency: str
    start: str
    end: str
    cash: float = 1_000_000.0
    provider: str = "jqdata"
    adj: str = "pre"
    fill_price: str = "next_open"
    resample_policy: str = "from_1m"
    cache_dir: str = "data/cache"
    sessions: tuple[tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "15:00"))
    costs: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self) -> None:
        if self.frequency not in FREQUENCIES:
            raise ValueError(f"Unsupported frequency: {self.frequency}")
        if self.fill_price != "next_open":
            raise ValueError("v1 only supports fill_price='next_open'")
        if not self.symbols:
            raise ValueError("At least one symbol is required")
