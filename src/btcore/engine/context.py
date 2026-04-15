from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from btcore.models import Bar


@dataclass(frozen=True)
class Order:
    symbol: str
    kind: str
    value: float
    created_at: datetime


@dataclass(frozen=True)
class PositionView:
    quantity: float = 0.0
    value: float = 0.0


class PositionMap(dict[str, PositionView]):
    def __missing__(self, key: str) -> PositionView:
        return PositionView()


@dataclass(frozen=True)
class PortfolioView:
    cash: float
    total_value: float
    positions: PositionMap


class StrategyContext:
    def __init__(self, frequency: str) -> None:
        self.freq = frequency
        self.now: datetime | None = None
        self.current_dt: datetime | None = None
        self._history: dict[str, list[Bar]] = {}
        self._orders: list[Order] = []
        self.logs: list[str] = []
        self.records: list[dict[str, float | str]] = []
        self.portfolio = PortfolioView(cash=0.0, total_value=0.0, positions=PositionMap())

    def set_now(self, now: datetime) -> None:
        self.now = now
        self.current_dt = now

    def set_portfolio(self, cash: float, positions: dict[str, float], prices: dict[str, float]) -> None:
        position_views = PositionMap({
            symbol: PositionView(quantity=quantity, value=quantity * prices.get(symbol, 0.0))
            for symbol, quantity in positions.items()
        })
        total_value = cash + sum(position.value for position in position_views.values())
        self.portfolio = PortfolioView(cash=cash, total_value=total_value, positions=position_views)

    def add_bar(self, bar: Bar) -> None:
        self._history.setdefault(bar.symbol, []).append(bar)

    def history(self, symbol: str, field: str, n: int) -> list[float]:
        if n <= 0:
            raise ValueError("history window must be positive")
        if field not in {"open", "high", "low", "close", "volume", "adj_factor"}:
            raise ValueError(f"Unsupported history field: {field}")
        bars = self._history.get(symbol, [])[-n:]
        return [float(getattr(bar, field)) for bar in bars]

    def order_value(self, symbol: str, value: float) -> None:
        if self.now is None:
            raise RuntimeError("Context time is not set")
        self._orders.append(Order(symbol=symbol, kind="value", value=value, created_at=self.now))

    def order_target_percent(self, symbol: str, percent: float) -> None:
        if self.now is None:
            raise RuntimeError("Context time is not set")
        self._orders.append(Order(symbol=symbol, kind="target_percent", value=percent, created_at=self.now))

    def order_target_value(self, symbol: str, value: float) -> None:
        if self.now is None:
            raise RuntimeError("Context time is not set")
        self._orders.append(Order(symbol=symbol, kind="target_value", value=value, created_at=self.now))

    def log(self, message: str) -> None:
        prefix = self.now.isoformat() if self.now else "NA"
        self.logs.append(f"{prefix} {message}")

    def pop_orders(self) -> list[Order]:
        orders = self._orders
        self._orders = []
        return orders
