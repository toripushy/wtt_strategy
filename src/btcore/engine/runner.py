from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from btcore.data.cache import BarCache
from btcore.data.providers import get_provider
from btcore.data.resample import resample_1m_to_5m
from btcore.engine.context import Order, StrategyContext
from btcore.metrics import compute_metrics
from btcore.models import BacktestConfig, Bar


@dataclass
class Trade:
    timestamp: str
    symbol: str
    quantity: float
    price: float
    gross_value: float
    cost: float


@dataclass
class BacktestResult:
    equity_curve: list[tuple[str, float]]
    trades: list[Trade]
    metrics: dict[str, float]
    logs: list[str]
    records: list[dict[str, float | str]]


def run(config: BacktestConfig, strategy_path: str | Path) -> BacktestResult:
    bars = _load_bars(config)
    bars_by_time: dict[object, list[Bar]] = {}
    for bar in bars:
        bars_by_time.setdefault(bar.timestamp, []).append(bar)

    strategy = _load_strategy(strategy_path)
    declared_frequency = getattr(strategy, "frequency", config.frequency)
    if declared_frequency != config.frequency:
        raise ValueError(f"Strategy frequency {declared_frequency!r} does not match config frequency {config.frequency!r}")

    context = StrategyContext(config.frequency)
    if hasattr(strategy, "initialize"):
        strategy.initialize(context)
    cash = config.cash
    positions = {symbol: 0.0 for symbol in config.symbols}
    pending_orders: list[Order] = []
    trades: list[Trade] = []
    equity_curve: list[tuple[str, float]] = []
    last_close: dict[str, float] = {}

    for timestamp in sorted(bars_by_time):
        current_bars = {bar.symbol: bar for bar in bars_by_time[timestamp]}
        cash, new_trades = _fill_orders(pending_orders, current_bars, positions, cash, config)
        trades.extend(new_trades)
        pending_orders = []

        context.set_now(timestamp)  # type: ignore[arg-type]
        context.set_portfolio(cash, positions, {symbol: bar.close for symbol, bar in current_bars.items()})
        for symbol in config.symbols:
            if symbol in current_bars:
                context.add_bar(current_bars[symbol])
                last_close[symbol] = current_bars[symbol].close

        strategy.on_bar(context, current_bars)
        pending_orders.extend(context.pop_orders())
        equity = cash + sum(positions[symbol] * last_close.get(symbol, 0.0) for symbol in config.symbols)
        equity_curve.append((timestamp.isoformat(), equity))  # type: ignore[union-attr]

    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=compute_metrics(equity_curve, config.cash), logs=context.logs, records=context.records)


def _load_bars(config: BacktestConfig) -> list[Bar]:
    cache = BarCache(config.cache_dir)
    cached = cache.read(config.provider, config.symbols, config.frequency, config.adj, config.start, config.end)
    if cached:
        return cached
    provider = get_provider(config.provider, config.sessions)
    fetch_freq = "1m" if config.frequency == "5m" and config.resample_policy == "from_1m" else config.frequency
    bars = provider.fetch_bars(config.symbols, fetch_freq, config.start, config.end, config.adj)
    if config.frequency == "5m" and fetch_freq == "1m":
        cache.write(config.provider, fetch_freq, config.adj, bars)
        bars = resample_1m_to_5m(bars)
    cache.write(config.provider, config.frequency, config.adj, bars)
    return bars


def _fill_orders(orders: list[Order], current_bars: dict[str, Bar], positions: dict[str, float], cash: float, config: BacktestConfig) -> tuple[float, list[Trade]]:
    trades: list[Trade] = []
    open_prices = {symbol: bar.open for symbol, bar in current_bars.items()}
    portfolio_value = cash + sum(positions[symbol] * open_prices.get(symbol, 0.0) for symbol in positions)
    for order in orders:
        bar = current_bars.get(order.symbol)
        if bar is None:
            continue
        price = bar.open
        if order.kind == "target_percent":
            target_value = portfolio_value * order.value
            current_value = positions[order.symbol] * price
            gross_value = target_value - current_value
        elif order.kind == "target_value":
            target_value = order.value
            current_value = positions[order.symbol] * price
            gross_value = target_value - current_value
        else:
            gross_value = order.value
        if abs(gross_value) < 1e-9:
            continue
        quantity = gross_value / price
        cost = _cost(abs(gross_value), gross_value < 0, config)
        cash -= gross_value + cost
        positions[order.symbol] += quantity
        trades.append(Trade(timestamp=bar.timestamp.isoformat(), symbol=order.symbol, quantity=quantity, price=price, gross_value=gross_value, cost=cost))
    return cash, trades


def _cost(abs_value: float, is_sell: bool, config: BacktestConfig) -> float:
    commission = max(abs_value * config.costs.commission_rate, config.costs.min_commission)
    stamp = abs_value * config.costs.stamp_duty_rate if is_sell else 0.0
    slippage = abs_value * config.costs.slippage_bps / 10_000
    return commission + stamp + slippage


def _load_strategy(path: str | Path):
    module_path = Path(path)
    if module_path.suffix == ".txt":
        loader = SourceFileLoader(module_path.stem, str(module_path))
        spec = importlib.util.spec_from_loader(module_path.stem, loader)
    else:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load strategy: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _strategy_from_module(module)


def _strategy_from_module(module: ModuleType):
    if hasattr(module, "Strategy"):
        return module.Strategy()
    if hasattr(module, "initialize") and hasattr(module, "handle_data"):
        return JoinQuantStrategy(module)
    raise RuntimeError("Strategy module must define a Strategy class")


class JoinQuantStrategy:
    frequency = "1d"

    def __init__(self, module: ModuleType) -> None:
        self.module = module

    def initialize(self, context: StrategyContext) -> None:
        import jqdata

        jqdata._set_context(context)
        self.module.initialize(context)

    def on_bar(self, context: StrategyContext, data: dict[str, Bar]) -> None:
        import jqdata

        jqdata._set_context(context)
        self.module.handle_data(context, data)
