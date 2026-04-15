from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any

_active_context = None


class _Globals(SimpleNamespace):
    pass


g = _Globals()


@dataclass(frozen=True)
class OrderCost:
    close_tax: float = 0.0
    open_commission: float = 0.0
    close_commission: float = 0.0
    min_commission: float = 0.0


def _set_context(context: Any) -> None:
    global _active_context
    _active_context = context


def set_benchmark(symbol: str) -> None:
    if _active_context is not None:
        _active_context.benchmark = symbol


def set_option(name: str, value: Any) -> None:
    if _active_context is not None:
        _active_context.options = getattr(_active_context, "options", {})
        _active_context.options[name] = value


def set_order_cost(cost: OrderCost, type: str = "stock") -> None:
    if _active_context is not None:
        _active_context.order_costs = getattr(_active_context, "order_costs", {})
        _active_context.order_costs[type] = cost


def get_security_info(symbol: str):
    return SimpleNamespace(code=symbol, start_date=date(1900, 1, 1))


def attribute_history(symbol: str, count: int, unit: str, fields: list[str] | tuple[str, ...]):
    if _active_context is None:
        raise RuntimeError("No active backtest context")
    return {field: _active_context.history(symbol, field, count) for field in fields}


def record(**kwargs: float) -> None:
    if _active_context is not None and _active_context.now is not None:
        _active_context.records.append({"timestamp": _active_context.now.isoformat(), **kwargs})


def order_target_value(symbol: str, value: float) -> None:
    if _active_context is None:
        raise RuntimeError("No active backtest context")
    _active_context.order_target_value(symbol, value)
