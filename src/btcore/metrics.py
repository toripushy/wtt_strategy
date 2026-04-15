from __future__ import annotations

import math


def compute_metrics(equity_curve: list[tuple[str, float]], initial_cash: float) -> dict[str, float]:
    if not equity_curve:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    values = [value for _, value in equity_curve]
    total_return = values[-1] / initial_cash - 1.0
    peak = values[0]
    max_drawdown = 0.0
    returns: list[float] = []
    previous = values[0]
    for value in values:
        peak = max(peak, value)
        if peak:
            max_drawdown = min(max_drawdown, value / peak - 1.0)
        if previous:
            returns.append(value / previous - 1.0)
        previous = value
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((item - mean) ** 2 for item in returns) / len(returns) if returns else 0.0
    sharpe = mean / math.sqrt(variance) if variance > 0 else 0.0
    return {"total_return": total_return, "max_drawdown": max_drawdown, "sharpe": sharpe}
