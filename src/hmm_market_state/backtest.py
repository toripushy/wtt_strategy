from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_regime_strategy(
    returns: pd.Series,
    positions: pd.Series,
    *,
    cost_bps: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Backtest a daily regime-based position series."""

    returns = returns.astype(float).copy()
    positions = positions.astype(float).reindex(returns.index).fillna(0.0)

    applied_position = positions.shift(1).fillna(0.0)
    gross_return = applied_position * returns
    turnover = applied_position.diff().abs().fillna(applied_position.abs())
    cost = turnover * (cost_bps / 10000.0)
    net_return = gross_return - cost

    equity = (1.0 + net_return).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    return pd.DataFrame(
        {
            "asset_return": returns,
            "position": positions,
            "applied_position": applied_position,
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=returns.index,
    )


def summarize_backtest(result: pd.DataFrame, *, trading_days: int = 252) -> dict[str, float]:
    """Summarize a backtest result frame."""

    net = result["net_return"].astype(float)
    equity = result["equity"].astype(float)
    total_return = float(equity.iloc[-1] - 1.0)
    mean_daily = float(net.mean())
    vol_daily = float(net.std(ddof=0))
    ann_return = float((1.0 + total_return) ** (trading_days / max(len(net), 1)) - 1.0)
    ann_vol = float(vol_daily * np.sqrt(trading_days))
    sharpe = float((mean_daily / vol_daily) * np.sqrt(trading_days)) if vol_daily > 0 else 0.0
    max_drawdown = float(result["drawdown"].min())
    hit_rate = float((net > 0).mean())
    turnover = float(result["turnover"].mean())

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "avg_turnover": turnover,
    }
