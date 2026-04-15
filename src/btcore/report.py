from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from btcore.engine import BacktestResult


def write_report(result: BacktestResult, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    with (root / "equity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "equity"])
        writer.writerows(result.equity_curve)
    with (root / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "symbol", "quantity", "price", "gross_value", "cost"])
        writer.writeheader()
        writer.writerows(asdict(trade) for trade in result.trades)
    if result.records:
        fieldnames = sorted({key for record in result.records for key in record})
        with (root / "records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result.records)
    summary = _summary_markdown(result)
    (root / "summary.md").write_text(summary, encoding="utf-8-sig")
    html = "<html><body><h1>Backtest Report</h1><pre>" + json.dumps(result.metrics, indent=2) + "</pre></body></html>"
    (root / "report.html").write_text(html, encoding="utf-8")
    return root


def _summary_markdown(result: BacktestResult) -> str:
    metrics = result.metrics
    lines = [
        "# 回测摘要",
        "",
        f"- 总收益率: {metrics.get('total_return', 0.0):.2%}",
        f"- 最大回撤: {metrics.get('max_drawdown', 0.0):.2%}",
        f"- 夏普比率: {metrics.get('sharpe', 0.0):.4f}",
        f"- 交易笔数: {len(result.trades)}",
        f"- 记录曲线点数: {len(result.records)}",
        f"- 起始净值: {result.equity_curve[0][1]:.2f}" if result.equity_curve else "- 起始净值: N/A",
        f"- 结束净值: {result.equity_curve[-1][1]:.2f}" if result.equity_curve else "- 结束净值: N/A",
        "",
        "## 文件说明",
        "",
        "- `equity.csv`: 每个回测时点的组合权益",
        "- `trades.csv`: 成交记录",
        "- `records.csv`: 策略 `record(...)` 输出的净值曲线，若策略有记录才生成",
        "- `metrics.json`: 机器可读指标",
    ]
    return "\n".join(lines) + "\n"
