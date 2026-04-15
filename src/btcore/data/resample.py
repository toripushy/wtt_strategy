from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from btcore.models import Bar


def resample_1m_to_5m(bars: list[Bar]) -> list[Bar]:
    grouped: dict[tuple[str, datetime], list[Bar]] = defaultdict(list)
    for bar in bars:
        bucket_minute = bar.timestamp.minute - (bar.timestamp.minute % 5)
        bucket = bar.timestamp.replace(minute=bucket_minute, second=0, microsecond=0)
        grouped[(bar.symbol, bucket)].append(bar)

    output: list[Bar] = []
    for (symbol, bucket), group in grouped.items():
        ordered = sorted(group, key=lambda item: item.timestamp)
        output.append(Bar(
            symbol=symbol,
            timestamp=bucket,
            open=ordered[0].open,
            high=max(item.high for item in ordered),
            low=min(item.low for item in ordered),
            close=ordered[-1].close,
            volume=sum(item.volume for item in ordered),
            adj_factor=ordered[-1].adj_factor,
        ))
    return sorted(output, key=lambda item: (item.timestamp, item.symbol))
