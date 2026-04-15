from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from btcore.models import Bar


class BarCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, provider: str, freq: str, adj: str, bars: list[Bar]) -> list[Path]:
        grouped: dict[tuple[str, str], list[Bar]] = defaultdict(list)
        for bar in bars:
            grouped[(bar.symbol, bar.timestamp.date().isoformat())].append(bar)
        paths: list[Path] = []
        for (symbol, day), group in grouped.items():
            path = self._path(provider, symbol, freq, adj, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["symbol", "timestamp", "open", "high", "low", "close", "volume", "adj_factor"])
                writer.writeheader()
                for bar in sorted(group, key=lambda item: item.timestamp):
                    writer.writerow({
                        "symbol": bar.symbol,
                        "timestamp": bar.timestamp.isoformat(),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "adj_factor": bar.adj_factor,
                    })
            paths.append(path)
        return paths

    def read(self, provider: str, symbols: tuple[str, ...], freq: str, adj: str, start: str, end: str) -> list[Bar]:
        bars: list[Bar] = []
        for symbol in symbols:
            symbol_root = self.root / provider / symbol / freq / f"adj={adj}"
            if not symbol_root.exists():
                continue
            for path in sorted(symbol_root.glob("date=*.csv")):
                day = path.stem.removeprefix("date=")
                if start <= day <= end:
                    bars.extend(self._read_file(path))
        return sorted(bars, key=lambda item: (item.timestamp, item.symbol))

    def _path(self, provider: str, symbol: str, freq: str, adj: str, day: str) -> Path:
        return self.root / provider / symbol / freq / f"adj={adj}" / f"date={day}.csv"

    def _read_file(self, path: Path) -> list[Bar]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [
                Bar(
                    symbol=row["symbol"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    adj_factor=float(row.get("adj_factor") or 1.0),
                )
                for row in reader
            ]
