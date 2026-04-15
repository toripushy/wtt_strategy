from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime

from btcore.calendar import calendar
from btcore.models import Bar


@dataclass
class JQDataAdapter:
    sessions: tuple[tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "15:00"))

    def fetch_bars(self, symbols: tuple[str, ...], freq: str, start: str, end: str, adj: str = "pre", fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")) -> list[Bar]:
        if freq not in {"1m", "5m", "1d"}:
            raise ValueError(f"Unsupported JQData frequency: {freq}")
        if adj not in {"pre", "post", "none"}:
            raise ValueError(f"Unsupported adj mode: {adj}")
        if os.getenv("JQDATA_USERNAME") and os.getenv("JQDATA_PASSWORD"):
            return self._fetch_with_sdk(symbols, freq, start, end, adj, fields)
        return self._sample_bars(symbols, freq, start, end)

    def _fetch_with_sdk(self, symbols: tuple[str, ...], freq: str, start: str, end: str, adj: str, fields: tuple[str, ...]) -> list[Bar]:
        try:
            import jqdatasdk as jq  # type: ignore
        except ImportError as exc:
            raise RuntimeError("JQData credentials are set, but jqdatasdk is not installed.") from exc
        jq.auth(os.environ["JQDATA_USERNAME"], os.environ["JQDATA_PASSWORD"])
        unit = {"1m": "1m", "5m": "5m", "1d": "1d"}[freq]
        start_date, end_date = _jq_date_range(freq, start, end, self.sessions)
        bars: list[Bar] = []
        for symbol in symbols:
            frame = jq.get_price(symbol, start_date=start_date, end_date=end_date, frequency=unit, fields=list(fields), fq=None if adj == "none" else adj)
            for timestamp, row in frame.iterrows():
                bars.append(Bar(symbol=symbol, timestamp=timestamp.to_pydatetime(), open=float(row["open"]), high=float(row["high"]), low=float(row["low"]), close=float(row["close"]), volume=float(row["volume"])))
        return sorted(bars, key=lambda item: (item.timestamp, item.symbol))

    def _sample_bars(self, symbols: tuple[str, ...], freq: str, start: str, end: str) -> list[Bar]:
        stamps = calendar("cn", freq, start, end, self.sessions)
        bars: list[Bar] = []
        for symbol_index, symbol in enumerate(symbols):
            base = 10.0 + symbol_index
            for index, stamp in enumerate(stamps):
                drift = math.sin(index / 11.0) * 0.03 + index * 0.0005
                open_price = base + drift
                close_price = open_price + math.sin(index / 7.0) * 0.02
                bars.append(Bar(
                    symbol=symbol,
                    timestamp=stamp,
                    open=round(open_price, 4),
                    high=round(max(open_price, close_price) + 0.03, 4),
                    low=round(min(open_price, close_price) - 0.03, 4),
                    close=round(close_price, 4),
                    volume=1000 + index * 3,
                ))
        return bars


def get_provider(name: str, sessions: tuple[tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "15:00"))) -> JQDataAdapter:
    if name.lower() != "jqdata":
        raise ValueError(f"Unsupported provider: {name}")
    return JQDataAdapter(sessions=sessions)


def _jq_date_range(freq: str, start: str, end: str, sessions: tuple[tuple[str, str], ...]) -> tuple[str, str]:
    if freq == "1d" or _has_time(start) or _has_time(end):
        return start, end
    return f"{start} {sessions[0][0]}:00", f"{end} {sessions[-1][1]}:00"


def _has_time(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return True
    return False
