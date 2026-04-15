from __future__ import annotations

from datetime import date, datetime, time, timedelta


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def trading_days(start: str, end: str) -> list[date]:
    current = parse_date(start)
    stop = parse_date(end)
    days: list[date] = []
    while current <= stop:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def calendar(exchange: str, freq: str, start: str, end: str, sessions: tuple[tuple[str, str], ...]) -> list[datetime]:
    if exchange.lower() not in {"xshg", "xshe", "cn", "china"}:
        raise ValueError(f"Unsupported exchange calendar: {exchange}")
    if freq == "1d":
        return [datetime.combine(day, time(15, 0)) for day in trading_days(start, end)]
    if freq not in {"1m", "5m"}:
        raise ValueError(f"Unsupported frequency: {freq}")
    step = timedelta(minutes=1 if freq == "1m" else 5)
    stamps: list[datetime] = []
    for day in trading_days(start, end):
        for start_text, end_text in sessions:
            current = datetime.combine(day, parse_time(start_text))
            session_end = datetime.combine(day, parse_time(end_text))
            while current <= session_end:
                stamps.append(current)
                current += step
    return stamps
