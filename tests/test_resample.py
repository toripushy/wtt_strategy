from datetime import datetime

from btcore.data.resample import resample_1m_to_5m
from btcore.models import Bar


def test_resample_1m_to_5m_ohlcv():
    bars = [
        Bar("000001.XSHE", datetime(2026, 1, 5, 9, 30 + index), 10 + index, 11 + index, 9 + index, 10.5 + index, 100)
        for index in range(5)
    ]

    result = resample_1m_to_5m(bars)

    assert len(result) == 1
    assert result[0].timestamp == datetime(2026, 1, 5, 9, 30)
    assert result[0].open == 10
    assert result[0].high == 15
    assert result[0].low == 9
    assert result[0].close == 14.5
    assert result[0].volume == 500
