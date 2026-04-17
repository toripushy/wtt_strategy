from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for market feature engineering."""

    close_col: str = "close"
    volume_col: str = "volume"
    vol_window: int = 20
    volume_window: int = 20
    use_volume: bool = False


def load_market_data_csv(
    path: str | Path,
    *,
    date_col: str = "date",
    index_col: str | None = None,
) -> pd.DataFrame:
    """Load market data from a CSV file."""

    frame = pd.read_csv(path)
    if date_col in frame.columns:
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame = frame.sort_values(date_col)
        frame = frame.set_index(date_col)
    elif index_col and index_col in frame.columns:
        frame[index_col] = pd.to_datetime(frame[index_col])
        frame = frame.sort_values(index_col)
        frame = frame.set_index(index_col)

    return frame


def build_market_features(
    data: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build HMM-ready features from OHLCV-like market data."""

    if config is None:
        config = FeatureConfig()

    frame = data.copy()
    if config.close_col not in frame.columns:
        raise KeyError(f"Missing required close column: {config.close_col}")

    close = pd.to_numeric(frame[config.close_col], errors="coerce")
    log_close = np.log(close.replace(0, np.nan))
    log_return = log_close.diff()
    rolling_vol = log_return.rolling(config.vol_window, min_periods=config.vol_window).std()

    features = pd.DataFrame(index=frame.index)
    features["log_return"] = log_return
    features["rolling_vol"] = rolling_vol

    if config.use_volume:
        if config.volume_col not in frame.columns:
            raise KeyError(f"Missing required volume column: {config.volume_col}")
        volume = pd.to_numeric(frame[config.volume_col], errors="coerce")
        log_volume = np.log(volume.replace(0, np.nan))
        volume_change = log_volume.diff()
        volume_vol = volume_change.rolling(
            config.volume_window, min_periods=config.volume_window
        ).std()
        features["volume_change"] = volume_change
        features["volume_vol"] = volume_vol

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    aligned_frame = frame.loc[features.index].copy()
    return features, aligned_frame


def ensure_datetime_index(frame: pd.DataFrame, index_candidates: Iterable[str]) -> pd.DataFrame:
    """Best-effort helper for demos/tests."""

    result = frame.copy()
    for candidate in index_candidates:
        if candidate in result.columns:
            result[candidate] = pd.to_datetime(result[candidate])
            result = result.sort_values(candidate).set_index(candidate)
            return result
    return result
