from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .features import FeatureConfig, build_market_features
from .model import GaussianHMM
from .strategy import RegimeMapper, summarize_regimes


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    regime_summary: pd.DataFrame


def walk_forward_regime_detection(
    data: pd.DataFrame,
    *,
    feature_config: FeatureConfig | None = None,
    train_window: int = 756,
    test_window: int = 63,
    step: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    position_map: dict[str, float] | None = None,
) -> WalkForwardResult:
    """Rolling train/predict regime detection without future leakage."""

    if feature_config is None:
        feature_config = FeatureConfig()
    if model_kwargs is None:
        model_kwargs = {}
    if position_map is None:
        position_map = {"bull": 1.0, "bear": 0.35, "risk": 0.0}
    if step is None:
        step = test_window

    features, _ = build_market_features(data, feature_config)
    if len(features) < train_window + test_window:
        raise ValueError("Not enough observations for the requested walk-forward split")

    prediction_chunks: list[pd.DataFrame] = []
    regime_summaries: list[pd.DataFrame] = []

    end = len(features) - test_window + 1
    for start in range(train_window, end, step):
        train_X = features.iloc[start - train_window : start].to_numpy(dtype=float)
        test_X = features.iloc[start : start + test_window].to_numpy(dtype=float)
        test_index = features.index[start : start + test_window]

        model = GaussianHMM(**model_kwargs)
        model.fit(train_X)
        train_states = model.predict(train_X)
        mapper = RegimeMapper(position_map=position_map).fit(train_X, train_states)

        test_states = model.predict_filtered(test_X)
        test_regimes = mapper.transform(test_states)
        positions = mapper.position_series(test_states, index=test_index)

        probs = model.predict_filtered_proba(test_X)
        chunk = pd.DataFrame(
            {
                "state": test_states,
                "regime": test_regimes,
                "position": positions,
            },
            index=test_index,
        )
        for state_id in range(probs.shape[1]):
            chunk[f"prob_state_{state_id}"] = probs[:, state_id]
        prediction_chunks.append(chunk)
        regime_summaries.append(summarize_regimes(train_X, train_states))

    predictions = pd.concat(prediction_chunks).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="last")]
    regime_summary = pd.concat(regime_summaries).groupby(level=0).mean(numeric_only=True)

    return WalkForwardResult(predictions=predictions, regime_summary=regime_summary)
