from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .features import FeatureConfig, build_market_features
from .model import GaussianHMM
from .strategy import RegimeMapper, summarize_regimes


@dataclass
class HMMMarketRegimePipeline:
    """End-to-end HMM pipeline for regime detection and strategy mapping.

    ``fit_transform`` fits on the full input before producing signals, so its
    output is for in-sample labeling. Use walk-forward detection for backtests.
    """

    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    position_map: dict[str, float] = field(
        default_factory=lambda: {"bull": 1.0, "bear": 0.35, "risk": 0.0}
    )

    model_: GaussianHMM | None = None
    mapper_: RegimeMapper | None = None
    feature_frame_: pd.DataFrame | None = None
    aligned_frame_: pd.DataFrame | None = None
    state_summary_: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame) -> "HMMMarketRegimePipeline":
        features, aligned = build_market_features(data, self.feature_config)
        model = GaussianHMM(**self.model_kwargs)
        model.fit(features.to_numpy(dtype=float))
        states = model.predict(features.to_numpy(dtype=float))
        mapper = RegimeMapper(position_map=self.position_map).fit(features.to_numpy(dtype=float), states)

        self.model_ = model
        self.mapper_ = mapper
        self.feature_frame_ = features
        self.aligned_frame_ = aligned
        self.state_summary_ = summarize_regimes(features.to_numpy(dtype=float), states)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.mapper_ is None:
            raise ValueError("Pipeline has not been fitted yet")
        features, aligned = build_market_features(data, self.feature_config)
        feature_values = features.to_numpy(dtype=float)
        states = self.model_.predict_filtered(feature_values)
        regimes = self.mapper_.transform(states)
        positions = self.mapper_.position_series(states, index=features.index)
        output = pd.DataFrame(
            {
                "state": states,
                "regime": regimes,
                "position": positions,
            },
            index=features.index,
        )
        probs = self.model_.predict_filtered_proba(feature_values)
        for state_id in range(probs.shape[1]):
            output[f"prob_state_{state_id}"] = probs[:, state_id]
        return output.join(aligned, how="left")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)
