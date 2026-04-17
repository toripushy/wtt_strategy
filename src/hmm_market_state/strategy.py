from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_POSITION_MAP = {
    "bull": 1.0,
    "bear": 0.35,
    "risk": 0.0,
}


@dataclass
class RegimeMapper:
    """Map anonymous HMM states to interpretable market regimes."""

    position_map: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_POSITION_MAP))

    state_to_regime_: dict[int, str] | None = None
    state_summary_: pd.DataFrame | None = None

    def fit(self, features: np.ndarray, states: np.ndarray) -> "RegimeMapper":
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
        frame["state"] = np.asarray(states, dtype=int)

        summary = (
            frame.groupby("state")
            .agg(
                count=("state", "size"),
                mean_return=("feature_0", "mean"),
                vol_return=("feature_0", "std"),
            )
            .fillna(0.0)
            .sort_index()
        )
        if "feature_1" in frame.columns:
            summary["mean_vol_feature"] = frame.groupby("state")["feature_1"].mean()
        else:
            summary["mean_vol_feature"] = np.nan

        state_to_regime: dict[int, str] = {}
        ordered_states = list(summary.sort_values(by=["vol_return", "mean_return"], ascending=[False, False]).index.astype(int))

        if len(ordered_states) == 1:
            state_to_regime[ordered_states[0]] = "bull"
        elif len(ordered_states) == 2:
            risk_state = int(summary["vol_return"].idxmax())
            other_state = int([s for s in ordered_states if s != risk_state][0])
            state_to_regime[risk_state] = "risk"
            state_to_regime[other_state] = "bull" if summary.loc[other_state, "mean_return"] >= 0 else "bear"
        else:
            risk_state = int(summary["vol_return"].idxmax())
            state_to_regime[risk_state] = "risk"

            remaining = summary.drop(index=risk_state)
            bull_state = int(remaining["mean_return"].idxmax())
            state_to_regime[bull_state] = "bull"

            for state in remaining.index.astype(int):
                if state != bull_state:
                    state_to_regime[int(state)] = "bear"

        for state in summary.index.astype(int):
            state_to_regime.setdefault(int(state), "risk")

        self.state_summary_ = summary
        self.state_to_regime_ = state_to_regime
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        if self.state_to_regime_ is None:
            raise ValueError("RegimeMapper has not been fitted yet")
        states = np.asarray(states, dtype=int)
        return np.array([self.state_to_regime_.get(int(s), "risk") for s in states], dtype=object)

    def position_series(self, states: np.ndarray, index: pd.Index | None = None) -> pd.Series:
        regimes = self.transform(states)
        positions = np.array([self.position_map.get(str(regime), 0.0) for regime in regimes], dtype=float)
        return pd.Series(positions, index=index, name="position")


def summarize_regimes(features: np.ndarray, states: np.ndarray) -> pd.DataFrame:
    """Return a compact summary table for fitted states."""

    frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    frame["state"] = np.asarray(states, dtype=int)
    summary = frame.groupby("state").agg(
        count=("state", "size"),
        mean_return=("feature_0", "mean"),
        vol_return=("feature_0", "std"),
        mean_feature_1=("feature_1", "mean") if features.shape[1] > 1 else ("feature_0", "mean"),
    )
    return summary.sort_index()
