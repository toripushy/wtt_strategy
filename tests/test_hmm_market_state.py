from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from hmm_market_state.backtest import backtest_regime_strategy, summarize_backtest
from hmm_market_state.features import FeatureConfig, build_market_features
from hmm_market_state.model import GaussianHMM
from hmm_market_state.pipeline import HMMMarketRegimePipeline
from hmm_market_state.strategy import RegimeMapper
from hmm_market_state.walk_forward import walk_forward_regime_detection


def _synthetic_price_frame(n: int = 420, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime_lengths = [n // 3, n // 3, n - 2 * (n // 3)]
    regime_specs = [
        (0.0012, 0.008, 1.0),
        (0.0001, 0.028, 1.8),
        (-0.0010, 0.012, 1.3),
    ]

    returns = []
    volumes = []
    for length, (mu, sigma, volume_scale) in zip(regime_lengths, regime_specs, strict=True):
        returns.append(mu + sigma * rng.standard_normal(length))
        volumes.append(
            np.exp(13.0 + volume_scale * rng.standard_normal(length) + 5.0 * abs(mu))
        )

    ret = np.concatenate(returns)
    volume = np.concatenate(volumes)
    close = 100.0 * np.exp(np.cumsum(ret))
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


def _synthetic_feature_matrix(n: int = 300, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.repeat(np.array([0, 1, 2]), repeats=[100, 100, 100])
    means = np.array(
        [
            [0.0015, 0.008],
            [0.0000, 0.030],
            [-0.0012, 0.012],
        ]
    )
    covars = np.array(
        [
            [0.00002, 0.00001],
            [0.00003, 0.00002],
            [0.00002, 0.00001],
        ]
    )
    X = np.vstack(
        [
            rng.normal(loc=means[state], scale=np.sqrt(covars[state]), size=(1, 2))
            for state in states
        ]
    )
    return X, states


class TestGaussianHMM(unittest.TestCase):
    def test_fit_and_predict_recovers_states(self) -> None:
        X, true_states = _synthetic_feature_matrix()
        model = GaussianHMM(n_states=3, n_iter=50, tol=1e-6, random_state=3)
        model.fit(X)
        self.assertGreater(len(model.loglik_history_), 1)
        self.assertGreater(model.loglik_history_[-1], model.loglik_history_[0] - 1e-8)

        pred = model.predict(X)

        mapping: dict[int, int] = {}
        for state in np.unique(pred):
            mask = pred == state
            label = int(np.bincount(true_states[mask]).argmax())
            mapping[int(state)] = label
        aligned = np.array([mapping[int(s)] for s in pred])
        accuracy = (aligned == true_states).mean()
        self.assertGreater(accuracy, 0.8)

    def test_save_and_load_round_trip(self) -> None:
        X, _ = _synthetic_feature_matrix()
        model = GaussianHMM(n_states=3, n_iter=20, random_state=5).fit(X)
        with TemporaryDirectory() as tmp_dir:
            path = model.save(f"{tmp_dir}/test_hmm_model")
            loaded = GaussianHMM.load(path)
        np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X), atol=1e-8)

    def test_predict_allows_single_observation_after_fit(self) -> None:
        X, _ = _synthetic_feature_matrix()
        model = GaussianHMM(n_states=3, n_iter=20, random_state=5).fit(X)

        pred = model.predict(X[:1])
        proba = model.predict_proba(X[:1])
        filtered_pred = model.predict_filtered(X[:1])
        filtered_proba = model.predict_filtered_proba(X[:1])

        self.assertEqual(pred.shape, (1,))
        self.assertEqual(filtered_pred.shape, (1,))
        self.assertEqual(proba.shape, (1, 3))
        self.assertEqual(filtered_proba.shape, (1, 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_allclose(filtered_proba.sum(axis=1), 1.0)

    def test_predict_rejects_feature_dimension_mismatch(self) -> None:
        X, _ = _synthetic_feature_matrix()
        model = GaussianHMM(n_states=3, n_iter=20, random_state=5).fit(X)

        with self.assertRaisesRegex(ValueError, "fitted with 2"):
            model.predict(X[:, :1])

    def test_filtered_predictions_are_prefix_stable(self) -> None:
        model = GaussianHMM(n_states=2)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.97, 0.03], [0.03, 0.97]])
        model.means_ = np.array([[0.0], [3.0]])
        model.covars_ = np.array([[1.0], [1.0]])
        model.n_features_ = 1
        X = np.array([[0.92522458], [4.64881656], [0.63442831], [0.02903342], [1.99957074]])

        prefix_proba = model.predict_filtered_proba(X[:3])
        extended_proba = model.predict_filtered_proba(X)[:3]
        np.testing.assert_allclose(prefix_proba, extended_proba)
        np.testing.assert_array_equal(model.predict_filtered(X[:3]), model.predict_filtered(X)[:3])


class TestFeatureAndPipeline(unittest.TestCase):
    def test_feature_engineering_and_backtest(self) -> None:
        data = _synthetic_price_frame()
        features, aligned = build_market_features(data, FeatureConfig(vol_window=20, use_volume=True))
        self.assertIn("log_return", features.columns)
        self.assertIn("rolling_vol", features.columns)
        self.assertGreater(len(features), 0)
        self.assertEqual(len(features), len(aligned))
        self.assertFalse(features.isna().any().any())

        positions = pd.Series(np.where(features["log_return"] > 0, 1.0, 0.2), index=features.index)
        result = backtest_regime_strategy(features["log_return"], positions, cost_bps=2.0)
        summary = summarize_backtest(result)
        self.assertIn("sharpe", summary)
        self.assertTrue(np.isfinite(summary["total_return"]))
        self.assertTrue(np.isfinite(summary["sharpe"]))

    def test_pipeline_fit_transform(self) -> None:
        data = _synthetic_price_frame()
        pipeline = HMMMarketRegimePipeline(
            feature_config=FeatureConfig(vol_window=20, use_volume=True),
            model_kwargs={"n_states": 3, "n_iter": 40, "random_state": 8},
        )
        output = pipeline.fit_transform(data)
        self.assertIn("state", output.columns)
        self.assertIn("regime", output.columns)
        self.assertIn("position", output.columns)
        self.assertGreater(len(output), 0)

    def test_walk_forward_detection(self) -> None:
        data = _synthetic_price_frame(n=480)
        result = walk_forward_regime_detection(
            data,
            feature_config=FeatureConfig(vol_window=20, use_volume=True),
            train_window=180,
            test_window=60,
            step=60,
            model_kwargs={"n_states": 3, "n_iter": 30, "random_state": 2},
        )
        self.assertGreater(len(result.predictions), 0)
        self.assertIn("position", result.predictions.columns)
        self.assertTrue(result.predictions.index.is_monotonic_increasing)
        self.assertIn("mean_return", result.regime_summary.columns)

    def test_walk_forward_uses_filtered_predictions(self) -> None:
        data = _synthetic_price_frame(n=300)
        feature_config = FeatureConfig(vol_window=20, use_volume=True)
        model_kwargs = {"n_states": 3, "n_iter": 10, "random_state": 2}
        train_window = 120
        test_window = 40

        result = walk_forward_regime_detection(
            data,
            feature_config=feature_config,
            train_window=train_window,
            test_window=test_window,
            step=test_window,
            model_kwargs=model_kwargs,
        )

        features, _ = build_market_features(data, feature_config)
        train_X = features.iloc[:train_window].to_numpy(dtype=float)
        test_X = features.iloc[train_window : train_window + test_window].to_numpy(dtype=float)
        expected_model = GaussianHMM(**model_kwargs).fit(train_X)
        expected_states = expected_model.predict_filtered(test_X)
        expected_probs = expected_model.predict_filtered_proba(test_X)
        first_chunk = result.predictions.iloc[:test_window]

        np.testing.assert_array_equal(first_chunk["state"].to_numpy(), expected_states)
        for state_id in range(expected_probs.shape[1]):
            np.testing.assert_allclose(first_chunk[f"prob_state_{state_id}"].to_numpy(), expected_probs[:, state_id])

    def test_regime_mapper(self) -> None:
        X, states = _synthetic_feature_matrix()
        mapper = RegimeMapper().fit(X, states)
        regimes = mapper.transform(states)
        self.assertEqual(len(regimes), len(states))
        self.assertIn("bull", set(regimes))


if __name__ == "__main__":
    unittest.main()
