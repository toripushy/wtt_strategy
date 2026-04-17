from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    shifted = np.exp(a - a_max)
    summed = np.sum(shifted, axis=axis, keepdims=True)
    out = np.log(summed) + a_max
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _log_gaussian_density_diag(
    X: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    reg_covar: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    covars = np.asarray(covars, dtype=float) + reg_covar

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if means.ndim != 2 or covars.ndim != 2:
        raise ValueError("means and covars must be 2D arrays")

    n_samples, _ = X.shape
    n_states = means.shape[0]
    log_prob = np.empty((n_samples, n_states), dtype=float)
    log_2pi = np.log(2.0 * np.pi)

    for k in range(n_states):
        var = np.clip(covars[k], reg_covar, None)
        diff = X - means[k]
        log_prob[:, k] = -0.5 * (
            np.sum(np.log(var) + log_2pi) + np.sum((diff * diff) / var, axis=1)
        )

    return log_prob


@dataclass
class GaussianHMM:
    """Minimal diagonal-covariance Gaussian HMM for market regime detection."""

    n_states: int = 3
    n_iter: int = 100
    n_init: int = 3
    tol: float = 1e-4
    reg_covar: float = 1e-6
    random_state: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.loglik_history_: list[float] = []
        self.n_features_: int | None = None

    def _check_array(self, X: np.ndarray, *, min_samples: int = 1) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) < min_samples:
            raise ValueError(f"Need at least {min_samples} observations")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        return X

    def _check_fit_X(self, X: np.ndarray) -> np.ndarray:
        X = self._check_array(X, min_samples=self.n_states)
        return X

    def _check_is_fitted(self) -> None:
        if any(v is None for v in (self.startprob_, self.transmat_, self.means_, self.covars_)):
            raise ValueError("Model has not been fitted yet")

    def _check_predict_X(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._check_array(X, min_samples=1)
        assert self.means_ is not None
        expected_features = self.n_features_ if self.n_features_ is not None else self.means_.shape[1]
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {expected_features}"
            )
        return X

    def _initialize(self, X: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, _ = X.shape

        if self.n_states == 3 and X.shape[1] >= 2:
            chosen = np.array(
                [
                    int(np.argmin(X[:, 0])),
                    int(np.argmax(X[:, 1])),
                    int(np.argmax(X[:, 0])),
                ],
                dtype=int,
            )
            if len(np.unique(chosen)) < 3:
                order = np.argsort(X[:, 0], kind="mergesort")
                positions = np.linspace(0, n_samples - 1, self.n_states).astype(int)
                chosen = order[positions]
        else:
            order = np.argsort(X[:, 0], kind="mergesort")
            positions = np.linspace(0, n_samples - 1, self.n_states).astype(int)
            chosen = order[positions]

        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states, dtype=float)
        self.transmat_ = np.full((self.n_states, self.n_states), 1.0 / self.n_states, dtype=float)
        if self.n_states > 1:
            np.fill_diagonal(self.transmat_, 0.75)
            off_diag = (1.0 - 0.75) / (self.n_states - 1)
            self.transmat_[:] = off_diag
            np.fill_diagonal(self.transmat_, 0.75)

        self.means_ = X[chosen].astype(float).copy()
        global_var = np.var(X, axis=0) + self.reg_covar
        global_var = np.where(global_var <= 0, 1.0, global_var)
        self.covars_ = np.tile(global_var, (self.n_states, 1)).astype(float)

        if self.n_states > 1:
            jitter = rng.normal(scale=0.05, size=self.means_.shape)
            self.means_ += jitter * np.std(X, axis=0, keepdims=True)

    def _forward(
        self, log_startprob: np.ndarray, log_transmat: np.ndarray, log_emlik: np.ndarray
    ) -> tuple[np.ndarray, float]:
        n_samples, n_states = log_emlik.shape
        log_alpha = np.empty((n_samples, n_states), dtype=float)
        log_alpha[0] = log_startprob + log_emlik[0]
        for t in range(1, n_samples):
            log_alpha[t] = log_emlik[t] + _logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
        log_prob = float(_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, log_prob

    def _backward(self, log_transmat: np.ndarray, log_emlik: np.ndarray) -> np.ndarray:
        n_samples, n_states = log_emlik.shape
        log_beta = np.empty((n_samples, n_states), dtype=float)
        log_beta[-1] = 0.0
        for t in range(n_samples - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_transmat + log_emlik[t + 1][None, :] + log_beta[t + 1][None, :],
                axis=1,
            )
        return log_beta

    def _e_step(self, X: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.means_ is not None
        assert self.covars_ is not None

        log_startprob = np.log(np.clip(self.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.transmat_, 1e-300, None))
        log_emlik = _log_gaussian_density_diag(X, self.means_, self.covars_, self.reg_covar)

        log_alpha, log_prob = self._forward(log_startprob, log_transmat, log_emlik)
        log_beta = self._backward(log_transmat, log_emlik)

        log_gamma = log_alpha + log_beta - log_prob
        gamma = np.exp(log_gamma)
        gamma = np.clip(gamma, 1e-300, None)
        gamma /= gamma.sum(axis=1, keepdims=True)
        return log_prob, gamma, log_emlik

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, log_emlik: np.ndarray) -> None:
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.means_ is not None
        assert self.covars_ is not None

        n_samples, _ = X.shape
        n_states = self.n_states
        weights = gamma.sum(axis=0)

        self.startprob_ = gamma[0] / np.sum(gamma[0])

        log_startprob = np.log(np.clip(self.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.transmat_, 1e-300, None))
        log_alpha, log_prob = self._forward(log_startprob, log_transmat, log_emlik)
        log_beta = self._backward(log_transmat, log_emlik)

        xi_sum = np.zeros((n_states, n_states), dtype=float)
        for t in range(n_samples - 1):
            log_xi_t = (
                log_alpha[t][:, None]
                + log_transmat
                + log_emlik[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - log_prob
            )
            xi_sum += np.exp(log_xi_t)

        row_sums = xi_sum.sum(axis=1, keepdims=True)
        self.transmat_ = np.zeros_like(xi_sum)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(xi_sum, row_sums, out=self.transmat_, where=row_sums > 0)
        zero_rows = np.where(row_sums.squeeze() <= 0)[0]
        if len(zero_rows) > 0:
            self.transmat_[zero_rows] = 1.0 / n_states

        global_var = np.var(X, axis=0) + self.reg_covar
        for k in range(n_states):
            weight = weights[k]
            if weight <= 1e-8:
                idx = int(np.random.default_rng(self.random_state).integers(0, n_samples))
                self.means_[k] = X[idx]
                self.covars_[k] = global_var
                continue

            mean = np.sum(gamma[:, k][:, None] * X, axis=0) / weight
            self.means_[k] = mean
            diff = X - mean
            var = np.sum(gamma[:, k][:, None] * (diff * diff), axis=0) / weight
            self.covars_[k] = np.clip(var, self.reg_covar, None)

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = self._check_fit_X(X)
        self.n_features_ = X.shape[1]
        self.loglik_history_ = []

        best_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]] | None = None
        best_log_prob = -np.inf

        for init_idx in range(max(1, self.n_init)):
            seed = None if self.random_state is None else self.random_state + init_idx * 1009
            rng = np.random.default_rng(seed)
            self._initialize(X, rng)

            history: list[float] = []
            prev_log_prob = -np.inf
            for iteration in range(self.n_iter):
                log_prob, gamma, log_emlik = self._e_step(X)
                self._m_step(X, gamma, log_emlik)
                history.append(float(log_prob))

                if self.verbose:
                    print(f"init={init_idx + 1} iteration={iteration + 1} loglik={log_prob:.6f}")

                if np.isfinite(prev_log_prob) and abs(log_prob - prev_log_prob) < self.tol:
                    break
                prev_log_prob = log_prob

            final_log_prob = history[-1]
            if final_log_prob > best_log_prob:
                best_log_prob = final_log_prob
                best_params = (
                    self.startprob_.copy(),
                    self.transmat_.copy(),
                    self.means_.copy(),
                    self.covars_.copy(),
                    history.copy(),
                )

        assert best_params is not None
        self.startprob_, self.transmat_, self.means_, self.covars_, self.loglik_history_ = best_params

        return self

    def score(self, X: np.ndarray) -> float:
        X = self._check_predict_X(X)
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.means_ is not None
        assert self.covars_ is not None
        log_startprob = np.log(np.clip(self.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.transmat_, 1e-300, None))
        log_emlik = _log_gaussian_density_diag(X, self.means_, self.covars_, self.reg_covar)
        _, log_prob = self._forward(log_startprob, log_transmat, log_emlik)
        return float(log_prob)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._check_predict_X(X)
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.means_ is not None
        assert self.covars_ is not None
        log_startprob = np.log(np.clip(self.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.transmat_, 1e-300, None))
        log_emlik = _log_gaussian_density_diag(X, self.means_, self.covars_, self.reg_covar)
        log_alpha, log_prob = self._forward(log_startprob, log_transmat, log_emlik)
        log_beta = self._backward(log_transmat, log_emlik)
        gamma = np.exp(log_alpha + log_beta - log_prob)
        gamma = np.clip(gamma, 1e-300, None)
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def predict_filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """Return forward-only state probabilities using observations up to each row."""

        X = self._check_predict_X(X)
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.means_ is not None
        assert self.covars_ is not None
        log_startprob = np.log(np.clip(self.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.transmat_, 1e-300, None))
        log_emlik = _log_gaussian_density_diag(X, self.means_, self.covars_, self.reg_covar)
        log_alpha, _ = self._forward(log_startprob, log_transmat, log_emlik)
        log_norm = _logsumexp(log_alpha, axis=1, keepdims=True)
        proba = np.exp(log_alpha - log_norm)
        proba = np.clip(proba, 1e-300, None)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict_filtered(self, X: np.ndarray) -> np.ndarray:
        """Return online state labels without using future observations."""

        return np.argmax(self.predict_filtered_proba(X), axis=1).astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_predict_X(X)
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.means_ is not None
        assert self.covars_ is not None

        log_startprob = np.log(np.clip(self.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.transmat_, 1e-300, None))
        log_emlik = _log_gaussian_density_diag(X, self.means_, self.covars_, self.reg_covar)

        n_samples, n_states = log_emlik.shape
        delta = np.empty((n_samples, n_states), dtype=float)
        psi = np.empty((n_samples, n_states), dtype=int)
        delta[0] = log_startprob + log_emlik[0]
        psi[0] = 0

        for t in range(1, n_samples):
            scores = delta[t - 1][:, None] + log_transmat
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emlik[t]

        states = np.empty(n_samples, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_samples - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states

    def to_dict(self) -> dict[str, Any]:
        if any(v is None for v in (self.startprob_, self.transmat_, self.means_, self.covars_)):
            raise ValueError("Model has not been fitted yet")
        return {
            "n_states": self.n_states,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "startprob": self.startprob_.tolist(),
            "transmat": self.transmat_.tolist(),
            "means": self.means_.tolist(),
            "covars": self.covars_.tolist(),
            "loglik_history": self.loglik_history_,
            "n_features": self.n_features_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GaussianHMM":
        model = cls(
            n_states=int(payload["n_states"]),
            n_iter=int(payload.get("n_iter", 100)),
            tol=float(payload.get("tol", 1e-4)),
            reg_covar=float(payload.get("reg_covar", 1e-6)),
            random_state=payload.get("random_state"),
            verbose=bool(payload.get("verbose", False)),
        )
        model.startprob_ = np.asarray(payload["startprob"], dtype=float)
        model.transmat_ = np.asarray(payload["transmat"], dtype=float)
        model.means_ = np.asarray(payload["means"], dtype=float)
        model.covars_ = np.asarray(payload["covars"], dtype=float)
        model.loglik_history_ = [float(v) for v in payload.get("loglik_history", [])]
        model.n_features_ = int(payload.get("n_features", model.means_.shape[1]))
        return model

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        if any(v is None for v in (self.startprob_, self.transmat_, self.means_, self.covars_)):
            raise ValueError("Model has not been fitted yet")
        np.savez_compressed(
            path,
            n_states=self.n_states,
            n_iter=self.n_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            random_state=-1 if self.random_state is None else self.random_state,
            verbose=int(self.verbose),
            startprob=self.startprob_,
            transmat=self.transmat_,
            means=self.means_,
            covars=self.covars_,
            loglik_history=np.asarray(self.loglik_history_, dtype=float),
            n_features=-1 if self.n_features_ is None else self.n_features_,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GaussianHMM":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            random_state = int(data["random_state"])
            if random_state < 0:
                random_state = None
            model = cls(
                n_states=int(data["n_states"]),
                n_iter=int(data["n_iter"]),
                tol=float(data["tol"]),
                reg_covar=float(data["reg_covar"]),
                random_state=random_state,
                verbose=bool(int(data["verbose"])),
            )
            model.startprob_ = np.asarray(data["startprob"], dtype=float)
            model.transmat_ = np.asarray(data["transmat"], dtype=float)
            model.means_ = np.asarray(data["means"], dtype=float)
            model.covars_ = np.asarray(data["covars"], dtype=float)
            model.loglik_history_ = list(np.asarray(data["loglik_history"], dtype=float))
            n_features = int(data["n_features"])
            model.n_features_ = None if n_features < 0 else n_features
        return model
