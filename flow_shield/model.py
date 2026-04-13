"""Policy models for learned continuous-space MAPF planning."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import ModelConfig, SimConfig
from .dataset import FEATURE_DIM, encode_joint_observation, normalize_observation_version
from .geometry import clip_by_norm
from .maps import GridMap


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


class NumpyAttentionPolicy:
    """A compact trainable single-head transformer-style velocity regressor."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 32,
        max_speed: float = 1.2,
        seed: int = 0,
    ):
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.max_speed = float(max_speed)
        self.observation_version = "legacy"
        self.observation_metadata: Dict[str, object] = {}
        rng = np.random.default_rng(seed)
        scale_in = 1.0 / np.sqrt(max(self.feature_dim, 1))
        scale_hidden = 1.0 / np.sqrt(max(self.d_model, 1))
        self.params: Dict[str, np.ndarray] = {
            "w_in": rng.normal(0.0, scale_in, size=(self.feature_dim, self.d_model)),
            "b_in": np.zeros(self.d_model, dtype=np.float64),
            "w_q": rng.normal(0.0, scale_hidden, size=(self.d_model, self.d_model)),
            "w_k": rng.normal(0.0, scale_hidden, size=(self.d_model, self.d_model)),
            "w_v": rng.normal(0.0, scale_hidden, size=(self.d_model, self.d_model)),
            "w_out": rng.normal(0.0, scale_hidden, size=(self.d_model, 2)),
            "b_out": np.zeros(2, dtype=np.float64),
        }
        self._adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_t = 0

    @classmethod
    def from_config(cls, config: ModelConfig, sim_config: SimConfig) -> "NumpyAttentionPolicy":
        return cls(
            feature_dim=config.feature_dim,
            d_model=config.d_model,
            max_speed=sim_config.max_speed,
            seed=config.seed,
        )

    def _forward(self, observations: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        p = self.params
        z = np.einsum("btf,fd->btd", observations, p["w_in"]) + p["b_in"]
        e = np.tanh(z)
        q = np.einsum("btd,de->bte", e, p["w_q"])
        k = np.einsum("btd,de->bte", e, p["w_k"])
        v = np.einsum("btd,de->bte", e, p["w_v"])
        scores = np.einsum("btd,bsd->bts", q, k) / np.sqrt(self.d_model)
        key_mask = masks[:, None, :]
        scores = np.where(key_mask, scores, -1e9)
        attention = _softmax(scores)
        hidden = np.einsum("bts,bsd->btd", attention, v)
        context = hidden[:, 0, :]
        out_pre = np.einsum("bd,do->bo", context, p["w_out"]) + p["b_out"]
        predictions = np.tanh(out_pre) * self.max_speed
        predictions = np.nan_to_num(
            predictions,
            nan=0.0,
            posinf=self.max_speed,
            neginf=-self.max_speed,
        )
        cache = {
            "observations": observations,
            "masks": masks,
            "z": z,
            "e": e,
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "attention": attention,
            "hidden": hidden,
            "context": context,
            "out_pre": out_pre,
            "predictions": predictions,
        }
        return predictions, cache

    def predict_batch(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        predictions, _ = self._forward(
            np.asarray(observations, dtype=np.float64),
            np.asarray(masks, dtype=bool),
        )
        return predictions

    def predict_joint(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        max_neighbors: int,
        sim_config: SimConfig,
        obstacle_map: GridMap | None = None,
        max_obstacle_tokens: int = 0,
        obstacle_context_range: float = 4.0,
        observation_version: str | None = None,
    ) -> np.ndarray:
        observations, masks = encode_joint_observation(
            positions,
            velocities,
            goals,
            radii,
            max_neighbors,
            sim_config,
            obstacle_map=obstacle_map,
            max_obstacle_tokens=max_obstacle_tokens,
            obstacle_context_range=obstacle_context_range,
            observation_version=observation_version or self.observation_version,
        )
        return self.predict_batch(observations, masks)

    def loss(self, observations: np.ndarray, masks: np.ndarray, targets: np.ndarray) -> float:
        predictions = self.predict_batch(observations, masks)
        return float(np.mean((predictions - targets) ** 2))

    def fit(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        config: ModelConfig,
        verbose: bool = False,
    ) -> Dict[str, list]:
        rng = np.random.default_rng(config.seed)
        n_samples = observations.shape[0]
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        val_count = int(round(n_samples * config.validation_split))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:] if val_count < n_samples else indices
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(config.epochs):
            rng.shuffle(train_indices)
            batch_losses = []
            for start in range(0, len(train_indices), config.batch_size):
                batch_idx = train_indices[start : start + config.batch_size]
                loss, grads = self._loss_and_grads(
                    observations[batch_idx],
                    masks[batch_idx],
                    targets[batch_idx],
                    weight_decay=config.weight_decay,
                )
                self._adam_step(grads, config.learning_rate)
                batch_losses.append(loss)
            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            if val_count:
                val_loss = self.loss(observations[val_indices], masks[val_indices], targets[val_indices])
            else:
                val_loss = train_loss
            history["train_loss"].append(train_loss)
            history["val_loss"].append(float(val_loss))
            if verbose:
                print(
                    f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f}"
                )
        return history

    def _loss_and_grads(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        weight_decay: float,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        predictions, cache = self._forward(observations, masks)
        batch = observations.shape[0]
        diff = predictions - targets
        mse = float(np.mean(diff**2))
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}

        d_pred = (2.0 / max(batch * 2, 1)) * diff
        tanh_out = predictions / self.max_speed
        d_out_pre = d_pred * self.max_speed * (1.0 - tanh_out**2)
        grads["w_out"] = np.einsum("bd,bo->do", cache["context"], d_out_pre)
        grads["b_out"] = np.sum(d_out_pre, axis=0)
        d_context = np.einsum("bo,do->bd", d_out_pre, self.params["w_out"])

        d_hidden = np.zeros_like(cache["hidden"])
        d_hidden[:, 0, :] = d_context
        d_attention = np.einsum("btd,bsd->bts", d_hidden, cache["v"])
        d_v = np.einsum("bts,btd->bsd", cache["attention"], d_hidden)

        row_dot = np.sum(d_attention * cache["attention"], axis=-1, keepdims=True)
        d_scores = cache["attention"] * (d_attention - row_dot)
        d_scores = np.where(cache["masks"][:, None, :], d_scores, 0.0)
        d_scores /= np.sqrt(self.d_model)
        d_q = np.einsum("bts,bsd->btd", d_scores, cache["k"])
        d_k = np.einsum("bts,btd->bsd", d_scores, cache["q"])

        e = cache["e"]
        grads["w_q"] = np.einsum("btd,bte->de", e, d_q)
        grads["w_k"] = np.einsum("btd,bte->de", e, d_k)
        grads["w_v"] = np.einsum("btd,bte->de", e, d_v)
        d_e = (
            np.einsum("bte,de->btd", d_q, self.params["w_q"])
            + np.einsum("bte,de->btd", d_k, self.params["w_k"])
            + np.einsum("bte,de->btd", d_v, self.params["w_v"])
        )
        d_z = d_e * (1.0 - e**2)
        d_z = np.where(cache["masks"][:, :, None], d_z, 0.0)
        grads["w_in"] = np.einsum("btf,btd->fd", observations, d_z)
        grads["b_in"] = np.sum(d_z, axis=(0, 1))

        if weight_decay > 0.0:
            for name, value in self.params.items():
                if name.startswith("w_"):
                    grads[name] += weight_decay * value
                    mse += float(0.5 * weight_decay * np.sum(value**2) / max(batch, 1))
        self._sanitize_grads(grads)
        self._clip_grads(grads, max_norm=5.0)
        return mse, grads

    def _sanitize_grads(self, grads: Dict[str, np.ndarray]) -> None:
        for name, grad in grads.items():
            grads[name] = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    def _clip_grads(self, grads: Dict[str, np.ndarray], max_norm: float) -> None:
        total = 0.0
        for grad in grads.values():
            total += float(np.sum(grad**2))
        norm = np.sqrt(total)
        if norm <= max_norm or norm <= 1e-12:
            return
        scale = max_norm / norm
        for name in grads:
            grads[name] *= scale

    def _adam_step(
        self,
        grads: Dict[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._adam_t += 1
        for name, grad in grads.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad**2)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
            self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        self._stabilize_params()

    def _stabilize_params(self) -> None:
        for name, value in self.params.items():
            self.params[name] = np.clip(
                np.nan_to_num(value, nan=0.0, posinf=3.0, neginf=-3.0),
                -3.0,
                3.0,
            )

    def save(self, path: str | Path, metadata: Dict[str, object] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(self.params)
        payload = metadata or {}
        observation_version = normalize_observation_version(
            payload.get("observation_version", self.observation_version)
        )
        arrays["metadata"] = np.array(
            json.dumps(
                {
                    "feature_dim": self.feature_dim,
                    "d_model": self.d_model,
                    "max_speed": self.max_speed,
                    "policy_type": "numpy_attention",
                    "observation_version": observation_version,
                    "observation_metadata": payload.get(
                        "observation_metadata",
                        self.observation_metadata,
                    ),
                    "metadata": payload,
                }
            )
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyAttentionPolicy":
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        model = cls(
            feature_dim=int(metadata["feature_dim"]),
            d_model=int(metadata["d_model"]),
            max_speed=float(metadata["max_speed"]),
            seed=0,
        )
        model.observation_version = normalize_observation_version(
            metadata.get("observation_version")
            or metadata.get("metadata", {}).get("observation_version")
            or metadata.get("metadata", {}).get("dataset_config", {}).get("observation_version")
        )
        model.observation_metadata = dict(metadata.get("observation_metadata", {}))
        for name in model.params:
            model.params[name] = np.asarray(data[name], dtype=np.float64)
        return model


class NumpyTransformerPolicy:
    """Scalable NumPy transformer-style policy for phase 4.

    The encoder is a fixed random multi-head self-attention stack, while the
    output head is trained with closed-form ridge regression. This keeps the
    path dependency-light and fast enough for scaled CPU smoke runs, while the
    artifact and rollout API remain compatible with the phase 1 policy.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_speed: float = 1.2,
        ridge_lambda: float = 1e-3,
        include_raw_features: bool = True,
        seed: int = 0,
    ):
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_speed = float(max_speed)
        self.ridge_lambda = float(ridge_lambda)
        self.include_raw_features = bool(include_raw_features)
        self.observation_version = "legacy"
        self.observation_metadata: Dict[str, object] = {}
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        rng = np.random.default_rng(seed)
        scale_in = 1.0 / np.sqrt(max(self.feature_dim, 1))
        scale_hidden = 1.0 / np.sqrt(max(self.d_model, 1))
        self.params: Dict[str, np.ndarray] = {
            "w_in": rng.normal(0.0, scale_in, size=(self.feature_dim, self.d_model)),
            "b_in": np.zeros(self.d_model, dtype=np.float64),
            "w_out": np.zeros((self.output_feature_dim, 2), dtype=np.float64),
            "b_out": np.zeros(2, dtype=np.float64),
        }
        for layer in range(self.num_layers):
            prefix = f"layer_{layer}_"
            self.params[prefix + "w_q"] = rng.normal(
                0.0, scale_hidden, size=(self.d_model, self.d_model)
            )
            self.params[prefix + "w_k"] = rng.normal(
                0.0, scale_hidden, size=(self.d_model, self.d_model)
            )
            self.params[prefix + "w_v"] = rng.normal(
                0.0, scale_hidden, size=(self.d_model, self.d_model)
            )
            self.params[prefix + "w_o"] = rng.normal(
                0.0, scale_hidden, size=(self.d_model, self.d_model)
            )
            self.params[prefix + "w_ff1"] = rng.normal(
                0.0, scale_hidden, size=(self.d_model, self.d_model)
            )
            self.params[prefix + "b_ff1"] = np.zeros(self.d_model, dtype=np.float64)
            self.params[prefix + "w_ff2"] = rng.normal(
                0.0, scale_hidden, size=(self.d_model, self.d_model)
            )
            self.params[prefix + "b_ff2"] = np.zeros(self.d_model, dtype=np.float64)

    @property
    def output_feature_dim(self) -> int:
        return self.d_model * 2 + (self.feature_dim if self.include_raw_features else 0)

    @classmethod
    def from_config(cls, config: ModelConfig, sim_config: SimConfig) -> "NumpyTransformerPolicy":
        return cls(
            feature_dim=config.feature_dim,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_speed=sim_config.max_speed,
            ridge_lambda=config.ridge_lambda,
            seed=config.seed,
        )

    def _encode_features(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64)
        masks = np.asarray(masks, dtype=bool)
        z = np.einsum("btf,fd->btd", observations, self.params["w_in"]) + self.params["b_in"]
        z = np.tanh(z)
        z = np.where(masks[:, :, None], z, 0.0)
        head_dim = self.d_model // self.num_heads

        for layer in range(self.num_layers):
            prefix = f"layer_{layer}_"
            q = np.einsum("btd,de->bte", z, self.params[prefix + "w_q"])
            k = np.einsum("btd,de->bte", z, self.params[prefix + "w_k"])
            v = np.einsum("btd,de->bte", z, self.params[prefix + "w_v"])
            q = q.reshape(q.shape[0], q.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(k.shape[0], k.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(v.shape[0], v.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
            scores = np.einsum("bhtd,bhsd->bhts", q, k) / np.sqrt(head_dim)
            scores = np.where(masks[:, None, None, :], scores, -1e9)
            attention = _softmax(scores)
            attended = np.einsum("bhts,bhsd->bhtd", attention, v)
            attended = attended.transpose(0, 2, 1, 3).reshape(z.shape[0], z.shape[1], self.d_model)
            attended = np.tanh(
                np.einsum("btd,de->bte", attended, self.params[prefix + "w_o"])
            )
            z = np.tanh(z + attended)
            ff = np.tanh(
                np.einsum("btd,de->bte", z, self.params[prefix + "w_ff1"])
                + self.params[prefix + "b_ff1"]
            )
            ff = (
                np.einsum("btd,de->bte", ff, self.params[prefix + "w_ff2"])
                + self.params[prefix + "b_ff2"]
            )
            z = np.tanh(z + ff)
            z = np.where(masks[:, :, None], z, 0.0)

        counts = np.maximum(np.sum(masks, axis=1, keepdims=True), 1)
        pooled = np.sum(z, axis=1) / counts
        self_token = z[:, 0, :]
        features = np.concatenate([self_token, pooled], axis=1)
        if self.include_raw_features:
            features = np.concatenate([features, observations[:, 0, :]], axis=1)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_batch(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        features = self._encode_features(observations, masks)
        predictions = np.einsum("bd,do->bo", features, self.params["w_out"]) + self.params["b_out"]
        return clip_by_norm(predictions, self.max_speed)

    def predict_joint(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        max_neighbors: int,
        sim_config: SimConfig,
        obstacle_map: GridMap | None = None,
        max_obstacle_tokens: int = 0,
        obstacle_context_range: float = 4.0,
        observation_version: str | None = None,
    ) -> np.ndarray:
        observations, masks = encode_joint_observation(
            positions,
            velocities,
            goals,
            radii,
            max_neighbors,
            sim_config,
            obstacle_map=obstacle_map,
            max_obstacle_tokens=max_obstacle_tokens,
            obstacle_context_range=obstacle_context_range,
            observation_version=observation_version or self.observation_version,
        )
        return self.predict_batch(observations, masks)

    def loss(self, observations: np.ndarray, masks: np.ndarray, targets: np.ndarray) -> float:
        predictions = self.predict_batch(observations, masks)
        return float(np.mean((predictions - targets) ** 2))

    def fit(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        config: ModelConfig,
        verbose: bool = False,
    ) -> Dict[str, object]:
        rng = np.random.default_rng(config.seed)
        n_samples = observations.shape[0]
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        val_count = int(round(n_samples * config.validation_split))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:] if val_count < n_samples else indices
        if len(train_indices) == 0:
            train_indices = indices

        feature_dim = self.output_feature_dim + 1
        xtx = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        xty = np.zeros((feature_dim, 2), dtype=np.float64)
        batch_size = max(1, int(config.batch_size))
        for start in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[start : start + batch_size]
            features = self._encode_features(observations[batch_idx], masks[batch_idx])
            design = np.concatenate(
                [features, np.ones((features.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            xtx += np.einsum("bi,bj->ij", design, design)
            xty += np.einsum("bi,bo->io", design, targets[batch_idx])

        regularizer = config.ridge_lambda * np.eye(feature_dim, dtype=np.float64)
        regularizer[-1, -1] = 0.0
        try:
            solution = np.linalg.solve(xtx + regularizer, xty)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(xtx + regularizer) @ xty
        self.params["w_out"] = solution[:-1]
        self.params["b_out"] = solution[-1]
        self._stabilize_output_head()

        train_loss = self._loss_on_indices(observations, masks, targets, train_indices, batch_size)
        val_loss = (
            self._loss_on_indices(observations, masks, targets, val_indices, batch_size)
            if val_count
            else train_loss
        )
        history = {
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "fit_method": "closed_form_ridge_output_head",
            "trained_parameters": ["w_out", "b_out"],
        }
        if verbose:
            print(
                "numpy_transformer_ridge "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )
        return history

    def _loss_on_indices(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
    ) -> float:
        if len(indices) == 0:
            return 0.0
        total = 0.0
        count = 0
        for start in range(0, len(indices), max(1, int(batch_size))):
            batch_idx = indices[start : start + batch_size]
            predictions = self.predict_batch(observations[batch_idx], masks[batch_idx])
            total += float(np.sum((predictions - targets[batch_idx]) ** 2))
            count += int(predictions.size)
        return float(total / max(count, 1))

    def _stabilize_output_head(self) -> None:
        for name in ("w_out", "b_out"):
            self.params[name] = np.clip(
                np.nan_to_num(self.params[name], nan=0.0, posinf=5.0, neginf=-5.0),
                -5.0,
                5.0,
            )

    def save(self, path: str | Path, metadata: Dict[str, object] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(self.params)
        payload = metadata or {}
        observation_version = normalize_observation_version(
            payload.get("observation_version", self.observation_version)
        )
        arrays["metadata"] = np.array(
            json.dumps(
                {
                    "feature_dim": self.feature_dim,
                    "d_model": self.d_model,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "max_speed": self.max_speed,
                    "ridge_lambda": self.ridge_lambda,
                    "include_raw_features": self.include_raw_features,
                    "policy_type": "numpy_transformer",
                    "observation_version": observation_version,
                    "observation_metadata": payload.get(
                        "observation_metadata",
                        self.observation_metadata,
                    ),
                    "metadata": payload,
                }
            )
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyTransformerPolicy":
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        model = cls(
            feature_dim=int(metadata["feature_dim"]),
            d_model=int(metadata["d_model"]),
            num_heads=int(metadata["num_heads"]),
            num_layers=int(metadata["num_layers"]),
            max_speed=float(metadata["max_speed"]),
            ridge_lambda=float(metadata.get("ridge_lambda", 1e-3)),
            include_raw_features=bool(metadata.get("include_raw_features", False)),
            seed=0,
        )
        model.observation_version = normalize_observation_version(
            metadata.get("observation_version")
            or metadata.get("metadata", {}).get("observation_version")
            or metadata.get("metadata", {}).get("dataset_config", {}).get("observation_version")
        )
        model.observation_metadata = dict(metadata.get("observation_metadata", {}))
        for name in model.params:
            model.params[name] = np.asarray(data[name], dtype=np.float64)
        return model


class TorchTransformerPolicy:
    """Trainable PyTorch transformer planner with the same rollout API.

    PyTorch remains an optional dependency. This class is selected explicitly
    with ``policy_type='torch_transformer'`` and is intended for Lambda/GPU
    training while preserving the existing observation, dataset, and shield
    evaluation pipeline.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        max_speed: float = 1.2,
        dropout: float = 0.0,
        device: str = "auto",
        seed: int = 0,
    ):
        if not _torch_available():
            raise ImportError(
                "PyTorch is required for policy_type='torch_transformer'. "
                "Install torch in the Lambda environment or use numpy_transformer."
            )
        import torch
        import torch.nn as nn

        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_speed = float(max_speed)
        self.dropout = float(dropout)
        self.observation_version = "legacy"
        self.observation_metadata: Dict[str, object] = {}
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        torch.manual_seed(int(seed))
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = str(device)
        self.device = torch.device(resolved)
        self.model = _TorchTransformerModule(
            feature_dim=self.feature_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_speed=self.max_speed,
            dropout=self.dropout,
        ).to(self.device)

    @classmethod
    def from_config(cls, config: ModelConfig, sim_config: SimConfig) -> "TorchTransformerPolicy":
        return cls(
            feature_dim=config.feature_dim,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_speed=sim_config.max_speed,
            dropout=config.dropout,
            device=config.torch_device,
            seed=config.seed,
        )

    def predict_batch(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            predictions = self.model(obs, mask)
        return predictions.detach().cpu().numpy().astype(np.float64)

    def predict_joint(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        max_neighbors: int,
        sim_config: SimConfig,
        obstacle_map: GridMap | None = None,
        max_obstacle_tokens: int = 0,
        obstacle_context_range: float = 4.0,
        observation_version: str | None = None,
    ) -> np.ndarray:
        observations, masks = encode_joint_observation(
            positions,
            velocities,
            goals,
            radii,
            max_neighbors,
            sim_config,
            obstacle_map=obstacle_map,
            max_obstacle_tokens=max_obstacle_tokens,
            obstacle_context_range=obstacle_context_range,
            observation_version=observation_version or self.observation_version,
        )
        return self.predict_batch(observations, masks)

    def loss(self, observations: np.ndarray, masks: np.ndarray, targets: np.ndarray) -> float:
        predictions = self.predict_batch(observations, masks)
        return float(np.mean((predictions - targets) ** 2))

    def fit(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        config: ModelConfig,
        verbose: bool = False,
    ) -> Dict[str, object]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        rng = np.random.default_rng(config.seed)
        n_samples = int(observations.shape[0])
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        val_count = int(round(n_samples * config.validation_split))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:] if val_count < n_samples else indices
        if len(train_indices) == 0:
            train_indices = indices

        obs = torch.as_tensor(observations, dtype=torch.float32)
        mask = torch.as_tensor(masks, dtype=torch.bool)
        target = torch.as_tensor(targets, dtype=torch.float32)
        train_ds = TensorDataset(obs[train_indices], mask[train_indices], target[train_indices])
        generator = torch.Generator()
        generator.manual_seed(int(config.seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=max(1, int(config.batch_size)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        history: Dict[str, object] = {
            "train_loss": [],
            "val_loss": [],
            "fit_method": "torch_transformer_adamw",
            "trained_parameters": ["all"],
            "device": str(self.device),
        }
        for epoch in range(int(config.epochs)):
            self.model.train()
            batch_losses = []
            for batch_obs, batch_mask, batch_target in train_loader:
                batch_obs = batch_obs.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_target = batch_target.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_obs, batch_mask)
                loss = torch.mean((pred - batch_target) ** 2)
                loss.backward()
                if config.grad_clip_norm and config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(config.grad_clip_norm),
                    )
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            val_loss = (
                self._loss_on_indices(obs, mask, target, val_indices, max(1, int(config.batch_size)))
                if val_count
                else train_loss
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(float(val_loss))
            if verbose:
                print(
                    f"torch_transformer epoch={epoch + 1:03d} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"device={self.device}"
                )
        return history

    def _loss_on_indices(self, obs, mask, target, indices: np.ndarray, batch_size: int) -> float:
        import torch

        if len(indices) == 0:
            return 0.0
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                pred = self.model(obs[batch_idx].to(self.device), mask[batch_idx].to(self.device))
                batch_target = target[batch_idx].to(self.device)
                total += float(torch.sum((pred - batch_target) ** 2).detach().cpu())
                count += int(pred.numel())
        return float(total / max(count, 1))

    def save(self, path: str | Path, metadata: Dict[str, object] | None = None) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = metadata or {}
        observation_version = normalize_observation_version(
            payload.get("observation_version", self.observation_version)
        )
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "metadata": {
                    "feature_dim": self.feature_dim,
                    "d_model": self.d_model,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "max_speed": self.max_speed,
                    "dropout": self.dropout,
                    "policy_type": "torch_transformer",
                    "observation_version": observation_version,
                    "observation_metadata": payload.get(
                        "observation_metadata",
                        self.observation_metadata,
                    ),
                    "metadata": payload,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TorchTransformerPolicy":
        import torch

        payload = torch.load(path, map_location="cpu")
        metadata = payload["metadata"]
        model = cls(
            feature_dim=int(metadata["feature_dim"]),
            d_model=int(metadata["d_model"]),
            num_heads=int(metadata["num_heads"]),
            num_layers=int(metadata["num_layers"]),
            max_speed=float(metadata["max_speed"]),
            dropout=float(metadata.get("dropout", 0.0)),
            device="cpu",
            seed=0,
        )
        model.model.load_state_dict(payload["state_dict"])
        model.observation_version = normalize_observation_version(
            metadata.get("observation_version")
            or metadata.get("metadata", {}).get("observation_version")
            or metadata.get("metadata", {}).get("dataset_config", {}).get("observation_version")
        )
        model.observation_metadata = dict(metadata.get("observation_metadata", {}))
        return model


class _TorchTransformerModule:
    def __new__(
        cls,
        feature_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_speed: float,
        dropout: float,
    ):
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.max_speed = float(max_speed)
                self.input = nn.Linear(int(feature_dim), int(d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=int(d_model),
                    nhead=int(num_heads),
                    dim_feedforward=int(d_model) * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
                self.head = nn.Sequential(
                    nn.LayerNorm(int(d_model) + int(feature_dim)),
                    nn.Linear(int(d_model) + int(feature_dim), int(d_model)),
                    nn.GELU(),
                    nn.Linear(int(d_model), 2),
                )

            def forward(self, observations, masks):
                x = self.input(observations)
                x = torch.where(masks[:, :, None], x, torch.zeros_like(x))
                encoded = self.encoder(x, src_key_padding_mask=~masks)
                self_token = encoded[:, 0, :]
                features = torch.cat([self_token, observations[:, 0, :]], dim=-1)
                return torch.tanh(self.head(features)) * self.max_speed

        return Module()


def make_policy(config: ModelConfig, sim_config: SimConfig):
    policy_type = config.policy_type.strip().lower().replace("-", "_")
    if policy_type in {"attention", "numpy_attention", "phase1_attention"}:
        return NumpyAttentionPolicy.from_config(config, sim_config)
    if policy_type in {"transformer", "numpy_transformer", "scaled_numpy_transformer"}:
        return NumpyTransformerPolicy.from_config(config, sim_config)
    if policy_type in {"torch_transformer", "pytorch_transformer"}:
        return TorchTransformerPolicy.from_config(config, sim_config)
    raise ValueError(
        "Unknown policy_type={!r}; expected 'numpy_attention', 'numpy_transformer', "
        "or 'torch_transformer'.".format(
            config.policy_type
        )
    )


def load_policy(path: str | Path):
    try:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        policy_type = metadata.get("policy_type", "numpy_attention")
    except Exception:
        if not _torch_available():
            raise ImportError(
                f"Could not load {path} as a NumPy policy and PyTorch is unavailable."
            )
        return TorchTransformerPolicy.load(path)
    if policy_type == "numpy_transformer":
        return NumpyTransformerPolicy.load(path)
    if policy_type == "torch_transformer":
        return TorchTransformerPolicy.load(path)
    return NumpyAttentionPolicy.load(path)


def policy_from_model(
    model: NumpyAttentionPolicy | NumpyTransformerPolicy | TorchTransformerPolicy,
    max_neighbors: int,
    sim_config: SimConfig,
    obstacle_map: GridMap | None = None,
    max_obstacle_tokens: int = 0,
    obstacle_context_range: float = 4.0,
    observation_version: str | None = None,
):
    def _policy(positions, velocities, goals, radii):
        return model.predict_joint(
            positions,
            velocities,
            goals,
            radii,
            max_neighbors,
            sim_config,
            obstacle_map=obstacle_map,
            max_obstacle_tokens=max_obstacle_tokens,
            obstacle_context_range=obstacle_context_range,
            observation_version=observation_version or getattr(model, "observation_version", "legacy"),
        )

    return _policy
