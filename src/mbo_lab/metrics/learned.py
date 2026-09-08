"""
Learned BookSpace history representations and latent-distance losses.

This module owns learned mappings

    X : [B, T, F] -> z : [B, D]

and losses defined directly on the resulting latent geometry.

It deliberately does NOT own:
- corpus/session loading,
- train/validation/test splits,
- pair traversal,
- X/Y/D construction,
- preprocessing,
- optimizers or training loops,
- checkpoints,
- KNN retrieval,
- compression.

All encoder families share the same public contract so experiment code can swap
architectures without changing downstream retrieval or compression code.

Only the linear history encoder required by the current experiment is present.
Future architectures belong here once their exact contracts are specified and
tested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderSpec:
    """
    Architecture-independent encoder configuration.

    Only `architecture` and `latent_dim` are universally required. Remaining
    fields are common knobs reserved for architecture families that need them.

    Input dimensions T and F are supplied separately by the data contract.
    """

    architecture: str
    latent_dim: int
    model_dim: int | None = None
    layers: int | None = None
    heads: int | None = None
    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"

    def __post_init__(self) -> None:
        architecture = self.architecture.strip().lower()
        if not architecture:
            raise ValueError("Encoder architecture must be nonempty")
        object.__setattr__(self, "architecture", architecture)

        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.model_dim is not None and self.model_dim < 1:
            raise ValueError("model_dim must be positive when supplied")
        if self.layers is not None and self.layers < 1:
            raise ValueError("layers must be positive when supplied")
        if self.heads is not None and self.heads < 1:
            raise ValueError("heads must be positive when supplied")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not self.activation.strip():
            raise ValueError("activation must be nonempty")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class HistoryEncoder(nn.Module):
    """
    Base contract for every learned BookSpace history encoder.

    Input:
        histories : [B, T, F]

    Output:
        z : [B, latent_dim]
    """

    def __init__(
        self,
        *,
        history: int,
        features: int,
        latent_dim: int,
        spec: EncoderSpec,
    ) -> None:
        super().__init__()

        if history < 1:
            raise ValueError("history must be positive")
        if features < 1:
            raise ValueError("features must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")

        self.history = int(history)
        self.features = int(features)
        self.latent_dim = int(latent_dim)
        self.spec = spec

    @property
    def input_dim(self) -> int:
        return self.history * self.features

    def _check_history_shape(self, histories: Tensor) -> None:
        if histories.ndim != 3:
            raise ValueError(
                "History encoder input must have shape [B, T, F]; "
                f"received {tuple(histories.shape)}"
            )

        if histories.shape[1] != self.history or histories.shape[2] != self.features:
            raise ValueError(
                "History shape does not match encoder contract: "
                f"expected [B, {self.history}, {self.features}], "
                f"received {tuple(histories.shape)}"
            )

    def encode(self, histories: Tensor) -> Tensor:
        return self(histories)

    def config_dict(self) -> dict[str, object]:
        return {
            "history": self.history,
            "features": self.features,
            "latent_dim": self.latent_dim,
            "spec": self.spec.as_dict(),
        }


class LinearHistoryEncoder(HistoryEncoder):
    """
    Learned linear projection of the full transformed history.

        X -> vec(X) -> W vec(X) + b -> z
    """

    def __init__(
        self,
        *,
        history: int,
        features: int,
        spec: EncoderSpec,
    ) -> None:
        super().__init__(
            history=history,
            features=features,
            latent_dim=spec.latent_dim,
            spec=spec,
        )

        self.projection = nn.Linear(
            self.input_dim,
            self.latent_dim,
            bias=spec.bias,
        )

    def forward(self, histories: Tensor) -> Tensor:
        self._check_history_shape(histories)
        flattened = histories.reshape(histories.shape[0], self.input_dim)
        return self.projection(flattened)


def build_encoder(
    spec: EncoderSpec,
    *,
    history: int,
    features: int,
) -> HistoryEncoder:
    """Construct an implemented history encoder from a shared config."""
    if spec.architecture != "linear":
        raise ValueError(
            f"Unsupported encoder architecture {spec.architecture!r}; available: linear"
        )
    return LinearHistoryEncoder(history=history, features=features, spec=spec)


def _check_latent_pair(left_z: Tensor, right_z: Tensor) -> None:
    if left_z.shape != right_z.shape:
        raise ValueError(
            "Latent representation shapes must match: "
            f"{tuple(left_z.shape)} != {tuple(right_z.shape)}"
        )

    if left_z.ndim < 1 or left_z.shape[-1] == 0:
        raise ValueError("Latent representations require a nonempty final dimension")


def latent_l2(left_z: Tensor, right_z: Tensor) -> Tensor:
    """
    Euclidean distance in learned representation space.

        d_theta(i,j) = ||z_i - z_j||_2
    """

    _check_latent_pair(left_z, right_z)
    return torch.linalg.vector_norm(left_z - right_z, ord=2, dim=-1)


def distance_mse(
    left_z: Tensor,
    right_z: Tensor,
    target_distance: Tensor,
) -> Tensor:
    """
    Initial BookSpace metric-learning objective.

        mean((||z_i-z_j||_2 - D_ij)**2)
    """

    predicted = latent_l2(left_z, right_z)

    if predicted.shape != target_distance.shape:
        raise ValueError(
            "Target-distance shape must match predicted distances: "
            f"{tuple(target_distance.shape)} != {tuple(predicted.shape)}"
        )

    return F.mse_loss(predicted, target_distance)
