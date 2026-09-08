"""
Fixed, non-learned BookSpace representations and target distances.

This module owns deterministic geometry only.

History representation
----------------------
The initial fixed baseline maps a transformed history directly to a flat vector:

    X : [..., T, F]
    z = vec(X) : [..., T * F]

No parameters are fitted and no information is discarded. This gives the fixed
baseline the same downstream contract as a learned history encoder:

    X -> z

so retrieval and compression code can operate on either representation without
special cases.

Future target geometry
----------------------
For cumulative future paths Y, the canonical target distance is:

    D_raw(i, j) = sqrt(mean_h((Y_i[h] - Y_j[h])**2))

A single positive scale fitted on training pairs gives:

    D(i, j) = D_raw(i, j) / s_D

Pair selection, split eligibility, corpus traversal, preprocessing and sampling
belong elsewhere.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Shared array validation
# ---------------------------------------------------------------------------


def _finite_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    return array


def eligible_pairs(anchors, history=256, horizon=128):
    """Compatibility wrapper; canonical implementation lives in samples."""
    from mbo_lab.samples import eligible_pairs as _eligible_pairs

    return _eligible_pairs(anchors, history, horizon)


# ---------------------------------------------------------------------------
# Fixed history representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlattenHistory:
    """
    Parameter-free history encoder.

    Maps:

        [..., T, F] -> [..., T * F]

    The input is expected to have already passed through the frozen
    train-fitted preprocessing pipeline. This class performs no normalization
    and fits no statistics.

    With the default BookSpace contract:

        [B, 256, 86] -> [B, 22016]
    """

    name: str = "flatten_history"

    def encode(self, histories) -> np.ndarray:
        histories = _finite_array(histories, name="histories")

        if histories.ndim < 2:
            raise ValueError(
                "Histories must have at least [time, feature] dimensions"
            )

        time_steps = histories.shape[-2]
        features = histories.shape[-1]

        if time_steps < 1 or features < 1:
            raise ValueError("History dimensions must be nonempty")

        return histories.reshape(*histories.shape[:-2], time_steps * features)

    def __call__(self, histories) -> np.ndarray:
        return self.encode(histories)


# ---------------------------------------------------------------------------
# Distances in representation space
# ---------------------------------------------------------------------------


def latent_l2(left, right) -> np.ndarray:
    """
    Euclidean distance between representations z.

    Inputs must have matching shape:

        left, right : [..., Z]

    Returns:

        distance : [...]

    This works for both the flattened fixed representation and, later, learned
    latent vectors converted to NumPy for evaluation.
    """

    left = _finite_array(left, name="left representations")
    right = _finite_array(right, name="right representations")

    if left.shape != right.shape:
        raise ValueError("Representation shapes must match")

    if left.ndim < 1 or left.shape[-1] == 0:
        raise ValueError("Representations must have a nonempty latent dimension")

    diff = left - right
    return np.sqrt(np.sum(diff * diff, axis=-1))


def fixed_history_distance(left_history, right_history) -> np.ndarray:
    """
    Convenience function for the initial fixed BookSpace metric:

        X_i -> vec(X_i)
        X_j -> vec(X_j)
        d(i,j) = ||vec(X_i) - vec(X_j)||_2

    Histories must already be transformed using training-fitted preprocessing.
    """

    encoder = FlattenHistory()

    left_z = encoder.encode(left_history)
    right_z = encoder.encode(right_history)

    return latent_l2(left_z, right_z)


# ---------------------------------------------------------------------------
# Future target geometry
# ---------------------------------------------------------------------------


def future_path_rms(left, right) -> np.ndarray:
    """
    RMS distance between cumulative future-return paths.

    Inputs:

        left, right : [..., H]

    Returns:

        D_raw : [...]

    Under the default contract H = 128.
    """

    left = _finite_array(left, name="left future paths")
    right = _finite_array(right, name="right future paths")

    if left.shape != right.shape:
        raise ValueError("Future-path shapes must match")

    if left.ndim < 1 or left.shape[-1] == 0:
        raise ValueError("Future paths must have a nonempty horizon")

    diff = left - right
    return np.sqrt(np.mean(diff * diff, axis=-1))


# ---------------------------------------------------------------------------
# Training-fitted target scale
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetScale:
    """
    Global scale for future-path target distances.

    Fit only from selected eligible training-pair D_raw values.

        s_D = median(D_raw[D_raw > 0])
        D   = D_raw / s_D

    `pair_count`, `zero_fraction` and `seed` are retained for provenance and
    reproducibility.
    """

    value: float
    pair_count: int
    zero_fraction: float
    seed: int

    @classmethod
    def fit(cls, training_distances, seed: int = 0) -> "TargetScale":
        values = _finite_array(
            training_distances,
            name="training target distances",
        ).astype(np.float64, copy=False)

        if values.ndim != 1 or len(values) == 0:
            raise ValueError(
                "Need a nonempty one-dimensional sample of training distances"
            )

        if np.any(values < 0):
            raise ValueError("Training target distances must be nonnegative")

        positive = values[values > 0]

        if len(positive) == 0:
            raise ValueError(
                "Training target distances are all zero; target scale is undefined"
            )

        scale = float(np.median(positive))

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Target scale must be finite and strictly positive")

        return cls(
            value=scale,
            pair_count=int(len(values)),
            zero_fraction=float(np.mean(values == 0)),
            seed=int(seed),
        )

    def transform(self, distances) -> np.ndarray:
        values = _finite_array(
            distances,
            name="target distances",
        ).astype(np.float64, copy=False)

        if np.any(values < 0):
            raise ValueError("Target distances must be nonnegative")

        return values / self.value

    def inverse_transform(self, distances) -> np.ndarray:
        """
        Recover raw future-distance units from scaled distances.
        """

        values = _finite_array(
            distances,
            name="scaled target distances",
        ).astype(np.float64, copy=False)

        if np.any(values < 0):
            raise ValueError("Scaled target distances must be nonnegative")

        return values * self.value


# ---------------------------------------------------------------------------
# Convenience reference helper
# ---------------------------------------------------------------------------


def future_pair_distances(
    futures,
    pairs,
    *,
    scale: TargetScale | None = None,
) -> np.ndarray:
    """
    Compute future-path distances for already-selected pairs.

    This function does NOT decide which pairs are eligible.

    Parameters
    ----------
    futures
        Array [N, H] containing cumulative future paths Y.

    pairs
        Integer array [P, 2] containing indices into `futures`.

    scale
        Optional frozen training TargetScale. If supplied, returns scaled D;
        otherwise returns D_raw.

    Returns
    -------
    distances
        Array [P].
    """

    futures = _finite_array(futures, name="future paths")

    if futures.ndim != 2 or futures.shape[1] == 0:
        raise ValueError("Futures must have shape [N, H] with H > 0")

    pairs = np.asarray(pairs, dtype=np.int64)

    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("Pairs must have nonempty shape [P, 2]")

    if np.any(pairs < 0) or np.any(pairs >= len(futures)):
        raise ValueError("Pair index out of range")

    raw = future_path_rms(
        futures[pairs[:, 0]],
        futures[pairs[:, 1]],
    )

    if scale is None:
        return raw

    return scale.transform(raw)


# Transitional names retained for the existing pipeline/tests. New code should
# use future_path_rms and future_pair_distances explicitly.
rms_distance = future_path_rms
pair_targets = future_pair_distances
