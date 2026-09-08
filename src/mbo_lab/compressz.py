"""Offline, streaming latent vector quantisation for BookSpace.

This module deliberately knows only about latent vectors ``z`` with shape
``[B, D]``.  It does not know how a latent was produced, what history or future
belongs to it, or how compressed states are later retrieved.

The initial compressor is classical vector quantisation with a codebook learned
by :class:`sklearn.cluster.MiniBatchKMeans`.

Two-pass protocol
-----------------
The encoder/representation is frozen before this module is used.

Pass 1 streams *training-split only* latents through ``partial_fit``.  The
compressor retains only bounded fitting state: a small bootstrap buffer, at most
one internal mini-batch, and the global MiniBatchKMeans codebook state.  It never
retains the latent corpus.

After ``finalize()``, the codebook is copied out and frozen.  The sklearn fitting
object is discarded, so centroids cannot subsequently adapt.

Pass 2 replays the corpus and calls ``encode`` (or ``quantize``) against that
fixed codebook.  Codes produced before finalization would refer to moving
centroids and are therefore intentionally not exposed.

Although MiniBatchKMeans consumes the frozen training corpus incrementally, this
is an *offline* prototype representation, not an online/adaptive compressor.
Once finalized, validation and test data can only be assigned to the fixed
training codebook.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

try:
    import sklearn
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin_min
except ImportError as exc:  # pragma: no cover - exercised only in broken environments.
    raise ImportError(
        "BookSpace latent compression requires scikit-learn. "
        "Install it in the project environment (for example: `uv add scikit-learn`)."
    ) from exc


_METHOD = "minibatch_kmeans"
_SERIALIZATION_VERSION = 1
_DEFAULT_STATS_RESERVOIR = 65_536

Float32Matrix = NDArray[np.float32]
Int32Vector = NDArray[np.int32]
Int64Vector = NDArray[np.int64]


class CompressionError(ValueError):
    """Invalid compression input, configuration, or artifact."""


class CompressionStateError(RuntimeError):
    """Operation is invalid for the compressor's current lifecycle state."""


def _plain_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_positive_int(name: str, value: object) -> int:
    if not _plain_int(value) or int(value) <= 0:
        raise CompressionError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _to_numpy(value: object, *, latent_float32: bool = False) -> NDArray[np.generic]:
    """Detach a Torch tensor if necessary, otherwise make a NumPy view/array.

    Torch is an optional dependency here.  CUDA inputs are transferred to CPU
    one batch at a time and no tensor/computation graph is retained.  Latent
    tensors are cast to float32; integer code tensors preserve their dtype.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - depends on caller environment.
        pass
    else:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if latent_float32:
                tensor = tensor.to(device="cpu", dtype=torch.float32)
            elif tensor.device.type != "cpu":
                tensor = tensor.to(device="cpu")
            return tensor.contiguous().numpy()

    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CompressionError("input could not be converted to a NumPy array") from exc


def _as_latent_batch(value: object, latent_dim: int) -> Float32Matrix:
    array = _to_numpy(value, latent_float32=True)
    if array.ndim != 2:
        raise CompressionError(
            f"z_batch must have shape [B, D]; received array with shape {array.shape}"
        )
    if array.shape[0] == 0:
        raise CompressionError("z_batch must contain at least one observation")
    if array.shape[1] != latent_dim:
        raise CompressionError(
            f"z_batch latent dimension is {array.shape[1]}, expected {latent_dim}"
        )
    if array.dtype.kind not in "biuf":
        raise CompressionError(f"z_batch must be numeric, got dtype {array.dtype}")

    try:
        result = np.asarray(array, dtype=np.float32, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompressionError("z_batch cannot be represented as float32") from exc

    if not np.isfinite(result).all():
        raise CompressionError("z_batch contains NaN or infinite values")
    return result


def _as_codes(value: object, n_prototypes: int) -> Int32Vector:
    array = _to_numpy(value)
    if array.ndim != 1:
        raise CompressionError(f"codes must have shape [B]; received shape {array.shape}")
    if array.size == 0:
        raise CompressionError("codes must contain at least one prototype ID")
    if array.dtype.kind not in "iu":
        raise CompressionError(f"codes must have an integer dtype, got {array.dtype}")

    codes64 = np.asarray(array, dtype=np.int64, order="C")
    if np.any(codes64 < 0) or np.any(codes64 >= n_prototypes):
        lo = int(codes64.min())
        hi = int(codes64.max())
        raise CompressionError(
            f"prototype IDs must lie in [0, {n_prototypes}); observed range [{lo}, {hi}]"
        )
    return codes64.astype(np.int32, copy=False)


def _readonly_float32_view(array: Float32Matrix) -> Float32Matrix:
    view = array.view()
    view.setflags(write=False)
    return view


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class CompressionSpec:
    """Configuration for the initial BookSpace latent compressor.

    ``batch_size`` is the compressor's *internal* MiniBatchKMeans update size.
    Calls to :meth:`MiniBatchKMeansCompressor.partial_fit` may contain any
    positive number of rows.  The compressor stages at most one internal batch,
    making the scientific update stream independent of external loader batch
    boundaries.

    ``n_init`` means the number of deterministic MiniBatchKMeans bootstrap
    candidates tried on a bounded initial training sample.  This explicit
    streaming treatment is used because sklearn's ``partial_fit`` path does not
    itself perform the estimator's multi-init ``fit`` loop.
    """

    method: str = _METHOD
    n_prototypes: int = 1024
    batch_size: int = 8192
    init_size: int | None = None
    n_init: int = 3
    reassignment_ratio: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method != _METHOD:
            raise CompressionError(
                f"unsupported compression method {self.method!r}; expected {_METHOD!r}"
            )

        n_prototypes = _validate_positive_int("n_prototypes", self.n_prototypes)
        batch_size = _validate_positive_int("batch_size", self.batch_size)
        _validate_positive_int("n_init", self.n_init)

        if n_prototypes > np.iinfo(np.int32).max:
            raise CompressionError("n_prototypes exceeds the supported int32 code range")
        if batch_size > np.iinfo(np.int32).max:
            raise CompressionError("batch_size is unreasonably large")

        if self.init_size is not None:
            init_size = _validate_positive_int("init_size", self.init_size)
            if init_size < n_prototypes:
                raise CompressionError(
                    "init_size must be at least n_prototypes so every codeword can be initialized"
                )

        if isinstance(self.reassignment_ratio, (bool, np.bool_)) or not isinstance(
            self.reassignment_ratio, (int, float, np.integer, np.floating)
        ):
            raise CompressionError("reassignment_ratio must be a finite number in [0, 1]")
        ratio = float(self.reassignment_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise CompressionError("reassignment_ratio must be a finite number in [0, 1]")

        if not _plain_int(self.seed):
            raise CompressionError("seed must be an integer")
        if not 0 <= int(self.seed) <= np.iinfo(np.uint32).max:
            raise CompressionError(
                f"seed must lie in [0, {np.iinfo(np.uint32).max}] for reproducible sklearn RNG"
            )


@dataclass(frozen=True, slots=True)
class CompressionStats:
    """Snapshot of streaming diagnostics from a frozen assignment pass."""

    observations_assigned: int
    active_prototype_count: int
    prototype_occupancy_counts: Int64Vector
    mean_squared_quantization_error: float
    mean_euclidean_distortion: float
    occupancy_entropy_bits: float
    effective_state_count: float
    p50_nearest_prototype_distance: float
    p90_nearest_prototype_distance: float
    p95_nearest_prototype_distance: float
    p99_nearest_prototype_distance: float
    max_nearest_prototype_distance: float
    percentile_sample_size: int


class CompressionStatsAccumulator:
    """Bounded-memory assignment diagnostics for one fixed codebook.

    Percentiles are estimated using random-priority reservoir sampling.  Each
    observed nearest-prototype distance receives an independent random priority;
    only the ``reservoir_size`` smallest priorities are retained.  Therefore
    memory is O(N + reservoir_size), not O(number of assigned observations).

    Exact streaming quantities are retained for occupancy counts, observation
    count, mean squared quantisation error, mean Euclidean distortion, and max.
    """

    def __init__(
        self,
        prototypes: Float32Matrix,
        *,
        reservoir_size: int = _DEFAULT_STATS_RESERVOIR,
        seed: int = 0,
    ) -> None:
        self._n_prototypes = int(prototypes.shape[0])
        self._latent_dim = int(prototypes.shape[1])
        self._prototypes = prototypes
        self._reservoir_size = _validate_positive_int("reservoir_size", reservoir_size)
        if not _plain_int(seed) or not 0 <= int(seed) <= np.iinfo(np.uint64).max:
            raise CompressionError("stats seed must be a non-negative integer")

        self._rng = np.random.default_rng(int(seed))
        self._occupancy = np.zeros(self._n_prototypes, dtype=np.int64)
        self._observations = 0
        self._sum_squared_distance = 0.0
        self._sum_distance = 0.0
        self._max_distance = 0.0
        self._sample_keys = np.empty(0, dtype=np.float64)
        self._sample_distances = np.empty(0, dtype=np.float32)

    @property
    def observations_assigned(self) -> int:
        return self._observations

    @property
    def reservoir_size(self) -> int:
        return self._reservoir_size

    def update(self, z_batch: object, codes: object) -> Self:
        """Accumulate diagnostics for already-frozen assignments.

        ``codes`` should normally be the result of ``compressor.encode(z_batch)``.
        Nearest-centroid search is intentionally not repeated here; distortion is
        measured from each supplied vector to its selected codeword.
        """

        z = _as_latent_batch(z_batch, self._latent_dim)
        code_ids = _as_codes(codes, self._n_prototypes)
        if code_ids.shape[0] != z.shape[0]:
            raise CompressionError(
                f"codes length {code_ids.shape[0]} does not match z_batch size {z.shape[0]}"
            )

        selected = self._prototypes[code_ids]
        diff = z - selected
        squared = np.sum(diff * diff, axis=1, dtype=np.float64)
        # Rounding in float32 subtraction cannot make this materially negative,
        # but maximum protects sqrt from a possible -0.0/tiny negative artifact.
        np.maximum(squared, 0.0, out=squared)
        distances = np.sqrt(squared)

        unique_codes, counts = np.unique(code_ids, return_counts=True)
        self._occupancy[unique_codes] += counts.astype(np.int64, copy=False)

        self._observations += int(z.shape[0])
        self._sum_squared_distance += float(np.sum(squared, dtype=np.float64))
        self._sum_distance += float(np.sum(distances, dtype=np.float64))
        self._max_distance = max(self._max_distance, float(np.max(distances)))

        self._update_reservoir(distances)
        return self

    def _update_reservoir(self, distances: NDArray[np.float64]) -> None:
        keys = self._rng.random(distances.shape[0])
        new_distances = distances.astype(np.float32, copy=False)

        all_keys = np.concatenate((self._sample_keys, keys))
        all_distances = np.concatenate((self._sample_distances, new_distances))
        if all_keys.size > self._reservoir_size:
            keep = np.argpartition(all_keys, self._reservoir_size - 1)[: self._reservoir_size]
            all_keys = all_keys[keep]
            all_distances = all_distances[keep]

        self._sample_keys = all_keys
        self._sample_distances = all_distances

    def summary(self) -> CompressionStats:
        if self._observations == 0:
            raise CompressionStateError("cannot summarize assignment statistics before update()")

        nonzero = self._occupancy[self._occupancy > 0].astype(np.float64)
        probabilities = nonzero / float(self._observations)
        entropy = float(-np.sum(probabilities * np.log2(probabilities), dtype=np.float64))
        effective = float(2.0**entropy)

        p50, p90, p95, p99 = np.percentile(
            self._sample_distances.astype(np.float64, copy=False), [50, 90, 95, 99]
        )
        occupancy = self._occupancy.copy()
        occupancy.setflags(write=False)

        return CompressionStats(
            observations_assigned=self._observations,
            active_prototype_count=int(np.count_nonzero(self._occupancy)),
            prototype_occupancy_counts=occupancy,
            mean_squared_quantization_error=self._sum_squared_distance / self._observations,
            mean_euclidean_distortion=self._sum_distance / self._observations,
            occupancy_entropy_bits=entropy,
            effective_state_count=effective,
            p50_nearest_prototype_distance=float(p50),
            p90_nearest_prototype_distance=float(p90),
            p95_nearest_prototype_distance=float(p95),
            p99_nearest_prototype_distance=float(p99),
            max_nearest_prototype_distance=self._max_distance,
            percentile_sample_size=int(self._sample_distances.size),
        )


class MiniBatchKMeansCompressor:
    """Streaming offline VQ codebook learned with MiniBatchKMeans."""

    def __init__(self, spec: CompressionSpec, latent_dim: int) -> None:
        if not isinstance(spec, CompressionSpec):
            raise CompressionError("spec must be a CompressionSpec")
        self.spec = spec
        self.latent_dim = _validate_positive_int("latent_dim", latent_dim)

        self._fit_observations = 0
        self._initialized = False
        self._finalized = False
        self._model: MiniBatchKMeans | None = None
        self._prototypes: Float32Matrix | None = None

        # Fixed-size warm-up and update staging.  Allocated lazily so merely
        # constructing a compressor does not reserve latent-sized buffers.
        self._warmup_target = self._resolve_warmup_target()
        self._warmup: Float32Matrix | None = None
        self._warmup_rows = 0
        self._pending: Float32Matrix | None = None
        self._pending_rows = 0

    def _resolve_warmup_target(self) -> int:
        if self.spec.init_size is not None:
            return int(self.spec.init_size)
        # Mirrors sklearn's MiniBatchKMeans heuristic when enough data exist:
        # roughly three internal batches, but never less than 3 * n_clusters.
        return max(3 * int(self.spec.batch_size), 3 * int(self.spec.n_prototypes))

    @property
    def is_fitted(self) -> bool:
        """Whether MiniBatchKMeans has initialized a global codebook."""

        return self._initialized

    @property
    def is_finalized(self) -> bool:
        """Whether the codebook is frozen and encoding is permitted."""

        return self._finalized

    @property
    def fit_observation_count(self) -> int:
        """Number of training latents accepted during pass 1."""

        return self._fit_observations

    @property
    def n_prototypes(self) -> int:
        return int(self.spec.n_prototypes)

    @property
    def prototypes(self) -> Float32Matrix:
        """Final codebook with shape ``[N, D]`` as a read-only float32 view."""

        self._require_finalized()
        assert self._prototypes is not None
        return _readonly_float32_view(self._prototypes)

    def partial_fit(self, z_batch: object) -> Self:
        """Consume one arbitrary-size training latent batch.

        External batch size may vary.  Internally, post-bootstrap updates are
        aligned to ``spec.batch_size`` across calls, so changing loader batch
        boundaries does not change the sequence of MiniBatchKMeans updates for
        the same ordered stream.

        No prototype IDs are returned: assignments are scientifically valid only
        after ``finalize()`` freezes the codebook.
        """

        if self._finalized:
            raise CompressionStateError("cannot fit a compressor after finalize()")

        z = _as_latent_batch(z_batch, self.latent_dim)
        start = 0

        if not self._initialized:
            start = self._consume_warmup(z)
            if self._warmup_rows == self._warmup_target:
                self._initialize_from_warmup()

        if self._initialized and start < z.shape[0]:
            self._consume_updates(z[start:])

        self._fit_observations += int(z.shape[0])
        return self

    def _consume_warmup(self, z: Float32Matrix) -> int:
        if self._warmup is None:
            self._warmup = np.empty(
                (self._warmup_target, self.latent_dim), dtype=np.float32, order="C"
            )

        take = min(self._warmup_target - self._warmup_rows, z.shape[0])
        if take:
            self._warmup[self._warmup_rows : self._warmup_rows + take] = z[:take]
            self._warmup_rows += int(take)
        return int(take)

    def _new_candidate(self, random_state: int) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=self.n_prototypes,
            init="k-means++",
            batch_size=int(self.spec.batch_size),
            init_size=self.spec.init_size,
            n_init=1,
            reassignment_ratio=float(self.spec.reassignment_ratio),
            random_state=random_state,
            compute_labels=False,
        )

    def _initialize_from_warmup(self) -> None:
        if self._warmup is None or self._warmup_rows < self.n_prototypes:
            raise CompressionStateError(
                f"need at least {self.n_prototypes} training vectors to initialize the codebook; "
                f"have {self._warmup_rows}"
            )

        data = np.ascontiguousarray(self._warmup[: self._warmup_rows], dtype=np.float32)
        seed_sequence = np.random.SeedSequence(int(self.spec.seed))
        candidate_seeds = seed_sequence.generate_state(int(self.spec.n_init), dtype=np.uint32)

        best_model: MiniBatchKMeans | None = None
        best_sse = math.inf
        for raw_seed in candidate_seeds:
            candidate = self._new_candidate(int(raw_seed))
            candidate.partial_fit(data)
            _, distances = pairwise_distances_argmin_min(
                data, candidate.cluster_centers_, metric="euclidean"
            )
            sse = float(np.dot(distances.astype(np.float64), distances.astype(np.float64)))
            if sse < best_sse:
                best_sse = sse
                best_model = candidate

        assert best_model is not None
        self._model = best_model
        self._initialized = True
        self._warmup = None
        self._warmup_rows = 0
        self._pending = np.empty(
            (int(self.spec.batch_size), self.latent_dim), dtype=np.float32, order="C"
        )
        self._pending_rows = 0

    def _consume_updates(self, z: Float32Matrix) -> None:
        assert self._model is not None
        assert self._pending is not None

        batch_size = int(self.spec.batch_size)
        cursor = 0
        total = z.shape[0]

        # First complete an update that was split across external calls.
        if self._pending_rows:
            needed = batch_size - self._pending_rows
            take = min(needed, total)
            self._pending[self._pending_rows : self._pending_rows + take] = z[:take]
            self._pending_rows += int(take)
            cursor += int(take)
            if self._pending_rows == batch_size:
                self._model.partial_fit(self._pending)
                self._pending_rows = 0

        # Full aligned chunks can be passed directly without an extra copy.
        while cursor + batch_size <= total:
            self._model.partial_fit(z[cursor : cursor + batch_size])
            cursor += batch_size

        # Retain only the final incomplete internal mini-batch.
        remainder = total - cursor
        if remainder:
            self._pending[:remainder] = z[cursor:]
            self._pending_rows = int(remainder)

    def finalize(self) -> Self:
        """Freeze the training codebook permanently.

        If the total training corpus is smaller than the configured bootstrap
        target, finalization initializes from all available training vectors as
        long as there are at least ``n_prototypes`` of them.
        """

        if self._finalized:
            raise CompressionStateError("compressor is already finalized")
        if self._fit_observations == 0:
            raise CompressionStateError("cannot finalize before any training vectors are fitted")

        if not self._initialized:
            if self._warmup_rows < self.n_prototypes:
                raise CompressionStateError(
                    f"cannot finalize {self.n_prototypes} prototypes from only "
                    f"{self._warmup_rows} training vectors"
                )
            self._initialize_from_warmup()

        assert self._model is not None
        if self._pending_rows:
            assert self._pending is not None
            self._model.partial_fit(self._pending[: self._pending_rows])
            self._pending_rows = 0

        centers = np.asarray(self._model.cluster_centers_, dtype=np.float32, order="C").copy()
        expected_shape = (self.n_prototypes, self.latent_dim)
        if centers.shape != expected_shape:
            raise CompressionStateError(
                f"sklearn returned prototype shape {centers.shape}, expected {expected_shape}"
            )
        if not np.isfinite(centers).all():
            raise CompressionStateError("final MiniBatchKMeans centroids contain NaN or Inf")

        centers.setflags(write=False)
        self._prototypes = centers
        self._finalized = True

        # This is the hard offline/frozen boundary.  Discard every object that
        # can update centroids or retain fitting batches.
        self._model = None
        self._warmup = None
        self._pending = None
        self._warmup_rows = 0
        self._pending_rows = 0
        return self

    def _require_finalized(self) -> None:
        if not self._finalized or self._prototypes is None:
            raise CompressionStateError(
                "compressor must be finalized before encode/decode/quantize"
            )

    def _nearest(self, z: Float32Matrix) -> tuple[Int32Vector, NDArray[np.float32]]:
        self._require_finalized()
        assert self._prototypes is not None
        codes, distances = pairwise_distances_argmin_min(z, self._prototypes, metric="euclidean")
        return (
            np.asarray(codes, dtype=np.int32),
            np.asarray(distances, dtype=np.float32),
        )

    def encode(self, z_batch: object) -> Int32Vector:
        """Return nearest fixed prototype IDs with shape ``[B]``."""

        self._require_finalized()
        z = _as_latent_batch(z_batch, self.latent_dim)
        codes, _ = self._nearest(z)
        return codes

    def decode(self, codes: object) -> Float32Matrix:
        """Reconstruct prototype vectors for integer IDs, shape ``[B, D]``."""

        self._require_finalized()
        assert self._prototypes is not None
        code_ids = _as_codes(codes, self.n_prototypes)
        return np.asarray(self._prototypes[code_ids], dtype=np.float32, order="C")

    def quantize(self, z_batch: object) -> tuple[Int32Vector, Float32Matrix]:
        """Return ``(codes, selected_prototypes)`` without a duplicate search."""

        self._require_finalized()
        assert self._prototypes is not None
        z = _as_latent_batch(z_batch, self.latent_dim)
        codes, _ = self._nearest(z)
        reconstructed = np.asarray(self._prototypes[codes], dtype=np.float32, order="C")
        return codes, reconstructed

    def make_stats_accumulator(
        self,
        *,
        reservoir_size: int = _DEFAULT_STATS_RESERVOIR,
        seed: int | None = None,
    ) -> CompressionStatsAccumulator:
        """Create diagnostics bound to this frozen codebook."""

        self._require_finalized()
        assert self._prototypes is not None
        stats_seed = int(self.spec.seed) if seed is None else seed
        return CompressionStatsAccumulator(
            self._prototypes,
            reservoir_size=reservoir_size,
            seed=stats_seed,
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Serialize a finalized compressor without encoder weights or latents."""

        self._require_finalized()
        assert self._prototypes is not None

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "serialization_version": _SERIALIZATION_VERSION,
            "spec": asdict(self.spec),
            "latent_dim": self.latent_dim,
            "fitted": self.is_fitted,
            "finalized": self.is_finalized,
            "fit_observation_count": self.fit_observation_count,
            "versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "mbo_lab": _package_version("mbo-lab"),
            },
        }

        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=destination.name + ".", delete=False
            ) as stream:
                temp_name = stream.name
                np.savez_compressed(
                    stream,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    prototypes=self._prototypes,
                )
            os.replace(temp_name, destination)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)

        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """Load a finalized compressor artifact."""

        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as artifact:
                if set(artifact.files) != {"metadata", "prototypes"}:
                    raise CompressionError(
                        f"unexpected compressor artifact fields: {sorted(artifact.files)}"
                    )
                raw_metadata = artifact["metadata"]
                if raw_metadata.ndim != 0:
                    raise CompressionError("compressor metadata must be a scalar JSON string")
                metadata = json.loads(str(raw_metadata.item()))
                prototypes = np.asarray(artifact["prototypes"], dtype=np.float32, order="C").copy()
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            if isinstance(exc, CompressionError):
                raise
            raise CompressionError(f"failed to load compressor artifact {source}") from exc

        if metadata.get("serialization_version") != _SERIALIZATION_VERSION:
            raise CompressionError(
                "unsupported compressor serialization version "
                f"{metadata.get('serialization_version')!r}"
            )
        if metadata.get("fitted") is not True or metadata.get("finalized") is not True:
            raise CompressionError("serialized compressor must be fitted and finalized")

        try:
            spec = CompressionSpec(**metadata["spec"])
            latent_dim = _validate_positive_int("latent_dim", metadata["latent_dim"])
            fit_count = int(metadata["fit_observation_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompressionError("invalid compressor metadata") from exc

        expected_shape = (int(spec.n_prototypes), latent_dim)
        if prototypes.shape != expected_shape:
            raise CompressionError(
                f"serialized prototype shape {prototypes.shape}, expected {expected_shape}"
            )
        if fit_count < int(spec.n_prototypes):
            raise CompressionError("serialized fit observation count is inconsistent with codebook")
        if not np.isfinite(prototypes).all():
            raise CompressionError("serialized prototypes contain NaN or Inf")

        prototypes.setflags(write=False)
        compressor = cls(spec, latent_dim)
        compressor._fit_observations = fit_count
        compressor._initialized = True
        compressor._finalized = True
        compressor._prototypes = prototypes
        compressor._model = None
        compressor._warmup = None
        compressor._pending = None
        compressor._warmup_rows = 0
        compressor._pending_rows = 0
        return compressor


def build_compressor(spec: CompressionSpec, latent_dim: int) -> MiniBatchKMeansCompressor:
    """Build the configured latent compressor."""

    if not isinstance(spec, CompressionSpec):
        raise CompressionError("spec must be a CompressionSpec")
    if spec.method != _METHOD:  # Defensive if construction validation is ever bypassed.
        raise CompressionError(f"unsupported compression method {spec.method!r}")
    return MiniBatchKMeansCompressor(spec, latent_dim)


def load_compressor(path: str | os.PathLike[str]) -> MiniBatchKMeansCompressor:
    """Load a finalized latent compressor artifact."""

    return MiniBatchKMeansCompressor.load(path)


def _smoke() -> None:
    """Executable acceptance smoke: ``python -m mbo_lab.compressz``."""

    rng = np.random.default_rng(1234)
    centers = np.asarray(
        [[-4.0, -4.0, 0.0], [4.0, -4.0, 1.0], [-4.0, 4.0, -1.0], [4.0, 4.0, 0.5]],
        dtype=np.float32,
    )
    labels = rng.integers(0, centers.shape[0], size=5000)
    z = centers[labels] + rng.normal(0.0, 0.35, size=(labels.size, centers.shape[1])).astype(
        np.float32
    )

    spec = CompressionSpec(
        n_prototypes=4,
        batch_size=127,
        init_size=381,
        n_init=3,
        seed=17,
    )
    compressor = build_compressor(spec, latent_dim=3)

    try:
        compressor.encode(z[:8])
    except CompressionStateError:
        pass
    else:  # pragma: no cover - executable assertion.
        raise AssertionError("encode() must fail before finalize()")

    # Intentionally irregular loader batches. Internal update boundaries remain 127 rows.
    boundaries = [73, 211, 17, 509, 1000, 3, 777, 2410]
    cursor = 0
    for width in boundaries:
        if cursor >= len(z):
            break
        compressor.partial_fit(z[cursor : min(cursor + width, len(z))])
        cursor += width
    if cursor < len(z):
        compressor.partial_fit(z[cursor:])

    compressor.finalize()
    prototypes_before_validation = compressor.prototypes.copy()
    codes = compressor.encode(z)
    decoded = compressor.decode(codes)
    codes2, decoded2 = compressor.quantize(z)

    assert codes.dtype.kind in "iu"
    assert int(codes.min()) >= 0 and int(codes.max()) < spec.n_prototypes
    assert np.array_equal(codes, codes2)
    assert np.array_equal(decoded, decoded2)
    assert np.array_equal(decoded, compressor.prototypes[codes])

    try:
        compressor.partial_fit(z[:8])
    except CompressionStateError:
        pass
    else:  # pragma: no cover - executable assertion.
        raise AssertionError("partial_fit() must fail after finalize()")

    stats = compressor.make_stats_accumulator(reservoir_size=257, seed=99)
    for start in range(0, len(z), 113):
        batch = z[start : start + 113]
        stats.update(batch, compressor.encode(batch))
    report = stats.summary()
    assert report.observations_assigned == len(z)
    assert report.percentile_sample_size <= 257
    assert int(report.prototype_occupancy_counts.sum()) == len(z)

    # Same seed + same ordered stream + same spec is reproducible even with different
    # external loader batch boundaries because updates are internally aligned.
    second = build_compressor(spec, latent_dim=3)
    for start in range(0, len(z), 333):
        second.partial_fit(z[start : start + 333])
    second.finalize()
    np.testing.assert_allclose(second.prototypes, compressor.prototypes, rtol=0.0, atol=0.0)
    assert np.array_equal(second.encode(z), codes)

    # Encoding held-out vectors cannot alter frozen prototypes.
    validation = rng.normal(size=(91, 3)).astype(np.float32)
    compressor.encode(validation)
    assert np.array_equal(compressor.prototypes, prototypes_before_validation)

    with tempfile.TemporaryDirectory() as directory:
        path = compressor.save(Path(directory) / "compressor.npz")
        loaded = load_compressor(path)
        assert np.array_equal(loaded.prototypes, compressor.prototypes)
        assert np.array_equal(loaded.encode(z), codes)

    print(
        json.dumps(
            {
                "status": "ok",
                "fit_observations": compressor.fit_observation_count,
                "active_prototypes": report.active_prototype_count,
                "mean_squared_quantization_error": report.mean_squared_quantization_error,
                "p99_nearest_prototype_distance": report.p99_nearest_prototype_distance,
                "effective_state_count": report.effective_state_count,
                "reservoir_size": report.percentile_sample_size,
            },
            indent=2,
            sort_keys=True,
        )
    )
