"""Diagnostic linear probes for the mechanics world model.

These metrics are *not* training signals — they exist so we can verify that
the unsupervised factorization actually captures the system's generalized
coordinates, which is the assumption underlying H1/H3. The two probes are:

* **Linear probe**: ordinary least squares from ``encoded_q`` (and
  ``encoded_qdot``) onto the simulator's ``qpos`` (and ``qvel``). We report
  R² so a value near 1 means the encoder's mechanics head is, up to a linear
  reparameterization, the true coordinate.
* **Energy tracking**: drift of ``E_learned(t) = T(q_t, qdot_t) + V(q_t)``
  computed by the model's ``MassMatrix`` and ``Potential`` over open-loop
  rollouts. It is *not* compared against the simulator's energy, and the
  normalization by ``|E(0)|`` depends on ``V``'s arbitrary additive constant,
  so treat it as a relative diagnostic only.
* **State probes** (``run_state_probes``): linear *and* MLP probes from any
  latent block (``q``, ``z``, the RSSM's ``h``/``z``, ...) onto identifiable
  targets — cart position plus ``(cos, sin)`` of the pole angle, and ``qvel``
  — with an episode-level train/test split. These replaced the legacy
  ``q -> raw qpos`` probe as the factorization check (October 2026 review).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from models.mechanics import MechanicsWorldModel, total_energy


@dataclass(frozen=True)
class LinearProbeResult:
    """OLS fit summary for one probe (q or qdot)."""

    r2_per_dim: NDArray[np.float64]  # [target_dim]
    r2_overall: float
    coef: NDArray[np.float64]  # [target_dim, encoded_dim + 1] (last col = intercept)
    rmse: NDArray[np.float64]  # [target_dim]


def fit_linear_probe(
    encoded: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> LinearProbeResult:
    """Fit a held-out OLS probe ``target ~= W @ encoded + b`` and return R² stats.

    ``encoded`` and ``target`` are flattened across (B, T) so each row is a
    single sample. We split deterministically by row index after a seeded
    shuffle so the train/test split is reproducible across calls.
    """

    if encoded.ndim != 2 or target.ndim != 2:
        raise ValueError("encoded and target must be 2-D [N, D]")
    if encoded.shape[0] != target.shape[0]:
        raise ValueError("encoded and target must have the same N")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must satisfy 0 < f < 1")

    rng = np.random.default_rng(seed)
    n_samples = encoded.shape[0]
    permutation = rng.permutation(n_samples)
    split = int(round(n_samples * train_fraction))
    train_idx = permutation[:split]
    test_idx = permutation[split:]
    if len(train_idx) < encoded.shape[1] + 1 or len(test_idx) < 1:
        raise ValueError(
            "not enough samples for the requested split: "
            f"N={n_samples}, train_fraction={train_fraction}",
        )

    x_train = np.concatenate(
        [encoded[train_idx], np.ones((len(train_idx), 1))], axis=1,
    )  # [N_train, D_enc + 1]
    y_train = target[train_idx]  # [N_train, D_tgt]
    coef, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    coef_t = coef.T  # [D_tgt, D_enc + 1]

    x_test = np.concatenate(
        [encoded[test_idx], np.ones((len(test_idx), 1))], axis=1,
    )
    y_test = target[test_idx]
    predictions = x_test @ coef_t.T  # [N_test, D_tgt]
    residuals = y_test - predictions
    ss_res = (residuals**2).sum(axis=0)
    ss_tot = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_dim = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
    r2_overall = float(
        1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12),
    )
    rmse = np.sqrt((residuals**2).mean(axis=0))
    return LinearProbeResult(
        r2_per_dim=r2_per_dim,
        r2_overall=r2_overall,
        coef=coef_t,
        rmse=rmse,
    )


STATE_PROBE_KINDS = ("linear", "mlp")


def pose_targets(
    qpos: NDArray[np.float64],
    angle_dims: Sequence[int],
) -> tuple[NDArray[np.float64], list[str]]:
    """Return ``[N, D']`` probe targets with each angle dim replaced by ``(cos, sin)``.

    Raw hinge angles are unbounded (the random-policy cartpole data spans
    about -32..16 rad), so no function of a single image can predict them;
    ``(cos, sin)`` is the image-identifiable version of the angle.
    """

    if qpos.ndim != 2:
        raise ValueError("qpos must be [N, D]")
    angle_set = set(angle_dims)
    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    for dim in range(qpos.shape[1]):
        if dim in angle_set:
            columns.extend([np.cos(qpos[:, dim]), np.sin(qpos[:, dim])])
            names.extend([f"cos_qpos{dim}", f"sin_qpos{dim}"])
        else:
            columns.append(qpos[:, dim])
            names.append(f"qpos{dim}")
    return np.stack(columns, axis=1), names


def group_train_test_split(
    groups: NDArray[np.int64],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], str]:
    """Split row indices so no group (episode) lands on both sides.

    Overlapping windows from one episode share frames, so a row-level split
    puts near-duplicate rows in train and test and inflates R². Falls back to
    a seeded row split, reported as ``"row"``, when there is only one group.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must satisfy 0 < f < 1")
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if unique.size < 2:
        permutation = rng.permutation(groups.shape[0])
        split = int(round(groups.shape[0] * train_fraction))
        return permutation[:split], permutation[split:], "row"
    shuffled = rng.permutation(unique)
    n_train = min(max(int(round(unique.size * train_fraction)), 1), unique.size - 1)
    train_mask = np.isin(groups, shuffled[:n_train])
    return np.flatnonzero(train_mask), np.flatnonzero(~train_mask), "episode"


def _r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> tuple[float, list[float]]:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    per_dim = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    overall = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
    return float(overall), [float(v) for v in per_dim]


def _predict_linear(
    x_train: NDArray[np.float64], y_train: NDArray[np.float64], x_test: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    return np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1) @ coef


def _predict_mlp(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    *,
    seed: int,
    hidden_size: int = 64,
    steps: int = 1500,
) -> NDArray[np.float64]:
    """Full-batch AdamW fit of a 2-hidden-layer MLP on standardized data (CPU)."""

    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0) + 1e-6
    xt = torch.from_numpy((x_train - x_mean) / x_std).float()
    yt = torch.from_numpy((y_train - y_mean) / y_std).float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(xt.shape[1], hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, yt.shape[1]),
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(net(xt), yt)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            xs = torch.from_numpy((x_test - x_mean) / x_std).float()
            prediction = net(xs).double().numpy()
    return prediction * y_std + y_mean


def fit_state_probe(
    encoded: NDArray[np.float64],
    target: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    *,
    kind: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Held-out R² of a ``linear`` (OLS) or ``mlp`` probe ``encoded -> target``."""

    if kind not in STATE_PROBE_KINDS:
        raise ValueError(f"kind must be one of {STATE_PROBE_KINDS}")
    if encoded.ndim != 2 or target.ndim != 2 or encoded.shape[0] != target.shape[0]:
        raise ValueError("encoded and target must be 2-D with matching N")
    x_train, x_test = encoded[train_idx], encoded[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    if kind == "linear":
        prediction = _predict_linear(x_train, y_train, x_test)
    else:
        prediction = _predict_mlp(x_train, y_train, x_test, seed=seed)
    overall, per_dim = _r2(y_test, prediction)
    return {"r2_overall": overall, "r2_per_dim": per_dim}


def run_state_probes(
    latents: dict[str, NDArray[np.float64]],
    physics_qpos: NDArray[np.float64],
    physics_qvel: NDArray[np.float64],
    sequence_groups: Sequence[int],
    *,
    target_angle_dims: Sequence[int] = (1,),
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    """Probe every latent block against identifiable pose and velocity targets.

    ``latents`` maps a name to a ``[B, T, D]`` array; ``physics_qpos`` /
    ``physics_qvel`` are ``[B, T, D_q]``; ``sequence_groups`` holds one episode
    id per sequence (``B`` entries) so the split never shares an episode.
    Reports linear and MLP R² so a nonlinear-but-informative chart is not
    mistaken for an empty one. The default ``target_angle_dims=(1,)`` is the
    cartpole hinge.
    """

    batch, steps = physics_qpos.shape[:2]
    if len(sequence_groups) != batch:
        raise ValueError("sequence_groups must have one entry per sequence")
    groups = np.repeat(np.asarray(sequence_groups, dtype=np.int64), steps)
    train_idx, test_idx, split_kind = group_train_test_split(
        groups, train_fraction=train_fraction, seed=seed,
    )
    pose, pose_names = pose_targets(flatten_target(physics_qpos), target_angle_dims)
    targets = {"pose": pose, "qvel": flatten_target(physics_qvel)}
    probes: dict[str, Any] = {}
    for latent_name, latent in latents.items():
        encoded = flatten_target(latent)
        probes[latent_name] = {
            target_name: {
                kind: fit_state_probe(
                    encoded, target, train_idx, test_idx, kind=kind, seed=seed,
                )
                for kind in STATE_PROBE_KINDS
            }
            for target_name, target in targets.items()
        }
    return {
        "split": split_kind,
        "num_train_rows": int(train_idx.size),
        "num_test_rows": int(test_idx.size),
        "num_groups": int(np.unique(groups).size),
        "pose_target_names": pose_names,
        "qvel_target_names": [f"qvel{d}" for d in range(physics_qvel.shape[-1])],
        "probes": probes,
    }


def state_probe_summary_rows(state_probes: dict[str, Any]) -> list[tuple[str, float]]:
    """Flatten ``run_state_probes`` output to ``(name, r2_overall)`` table rows."""

    rows: list[tuple[str, float]] = []
    for latent_name, by_target in state_probes.get("probes", {}).items():
        for target_name, by_kind in by_target.items():
            for kind, result in by_kind.items():
                rows.append(
                    (f"state_probes.{latent_name}->{target_name}.{kind}", result["r2_overall"]),
                )
    return rows


@torch.no_grad()
def encode_dataset(
    model: MechanicsWorldModel,
    observations: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the encoder on a [B, T, 3, H, W] tensor and return ``(q, qdot, z)``.

    All three returned arrays are flattened to ``[B * T, D]`` ``float64``
    NumPy arrays so downstream OLS works without device shuffling.
    """

    if observations.ndim != 5:
        raise ValueError("observations must have shape [B, T, C, H, W]")
    model.eval()
    posterior = model.encoder(observations.to(device))
    q = posterior.q_mean.detach().cpu().double().numpy()
    qdot = posterior.qdot_mean.detach().cpu().double().numpy()
    z = posterior.z_mean.detach().cpu().double().numpy()
    return _flatten(q), _flatten(qdot), _flatten(z)


def _flatten(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr.reshape(-1, arr.shape[-1])


def flatten_target(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten a ``[B, T, D]`` numpy array of physics targets to ``[B*T, D]``."""

    if arr.ndim != 3:
        raise ValueError("target array must be [B, T, D]")
    return arr.reshape(-1, arr.shape[-1])


@torch.no_grad()
def compute_learned_energy(
    model: MechanicsWorldModel,
    q: torch.Tensor,
    qdot: torch.Tensor,
) -> NDArray[np.float64]:
    """Return ``E_learned(t) = T + V`` from the model's mass matrix + potential.

    ``q`` and ``qdot`` are ``[B, T, d]`` posterior or imagined samples.
    Returned array is ``[B, T]`` ``float64``.
    """

    if q.shape != qdot.shape:
        raise ValueError("q and qdot must share shape")
    energy = total_energy(model.mass_matrix_module, model.potential_module, q, qdot)
    return energy.detach().cpu().double().numpy()


def normalized_energy_drift(
    energy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(E(t) - E(0)) / max(|E(0)|, eps)`` per sequence — diagnostic only."""

    if energy.ndim != 2:
        raise ValueError("energy must be [B, T]")
    reference = np.maximum(np.abs(energy[:, :1]), 1e-3)
    return (energy - energy[:, :1]) / reference
