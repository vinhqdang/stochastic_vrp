#!/usr/bin/env python3
"""

Resume an interrupted run (same CLI/settings and output prefix):
  python go_nogo_orl_certified_dethloff_v3_resume.py full dir=Dethloff resume=1 overwrite=0

Completed cells are skipped.  Any orphan/duplicate run rows from an interrupted
write are cleaned before continuation; the unfinished cell is rerun from scratch.
go_nogo_orl_certified_dethloff.py

Self-contained GO/NO-GO harness for the research direction:

    multivariate affine route risk
        -> one global projection direction
        -> exact one-dimensional O(p log N) empirical-CVaR kernel
        -> deterministic two-sided certificate
        -> exact full-scenario fallback for ambiguous routes.

The harness is distilled from the uploaded ORL-extension implementation and
copies the required Dethloff parser, affine overlay, route kernels, certificate,
and fixed-iteration ALNS directly into this file. It does not import the old
solver or code-orl-extend.txt.

What it tests
-------------
1. Decision exactness: every certified decision is audited in a separate,
   untimed audit run; ambiguous routes always use the full evaluator.
2. Search identity: paired FULL and CERT runs must have identical route-check
   counts, final plans, fleet size, distance, and objective.
3. Computational value: gate-only and end-to-end solver speed-ups.
4. Coverage: fraction of route checks decided without a full multivariate pass.

Outputs
-------
<prefix>_runs.csv      one row per timed FULL/CERT pair
<prefix>_cells.csv     median/IQR summary per instance/profile/policy
<prefix>_summary.txt   GO / BORDERLINE / NO-GO decision and aggregate metrics

Windows examples
----------------
Quick research decision (recommended first):

    python go_nogo_orl_certified_dethloff.py quick ^
      dir="C:\\path\\to\\Dethloff"

Very small sanity run:

    python go_nogo_orl_certified_dethloff.py smoke ^
      dir="C:\\path\\to\\Dethloff"

Full benchmark:

    python go_nogo_orl_certified_dethloff.py full ^
      dir="C:\\path\\to\\Dethloff"

Useful overrides:

    instances=CON3-4,CON8-4,SCA3-4,SCA8-4
    profiles=moderate
    policy=BOTH
    N=20000 d=10 iters=5 reps=2 train=200
    direction=SLOPE out=my_probe

Only NumPy is required.
"""

from __future__ import annotations

import bisect
import csv
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ============================================================================
# Defaults and presets
# ============================================================================

ALPHA = 0.90
CV = 0.25
EPS_FRAC = 0.15
FACTOR_DIMENSION = 10
SUPPORT = 2.5
SEED = 2026


@dataclass(frozen=True)
class OverlayProfile:
    name: str
    spatial_decay: float
    common_weight: float


PROFILES = {
    "concentrated": OverlayProfile("concentrated", 2.0, 1.50),
    "moderate": OverlayProfile("moderate", 1.0, 1.00),
    "diffuse": OverlayProfile("diffuse", 0.50, 0.65),
}


MODE_PRESETS = {
    "smoke": {
        "instances": ["CON3-4", "SCA8-4"],
        "profiles": ["moderate"],
        "policy": "SAA",
        "N": 5_000,
        "d": 10,
        "iters": 2,
        "reps": 1,
        "train": 100,
        "audit_iters": 1,
    },
    "quick": {
        "instances": ["CON3-4", "CON8-4", "SCA3-4", "SCA8-4"],
        "profiles": ["moderate"],
        "policy": "BOTH",
        "N": 20_000,
        "d": 10,
        "iters": 5,
        "reps": 2,
        "train": 200,
        "audit_iters": 2,
    },
    "full": {
        "instances": ["ALL"],
        "profiles": ["concentrated", "moderate", "diffuse"],
        "policy": "BOTH",
        "N": 50_000,
        "d": 10,
        "iters": 50,
        "reps": 3,
        "train": 200,
        "audit_iters": 2,
    },
}


# GO thresholds are intentionally modest: this is a conference-sized extension,
# not a claim that every instance must accelerate.
DEFAULT_THRESHOLDS = {
    "strong_median_gate": 1.50,
    "strong_median_solver": 1.25,
    "strong_solver_win_rate": 0.75,
    "strong_median_coverage": 0.45,
    "weak_median_gate": 1.20,
    "weak_median_solver": 1.05,
    "weak_solver_win_rate": 0.55,
    "weak_median_coverage": 0.25,
}


# ============================================================================
# Dethloff-compatible parser
# ============================================================================


def _lines_after(text: str, tag: str) -> list[str]:
    output: list[str] = []
    capturing = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not capturing:
            if stripped.upper().startswith(tag):
                capturing = True
            continue
        if any(character.isalpha() for character in stripped):
            break
        output.append(stripped)
    return output


def _header_int(text: str, key: str) -> int | None:
    for line in text.splitlines():
        if line.strip().upper().startswith(key):
            match = re.findall(r"-?\d+", line)
            if match:
                return int(match[-1])
    return None


def _header_float(text: str, key: str) -> float | None:
    for line in text.splitlines():
        if line.strip().upper().startswith(key):
            match = re.findall(r"-?\d+\.?\d*", line)
            if match:
                return float(match[-1])
    return None


def _parse_pickup_delivery(text: str, n: int) -> np.ndarray:
    demands = np.zeros((n, 2), dtype=float)
    for row in _lines_after(text, "PICKUP_AND_DELIVERY_SECTION"):
        tokens = row.split()
        if len(tokens) < 3:
            continue
        index = int(float(tokens[0])) - 1
        if 0 <= index < n:
            demands[index, 0] = float(tokens[-2])
            demands[index, 1] = float(tokens[-1])
    return demands


def parse_dethloff(path: str | Path) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    text = Path(path).read_text(errors="ignore")
    n = _header_int(text, "DIMENSION")
    capacity = _header_float(text, "CAPACITY")
    tokens: list[str] = []
    for row in _lines_after(text, "EDGE_WEIGHT_SECTION"):
        tokens.extend(row.split())
    values = [int(float(token)) for token in tokens]
    if n is None or capacity is None:
        raise ValueError(f"Missing DIMENSION or CAPACITY in {path}")
    if len(values) != n * n:
        raise ValueError(
            f"EDGE_WEIGHT_SECTION in {path}: got {len(values)} values, expected {n*n}"
        )
    # The distributed Dethloff files store every numerical quantity at a
    # common 10,000x scale.  Normalize the complete instance here—not only
    # the reported distance—so distance, pickup/delivery demand, capacity,
    # CVaR values, and certificate widths are all in the original units.
    data_scale = 10_000.0
    distance = np.asarray(values, dtype=np.float64).reshape(n, n) / data_scale
    demands = _parse_pickup_delivery(text, n) / data_scale
    capacity = float(capacity) / data_scale
    return distance, demands, capacity, n, 1.0


# ============================================================================
# Multivariate bounded affine demand overlay
# ============================================================================


@dataclass(frozen=True)
class AffineInstance:
    name: str
    source: str
    distance: np.ndarray
    capacity: float
    n: int
    scale: float
    X: np.ndarray
    d0: np.ndarray
    p0: np.ndarray
    dvec: np.ndarray
    pvec: np.ndarray
    delivery_values: np.ndarray  # shape (n, N)
    pickup_values: np.ndarray    # shape (n, N)


def bounded_factor_samples(N: int, dimension: int, support: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.clip(rng.normal(size=(N, dimension)), -support, support)
    X -= X.mean(axis=0, keepdims=True)
    maximum = np.max(np.abs(X), axis=0)
    correction = np.maximum(1.0, maximum / support)
    X /= correction[None, :]
    return X


def spatial_basis_from_distance(
    customer_distance: np.ndarray,
    components: int,
    decay: float,
) -> np.ndarray:
    positive = customer_distance[customer_distance > 1e-12]
    length = float(np.median(positive)) if positive.size else 1.0
    length = max(length, 1e-12)
    kernel = np.exp(-customer_distance / length)
    kernel = 0.5 * (kernel + kernel.T)
    values, vectors = np.linalg.eigh(kernel)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    take = min(components, vectors.shape[1])
    weights = np.power(values[:take] + 1e-12, decay / 2.0)
    basis = vectors[:, :take] * weights[None, :]
    if take < components:
        basis = np.pad(basis, ((0, 0), (0, components - take)))
    return basis


def row_l1_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.sum(np.abs(matrix), axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return matrix / norm


def make_affine_instance(
    path: str | Path,
    profile: OverlayProfile,
    N: int,
    dimension: int,
    cv: float,
    support: float,
    seed: int,
    data_root: str | Path,
) -> AffineInstance:
    distance, demands, capacity, n, scale = parse_dethloff(path)
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if not (0.0 < cv < 0.95):
        raise ValueError("cv must lie in (0,0.95)")

    customer_count = n - 1
    X = bounded_factor_samples(N, dimension, support, seed)
    customer_distance = distance[1:, 1:].astype(float)
    basis = spatial_basis_from_distance(
        customer_distance,
        components=dimension - 1,
        decay=profile.spatial_decay,
    )

    delivery_raw = np.column_stack(
        (np.full(customer_count, profile.common_weight), basis)
    )
    rng = np.random.default_rng(seed + 1)
    rotation, _ = np.linalg.qr(rng.normal(size=(dimension - 1, dimension - 1)))
    pickup_spatial = basis @ rotation
    pickup_raw = np.column_stack(
        (np.full(customer_count, 0.85 * profile.common_weight), pickup_spatial)
    )
    delivery_loading = row_l1_normalize(delivery_raw)
    pickup_loading = row_l1_normalize(pickup_raw)

    delivery_cv = np.minimum(
        cv * rng.uniform(0.80, 1.20, size=customer_count), 0.90
    )
    pickup_cv = np.minimum(
        cv * rng.uniform(0.80, 1.20, size=customer_count), 0.90
    )

    d0 = demands[:, 0].astype(float).copy()
    p0 = demands[:, 1].astype(float).copy()
    dvec = np.zeros((n, dimension), dtype=float)
    pvec = np.zeros((n, dimension), dtype=float)
    dvec[1:] = d0[1:, None] * delivery_cv[:, None] * delivery_loading / support
    pvec[1:] = p0[1:, None] * pickup_cv[:, None] * pickup_loading / support

    delivery_values = d0[:, None] + dvec @ X.T
    pickup_values = p0[:, None] + pvec @ X.T
    if delivery_values.min() < -1e-9 or pickup_values.min() < -1e-9:
        raise AssertionError("Bounded affine positivity construction failed")

    source = str(Path(path).resolve().relative_to(Path(data_root).resolve()))
    label = str(Path(source).with_suffix(""))
    return AffineInstance(
        name=label,
        source=source,
        distance=distance,
        capacity=capacity,
        n=n,
        scale=scale,
        X=X,
        d0=d0,
        p0=p0,
        dvec=dvec,
        pvec=pvec,
        delivery_values=delivery_values,
        pickup_values=pickup_values,
    )


# ============================================================================
# Route risk, exact empirical CVaR, and ORL one-dimensional kernel
# ============================================================================


def route_cost(route: Sequence[int], distance: np.ndarray) -> float:
    if not route:
        return 0.0
    cost = distance[0, route[0]] + distance[route[-1], 0]
    for index in range(len(route) - 1):
        cost += distance[route[index], route[index + 1]]
    return float(cost)


def route_peaks(
    route: Sequence[int],
    delivery_values: np.ndarray,
    pickup_values: np.ndarray,
) -> np.ndarray:
    if not route:
        return np.zeros(delivery_values.shape[1], dtype=float)
    delivery = delivery_values[list(route)].T
    pickup = pickup_values[list(route)].T
    total_delivery = delivery.sum(axis=1)
    middle = (
        total_delivery[:, None]
        - np.cumsum(delivery, axis=1)
        + np.cumsum(pickup, axis=1)
    )
    return np.maximum(total_delivery, middle.max(axis=1))


def tail_parameters(alpha: float, n: int) -> tuple[float, int, float]:
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must lie in [0,1)")
    if n <= 0:
        raise ValueError("sample must be nonempty")
    tail_mass = (1.0 - alpha) * n
    nearest = float(round(tail_mass))
    tolerance = 16.0 * max(math.ulp(tail_mass), math.ulp(nearest))
    if abs(tail_mass - nearest) <= tolerance:
        tail_mass = nearest
    count = min(n, max(1, int(math.ceil(tail_mass))))
    boundary_weight = min(1.0, max(0.0, tail_mass - (count - 1)))
    return tail_mass, count, boundary_weight


def empirical_cvar(values: Sequence[float] | np.ndarray, alpha: float) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    n = array.size
    tail_mass, count, boundary_weight = tail_parameters(alpha, n)
    tail = np.partition(array, n - count)[n - count:]
    boundary = float(tail.min())
    return (
        float(tail.sum()) - (1.0 - boundary_weight) * boundary
    ) / tail_mass


def upper_envelope(lines: Iterable[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = sorted(lines, key=lambda item: (item[1], item[0]))
    unique: list[tuple[float, float]] = []
    for intercept, slope in records:
        intercept = float(intercept)
        slope = float(slope)
        if unique and slope == unique[-1][1]:
            if intercept > unique[-1][0]:
                unique[-1] = (intercept, slope)
            continue
        unique.append((intercept, slope))
    if not unique:
        raise ValueError("At least one line is required")

    def crossing(left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - right[0]) / (right[1] - left[1])

    stack: list[tuple[float, float]] = []
    starts: list[float] = []
    for line in unique:
        while stack:
            x = crossing(stack[-1], line)
            if starts and x <= starts[-1]:
                stack.pop()
                starts.pop()
            else:
                break
        if not stack:
            stack.append(line)
            starts.append(-math.inf)
        else:
            starts.append(crossing(stack[-1], line))
            stack.append(line)

    intercepts = np.asarray([line[0] for line in stack], dtype=float)
    slopes = np.asarray([line[1] for line in stack], dtype=float)
    breakpoints = np.asarray(starts + [math.inf], dtype=float)
    return intercepts, slopes, breakpoints


def cvar_os(
    envelope: tuple[np.ndarray, np.ndarray, np.ndarray],
    z_sorted: np.ndarray,
    prefix: np.ndarray,
    alpha: float,
) -> float:
    """Exact empirical CVaR of a scalar max-affine loss after shared sorting."""
    intercepts, slopes, breakpoints = envelope
    n = len(z_sorted)
    pieces = len(intercepts)
    tail_mass, count, boundary_weight = tail_parameters(alpha, n)

    def value(index: int) -> float:
        z = z_sorted[index]
        piece = bisect.bisect_right(breakpoints, z) - 1
        piece = min(max(piece, 0), pieces - 1)
        return float(intercepts[piece] + slopes[piece] * z)

    sign_change = bisect.bisect_left(slopes, 0.0)
    if sign_change == 0:
        valley = 0
    elif sign_change >= pieces:
        valley = n - 1
    else:
        position = bisect.bisect_left(z_sorted, breakpoints[sign_change])
        candidates = [
            index
            for index in (position - 1, position, position + 1)
            if 0 <= index < n
        ]
        valley = min(candidates, key=value)

    left_size = valley + 1
    right_size = n - 1 - valley

    def left_arm(k: int) -> float:
        if k <= 0:
            return math.inf
        if k > left_size:
            return -math.inf
        return value(k - 1)

    def right_arm(k: int) -> float:
        if k <= 0:
            return math.inf
        if k > right_size:
            return -math.inf
        return value(n - k)

    low = max(0, count - right_size)
    high = min(count, left_size)
    lo, hi, best = low, high, low - 1
    while lo <= hi:
        middle = (lo + hi) // 2
        if left_arm(middle) >= right_arm(count - middle + 1):
            best = middle
            lo = middle + 1
        else:
            hi = middle - 1
    selected_left = max(best, low)
    selected_right = count - selected_left

    def range_sum(first: int, last: int) -> float:
        if first > last:
            return 0.0
        total = 0.0
        index = first
        while index <= last:
            z = z_sorted[index]
            piece = bisect.bisect_right(breakpoints, z) - 1
            piece = min(max(piece, 0), pieces - 1)
            end = bisect.bisect_left(z_sorted, breakpoints[piece + 1]) - 1
            end = min(max(end, index), last)
            total += (
                intercepts[piece] * (end - index + 1)
                + slopes[piece] * (prefix[end + 1] - prefix[index])
            )
            index = end + 1
        return float(total)

    selected_sum = range_sum(0, selected_left - 1) + range_sum(
        n - selected_right, n - 1
    )
    boundary = min(left_arm(selected_left), right_arm(selected_right))
    return (
        selected_sum - (1.0 - boundary_weight) * boundary
    ) / tail_mass


def route_affine_lines(
    instance: AffineInstance,
    route: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    route = list(route)
    length = len(route)
    intercepts = np.empty(length + 1, dtype=float)
    slopes = np.empty((length + 1, instance.X.shape[1]), dtype=float)

    intercept = float(instance.d0[route].sum())
    slope = instance.dvec[route].sum(axis=0)
    intercepts[0] = intercept
    slopes[0] = slope
    for index, customer in enumerate(route, 1):
        intercept += instance.p0[customer] - instance.d0[customer]
        slope = slope + instance.pvec[customer] - instance.dvec[customer]
        intercepts[index] = intercept
        slopes[index] = slope
    return intercepts, slopes


# ============================================================================
# Global direction and exact projection certificate
# ============================================================================


def training_routes(instance: AffineInstance, route_count: int, seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    customers = np.arange(1, instance.n)
    positive = instance.d0[1:][instance.d0[1:] > 0]
    mean_delivery = float(np.mean(positive)) if positive.size else 1.0
    target_length = max(2, int(round(instance.capacity / mean_delivery)))
    maximum_length = min(
        instance.n - 1,
        max(8, int(math.ceil(1.7 * target_length))),
    )
    routes: list[tuple[int, ...]] = []
    for _ in range(route_count):
        length = int(rng.integers(2, maximum_length + 1))
        sampled = rng.choice(customers, size=length, replace=False)
        routes.append(tuple(int(customer) for customer in sampled))
    return routes


def slope_second_moment(instance: AffineInstance, routes: Sequence[Sequence[int]]) -> np.ndarray:
    dimension = instance.X.shape[1]
    moment = np.zeros((dimension, dimension), dtype=float)
    count = 0
    for route in routes:
        _, slopes = route_affine_lines(instance, route)
        moment += slopes.T @ slopes
        count += slopes.shape[0]
    if count:
        moment /= count
    return moment


def matrix_geometry(matrix: np.ndarray) -> tuple[float, float, float]:
    values = np.maximum(np.linalg.eigvalsh(matrix), 0.0)[::-1]
    total = float(values.sum())
    if total <= 1e-15:
        return 0.0, 0.0, 0.0
    weights = values / total
    positive = weights[weights > 1e-15]
    effective_rank = float(1.0 / np.sum(weights * weights))
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    top_share = float(weights[0])
    return effective_rank, entropy_rank, top_share


@dataclass(frozen=True)
class CertificateCache:
    alpha: float
    u: np.ndarray
    mu: np.ndarray
    z_sorted: np.ndarray
    z_prefix: np.ndarray
    residual_cvar: float
    explained_variance: float
    effective_rank: float
    entropy_rank: float
    top_share: float
    direction: str
    prep_seconds: float


def build_certificate_cache(
    instance: AffineInstance,
    alpha: float,
    direction: str,
    train_route_count: int,
    seed: int,
) -> CertificateCache:
    start = time.perf_counter()
    X = np.asarray(instance.X, dtype=float)
    mu = X.mean(axis=0)
    centered = X - mu[None, :]
    routes = training_routes(instance, train_route_count, seed)
    slope_moment = slope_second_moment(instance, routes)
    geometry = matrix_geometry(slope_moment)

    key = direction.upper()
    if key == "SLOPE":
        values, vectors = np.linalg.eigh(slope_moment)
        u = vectors[:, int(np.argmax(values))]
    elif key == "PCA":
        covariance = centered.T @ centered / max(1, centered.shape[0])
        values, vectors = np.linalg.eigh(covariance)
        u = vectors[:, int(np.argmax(values))]
    elif key == "DELIVERY":
        delivery_sums = [
            instance.dvec[list(route)].sum(axis=0)
            for route in routes
        ]
        matrix = np.asarray(delivery_sums, dtype=float)
        delivery_moment = matrix.T @ matrix / max(1, len(matrix))
        values, vectors = np.linalg.eigh(delivery_moment)
        u = vectors[:, int(np.argmax(values))]
    else:
        raise ValueError("direction must be SLOPE, PCA, or DELIVERY")

    norm = float(np.linalg.norm(u))
    if norm <= 1e-15:
        raise ValueError("Direction construction returned zero")
    u = u / norm

    z = centered @ u
    residual_sq = np.maximum(
        np.einsum("ij,ij->i", centered, centered) - z * z,
        0.0,
    )
    residual_norm = np.sqrt(residual_sq)
    z_sorted = np.sort(z)
    z_prefix = np.concatenate(([0.0], np.cumsum(z_sorted)))
    residual_cvar = empirical_cvar(residual_norm, alpha)
    total_variance = float(np.sum(centered * centered))
    explained = float(np.dot(z, z)) / total_variance if total_variance > 0 else 1.0

    return CertificateCache(
        alpha=float(alpha),
        u=u,
        mu=mu,
        z_sorted=z_sorted,
        z_prefix=z_prefix,
        residual_cvar=float(residual_cvar),
        explained_variance=float(explained),
        effective_rank=geometry[0],
        entropy_rank=geometry[1],
        top_share=geometry[2],
        direction=key,
        prep_seconds=time.perf_counter() - start,
    )


@dataclass(frozen=True)
class RouteCertificate:
    projected_cvar: float
    gamma: float
    radius: float
    lower: float
    upper: float
    pieces: int


def route_certificate(
    instance: AffineInstance,
    route: Sequence[int],
    cache: CertificateCache,
) -> RouteCertificate:
    intercepts, slopes = route_affine_lines(instance, route)
    projected_intercepts = intercepts + slopes @ cache.mu
    projected_slopes = slopes @ cache.u
    envelope = upper_envelope(zip(projected_intercepts, projected_slopes))
    projected_cvar = cvar_os(
        envelope,
        cache.z_sorted,
        cache.z_prefix,
        cache.alpha,
    )
    orthogonal_sq = np.maximum(
        np.einsum("ij,ij->i", slopes, slopes)
        - projected_slopes * projected_slopes,
        0.0,
    )
    gamma = float(np.sqrt(orthogonal_sq).max())
    radius = gamma * cache.residual_cvar
    return RouteCertificate(
        projected_cvar=float(projected_cvar),
        gamma=gamma,
        radius=float(radius),
        lower=float(projected_cvar - radius),
        upper=float(projected_cvar + radius),
        pieces=len(envelope[0]),
    )


# ============================================================================
# FULL and decision-exact CERT gates
# ============================================================================


class FullGate:
    mode = "FULL"

    def __init__(self, instance: AffineInstance, capacity: float, alpha: float):
        self.instance = instance
        self.capacity = float(capacity)
        self.alpha = float(alpha)
        self.calls = 0
        self.route_length_sum = 0
        self.gate_seconds = 0.0

    def feasible(self, route: Sequence[int]) -> bool:
        if not route:
            return True
        self.calls += 1
        self.route_length_sum += len(route)
        start = time.perf_counter()
        value = empirical_cvar(
            route_peaks(
                route,
                self.instance.delivery_values,
                self.instance.pickup_values,
            ),
            self.alpha,
        )
        self.gate_seconds += time.perf_counter() - start
        return value <= self.capacity + 1e-9


class CertifiedGate:
    mode = "CERT"

    def __init__(
        self,
        instance: AffineInstance,
        capacity: float,
        alpha: float,
        cache: CertificateCache,
        audit: bool = False,
    ):
        self.instance = instance
        self.capacity = float(capacity)
        self.alpha = float(alpha)
        self.cache = cache
        self.audit = bool(audit)
        self.calls = 0
        self.route_length_sum = 0
        self.certified_feasible = 0
        self.certified_infeasible = 0
        self.fallbacks = 0
        self.pieces_sum = 0
        self.certificate_seconds = 0.0
        self.fallback_seconds = 0.0
        self.audit_seconds = 0.0
        self.audit_checks = 0

    @property
    def coverage(self) -> float:
        if not self.calls:
            return 0.0
        return (
            self.certified_feasible + self.certified_infeasible
        ) / self.calls

    @property
    def gate_seconds(self) -> float:
        return self.certificate_seconds + self.fallback_seconds

    def full_value(self, route: Sequence[int]) -> float:
        return empirical_cvar(
            route_peaks(
                route,
                self.instance.delivery_values,
                self.instance.pickup_values,
            ),
            self.alpha,
        )

    def full_decision(self, route: Sequence[int]) -> bool:
        return self.full_value(route) <= self.capacity + 1e-9

    def feasible(self, route: Sequence[int]) -> bool:
        if not route:
            return True
        self.calls += 1
        self.route_length_sum += len(route)

        start = time.perf_counter()
        cert = route_certificate(self.instance, route, self.cache)
        self.certificate_seconds += time.perf_counter() - start
        self.pieces_sum += cert.pieces

        padding = (
            64.0
            * np.finfo(float).eps
            * (
                1.0
                + abs(self.capacity)
                + abs(cert.projected_cvar)
                + abs(cert.radius)
            )
        )

        fell_back = False
        if cert.upper + padding <= self.capacity:
            decision = True
            self.certified_feasible += 1
        elif cert.lower - padding > self.capacity:
            decision = False
            self.certified_infeasible += 1
        else:
            fell_back = True
            self.fallbacks += 1
            start = time.perf_counter()
            decision = self.full_decision(route)
            self.fallback_seconds += time.perf_counter() - start

        if self.audit and not fell_back:
            start = time.perf_counter()
            truth_value = self.full_value(route)
            truth = truth_value <= self.capacity + 1e-9
            self.audit_seconds += time.perf_counter() - start
            self.audit_checks += 1
            tolerance = 2e-8 * max(1.0, abs(truth_value), abs(cert.lower), abs(cert.upper))
            if truth_value < cert.lower - tolerance or truth_value > cert.upper + tolerance:
                raise AssertionError(
                    "CVaR bracket violation: "
                    f"route={list(route)} truth={truth_value} "
                    f"lower={cert.lower} upper={cert.upper}"
                )
            if decision != truth:
                raise AssertionError(
                    "Certificate decision mismatch: "
                    f"route={list(route)} truth={truth} cert={decision} "
                    f"lower={cert.lower} upper={cert.upper} cap={self.capacity}"
                )
        return decision


# ============================================================================
# Fixed-iteration Dethloff solver core copied from the extension implementation
# ============================================================================


def cw_init(distance: np.ndarray, gate: FullGate | CertifiedGate, n: int) -> list[list[int]]:
    routes = [
        [customer]
        for customer in range(1, n)
        if gate.feasible([customer])
    ]
    placed = {customer for route in routes for customer in route}
    leftovers = [customer for customer in range(1, n) if customer not in placed]

    savings: list[tuple[float, int, int]] = []
    for a in range(1, n):
        for b in range(a + 1, n):
            savings.append(
                (distance[0, a] + distance[0, b] - distance[a, b], a, b)
            )
    savings.sort(reverse=True)

    routes_by_id = {index: route for index, route in enumerate(routes)}
    where = {route[0]: index for index, route in enumerate(routes)}
    for saving, a, b in savings:
        if saving <= 0:
            break
        ra, rb = where.get(a), where.get(b)
        if ra is None or rb is None or ra == rb:
            continue
        route_a = routes_by_id[ra]
        route_b = routes_by_id[rb]
        if route_a[-1] == a and route_b[0] == b:
            merged = route_a + route_b
        elif route_a[0] == a and route_b[-1] == b:
            merged = route_b + route_a
        elif route_a[-1] == a and route_b[-1] == b:
            merged = route_a + route_b[::-1]
        elif route_a[0] == a and route_b[0] == b:
            merged = route_a[::-1] + route_b
        else:
            continue
        if not gate.feasible(merged):
            continue
        routes_by_id[ra] = merged
        for customer in route_b:
            where[customer] = ra
        del routes_by_id[rb]

    solution = list(routes_by_id.values())
    for customer in leftovers:
        solution.append([customer])
    return solution


def two_opt_gate(
    route: list[int],
    distance: np.ndarray,
    gate: FullGate | CertifiedGate,
) -> list[int]:
    if len(route) < 4:
        return route
    current = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(current) - 1):
            for k in range(i + 1, len(current)):
                a = current[i - 1] if i > 0 else 0
                b = current[i]
                c = current[k]
                d = current[k + 1] if k + 1 < len(current) else 0
                if a == c or b == d:
                    continue
                delta = (
                    distance[a, c]
                    + distance[b, d]
                    - distance[a, b]
                    - distance[c, d]
                )
                if delta < -1e-9:
                    candidate = (
                        current[:i]
                        + current[i:k + 1][::-1]
                        + current[k + 1:]
                    )
                    if gate.feasible(candidate):
                        current = candidate
                        improved = True
                        break
            if improved:
                break
    return current


def relocate_gate(
    solution: list[list[int]],
    distance: np.ndarray,
    gate: FullGate | CertifiedGate,
) -> list[list[int]]:
    improved = True
    while improved:
        improved = False
        for route_index in range(len(solution)):
            route = solution[route_index]
            for position in range(len(route)):
                customer = route[position]
                a = route[position - 1] if position > 0 else 0
                b = route[position + 1] if position + 1 < len(route) else 0
                gain = (
                    distance[a, customer]
                    + distance[customer, b]
                    - distance[a, b]
                )
                for target_index in range(len(solution)):
                    if target_index == route_index:
                        continue
                    target = solution[target_index]
                    for insertion_position in range(len(target) + 1):
                        u = target[insertion_position - 1] if insertion_position > 0 else 0
                        v = target[insertion_position] if insertion_position < len(target) else 0
                        delta = (
                            distance[u, customer]
                            + distance[customer, v]
                            - distance[u, v]
                            - gain
                        )
                        if delta < -1e-9:
                            new_route = route[:position] + route[position + 1:]
                            new_target = (
                                target[:insertion_position]
                                + [customer]
                                + target[insertion_position:]
                            )
                            if gate.feasible(new_route) and gate.feasible(new_target):
                                solution[route_index] = new_route
                                solution[target_index] = new_target
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        solution = [route for route in solution if route]
    return solution


def greedy_insert(
    solution: list[list[int]],
    customer: int,
    distance: np.ndarray,
    gate: FullGate | CertifiedGate,
) -> tuple[int | None, int | None, float]:
    best: tuple[int | None, int | None, float] = (None, None, math.inf)
    for route_index, route in enumerate(solution):
        for position in range(len(route) + 1):
            candidate = route[:position] + [customer] + route[position:]
            if not gate.feasible(candidate):
                continue
            u = route[position - 1] if position > 0 else 0
            v = route[position] if position < len(route) else 0
            delta = (
                distance[u, customer]
                + distance[customer, v]
                - distance[u, v]
            )
            if delta < best[2]:
                best = (route_index, position, float(delta))
    return best


def ruin_recreate(
    solution: list[list[int]],
    distance: np.ndarray,
    gate: FullGate | CertifiedGate,
    rng: random.Random,
    q_frac: float = 0.2,
) -> list[list[int]]:
    solution = [route[:] for route in solution]
    customers = [customer for route in solution for customer in route]
    q = max(1, int(q_frac * len(customers)))
    removed = rng.sample(customers, min(q, len(customers)))
    removed_set = set(removed)
    solution = [
        [customer for customer in route if customer not in removed_set]
        for route in solution
    ]
    solution = [route for route in solution if route]
    rng.shuffle(removed)
    for customer in removed:
        route_index, position, _ = greedy_insert(
            solution,
            customer,
            distance,
            gate,
        )
        if route_index is None:
            solution.append([customer])
        else:
            assert position is not None
            solution[route_index].insert(position, customer)
    return solution


def local_search(
    solution: list[list[int]],
    distance: np.ndarray,
    gate: FullGate | CertifiedGate,
) -> list[list[int]]:
    solution = [
        two_opt_gate(route, distance, gate)
        for route in solution
        if route
    ]
    solution = relocate_gate(solution, distance, gate)
    return [route for route in solution if route]


def econ_cost(
    solution: list[list[int]],
    distance: np.ndarray,
    omega_vehicle: float,
) -> float:
    return (
        sum(route_cost(route, distance) for route in solution)
        + omega_vehicle * sum(1 for route in solution if route)
    )


def solve_fixed_iterations(
    distance: np.ndarray,
    gate: FullGate | CertifiedGate,
    n: int,
    omega_vehicle: float,
    iterations: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    current = local_search(cw_init(distance, gate, n), distance, gate)
    best = [route[:] for route in current]
    best_cost = econ_cost(best, distance, omega_vehicle)
    for _ in range(iterations):
        candidate = local_search(
            ruin_recreate(
                best,
                distance,
                gate,
                rng,
                rng.choice([0.1, 0.15, 0.2, 0.3]),
            ),
            distance,
            gate,
        )
        candidate_cost = econ_cost(candidate, distance, omega_vehicle)
        if candidate_cost < best_cost - 1e-9:
            best = [route[:] for route in candidate]
            best_cost = candidate_cost
    return [route for route in best if route]


# ============================================================================
# Paired benchmark and exactness audit
# ============================================================================


def canonical_plan(plan: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(route) for route in plan))


def run_solver_once(
    instance: AffineInstance,
    capacity: float,
    alpha: float,
    cache: CertificateCache,
    iterations: int,
    seed: int,
    method: str,
    audit: bool = False,
) -> dict[str, object]:
    if method == "FULL":
        gate: FullGate | CertifiedGate = FullGate(instance, capacity, alpha)
    elif method == "CERT":
        gate = CertifiedGate(instance, capacity, alpha, cache, audit=audit)
    else:
        raise ValueError(method)

    positive_distances = instance.distance[instance.distance > 0]
    omega_vehicle = float(np.mean(positive_distances)) if positive_distances.size else 1.0
    start = time.perf_counter()
    plan = solve_fixed_iterations(
        instance.distance,
        gate,
        instance.n,
        omega_vehicle,
        iterations,
        seed,
    )
    total_seconds = time.perf_counter() - start
    distance_raw = sum(route_cost(route, instance.distance) for route in plan)
    result: dict[str, object] = {
        "method": method,
        "plan": plan,
        "plan_key": canonical_plan(plan),
        "K": len(plan),
        "distance_raw": distance_raw,
        "objective_raw": econ_cost(plan, instance.distance, omega_vehicle),
        "calls": gate.calls,
        "gate_seconds": gate.gate_seconds,
        "total_seconds": total_seconds,
        "mean_route_length_checked": (
            gate.route_length_sum / gate.calls if gate.calls else 0.0
        ),
    }
    if isinstance(gate, CertifiedGate):
        result.update(
            {
                "coverage": gate.coverage,
                "certified_feasible": gate.certified_feasible,
                "certified_infeasible": gate.certified_infeasible,
                "fallbacks": gate.fallbacks,
                "certificate_seconds": gate.certificate_seconds,
                "fallback_seconds": gate.fallback_seconds,
                "audit_seconds": gate.audit_seconds,
                "audit_checks": gate.audit_checks,
                "mean_pieces": gate.pieces_sum / gate.calls if gate.calls else 0.0,
            }
        )
    else:
        result.update(
            {
                "coverage": 0.0,
                "certified_feasible": 0,
                "certified_infeasible": 0,
                "fallbacks": gate.calls,
                "certificate_seconds": 0.0,
                "fallback_seconds": gate.gate_seconds,
                "audit_seconds": 0.0,
                "audit_checks": 0,
                "mean_pieces": 0.0,
            }
        )
    return result


def assert_pair_identity(
    full: dict[str, object],
    cert: dict[str, object],
    context: str,
) -> None:
    if full["plan_key"] != cert["plan_key"]:
        raise AssertionError(f"{context}: FULL/CERT final plans differ")
    if full["calls"] != cert["calls"]:
        raise AssertionError(f"{context}: FULL/CERT route-check counts differ")
    if full["K"] != cert["K"]:
        raise AssertionError(f"{context}: FULL/CERT fleet sizes differ")
    if abs(float(full["distance_raw"]) - float(cert["distance_raw"])) > 1e-8:
        raise AssertionError(f"{context}: FULL/CERT distances differ")
    if abs(float(full["objective_raw"]) - float(cert["objective_raw"])) > 1e-8:
        raise AssertionError(f"{context}: FULL/CERT objectives differ")


def median_iqr(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    )


def benchmark_cell(
    instance: AffineInstance,
    profile: str,
    policy: str,
    capacity: float,
    alpha: float,
    cache: CertificateCache,
    iterations: int,
    repetitions: int,
    seed: int,
    audit_iterations: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    context = f"{instance.name}/{profile}/{policy}"

    # Separate untimed audit: every nonfallback certificate decision encountered
    # here is checked against the full multivariate CVaR value and bracket.
    audit_seed = seed + 771_001
    full_audit = run_solver_once(
        instance,
        capacity,
        alpha,
        cache,
        audit_iterations,
        audit_seed,
        "FULL",
        audit=False,
    )
    cert_audit = run_solver_once(
        instance,
        capacity,
        alpha,
        cache,
        audit_iterations,
        audit_seed,
        "CERT",
        audit=True,
    )
    assert_pair_identity(full_audit, cert_audit, context + "/audit")

    # Tiny warm-up, excluded from timing.
    if iterations > 0:
        warm_seed = seed + 880_003
        warm_iterations = min(1, iterations)
        warm_full = run_solver_once(
            instance, capacity, alpha, cache, warm_iterations, warm_seed, "FULL"
        )
        warm_cert = run_solver_once(
            instance, capacity, alpha, cache, warm_iterations, warm_seed, "CERT"
        )
        assert_pair_identity(warm_full, warm_cert, context + "/warmup")

    timing_order_rng = random.Random(seed + sum(map(ord, context)))
    run_rows: list[dict[str, object]] = []
    full_runs: list[dict[str, object]] = []
    cert_runs: list[dict[str, object]] = []

    for repetition in range(repetitions):
        run_seed = seed + 10_007 * repetition
        order = ["FULL", "CERT"]
        timing_order_rng.shuffle(order)
        current: dict[str, dict[str, object]] = {}
        for method in order:
            current[method] = run_solver_once(
                instance,
                capacity,
                alpha,
                cache,
                iterations,
                run_seed,
                method,
                audit=False,
            )
        full = current["FULL"]
        cert = current["CERT"]
        assert_pair_identity(full, cert, context + f"/rep{repetition}")

        gate_speedup = float(full["gate_seconds"]) / max(
            float(cert["gate_seconds"]), 1e-15
        )
        solver_speedup = float(full["total_seconds"]) / max(
            float(cert["total_seconds"]), 1e-15
        )
        solver_speedup_with_prep = float(full["total_seconds"]) / max(
            float(cert["total_seconds"]) + cache.prep_seconds,
            1e-15,
        )

        row = {
            "instance": instance.name,
            "source": instance.source,
            "profile": profile,
            "policy": policy,
            "repetition": repetition,
            "seed": run_seed,
            "N": instance.X.shape[0],
            "dimension": instance.X.shape[1],
            "alpha": alpha,
            "capacity": capacity,
            "iterations": iterations,
            "direction": cache.direction,
            "calls": cert["calls"],
            "coverage": cert["coverage"],
            "certified_feasible": cert["certified_feasible"],
            "certified_infeasible": cert["certified_infeasible"],
            "fallbacks": cert["fallbacks"],
            "mean_pieces": cert["mean_pieces"],
            "mean_route_length_checked": cert["mean_route_length_checked"],
            "K": cert["K"],
            "distance": float(cert["distance_raw"]) / instance.scale,
            "objective_raw": cert["objective_raw"],
            "full_gate_seconds": full["gate_seconds"],
            "cert_gate_seconds": cert["gate_seconds"],
            "full_solver_seconds": full["total_seconds"],
            "cert_solver_seconds": cert["total_seconds"],
            "gate_speedup": gate_speedup,
            "solver_speedup": solver_speedup,
            "solver_speedup_with_prep": solver_speedup_with_prep,
            "prep_seconds": cache.prep_seconds,
            "explained_variance": cache.explained_variance,
            "effective_rank": cache.effective_rank,
            "entropy_rank": cache.entropy_rank,
            "top_share": cache.top_share,
            "residual_cvar": cache.residual_cvar,
            "audit_checks": cert_audit["audit_checks"],
            "identity_pass": 1,
        }
        run_rows.append(row)
        full_runs.append(full)
        cert_runs.append(cert)
        print(
            f"        rep {repetition + 1}/{repetitions}: "
            f"gate={gate_speedup:.2f}x solver={solver_speedup:.2f}x "
            f"cover={float(cert['coverage']):.3f} "
            f"calls={int(cert['calls'])} assert=PASS"
        )

    gate_speedup, gate_speedup_iqr = median_iqr(
        float(row["gate_speedup"]) for row in run_rows
    )
    solver_speedup, solver_speedup_iqr = median_iqr(
        float(row["solver_speedup"]) for row in run_rows
    )
    solver_with_prep, solver_with_prep_iqr = median_iqr(
        float(row["solver_speedup_with_prep"]) for row in run_rows
    )
    coverage, coverage_iqr = median_iqr(
        float(row["coverage"]) for row in run_rows
    )
    full_gate, full_gate_iqr = median_iqr(
        float(run["gate_seconds"]) for run in full_runs
    )
    cert_gate, cert_gate_iqr = median_iqr(
        float(run["gate_seconds"]) for run in cert_runs
    )
    full_solver, full_solver_iqr = median_iqr(
        float(run["total_seconds"]) for run in full_runs
    )
    cert_solver, cert_solver_iqr = median_iqr(
        float(run["total_seconds"]) for run in cert_runs
    )
    reference = run_rows[0]
    median_calls = int(round(float(np.median([float(row["calls"]) for row in run_rows]))))
    median_cert_feasible = int(round(float(np.median([float(row["certified_feasible"]) for row in run_rows]))))
    median_cert_infeasible = int(round(float(np.median([float(row["certified_infeasible"]) for row in run_rows]))))
    median_fallbacks = int(round(float(np.median([float(row["fallbacks"]) for row in run_rows]))))
    median_pieces = float(np.median([float(row["mean_pieces"]) for row in run_rows]))
    median_route_length = float(np.median([float(row["mean_route_length_checked"]) for row in run_rows]))
    median_K = int(round(float(np.median([float(row["K"]) for row in run_rows]))))
    median_distance = float(np.median([float(row["distance"]) for row in run_rows]))
    median_objective = float(np.median([float(row["objective_raw"]) for row in run_rows]))

    cell_row = {
        "instance": instance.name,
        "source": instance.source,
        "profile": profile,
        "policy": policy,
        "N": instance.X.shape[0],
        "dimension": instance.X.shape[1],
        "alpha": alpha,
        "capacity": capacity,
        "iterations": iterations,
        "repetitions": repetitions,
        "direction": cache.direction,
        "prep_seconds": cache.prep_seconds,
        "explained_variance": cache.explained_variance,
        "effective_rank": cache.effective_rank,
        "entropy_rank": cache.entropy_rank,
        "top_share": cache.top_share,
        "residual_cvar": cache.residual_cvar,
        "calls": median_calls,
        "coverage": coverage,
        "coverage_iqr": coverage_iqr,
        "certified_feasible": median_cert_feasible,
        "certified_infeasible": median_cert_infeasible,
        "fallbacks": median_fallbacks,
        "mean_pieces": median_pieces,
        "mean_route_length_checked": median_route_length,
        "K": median_K,
        "distance": median_distance,
        "objective_raw": median_objective,
        "full_gate_seconds": full_gate,
        "full_gate_iqr": full_gate_iqr,
        "cert_gate_seconds": cert_gate,
        "cert_gate_iqr": cert_gate_iqr,
        "gate_speedup": gate_speedup,
        "gate_speedup_iqr": gate_speedup_iqr,
        "full_solver_seconds": full_solver,
        "full_solver_iqr": full_solver_iqr,
        "cert_solver_seconds": cert_solver,
        "cert_solver_iqr": cert_solver_iqr,
        "solver_speedup": solver_speedup,
        "solver_speedup_iqr": solver_speedup_iqr,
        "solver_speedup_with_prep": solver_with_prep,
        "solver_speedup_with_prep_iqr": solver_with_prep_iqr,
        "audit_checks": cert_audit["audit_checks"],
        "identity_pass": 1,
    }
    return cell_row, run_rows


# ============================================================================
# CSV, decision rule, and CLI
# ============================================================================


RUN_FIELDS = [
    "instance", "source", "profile", "policy", "repetition", "seed",
    "N", "dimension", "alpha", "capacity", "iterations", "direction",
    "calls", "coverage", "certified_feasible", "certified_infeasible",
    "fallbacks", "mean_pieces", "mean_route_length_checked", "K",
    "distance", "objective_raw", "full_gate_seconds", "cert_gate_seconds",
    "full_solver_seconds", "cert_solver_seconds", "gate_speedup",
    "solver_speedup", "solver_speedup_with_prep", "prep_seconds",
    "explained_variance", "effective_rank", "entropy_rank", "top_share",
    "residual_cvar", "audit_checks", "identity_pass",
]

CELL_FIELDS = [
    "instance", "source", "profile", "policy", "N", "dimension", "alpha",
    "capacity", "iterations", "repetitions", "direction", "prep_seconds",
    "explained_variance", "effective_rank", "entropy_rank", "top_share",
    "residual_cvar", "calls", "coverage", "coverage_iqr",
    "certified_feasible", "certified_infeasible", "fallbacks", "mean_pieces",
    "mean_route_length_checked", "K", "distance", "objective_raw",
    "full_gate_seconds", "full_gate_iqr", "cert_gate_seconds", "cert_gate_iqr",
    "gate_speedup", "gate_speedup_iqr", "full_solver_seconds",
    "full_solver_iqr", "cert_solver_seconds", "cert_solver_iqr",
    "solver_speedup", "solver_speedup_iqr", "solver_speedup_with_prep",
    "solver_speedup_with_prep_iqr", "audit_checks", "identity_pass",
]


def write_rows(path: str | Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rewrite_rows(path: str | Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _norm_label(value: object) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _cell_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        _norm_label(row.get("instance", "")),
        str(row.get("profile", "")).strip().lower(),
        str(row.get("policy", "")).strip().upper(),
    )


def _validate_resume_row(row: dict[str, object], config: dict[str, object]) -> None:
    checks = [
        ("N", int(float(row["N"])), int(config["N"])),
        ("dimension", int(float(row["dimension"])), int(config["d"])),
        ("iterations", int(float(row["iterations"])), int(config["iters"])),
        ("repetitions", int(float(row["repetitions"])), int(config["reps"])),
        ("direction", str(row["direction"]).upper(), str(config["direction"]).upper()),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"resume output is incompatible: {name}={actual!r}, expected {expected!r}"
            )
    if not math.isclose(float(row["alpha"]), float(config["alpha"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"resume output is incompatible: alpha={row['alpha']}, expected {config['alpha']}"
        )


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: Sequence[str]) -> dict[str, object]:
    mode = "quick"
    arguments = list(argv)
    if arguments and "=" not in arguments[0]:
        mode = arguments.pop(0).lower()
    if mode not in MODE_PRESETS:
        raise ValueError("first argument must be smoke, quick, or full")

    preset = MODE_PRESETS[mode]
    config: dict[str, object] = {
        "mode": mode,
        "dir": "Dethloff",
        "instances": list(preset["instances"]),
        "profiles": list(preset["profiles"]),
        "policy": preset["policy"],
        "N": preset["N"],
        "d": preset["d"],
        "iters": preset["iters"],
        "reps": preset["reps"],
        "train": preset["train"],
        "audit_iters": preset["audit_iters"],
        "direction": "SLOPE",
        "seed": SEED,
        "cv": CV,
        "alpha": ALPHA,
        "eps": EPS_FRAC,
        "support": SUPPORT,
        "out": f"go_nogo_orl_cert_{mode}",
        "overwrite": True,
        "resume": False,
        **DEFAULT_THRESHOLDS,
    }

    integer_keys = {"N", "d", "iters", "reps", "train", "audit_iters", "seed"}
    float_keys = {
        "cv", "alpha", "eps", "support",
        "strong_median_gate", "strong_median_solver",
        "strong_solver_win_rate", "strong_median_coverage",
        "weak_median_gate", "weak_median_solver",
        "weak_solver_win_rate", "weak_median_coverage",
    }
    string_keys = {"dir", "policy", "direction", "out"}

    for argument in arguments:
        if "=" not in argument:
            raise ValueError(f"Unknown argument: {argument}")
        key, value = argument.split("=", 1)
        if key in integer_keys:
            config[key] = int(value)
        elif key in float_keys:
            config[key] = float(value)
        elif key in string_keys:
            config[key] = value
        elif key in {"instances", "profiles"}:
            config[key] = parse_list(value)
        elif key in {"overwrite", "resume"}:
            config[key] = parse_bool(value)
        else:
            raise ValueError(f"Unknown argument: {argument}")

    config["policy"] = str(config["policy"]).upper()
    config["direction"] = str(config["direction"]).upper()
    config["profiles"] = [str(value).lower() for value in config["profiles"]]
    config["instances"] = [str(value) for value in config["instances"]]

    if config["policy"] not in {"SAA", "WDRO", "BOTH"}:
        raise ValueError("policy must be SAA, WDRO, or BOTH")
    if config["direction"] not in {"SLOPE", "PCA", "DELIVERY"}:
        raise ValueError("direction must be SLOPE, PCA, or DELIVERY")
    for profile in config["profiles"]:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
    if int(config["N"]) <= 0 or int(config["d"]) < 2:
        raise ValueError("N must be positive and d must be at least 2")
    if int(config["iters"]) < 0 or int(config["reps"]) <= 0:
        raise ValueError("iters must be nonnegative and reps positive")
    if int(config["audit_iters"]) < 0:
        raise ValueError("audit_iters must be nonnegative")
    if not (0.0 < float(config["alpha"]) < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    if not (0.0 <= float(config["eps"]) < 1.0):
        raise ValueError("eps must lie in [0,1)")
    return config


def discover_files(data_dir: str | Path, requested: Sequence[str]) -> list[Path]:
    root = Path(data_dir)
    all_files = sorted(root.rglob("*.vrpspd"))
    if not all_files:
        raise FileNotFoundError(f"No .vrpspd files found under {root}")
    if any(name.upper() == "ALL" for name in requested):
        return all_files

    selected: list[Path] = []
    missing: list[str] = []
    for name in requested:
        target = Path(name).stem.lower()
        matches = [path for path in all_files if path.stem.lower() == target]
        if not matches:
            missing.append(name)
            continue
        if len(matches) > 1:
            print(
                f"WARNING: {name} matched {len(matches)} files; all will be run: "
                + ", ".join(str(path.relative_to(root)) for path in matches)
            )
        selected.extend(matches)
    if missing:
        raise FileNotFoundError("Missing requested instances: " + ", ".join(missing))

    # Stable de-duplication.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def self_test() -> None:
    rng = np.random.default_rng(731)

    # Fractional CVaR against a direct sorted implementation.
    values = rng.normal(size=137)
    for alpha in [0.0, 0.5, 0.9, 0.913, 0.99]:
        tail_mass, count, weight = tail_parameters(alpha, len(values))
        ordered = np.sort(values)[::-1]
        direct = (ordered[: count - 1].sum() + weight * ordered[count - 1]) / tail_mass
        fast = empirical_cvar(values, alpha)
        if not np.isclose(direct, fast, atol=1e-12, rtol=1e-12):
            raise AssertionError(f"fractional CVaR self-test failed at alpha={alpha}")

    # ORL max-affine scalar evaluator against a direct scenario pass.
    z = np.sort(rng.normal(size=401))
    prefix = np.concatenate(([0.0], np.cumsum(z)))
    for _ in range(40):
        lines = [(float(rng.normal()), float(rng.normal())) for _ in range(12)]
        envelope = upper_envelope(lines)
        direct_values = np.max(
            np.asarray([a + b * z for a, b in lines]),
            axis=0,
        )
        for alpha in [0.0, 0.73, 0.9, 0.937]:
            direct = empirical_cvar(direct_values, alpha)
            exact = cvar_os(envelope, z, prefix, alpha)
            if not np.isclose(direct, exact, atol=2e-10, rtol=2e-10):
                raise AssertionError(
                    f"O(p log N) evaluator self-test failed: direct={direct}, exact={exact}"
                )

    # Generic multivariate projection bracket.
    N, d, K = 503, 6, 9
    X = rng.normal(size=(N, d))
    X -= X.mean(axis=0, keepdims=True)
    a = rng.normal(size=K)
    b = rng.normal(size=(K, d))
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    z_raw = X @ u
    order = np.argsort(z_raw)
    z_sorted = z_raw[order]
    prefix = np.concatenate(([0.0], np.cumsum(z_sorted)))
    envelope = upper_envelope(zip(a, b @ u))
    projected = cvar_os(envelope, z_sorted, prefix, 0.9)
    residual = X - np.outer(z_raw, u)
    ce = empirical_cvar(np.linalg.norm(residual, axis=1), 0.9)
    gamma = np.sqrt(np.maximum(np.sum(b * b, axis=1) - (b @ u) ** 2, 0.0)).max()
    full = empirical_cvar(np.max(a[None, :] + X @ b.T, axis=1), 0.9)
    radius = gamma * ce
    tolerance = 2e-10 * max(1.0, abs(full), abs(projected), abs(radius))
    if full < projected - radius - tolerance or full > projected + radius + tolerance:
        raise AssertionError("multivariate CVaR bracket self-test failed")

    print(
        "SELFTEST PASS: fractional CVaR, exact O(p log N) max-affine CVaR, "
        "and multivariate projection bracket"
    )


def aggregate_decision(
    cell_rows: Sequence[dict[str, object]],
    thresholds: dict[str, float],
) -> tuple[str, dict[str, float], list[str]]:
    gate = np.asarray([float(row["gate_speedup"]) for row in cell_rows], dtype=float)
    solver = np.asarray([float(row["solver_speedup"]) for row in cell_rows], dtype=float)
    solver_with_prep = np.asarray(
        [float(row["solver_speedup_with_prep"]) for row in cell_rows],
        dtype=float,
    )
    coverage = np.asarray([float(row["coverage"]) for row in cell_rows], dtype=float)
    identities = np.asarray([int(row["identity_pass"]) for row in cell_rows], dtype=int)

    metrics = {
        "cells": float(len(cell_rows)),
        "identity_rate": float(np.mean(identities)) if identities.size else 0.0,
        "median_gate_speedup": float(np.median(gate)),
        "min_gate_speedup": float(np.min(gate)),
        "median_solver_speedup": float(np.median(solver)),
        "min_solver_speedup": float(np.min(solver)),
        "solver_win_rate": float(np.mean(solver > 1.0)),
        "median_solver_speedup_with_prep": float(np.median(solver_with_prep)),
        "median_coverage": float(np.median(coverage)),
        "min_coverage": float(np.min(coverage)),
    }

    reasons: list[str] = []
    exact = metrics["identity_rate"] == 1.0
    strong = (
        exact
        and metrics["median_gate_speedup"] >= thresholds["strong_median_gate"]
        and metrics["median_solver_speedup"] >= thresholds["strong_median_solver"]
        and metrics["solver_win_rate"] >= thresholds["strong_solver_win_rate"]
        and metrics["median_coverage"] >= thresholds["strong_median_coverage"]
    )
    weak = (
        exact
        and metrics["median_gate_speedup"] >= thresholds["weak_median_gate"]
        and metrics["median_solver_speedup"] >= thresholds["weak_median_solver"]
        and metrics["solver_win_rate"] >= thresholds["weak_solver_win_rate"]
        and metrics["median_coverage"] >= thresholds["weak_median_coverage"]
    )

    if not exact:
        reasons.append("decision/search identity failed")
        return "NO-GO", metrics, reasons
    if strong:
        reasons.append("decision-exact and clear end-to-end acceleration")
        return "GO", metrics, reasons
    if weak:
        reasons.append("positive but not yet strong enough; run the full benchmark")
        return "BORDERLINE-GO", metrics, reasons

    if metrics["median_gate_speedup"] < thresholds["weak_median_gate"]:
        reasons.append("certificate gate is not faster enough")
    if metrics["median_solver_speedup"] < thresholds["weak_median_solver"]:
        reasons.append("end-to-end solver acceleration is too small")
    if metrics["solver_win_rate"] < thresholds["weak_solver_win_rate"]:
        reasons.append("too few cells accelerate")
    if metrics["median_coverage"] < thresholds["weak_median_coverage"]:
        reasons.append("certificate coverage is too low")
    return "NO-GO", metrics, reasons


def format_summary(
    status: str,
    metrics: dict[str, float],
    reasons: Sequence[str],
    config: dict[str, object],
    failed_cells: int,
) -> str:
    lines = [
        "=" * 92,
        "GO / NO-GO SUMMARY — CERTIFIED MULTIVARIATE SCREENING WITH ORL KERNEL",
        "=" * 92,
        f"STATUS: {status}",
        f"mode={config['mode']}  cells={int(metrics.get('cells', 0))}  failed_cells={failed_cells}",
        f"identity rate:                  {metrics.get('identity_rate', float('nan')):.3f}",
        f"median gate speed-up:           {metrics.get('median_gate_speedup', float('nan')):.3f}x",
        f"minimum gate speed-up:          {metrics.get('min_gate_speedup', float('nan')):.3f}x",
        f"median solver speed-up:         {metrics.get('median_solver_speedup', float('nan')):.3f}x",
        f"minimum solver speed-up:        {metrics.get('min_solver_speedup', float('nan')):.3f}x",
        f"solver win rate:                {metrics.get('solver_win_rate', float('nan')):.3f}",
        f"median solver speed-up + prep:  {metrics.get('median_solver_speedup_with_prep', float('nan')):.3f}x",
        f"median certification coverage:  {metrics.get('median_coverage', float('nan')):.3f}",
        f"minimum certification coverage: {metrics.get('min_coverage', float('nan')):.3f}",
        "-" * 92,
        "Reason: " + ("; ".join(reasons) if reasons else "n/a"),
    ]
    if config["mode"] == "smoke":
        lines.append(
            "NOTE: smoke mode is a correctness/sanity run; use quick before making the research decision."
        )
    lines.extend(
        [
            "-" * 92,
            "Interpretation:",
            "  GO            = enough signal for a conference-sized paper and full benchmark.",
            "  BORDERLINE-GO = keep the direction, but confirm on all instances/profiles.",
            "  NO-GO         = do not write the paper around computational acceleration.",
            "=" * 92,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    try:
        config = parse_args(sys.argv[1:])
    except Exception as error:
        print("ARGUMENT ERROR:", error)
        raise SystemExit(2)

    self_test()

    data_dir = Path(str(config["dir"]))
    files = discover_files(data_dir, config["instances"])
    policies = ["SAA", "WDRO"] if config["policy"] == "BOTH" else [str(config["policy"])]

    prefix = str(config["out"])
    runs_path = Path(prefix + "_runs.csv")
    cells_path = Path(prefix + "_cells.csv")
    summary_path = Path(prefix + "_summary.txt")
    resume = bool(config["resume"])
    if resume and bool(config["overwrite"]):
        print("RESUME: forcing overwrite=0 so completed cells are preserved.")
        config["overwrite"] = False
    if bool(config["overwrite"]):
        for path in [runs_path, cells_path, summary_path]:
            if path.exists():
                path.unlink()

    print("\n" + "=" * 112)
    print("CERTIFIED MULTIVARIATE CVaR GO/NO-GO — exact ORL projection kernel + exact fallback")
    print("=" * 112)
    print(
        f"mode={config['mode']} files={len(files)} profiles={','.join(config['profiles'])} "
        f"policies={','.join(policies)} N={config['N']} d={config['d']} "
        f"iters={config['iters']} reps={config['reps']} direction={config['direction']} "
        f"resume={int(resume)}"
    )
    print(
        f"alpha={config['alpha']} cv={config['cv']} support={config['support']} "
        f"WDRO eps_frac={config['eps']} train_routes={config['train']}"
    )
    print("Protocol: separate exactness audit, randomized paired timing order, different seed per repetition.")
    print("-" * 112)

    expected_instances = {
        _norm_label(str(path.relative_to(data_dir).with_suffix("")))
        for path in files
    }
    allowed_keys = {
        (instance_name, profile, policy)
        for instance_name in expected_instances
        for profile in config["profiles"]
        for policy in policies
    }

    cell_rows: list[dict[str, object]] = []
    completed_keys: set[tuple[str, str, str]] = set()
    if resume:
        loaded_cells = read_rows(cells_path)
        deduped_cells: dict[tuple[str, str, str], dict[str, object]] = {}
        for row in loaded_cells:
            key = _cell_key(row)
            if key not in allowed_keys:
                continue
            _validate_resume_row(row, config)
            deduped_cells[key] = row
        cell_rows = list(deduped_cells.values())
        completed_keys = set(deduped_cells)

        # Remove duplicate/orphan run rows left by an interrupted write.  A run
        # is retained only when its aggregate cell row exists.
        loaded_runs = read_rows(runs_path)
        deduped_runs: dict[tuple[tuple[str, str, str], int], dict[str, object]] = {}
        for row in loaded_runs:
            key = _cell_key(row)
            if key not in completed_keys:
                continue
            repetition = int(float(row.get("repetition", -1)))
            deduped_runs[(key, repetition)] = row
        if loaded_cells or loaded_runs:
            rewrite_rows(cells_path, cell_rows, CELL_FIELDS)
            rewrite_rows(runs_path, list(deduped_runs.values()), RUN_FIELDS)
        if summary_path.exists():
            summary_path.unlink()
        print(
            f"RESUME: loaded {len(cell_rows)} completed cells; "
            f"{len(allowed_keys) - len(completed_keys)} remain."
        )

    failed_cells = 0
    total_cells = len(files) * len(config["profiles"]) * len(policies)
    cell_index = 0

    for file_index, file_path in enumerate(files, 1):
        relative = file_path.relative_to(data_dir)
        print(f"\n[{file_index}/{len(files)}] {relative}")
        for profile_name in config["profiles"]:
            profile = PROFILES[profile_name]
            print(f"    profile={profile_name}")
            try:
                scenario_start = time.perf_counter()
                # Stable but instance-specific overlay seed.
                overlay_seed = int(config["seed"]) + sum(map(ord, str(relative)))
                instance = make_affine_instance(
                    file_path,
                    profile,
                    int(config["N"]),
                    int(config["d"]),
                    float(config["cv"]),
                    float(config["support"]),
                    overlay_seed,
                    data_dir,
                )
                scenario_seconds = time.perf_counter() - scenario_start
                cache = build_certificate_cache(
                    instance,
                    float(config["alpha"]),
                    str(config["direction"]),
                    int(config["train"]),
                    int(config["seed"]) + 1000,
                )
                print(
                    f"      scenario={scenario_seconds:.2f}s prep={cache.prep_seconds:.2f}s "
                    f"EV1={cache.explained_variance:.3f} effective_rank={cache.effective_rank:.3f} "
                    f"top_share={cache.top_share:.3f}"
                )

                for policy in policies:
                    cell_index += 1
                    capacity = (
                        instance.capacity
                        if policy == "SAA"
                        else instance.capacity * (1.0 - float(config["eps"]))
                    )
                    key = (_norm_label(instance.name), profile_name, policy)
                    if key in completed_keys:
                        print(
                            f"      [{cell_index}/{total_cells}] {policy}: "
                            f"capacity={capacity:.2f} SKIP (already complete)"
                        )
                        continue
                    print(
                        f"      [{cell_index}/{total_cells}] {policy}: capacity={capacity:.2f}"
                    )
                    try:
                        cell, runs = benchmark_cell(
                            instance,
                            profile_name,
                            policy,
                            capacity,
                            float(config["alpha"]),
                            cache,
                            int(config["iters"]),
                            int(config["reps"]),
                            int(config["seed"]) + 50_000 * cell_index,
                            int(config["audit_iters"]),
                        )
                        write_rows(runs_path, runs, RUN_FIELDS)
                        write_rows(cells_path, [cell], CELL_FIELDS)
                        cell_rows.append(cell)
                        completed_keys.add(key)
                        print(
                            f"      => gate={float(cell['gate_speedup']):.2f}x "
                            f"solver={float(cell['solver_speedup']):.2f}x "
                            f"cover={float(cell['coverage']):.3f} "
                            f"calls={int(cell['calls'])} K={int(cell['K'])} "
                            f"dist={float(cell['distance']):.2f} assert=PASS"
                        )
                    except Exception as error:
                        failed_cells += 1
                        print(f"      CELL ERROR: {type(error).__name__}: {error}")
                        import traceback
                        traceback.print_exc()
            except Exception as error:
                # All policy cells for this profile are failed.
                failed_cells += len(policies)
                cell_index += len(policies)
                print(f"    PROFILE ERROR: {type(error).__name__}: {error}")
                import traceback
                traceback.print_exc()

    if not cell_rows:
        raise SystemExit("No successful cells; cannot make a GO/NO-GO decision")

    thresholds = {
        key: float(config[key])
        for key in DEFAULT_THRESHOLDS
    }
    status, metrics, reasons = aggregate_decision(cell_rows, thresholds)
    if failed_cells:
        status = "NO-GO"
        reasons = list(reasons) + [f"{failed_cells} benchmark cell(s) failed"]

    if config["mode"] == "smoke":
        if failed_cells or metrics.get("identity_rate", 0.0) < 1.0:
            status = "SMOKE-FAIL"
            reasons = list(reasons) + ["smoke correctness check failed"]
        else:
            status = "SMOKE-PASS"
            reasons = ["correctness and paired-search identity passed; run quick for the research decision"]

    summary = format_summary(status, metrics, reasons, config, failed_cells)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"wrote {runs_path}")
    print(f"wrote {cells_path}")
    print(f"wrote {summary_path}")

    # Nonzero exit only for correctness/runtime failures, not for a scientific NO-GO.
    if failed_cells:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
