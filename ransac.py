"""
Python port of kpt::RANSACZ and homography helpers (ransac.cpp / ransac.hpp).

- ``ransac_classic``: 4-point RANSAC with default :class:`UniformSampler4`; no convexity or
  edge pre-checks (no OpenCV required for those steps).
- ``ransac_optimized``: same pipeline with default :class:`RectangularRandomSampler4` and
  geometry filters (requires OpenCV for convex hull).
"""

from __future__ import annotations

import math
from typing import List, Protocol, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from sampler import RectangularRandomSampler4, Sampler4, UniformSampler4


def homography_mul(H: np.ndarray, p: Tuple[float, float]) -> Tuple[float, float]:
    """Apply 3x3 homography to 2D point (inhomogeneous). H must be float64 3x3."""
    x, y = float(p[0]), float(p[1])
    m = H.reshape(3, 3).astype(np.float64)
    xh = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    yh = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    zh = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return (float(xh / zh), float(yh / zh))


def _points_to_xy_array(
    src: Sequence[Tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return np.asarray(src, dtype=np.float64).reshape(-1, 2)
    return np.array([[float(p[0]), float(p[1])] for p in src], dtype=np.float64)


def argsort_indices(points: np.ndarray, indices: List[int]) -> None:
    """Sort indices in-place by polar angle around centroid of points[indices]."""
    n = len(indices)
    if n == 0:
        return
    sel = points[np.array(indices, dtype=np.intp)]
    center = sel.mean(axis=0)
    angles = np.degrees(np.arctan2(sel[:, 1] - center[1], sel[:, 0] - center[0])) + 360.0
    order = np.argsort(angles, kind="stable")
    indices[:] = [indices[i] for i in order]


def is_convex(points: np.ndarray, indices: Sequence[int]) -> bool:
    if cv2 is None:
        raise ImportError("is_convex requires opencv-python (cv2)")
    n = len(indices)
    if n == 0:
        return False
    pts = np.array([[points[i, 0], points[i, 1]] for i in indices], dtype=np.float32)
    hull = cv2.convexHull(pts)
    return int(hull.shape[0]) == n


def check_edge_lengths(indices: Sequence[int], src: np.ndarray, bound: float) -> bool:
    s = len(indices)
    b2 = bound * bound
    for s_i in range(s):
        i0 = indices[s_i]
        i1 = indices[(s_i + 1) % s]
        d = src[i1] - src[i0]
        if float(d.dot(d)) < b2:
            return False
    return True


def check_self_intersection(k: Sequence[int], src: np.ndarray) -> bool:
    t = list(k)
    argsort_indices(src, t)
    k0 = k[0]
    n = len(k)
    idx1 = next((i for i in range(n) if t[i] == k0), 0)
    for i in range(1, n):
        if k[i] != t[(i + idx1) % n]:
            return False
    return True


def calculate_normalization_matrix(win_size: Tuple[int, int]) -> np.ndarray:
    radius = 2.0
    tx, ty = float(win_size[0]), float(win_size[1])
    assert ty > 1 and tx > 1
    s_scale = radius / ty
    dx = -tx * s_scale / 2.0
    dy = -radius / 2.0
    return np.array([[s_scale, 0.0, dx], [0.0, s_scale, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_to_circle(src: np.ndarray, T: np.ndarray) -> np.ndarray:
    s = float(T[0, 0])
    tx = float(T[0, 2])
    ty = float(T[1, 2])
    out = np.empty_like(src, dtype=np.float64)
    out[:, 0] = src[:, 0] * s + tx
    out[:, 1] = src[:, 1] * s + ty
    return out


def ata(A: np.ndarray) -> np.ndarray:
    """A.T @ A for DLT stacked rows (same result as optimized C++ AtA)."""
    return A.T @ A


def find_homography_9x9_lms(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography from SVD on A.T@A (minimal singular vector); h33 normalized to 1 after."""
    n = src.shape[0]
    assert dst.shape[0] == n and n >= 4
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = float(src[i, 0]), float(src[i, 1])
        X, Y = float(dst[i, 0]), float(dst[i, 1])
        A[2 * i] = [x, y, 1, 0, 0, 0, -X * x, -X * y, -X]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y]
    a1 = ata(A)
    _, _, vh = np.linalg.svd(a1, full_matrices=True)
    x = vh[-1, :].reshape(9, 1)
    H = x.reshape(3, 3)
    H = H / H[2, 2]
    return H


def find_homography_9x9_linear(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT with constraint h33 = 1 via extra linear row."""
    n = src.shape[0]
    assert dst.shape[0] == n and n >= 4
    A = np.zeros((2 * n + 1, 9), dtype=np.float64)
    for i in range(n):
        x, y = float(src[i, 0]), float(src[i, 1])
        X, Y = float(dst[i, 0]), float(dst[i, 1])
        A[2 * i] = [x, y, 1, 0, 0, 0, -X * x, -X * y, -X]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y]
    A[-1, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    b = np.zeros((2 * n + 1, 1), dtype=np.float64)
    b[-1, 0] = 1.0
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = x.reshape(3, 3)
    H = H / H[2, 2]
    return H


def find_homography_8x8(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Affine-part DLT with h33 fixed to 1 (8 DOF)."""
    n = src.shape[0]
    assert dst.shape[0] == n and n >= 4
    A = np.zeros((2 * n, 8), dtype=np.float64)
    b = np.zeros((2 * n, 1), dtype=np.float64)
    for i in range(n):
        x, y = float(src[i, 0]), float(src[i, 1])
        X, Y = float(dst[i, 0]), float(dst[i, 1])
        A[2 * i] = [x, y, 1, 0, 0, 0, -X * x, -X * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -Y * x, -Y * y]
        b[2 * i] = X
        b[2 * i + 1] = Y
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.zeros((3, 3), dtype=np.float64)
    H.flat[:8] = x.flat
    H[2, 2] = 1.0
    return H


def estimate_inliers(
    H: np.ndarray,
    h_inv: np.ndarray,
    scale_s: float,
    a: np.ndarray,
    b: np.ndarray,
    estimate_by_lms: bool,
    t_dist: float,
    max_inliers_hint: int,
    mask: np.ndarray,
) -> Tuple[bool, int, float]:
    n = a.shape[0]
    u_dist = 2.0 * t_dist
    distances = np.zeros(n, dtype=np.float64)
    mean_i = 0.0
    inliers = 0

    for i in range(n):
        fp = np.array(homography_mul(H, (a[i, 0], a[i, 1])), dtype=np.float64)
        bp = np.array(homography_mul(h_inv, (b[i, 0], b[i, 1])), dtype=np.float64)
        ef = fp - b[i]
        eb = bp - a[i]
        d0 = float(np.linalg.norm(ef))
        d1 = float(np.linalg.norm(eb))
        dist = min(d0, d1)
        flag = dist < t_dist
        mask[i] = 255 if flag else 0

        if not flag and dist < u_dist:
            v0 = scale_s * (b[i] - a[i])
            v1 = scale_s * (fp - a[i])
            d0n = float(np.linalg.norm(v0))
            d1n = float(np.linalg.norm(v1))
            sc = d0n * d1n
            if d0n > 1.0 and d1n > 1.0 and sc > 0:
                c = float(np.dot(v0, v1)) / sc
                c = max(-1.0, min(1.0, c))
                ang = math.degrees(math.acos(c))
                if ang < 0:
                    ang += 360.0
                r = d0n / d1n if d0n < d1n else d1n / d0n
                reps = 0.1
                aeps = 5.0
                if r > (1.0 - reps) and ang < aeps:
                    flag = True
                    mask[i] = 200

        distances[i] = dist
        if flag:
            mean_i += dist
            inliers += 1

    if inliers <= 3:
        return False, inliers, 0.0
    mean_i /= inliers
    stddev = 0.0
    for i in range(n):
        if mask[i]:
            d = distances[i] - mean_i
            stddev += d * d
    stddev = float(stddev / max(inliers - 1, 1))
    return True, inliers, stddev


def dlte_estimate(
    estimate_by_lms: bool,
    scale_s: float,
    a: np.ndarray,
    b: np.ndarray,
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    t_dist: float,
    max_inliers_hint: int,
) -> Tuple[np.ndarray | None, int, float, np.ndarray]:
    if estimate_by_lms:
        H = find_homography_9x9_lms(samples_a, samples_b)
    else:
        H = find_homography_8x8(samples_a, samples_b)
    try:
        h_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, 0, 0.0, np.zeros(len(a), dtype=np.uint8)

    mask = np.zeros(len(a), dtype=np.uint8)
    ok, inliers, stddev = estimate_inliers(
        H, h_inv, scale_s, a, b, estimate_by_lms, t_dist, max_inliers_hint, mask
    )
    if not ok:
        return None, inliers, stddev, mask
    return H, inliers, stddev, mask


class Sampler4Protocol(Protocol):
    def init(self, samples: Sequence[Tuple[float, float]], win_size: Tuple[int, int]) -> None: ...
    def sample(self, k: List[int]) -> None: ...


def _ransac_homography_core(
    sampler: Sampler4Protocol,
    src: Sequence[Tuple[float, float]] | np.ndarray,
    dst: Sequence[Tuple[float, float]] | np.ndarray,
    win_size: Tuple[int, int],
    *,
    hint: np.ndarray | None,
    refine: bool,
    geometry_filters: bool,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Shared 4-point RANSAC homography (normalized coordinates, DLT + optional refine).

    If ``geometry_filters`` is True, applies argsort, edge-length, convexity, and
    self-intersection checks (requires OpenCV). If False, only ``sampler.sample`` is used.
    """
    if geometry_filters and cv2 is None:
        raise ImportError(
            "ransac_optimized requires opencv-python (cv2) for convex hull checks"
        )

    s_count = 4
    p_target = 0.99
    t_dist = 0.091
    estimate_by_lms = True

    src_a = _points_to_xy_array(src)
    dst_a = _points_to_xy_array(dst)
    n = src_a.shape[0]
    if n != dst_a.shape[0] or n < s_count:
        return None, np.array([], dtype=np.uint8)

    max_mask = np.full(n, 255, dtype=np.uint8)

    T = calculate_normalization_matrix(win_size)
    scale = 1.0 / float(T[0, 0])
    t_inv = np.linalg.inv(T)

    a = project_to_circle(src_a, T)
    b = project_to_circle(dst_a, T)

    min_std = 1e9
    max_inliers = -1
    best_h: np.ndarray | None = None
    hi: np.ndarray | None = None

    max_iterations = min(20, n - s_count + 1)
    mask = np.zeros(n, dtype=np.uint8)

    if hint is not None and hint.size > 0:
        h_prob = T @ hint.reshape(3, 3) @ t_inv
        try:
            h_inv = np.linalg.inv(h_prob)
        except np.linalg.LinAlgError:
            h_inv = None
        if h_inv is not None:
            ok, inl, stdv = estimate_inliers(
                h_prob, h_inv, scale, a, b, estimate_by_lms, t_dist, max_inliers, mask
            )
            if ok:
                max_inliers = inl
                min_std = stdv
                max_mask = mask.copy()
                hi = best_h = h_prob.copy()

    ht, inl0, std0, mask = dlte_estimate(
        estimate_by_lms, scale, a, b, a, b, t_dist, 0
    )
    if ht is not None:
        if inl0 > max_inliers or (inl0 == max_inliers and min_std > std0):
            max_inliers = inl0
            min_std = std0
            max_mask = mask.copy()
            hi = best_h = ht

    if hi is None:
        return None, max_mask

    e = 1.0 - float(max_inliers) / n
    if e > 1e-12:
        inner = 1.0 - (1.0 - e) ** s_count
        if inner > 1e-15 and inner < 1.0:
            denom = math.log(inner)
            if denom < -1e-15:
                max_iterations = min(max_iterations, int(math.log(1.0 - p_target) / denom))

    sampler.init([(float(src_a[i, 0]), float(src_a[i, 1])) for i in range(n)], win_size)
    k: List[int] = [0, 1, 2, 3]
    bound = win_size[1] / 20.0

    iteration = 0
    while iteration < max_iterations:
        sampler.sample(k)

        if geometry_filters:
            argsort_indices(src_a, k)
            if not check_edge_lengths(k, src_a, bound):
                iteration += 1
                continue
            if not is_convex(src_a, k):
                iteration += 1
                continue
            if not check_edge_lengths(k, dst_a, bound):
                iteration += 1
                continue
            if not is_convex(dst_a, k):
                iteration += 1
                continue
            if not check_self_intersection(k, dst_a):
                iteration += 1
                continue

        samples_a = np.array([a[i] for i in k], dtype=np.float64)
        samples_b = np.array([b[i] for i in k], dtype=np.float64)
        H, inl, stdv, mask = dlte_estimate(
            estimate_by_lms, scale, a, b, samples_a, samples_b, t_dist, max_inliers
        )
        if H is None:
            iteration += 1
            continue
        if inl > max_inliers or (inl == max_inliers and min_std > stdv):
            max_inliers = inl
            min_std = stdv
            max_mask = mask.copy()
            best_h = H
            e2 = 1.0 - float(inl) / n
            if e2 > 1e-12:
                inner2 = 1.0 - (1.0 - e2) ** s_count
                if inner2 > 1e-15 and inner2 < 1.0:
                    denom2 = math.log(inner2)
                    if denom2 < -1e-15:
                        max_iterations = min(
                            max_iterations, int(math.log(1.0 - p_target) / denom2)
                        )
        iteration += 1

    if refine and max_inliers >= s_count and best_h is not None:
        samples_a = a[max_mask.astype(bool)]
        samples_b = b[max_mask.astype(bool)]
        H2, _, _, mask2 = dlte_estimate(
            estimate_by_lms, scale, a, b, samples_a, samples_b, t_dist, 0
        )
        if H2 is not None:
            best_h = H2
            max_mask = mask2

    if best_h is None:
        return None, max_mask

    ho = t_inv @ best_h @ T
    ho = ho / ho[2, 2]
    return ho, max_mask


def ransac_classic(
    src: Sequence[Tuple[float, float]] | np.ndarray,
    dst: Sequence[Tuple[float, float]] | np.ndarray,
    win_size: Tuple[int, int],
    *,
    sampler: Sampler4 | None = None,  # if None: UniformSampler4(seed)
    hint: np.ndarray | None = None,
    refine: bool = True,
    seed: int | None = None,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    4-point RANSAC homography without geometry pre-filters (no OpenCV required for those).

    Default ``sampler`` is :class:`UniformSampler4` with ``seed`` applied when no custom
    sampler is passed. ``seed`` is ignored if ``sampler`` is provided.
    If ``hint`` is None or empty, the hint bootstrap step is skipped.
    """
    use = sampler if sampler is not None else UniformSampler4(seed)
    return _ransac_homography_core(
        use,
        src,
        dst,
        win_size,
        hint=hint,
        refine=refine,
        geometry_filters=False,
    )

ransacc = ransac_classic

def ransac_optimized(
    src: Sequence[Tuple[float, float]] | np.ndarray,
    dst: Sequence[Tuple[float, float]] | np.ndarray,
    win_size: Tuple[int, int],
    *,
    sampler: Sampler4 | None = None,  # if None: RectangularRandomSampler4(seed)
    hint: np.ndarray | None = None,
    refine: bool = True,
    seed: int | None = None,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    4-point RANSAC with convexity, edge-length, and self-intersection checks (needs cv2).

    Default ``sampler`` is :class:`RectangularRandomSampler4` with ``seed`` when no custom
    sampler is passed. ``seed`` is ignored if ``sampler`` is provided.
    If ``hint`` is None or empty, the hint bootstrap step is skipped.
    """
    use = sampler if sampler is not None else RectangularRandomSampler4(seed)
    return _ransac_homography_core(
        use,
        src,
        dst,
        win_size,
        hint=hint,
        refine=refine,
        geometry_filters=True,
    )


ransacz = ransac_optimized


__all__ = [
    "homography_mul",
    "ransac_classic",
    "ransac_optimized",
    "ransacz",
    "ransacc",
    "RectangularRandomSampler4",
    "UniformSampler4",
    "calculate_normalization_matrix",
    "find_homography_8x8",
    "find_homography_9x9_lms",
    "find_homography_9x9_linear",
]
