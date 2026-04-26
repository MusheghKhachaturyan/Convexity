# Convexity

## A Python project for homography estimation using classic and optimized RANSAC methods.

## What you can do with this project

### Estimate a planar homography from point correspondences

Given **matching 2D points** in a source image and a destination image (at least four pairs, same count and order), you can recover a **3×3 homography** \(H\) that maps source pixels to destination pixels (up to scale). Typical uses:

- **Wide-baseline or video alignment** — relate two views of a (roughly) planar scene or a rotating camera.
- **Stitching / mosaicking** — same math as many stitchers, exposed as a small library instead of a full pipeline.
- **Benchmarking** — compare a **minimal** RANSAC variant against one that **rejects bad quadrilaterals** before running expensive DLT steps.

**Inputs**

- `src`, `dst` — sequences of `(x, y)` or `Nx2` NumPy arrays (paired rows are one correspondence).
- `win_size` — `(width, height)` of the **source** image (or the coordinate frame you care about); used for normalization and for geometry thresholds (e.g. minimum edge length is tied to image height).

**Outputs**

- `H` — `3×3` `float64` homography in **pixel** coordinates (or `None` if estimation fails).
- `mask` — length-`N` `uint8` array: inlier / soft-inlier flags from the symmetric transfer inlier rule used inside RANSAC (see `estimate_inliers` in `ransac.py`).

**Aliases** — `ransacc` is the same as `ransac_classic`; `ransacz` is the same as `ransac_optimized`.

### Two RANSAC modes: classic vs optimized

Both modes share the same **core loop**: Hartley normalization, projective normalization of points, random **4-point** minimal sets, homography from DLT / LMS, symmetric transfer distance for inliers, adaptive iteration cap toward 99% confidence, and optional **refinement** over all inliers at the end.

| Capability | `ransac_classic` | `ransac_optimized` |
|------------|------------------|---------------------|
| Default 4-point sampler | `UniformSampler4` (uniform over index sets) | `RectangularRandomSampler4` (spatially biased toward corners via sorted lists + folded random index) |
| Reject sample if the 4 source points form a bad quad (too short edges, non-convex, self-intersecting) | No | Yes (needs **OpenCV** for convex hull) |
| OpenCV required | Only if you call convex-hull helpers or optimized mode | Yes |

Use **classic** when you want fewer dependencies or maximum compatibility. Use **optimized** when matches are noisy and **rejecting degenerate quadrilaterals** before solving homographies improves stability (as in the original kpt design).

### Optional homography “hints”

Both entry points accept `hint: np.ndarray | None` — a **3×3** homography (same convention as the output). When provided, the implementation **bootstraps** the best inlier count / score from that hint before RANSAC iterations (see `_ransac_homography_core`). The demo script can fill this with **`cv2.findHomography(..., RANSAC)`** so you can study how much the kpt-style RANSAC adds on top of OpenCV’s estimate.

### Custom samplers

You are not locked to the defaults. Pass any object implementing the `Sampler4` interface (`init(samples, win_size)`, then `sample(k)` filling four indices in-place):

- **`UniformSampler4`** — uniform minimal sets over point indices.
- **`BernoulliRandomSampler4`** — binomial-based bias along top-left / top-right sorted lists.
- **`RectangularRandomSampler4`** — extends the Bernoulli sampler with a **piecewise-linear** folded index (default for optimized RANSAC).

Defined in `sampler.py`.

### Lower-level building blocks (library use)

For custom tools or research, `ransac.py` also exposes homography and linear-algebra helpers used by the RANSAC loop, for example:

- `homography_mul` — apply `H` to a 2D point (inhomogeneous).
- `calculate_normalization_matrix`, `project_to_circle` — coordinate preprocessing.
- `find_homography_8x8`, `find_homography_9x9_linear`, `find_homography_9x9_lms` — solve homography from four or more correspondences.
- `estimate_inliers`, `dlte_estimate` — scoring and refinement-style estimation.

These follow the original C++ port’s structure; read docstrings in `ransac.py` for exact behavior.

### Interactive demo: ORB matches on video

`test_compare_time.py` is a **small application**, not just a unit test:

- Reads a video, runs **ORB** on a **reference** frame and each **current** frame, matches descriptors, and runs **both** `ransac_classic` and `ransac_optimized` on the same matches.
- Prints and overlays **wall time**, **inlier counts**, **forward** and **symmetric** transfer RMSE on inliers, and Frobenius norm of the difference between the two homographies when both succeed.
- Draws a **match visualization** (up to 60 matches) for inspection.

**Controls**

- Any other key — advance to the next frame (after a successful match step).
- **q** — quit.
- **r** — make the **current** frame the new reference (re-detect ORB on it).
- **h** — toggle **OpenCV RANSAC hints** at runtime (`--use-hints` only sets the initial state).

**Useful CLI flags**

| Flag | Purpose |
|------|---------|
| `--seed` | RNG seed for both RANSAC runs (default `42`). |
| `--max-matches` | Cap how many best ORB matches are passed in (default: all). |
| `--headless` | No GUI window; metrics in the terminal with text prompts. |
| `--use-hints` | Start with OpenCV `findHomography` as `hint=`. |
| `--hint-ransac-threshold` | Reprojection threshold (pixels) for that OpenCV hint (default `3`). |

## Requirements

- **NumPy** — always required.
- **OpenCV** (`opencv-python`) — required for `ransac_optimized`, convex-hull-based helpers, and the video demo.

```bash
pip install numpy opencv-python
```

## Layout

| File | Role |
|------|------|
| `ransac.py` | Homography helpers, inlier scoring, `ransac_classic` / `ransac_optimized`. |
| `sampler.py` | `UniformSampler4`, `BernoulliRandomSampler4`, `RectangularRandomSampler4`. |
| `test_compare_time.py` | Interactive ORB + video comparison harness. |

## Minimal import example

From the project directory (so `ransac` / `sampler` resolve):

```python
from ransac import ransac_classic, ransac_optimized
from sampler import UniformSampler4, RectangularRandomSampler4

# src_pts, dst_pts: list[tuple[float, float]] or Nx2 float arrays
# win_w, win_h: size of the source image in pixels
H, mask = ransac_classic(src_pts, dst_pts, (win_w, win_h), seed=0)
# H_opt, mask_opt = ransac_optimized(src_pts, dst_pts, (win_w, win_h), seed=0)
```

## Demo commands

```bash
python test_compare_time.py --video path/to/video.mp4
python test_compare_time.py --video path/to/video.mp4 --use-hints
```
