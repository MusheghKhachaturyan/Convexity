"""
Compare ``ransac_classic`` vs ``ransac_optimized`` on ORB matches while stepping through a video.

- Each step compares the **reference** image to the **next** frame from the video.
- Reference is initially the first frame. Press **r** to set the current frame (and its ORB
  keypoints/descriptors) as the new reference for all following comparisons.
- Press **q** to quit; any other key loads the next frame and runs the comparison again.

Usage:
  python test_compare_time.py --video path/to/video.mp4
  python test_compare_time.py --video path/to/video.mp4 --use-hints   # start with hints; h toggles in GUI
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import cv2

from ransac import homography_mul, ransac_classic, ransac_optimized


def match_orb_pair(
    kp1: list[cv2.KeyPoint],
    des1: np.ndarray | None,
    kp2: list[cv2.KeyPoint],
    des2: np.ndarray | None,
    max_matches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Too few keypoints/descriptors on one side")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    if max_matches is not None:
        matches = matches[:max_matches]
    if len(matches) < 4:
        raise RuntimeError(f"Need at least 4 matches, got {len(matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def forward_transfer_rmse(
    H: np.ndarray | None,
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    if H is None:
        return float("nan"), 0
    inl = np.asarray(mask) > 0
    n = int(np.sum(inl))
    if n == 0:
        return float("nan"), 0
    H3 = H.reshape(3, 3).astype(np.float64)
    err_sum = 0.0
    for i in np.where(inl)[0]:
        px, py = homography_mul(H3, (float(src[i, 0]), float(src[i, 1])))
        dx = px - float(dst[i, 0])
        dy = py - float(dst[i, 1])
        err_sum += dx * dx + dy * dy
    return float(np.sqrt(err_sum / n)), n


def symmetric_transfer_rmse(
    H: np.ndarray | None,
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    if H is None:
        return float("nan"), 0
    inl = np.asarray(mask) > 0
    n = int(np.sum(inl))
    if n == 0:
        return float("nan"), 0
    try:
        h_inv = np.linalg.inv(H.reshape(3, 3))
    except np.linalg.LinAlgError:
        return float("nan"), n
    err_sum = 0.0
    for i in np.where(inl)[0]:
        fp = homography_mul(H, (float(src[i, 0]), float(src[i, 1])))
        bp = homography_mul(h_inv, (float(dst[i, 0]), float(dst[i, 1])))
        d0 = np.hypot(fp[0] - dst[i, 0], fp[1] - dst[i, 1])
        d1 = np.hypot(bp[0] - src[i, 0], bp[1] - src[i, 1])
        err_sum += min(d0, d1) ** 2
    return float(np.sqrt(err_sum / n)), n


def opencv_hint_homography(
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    ransac_threshold: float = 3.0,
) -> np.ndarray | None:
    """3x3 float64 homography from OpenCV RANSAC, or None if estimation fails."""
    H, _mask = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, ransac_threshold)
    if H is None or H.size != 9:
        return None
    return np.asarray(H, dtype=np.float64).reshape(3, 3)


def draw_metrics_overlay(bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(
            out,
            line[:100],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return out


def run_interactive(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}", file=sys.stderr)
        return 1

    orb = cv2.ORB_create(nfeatures=2000)
    ok, ref_bgr = cap.read()
    if not ok:
        print("Could not read first frame", file=sys.stderr)
        cap.release()
        return 1

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
    ref_frame_idx = 0
    frame_num = 0

    win_size = (ref_bgr.shape[1], ref_bgr.shape[0])
    headless: bool = args.headless

    if not headless:
        cv2.namedWindow("ransac_compare", cv2.WINDOW_NORMAL)

    use_hints = bool(args.use_hints)

    print(
        "Reference = first frame. Keys: [any] next | [h] toggle OpenCV hints | "
        "[r] current -> new reference | [q] quit"
    )
    print("-" * 70)

    while True:
        ok, cur_bgr = cap.read()
        if not ok:
            print("End of video")
            break

        frame_num += 1
        cur_frame_idx = frame_num

        cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        cur_kp, cur_des = orb.detectAndCompute(cur_gray, None)

        try:
            pts_ref, pts_cur = match_orb_pair(ref_kp, ref_des, cur_kp, cur_des, args.max_matches)
        except RuntimeError as e:
            print(f"\nframe {cur_frame_idx} vs ref {ref_frame_idx}: {e}", file=sys.stderr)
            overlay = draw_metrics_overlay(
                cur_bgr,
                [
                    f"ref#{ref_frame_idx} -> #{cur_frame_idx}",
                    str(e),
                    "r=rebase q=quit other=next",
                ],
            )
            if not headless:
                cv2.imshow("ransac_compare", overlay)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    ref_bgr = cur_bgr.copy()
                    ref_gray = cur_gray.copy()
                    ref_kp, ref_des = cur_kp, cur_des
                    ref_frame_idx = cur_frame_idx
                    print(f"Reference set to frame index {ref_frame_idx}")
            continue

        n = pts_ref.shape[0]
        src = [(float(x), float(y)) for x, y in pts_ref]
        dst = [(float(x), float(y)) for x, y in pts_cur]

        quit_app = False
        while True:
            hint: np.ndarray | None = None
            if use_hints:
                hint = opencv_hint_homography(
                    pts_ref, pts_cur, ransac_threshold=args.hint_ransac_threshold
                )

            t0 = time.perf_counter()
            h_classic, mask_c = ransac_classic(
                src,
                dst,
                win_size,
                seed=args.seed,
                hint=hint,
            )
            t_classic = time.perf_counter() - t0

            t0 = time.perf_counter()
            h_opt, mask_o = ransac_optimized(
                src,
                dst,
                win_size,
                seed=args.seed,
                hint=hint,
            )
            t_opt = time.perf_counter() - t0

            fwd_c, inl_c = forward_transfer_rmse(h_classic, pts_ref, pts_cur, mask_c)
            fwd_o, inl_o = forward_transfer_rmse(h_opt, pts_ref, pts_cur, mask_o)
            sym_c, _ = symmetric_transfer_rmse(h_classic, pts_ref, pts_cur, mask_c)
            sym_o, _ = symmetric_transfer_rmse(h_opt, pts_ref, pts_cur, mask_o)

            dhf = ""
            if h_classic is not None and h_opt is not None:
                dhf = f" ||dH||_F={np.linalg.norm(h_classic - h_opt, ord='fro'):.4f}"

            hint_note = "hint=OpenCV+RANSAC" if (use_hints and hint is not None) else (
                "hint=failed" if use_hints else "hint=off"
            )
            # Line 1: status. Then aligned table: header + classic row (red) + optim row (green) + delta
            line1 = (
                f"ref {ref_frame_idx}->{cur_frame_idx} | matches {n} | {hint_note}   "
                f"[any] next  [h] hints  [r] rebase ref  [q] quit"
            )
            neutral = (220, 255, 255)
            red = (60, 60, 255)
            green = (60, 220, 60)
            delta_color = (0, 255, 255)  # yellow — easy to spot vs red/green

            # Fixed-width columns so time / inliers / errors line up for visual compare
            col_hdr = (
                f"{'':12s}"
                f"{'time (ms)':>10}  "
                f"{'inliers':>8}  "
                f"{'fwd':>8}  "
                f"{'sym':>8}"
            )
            row_classic = (
                f"{'classic':12s}"
                f"{t_classic * 1000:10.1f}  "
                f"{inl_c:8d}  "
                f"{fwd_c:8.3f}  "
                f"{sym_c:8.3f}"
            )
            row_optim = (
                f"{'optim':12s}"
                f"{t_opt * 1000:10.1f}  "
                f"{inl_o:8d}  "
                f"{fwd_o:8.3f}  "
                f"{sym_o:8.3f}"
            )
            dt_ms = (t_opt - t_classic) * 1000.0
            di = inl_o - inl_c
            line_delta = (
                f"{'delta (opt-classic)':12s}"
                f"{dt_ms:+10.1f}  "
                f"{di:+8d}  "
                f"{fwd_o - fwd_c:+8.3f}  "
                f"{sym_o - sym_c:+8.3f}"
            )

            overlay_rows: list[tuple[str, tuple[int, int, int]]] = [
                (col_hdr, neutral),
                (row_classic, red),
                (row_optim, green),
                (line_delta, delta_color),
            ]
            if dhf:
                overlay_rows.append((dhf.strip(), neutral))

            print(f"\n=== Current frame {cur_frame_idx} vs reference frame {ref_frame_idx} ===")
            print(f"{'Time ms (cl/opt)':<24} {t_classic*1000:>10.3f} {t_opt*1000:>10.3f}")
            print(f"{'Inliers (cl/opt)':<24} {inl_c:>10} {inl_o:>10}")
            print(f"{'Fwd RMSE (cl/opt)':<24} {fwd_c:>10.4f} {fwd_o:>10.4f}")
            print(f"{'Sym RMSE (cl/opt)':<24} {sym_c:>10.4f} {sym_o:>10.4f}")
            if dhf:
                print(dhf.strip())

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            dm = bf.match(ref_des, cur_des)
            dm = sorted(dm, key=lambda m: m.distance)
            if args.max_matches is not None:
                dm = dm[: args.max_matches]
            dm = dm[: min(60, len(dm))]
            vis = cv2.drawMatches(ref_bgr, ref_kp, cur_bgr, cur_kp, dm, None, flags=2)

            w_vis = vis.shape[1]
            if w_vis > 1600:
                scale = 1600 / w_vis
                vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            thickness = 1
            line_gap = 4
            (_, h_line1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            block_h = h_line1 + line_gap
            for txt, _c in overlay_rows:
                (_, hh), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                block_h += hh + line_gap
            text_h = block_h + 14
            bar = np.zeros((text_h, vis.shape[1], 3), dtype=np.uint8)
            y = 18
            cv2.putText(bar, line1, (8, y), font, font_scale, neutral, thickness, cv2.LINE_AA)
            (_, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            y += h1 + line_gap
            for txt, color in overlay_rows:
                cv2.putText(bar, txt, (8, y), font, font_scale, color, thickness, cv2.LINE_AA)
                (_, hh), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                y += hh + line_gap
            vis = np.vstack([bar, vis])

            if not headless:
                cv2.imshow("ransac_compare", vis)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("h"):
                    use_hints = not use_hints
                    print(f"Hints: {'ON' if use_hints else 'OFF'}")
                    continue
                if key == ord("q"):
                    quit_app = True
                    break
                if key == ord("r"):
                    ref_bgr = cur_bgr.copy()
                    ref_gray = cur_gray.copy()
                    ref_kp, ref_des = cur_kp, cur_des
                    ref_frame_idx = cur_frame_idx
                    print(f"Reference set to frame index {ref_frame_idx} (ORB recomputed on that frame)")
                break
            else:
                cmd = input("Enter=next  h=toggle  r=rebase  q=quit: ").strip().lower()
                if cmd in ("h", "hint"):
                    use_hints = not use_hints
                    print(f"Hints: {'ON' if use_hints else 'OFF'}")
                    continue
                if cmd == "q":
                    quit_app = True
                    break
                if cmd == "r":
                    ref_bgr = cur_bgr.copy()
                    ref_gray = cur_gray.copy()
                    ref_kp, ref_des = cur_kp, cur_des
                    ref_frame_idx = cur_frame_idx
                    print(f"Reference set to frame index {ref_frame_idx}")
                break

        if quit_app:
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare ransac_classic vs ransac_optimized on ORB matches (interactive video)"
    )
    p.add_argument("--video", type=Path, required=True, help="Input video")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for both RANSAC variants")
    p.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Cap number of best ORB matches (default: all)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No OpenCV window; print metrics and use text prompts (Enter/r/q)",
    )
    p.add_argument(
        "--use-hints",
        action="store_true",
        help="Start with OpenCV findHomography(RANSAC) hints ON; press h in the viewer to toggle at runtime",
    )
    p.add_argument(
        "--hint-ransac-threshold",
        type=float,
        default=3.0,
        help="Reprojection threshold (px) for OpenCV hint homography when --use-hints (default: 3)",
    )
    args = p.parse_args()

    if not args.video.is_file():
        print(f"File not found: {args.video}", file=sys.stderr)
        return 1

    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
