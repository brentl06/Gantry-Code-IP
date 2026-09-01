#!/usr/bin/env python3
"""
RealSense D435i live viewer with sand-plane depth correction.

Controls:
- r: collect reference frames and fit the visible sand plane
- s: save the current raw depth and corrected height frame
- q or Esc: quit

The corrected pane shows signed distance from the fitted reference plane.
White is near the fitted plane, red is positive, and blue is negative.
Validate the sign with a known mound/depression before treating red/blue as
absolute up/down in world coordinates.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class StreamCfg:
    w: int
    h: int
    fps: int


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


DEFAULT_DEPTH = StreamCfg(848, 480, 30)
DEFAULT_COLOR = StreamCfg(848, 480, 30)
DEFAULT_RS_CONFIG = Path(__file__).resolve().parents[2] / "rs_config_gui.json"


def _try_set(opt_owner, option, value) -> bool:
    try:
        opt_owner.set_option(option, value)
        return True
    except Exception as exc:
        print(f"[WARN] Could not set {option} to {value}: {exc}")
        return False


def _make_colorizer(min_m, max_m, scheme: str, hist_eq: bool) -> rs.colorizer:
    cz = rs.colorizer()
    scheme_map = {
        "jet": 0,
        "classic": 1,
        "white_to_black": 2,
        "black_to_white": 3,
        "bio": 4,
        "cold": 5,
        "warm": 6,
        "quantized": 7,
        "pattern": 8,
        "turbo": 9,
    }
    _try_set(cz, rs.option.color_scheme, float(scheme_map.get(scheme, 0)))
    if min_m is not None:
        _try_set(cz, rs.option.min_distance, float(min_m))
    if max_m is not None:
        _try_set(cz, rs.option.max_distance, float(max_m))
    _try_set(cz, rs.option.histogram_equalization_enabled, 1.0 if hist_eq else 0.0)
    return cz


def _load_gui_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _stream_cfg_from_gui_config(gui_config: Dict[str, Any], fallback: StreamCfg) -> StreamCfg:
    viewer = gui_config.get("viewer", {})
    return StreamCfg(
        int(viewer.get("stream-width", fallback.w)),
        int(viewer.get("stream-height", fallback.h)),
        int(viewer.get("stream-fps", fallback.fps)),
    )


def _apply_gui_config(device: rs.device, config_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"RealSense GUI config not found: {config_path}")

    raw_config = config_path.read_text(encoding="utf-8")
    try:
        advanced = rs.rs400_advanced_mode(device)
    except Exception as exc:
        print(f"[WARN] Device does not expose D400 advanced mode; GUI config not loaded: {exc}")
        return

    try:
        if not advanced.is_enabled():
            print("[WARN] D400 advanced mode is disabled; GUI JSON was not loaded.")
            print("[WARN] Open Intel RealSense Viewer once, enable Advanced Mode, then rerun this script.")
            return
        advanced.load_json(raw_config)
        print(f"Loaded RealSense GUI config: {config_path}")
    except Exception as exc:
        print(f"[WARN] Failed to load RealSense GUI config {config_path}: {exc}")


def _get_depth_intrinsics(profile: rs.pipeline_profile) -> Intrinsics:
    stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = stream_profile.get_intrinsics()
    return Intrinsics(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        ppx=intr.ppx,
        ppy=intr.ppy,
    )


def _get_color_intrinsics(profile: rs.pipeline_profile) -> Intrinsics:
    stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream_profile.get_intrinsics()
    return Intrinsics(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        ppx=intr.ppx,
        ppy=intr.ppy,
    )


def _pixel_rays(intr: Intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    v, u = np.indices((intr.height, intr.width), dtype=np.float32)
    x_coeff = (u - intr.ppx) / intr.fx
    y_coeff = (v - intr.ppy) / intr.fy
    return x_coeff, y_coeff


def _points_from_depth(depth_raw: np.ndarray, depth_scale: float, x_coeff: np.ndarray, y_coeff: np.ndarray):
    z = depth_raw.astype(np.float32) * depth_scale
    x = x_coeff * z
    y = y_coeff * z
    return x, y, z


def _fit_mask(
    depth_raw: np.ndarray,
    *,
    top_crop: float,
    bottom_crop: float,
    side_crop: float,
    min_depth_m: float,
    max_depth_m: float,
    depth_scale: float,
) -> np.ndarray:
    h, w = depth_raw.shape
    depth_m = depth_raw.astype(np.float32) * depth_scale
    mask = (depth_raw > 0) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    mask[: int(h * top_crop), :] = False
    mask[int(h * bottom_crop) :, :] = False
    mask[:, : int(w * side_crop)] = False
    mask[:, int(w * (1.0 - side_crop)) :] = False
    return mask


def _fit_plane_ransac(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int,
    iterations: int,
    threshold_m: float,
    seed: int = 7,
) -> np.ndarray:
    points = np.column_stack([x[mask], y[mask], z[mask]])
    if len(points) < 1000:
        raise RuntimeError(f"Not enough valid sand points to fit plane: {len(points)}")

    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[rng.choice(len(points), size=max_points, replace=False)]

    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        residuals = np.abs(points @ normal + d)
        inliers = residuals < threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 1000:
        raise RuntimeError("RANSAC could not find a stable sand plane.")

    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    d = -float(normal @ centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def _signed_plane_distance_mm(x: np.ndarray, y: np.ndarray, z: np.ndarray, plane: np.ndarray) -> np.ndarray:
    a, b, c, d = plane
    dist_m = (a * x + b * y + c * z + d) / np.linalg.norm(plane[:3])
    return dist_m * 1000.0


def _height_to_bgr(height_mm: np.ndarray, valid_mask: np.ndarray, range_mm: float) -> np.ndarray:
    norm = np.clip((height_mm + range_mm) / (2.0 * range_mm), 0.0, 1.0)
    bgr = np.zeros((*height_mm.shape, 3), dtype=np.uint8)

    low = norm < 0.5
    high = ~low
    t_low = norm[low] / 0.5
    bgr[low, 0] = 255
    bgr[low, 1] = np.clip(255 * t_low, 0, 255).astype(np.uint8)
    bgr[low, 2] = np.clip(255 * t_low, 0, 255).astype(np.uint8)

    t_high = (norm[high] - 0.5) / 0.5
    bgr[high, 0] = np.clip(255 * (1.0 - t_high), 0, 255).astype(np.uint8)
    bgr[high, 1] = np.clip(255 * (1.0 - t_high), 0, 255).astype(np.uint8)
    bgr[high, 2] = 255

    bgr[~valid_mask] = (0, 0, 0)
    return bgr


def _draw_label(image: np.ndarray, text: str, y: int = 36) -> None:
    cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    if image.shape[0] == target_h:
        return image
    scale = target_h / image.shape[0]
    target_w = int(round(image.shape[1] * scale))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _save_frame(save_dir: Path, idx: int, depth_raw: np.ndarray, height_mm: np.ndarray, height_bgr: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = save_dir / f"plane_corrected_{idx:04d}"
    np.save(str(stem) + "_raw_depth.npy", depth_raw)
    np.save(str(stem) + "_height_mm.npy", height_mm)
    cv2.imwrite(str(stem) + "_height_preview.png", height_bgr)
    print(f"[INFO] Saved {stem}_*.npy/png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rs-config",
        type=Path,
        default=DEFAULT_RS_CONFIG,
        help="Intel RealSense Viewer JSON config to load onto the device.",
    )
    ap.add_argument("--no-rs-config", action="store_true", help="Do not load the Intel RealSense Viewer JSON config.")
    ap.add_argument("--serial", type=str, default=None, help="RealSense serial number. Omit to use first device.")
    ap.add_argument("--min-depth", type=float, default=None, help="Raw depth colorizer min distance in meters.")
    ap.add_argument("--max-depth", type=float, default=None, help="Raw depth colorizer max distance in meters.")
    ap.add_argument(
        "--scheme",
        type=str,
        default="jet",
        choices=["jet", "classic", "white_to_black", "black_to_white", "bio", "cold", "warm", "quantized", "pattern", "turbo"],
        help="Raw depth color scheme.",
    )
    ap.add_argument("--hist-eq", dest="hist_eq", action="store_true", default=False, help="Enable raw-depth histogram equalization.")
    ap.add_argument("--no-hist-eq", dest="hist_eq", action="store_false", help="Disable raw-depth histogram equalization.")
    ap.add_argument("--post", action="store_true", help="Enable spatial/temporal/hole-fill depth filters.")
    ap.add_argument("--emitter", type=int, default=None, choices=[0, 1], help="Force emitter_enabled (0/1).")
    ap.add_argument("--laser", type=float, default=None, help="Force laser_power.")
    ap.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["default", "high_accuracy", "high_density", "medium_density"],
        help="Force visual_preset on depth sensor.",
    )
    ap.add_argument("--depth-w", type=int, default=DEFAULT_DEPTH.w)
    ap.add_argument("--depth-h", type=int, default=DEFAULT_DEPTH.h)
    ap.add_argument("--depth-fps", type=int, default=DEFAULT_DEPTH.fps)
    ap.add_argument("--color-w", type=int, default=DEFAULT_COLOR.w)
    ap.add_argument("--color-h", type=int, default=DEFAULT_COLOR.h)
    ap.add_argument("--color-fps", type=int, default=DEFAULT_COLOR.fps)

    ap.add_argument("--reference-frames", type=int, default=60, help="Frames to median-stack when fitting the reference plane.")
    ap.add_argument("--height-range-mm", type=float, default=80.0, help="Corrected display clamp: +/- this many mm.")
    ap.add_argument("--top-crop", type=float, default=0.30, help="Ignore this top image fraction while fitting the plane.")
    ap.add_argument("--bottom-crop", type=float, default=0.98, help="Ignore rows below this image fraction while fitting the plane.")
    ap.add_argument("--side-crop", type=float, default=0.06, help="Ignore this fraction at left and right while fitting the plane.")
    ap.add_argument("--fit-min-depth", type=float, default=0.15, help="Minimum valid depth in meters for plane fitting.")
    ap.add_argument("--fit-max-depth", type=float, default=3.0, help="Maximum valid depth in meters for plane fitting.")
    ap.add_argument("--ransac-iters", type=int, default=250)
    ap.add_argument("--ransac-threshold-mm", type=float, default=8.0)
    ap.add_argument("--ransac-max-points", type=int, default=40000)
    ap.add_argument("--save-dir", type=Path, default=Path("plane_corrected_captures"))
    args = ap.parse_args()

    gui_config = None if args.no_rs_config else _load_gui_config(args.rs_config)
    if gui_config is not None:
        gui_stream = _stream_cfg_from_gui_config(gui_config, DEFAULT_DEPTH)
        args.depth_w = gui_stream.w
        args.depth_h = gui_stream.h
        args.depth_fps = gui_stream.fps
        args.color_w = gui_stream.w
        args.color_h = gui_stream.h
        args.color_fps = gui_stream.fps

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.depth, args.depth_w, args.depth_h, rs.format.z16, args.depth_fps)
    cfg.enable_stream(rs.stream.color, args.color_w, args.color_h, rs.format.bgr8, args.color_fps)

    profile = pipeline.start(cfg)
    intr = _get_color_intrinsics(profile)
    x_coeff, y_coeff = _pixel_rays(intr)

    dev = profile.get_device()
    if gui_config is not None:
        _apply_gui_config(dev, args.rs_config)
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale (meters per unit): {depth_scale}")
    print("Controls: r=fit reference plane, s=save current corrected frame, q/Esc=quit")

    if args.emitter is not None:
        _try_set(depth_sensor, rs.option.emitter_enabled, float(args.emitter))
    if args.laser is not None:
        _try_set(depth_sensor, rs.option.laser_power, float(args.laser))
    if args.preset is not None:
        preset_map = {
            "default": 0,
            "high_accuracy": 3,
            "high_density": 4,
            "medium_density": 5,
        }
        _try_set(depth_sensor, rs.option.visual_preset, float(preset_map[args.preset]))

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    colorizer = _make_colorizer(args.min_depth, args.max_depth, args.scheme, args.hist_eq)
    align = rs.align(rs.stream.color)

    plane: Optional[np.ndarray] = None
    collecting = False
    reference_frames = []
    save_idx = 0
    last_depth_raw: Optional[np.ndarray] = None
    last_height_mm: Optional[np.ndarray] = None
    last_height_bgr: Optional[np.ndarray] = None

    win = "RealSense live plane correction"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            if args.post:
                depth = spatial.process(depth)
                depth = temporal.process(depth)
                depth = hole.process(depth)

            depth_raw = np.asanyarray(depth.get_data()).copy()
            color_img = np.asanyarray(color.get_data())
            depth_color = colorizer.colorize(depth)
            depth_rgb = np.asanyarray(depth_color.get_data())
            depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)

            if collecting:
                reference_frames.append(depth_raw)
                if len(reference_frames) >= args.reference_frames:
                    reference_depth = np.median(np.stack(reference_frames, axis=0), axis=0).astype(np.float32)
                    x_ref, y_ref, z_ref = _points_from_depth(reference_depth, depth_scale, x_coeff, y_coeff)
                    mask = _fit_mask(
                        reference_depth,
                        top_crop=args.top_crop,
                        bottom_crop=args.bottom_crop,
                        side_crop=args.side_crop,
                        min_depth_m=args.fit_min_depth,
                        max_depth_m=args.fit_max_depth,
                        depth_scale=depth_scale,
                    )
                    plane = _fit_plane_ransac(
                        x_ref,
                        y_ref,
                        z_ref,
                        mask,
                        max_points=args.ransac_max_points,
                        iterations=args.ransac_iters,
                        threshold_m=args.ransac_threshold_mm * 0.001,
                    )
                    collecting = False
                    reference_frames.clear()
                    print(f"[INFO] Plane fitted [a, b, c, d]: {plane}")

            if plane is not None:
                x, y, z = _points_from_depth(depth_raw, depth_scale, x_coeff, y_coeff)
                valid = depth_raw > 0
                height_mm = _signed_plane_distance_mm(x, y, z, plane)
                height_bgr = _height_to_bgr(height_mm, valid, args.height_range_mm)
                _draw_label(height_bgr, f"Corrected height +/-{args.height_range_mm:.0f} mm")
                last_height_mm = height_mm
                last_height_bgr = height_bgr
            else:
                height_bgr = np.zeros_like(depth_bgr)
                text = "Press r to fit reference plane"
                if collecting:
                    text = f"Collecting reference {len(reference_frames)}/{args.reference_frames}"
                _draw_label(height_bgr, text)
                last_height_mm = None
                last_height_bgr = None

            last_depth_raw = depth_raw
            _draw_label(color_img, "Color")
            _draw_label(depth_bgr, "Raw depth")

            h = min(color_img.shape[0], depth_bgr.shape[0], height_bgr.shape[0])
            canvas = np.hstack([
                _resize_to_height(color_img, h),
                _resize_to_height(depth_bgr, h),
                _resize_to_height(height_bgr, h),
            ])

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("r"):
                collecting = True
                plane = None
                reference_frames.clear()
                print(f"[INFO] Collecting {args.reference_frames} reference frames...")
            if k == ord("s"):
                if last_depth_raw is None or last_height_mm is None or last_height_bgr is None:
                    print("[WARN] Fit a plane before saving corrected frames.")
                else:
                    _save_frame(args.save_dir, save_idx, last_depth_raw, last_height_mm, last_height_bgr)
                    save_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
