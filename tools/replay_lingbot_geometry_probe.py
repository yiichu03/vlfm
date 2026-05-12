#!/usr/bin/env python3
"""Geometry-only replay probe for LingBot exports into VLFM obstacle maps.

The script intentionally avoids Habitat, PointNav, policy/VLM servers, semantic
value maps, and learned frontier scoring. It only tests whether LingBot depth
and relative camera pose can drive VLFM's existing ObstacleMap/frontier path.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VLFM_IMPORT_ERROR: Optional[str] = None
try:
    from vlfm.mapping.obstacle_map import ObstacleMap, filter_points_by_height
    from vlfm.utils.geometry_utils import extract_yaw, get_fov, transform_points
except Exception:
    VLFM_IMPORT_ERROR = traceback.format_exc()
    ObstacleMap = None  # type: ignore[assignment]
    filter_points_by_height = None  # type: ignore[assignment]
    extract_yaw = None  # type: ignore[assignment]
    get_fov = None  # type: ignore[assignment]
    transform_points = None  # type: ignore[assignment]


OPENCV_TO_VLFM_ROT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass
class StepResult:
    frame_path: str
    output_npz: str
    rgb_shape: Tuple[int, int, int]
    depth_shape: Tuple[int, int]
    inverse_error: float
    finite_projected_points: int
    robot_xy: List[float]
    robot_heading: float
    frontier_count: Optional[int]


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay LingBot geometry into VLFM obstacle/frontier maps.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--axis_mode", type=str, default="auto", choices=["auto", "opencv_first_camera", "identity"])
    parser.add_argument("--geometry_only", type=str_to_bool, default=True)
    parser.add_argument("--map_size", type=int, default=1000)
    parser.add_argument("--pixels_per_meter", type=int, default=20)
    parser.add_argument("--min_obstacle_height", type=float, default=-0.6)
    parser.add_argument("--max_obstacle_height", type=float, default=0.6)
    parser.add_argument("--agent_radius", type=float, default=0.18)
    parser.add_argument("--obstacle_map_area_threshold", type=float, default=1.5)
    parser.add_argument("--hole_area_thresh", type=int, default=100000)
    return parser.parse_args()


def load_frames(input_dir: Path, max_steps: int, stride: int) -> List[Path]:
    frame_paths = sorted(input_dir.glob("frame_*.npz"))
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    frame_paths = frame_paths[::stride]
    if max_steps > 0:
        frame_paths = frame_paths[:max_steps]
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.npz files found in {input_dir}")
    return frame_paths


def squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape HxW or HxWx1, got {depth.shape}")
    return depth.astype(np.float32, copy=False)


def ensure_4x4(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"{name} must be 4x4 or 3x4, got {matrix.shape}")


def opencv_to_vlfm_transform() -> np.ndarray:
    """Return homogeneous transform from OpenCV camera axes to VLFM camera axes.

    OpenCV camera axes are x-right, y-down, z-forward.
    VLFM local camera axes are x-forward, y-left, z-up.
    """
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = OPENCV_TO_VLFM_ROT
    return transform


def lingbot_pose_to_vlfm_episodic(
    c2w_current: np.ndarray,
    w2c_first: np.ndarray,
    axis_mode: str,
) -> np.ndarray:
    """Convert LingBot OpenCV C2W into a VLFM camera-to-episodic transform.

    The episodic frame is anchored at the first LingBot camera pose. For the
    default/auto path, a relative OpenCV camera transform is converted into the
    VLFM x-forward/y-left/z-up camera convention.
    """
    relative_cv = w2c_first @ c2w_current
    if axis_mode == "identity":
        return relative_cv.astype(np.float32)
    if axis_mode in {"auto", "opencv_first_camera"}:
        cv_to_vlfm = opencv_to_vlfm_transform()
        vlfm_to_cv = np.linalg.inv(cv_to_vlfm)
        return (cv_to_vlfm @ relative_cv @ vlfm_to_cv).astype(np.float32)
    raise ValueError(f"Unsupported axis_mode {axis_mode!r}")


def normalize_depth_for_vlfm(
    depth: np.ndarray,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric_depth = depth.astype(np.float32, copy=False) * np.float32(depth_scale)
    valid = np.isfinite(metric_depth) & (metric_depth >= min_depth) & (metric_depth <= max_depth)
    normalized = np.ones_like(metric_depth, dtype=np.float32)
    if max_depth <= min_depth:
        raise ValueError("--max_depth must be greater than --min_depth")
    normalized[valid] = (metric_depth[valid] - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0.0, 1.0)
    return metric_depth, normalized, valid


def backproject_depth_to_vlfm_camera(
    depth_metric: np.ndarray,
    intrinsic: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Back-project metric depth into VLFM local camera coordinates.

    This mirrors VLFM's `get_point_cloud` convention while honoring the exported
    intrinsic principal point: local `(x, y, z) = (forward, left, up)`.
    """
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    v, u = np.where(valid_mask)
    z = depth_metric[v, u]
    x_right = (u.astype(np.float32) - cx) * z / fx
    y_down = (v.astype(np.float32) - cy) * z / fy
    return np.stack((z, -x_right, -y_down), axis=-1).astype(np.float32)


def local_transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    if transform_points is not None:
        return transform_points(transform, points)
    homogeneous = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3:]


def local_extract_yaw(transform: np.ndarray) -> float:
    if extract_yaw is not None:
        return float(extract_yaw(transform))
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def local_get_hfov(fx: float, width: int) -> float:
    if get_fov is not None:
        return float(get_fov(fx, width))
    return float(2.0 * math.atan((width / 2.0) / fx))


def xy_to_px(points: np.ndarray, size: int, pixels_per_meter: int) -> np.ndarray:
    px = np.rint(points[:, ::-1] * pixels_per_meter) + np.array([size // 2, size // 2])
    px[:, 0] = size - px[:, 0]
    return px.astype(np.int32)


def draw_depth(depth: np.ndarray, valid: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    denom = max(max_depth - min_depth, 1e-6)
    norm = np.clip((depth - min_depth) / denom, 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    img[~valid] = 0
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)


def draw_confidence(conf: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if conf is None:
        return None
    conf = np.asarray(conf, dtype=np.float32)
    finite = np.isfinite(conf)
    if not np.any(finite):
        return np.zeros((*conf.shape[:2], 3), dtype=np.uint8)
    lo = float(np.nanmin(conf[finite]))
    hi = float(np.nanmax(conf[finite]))
    denom = max(hi - lo, 1e-6)
    img = np.zeros(conf.shape[:2], dtype=np.uint8)
    img[finite] = np.clip((conf[finite] - lo) / denom * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)


def draw_projected_points_bev(
    xy_points: np.ndarray,
    robot_xy: np.ndarray,
    map_size: int,
    pixels_per_meter: int,
) -> np.ndarray:
    image = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    if xy_points.size:
        px = xy_to_px(xy_points, map_size, pixels_per_meter)
        in_bounds = (
            (px[:, 0] >= 0)
            & (px[:, 0] < map_size)
            & (px[:, 1] >= 0)
            & (px[:, 1] < map_size)
        )
        px = px[in_bounds]
        image[px[:, 1], px[:, 0]] = (0, 0, 0)
    robot_px = xy_to_px(robot_xy.reshape(1, 2), map_size, pixels_per_meter)[0]
    if 0 <= robot_px[0] < map_size and 0 <= robot_px[1] < map_size:
        cv2.circle(image, tuple(int(v) for v in robot_px), 5, (0, 0, 255), -1)
    return cv2.flip(image, 0)


def draw_frontiers(
    obstacle_vis: Optional[np.ndarray],
    frontiers: Optional[np.ndarray],
    map_size: int,
    pixels_per_meter: int,
) -> Optional[np.ndarray]:
    if obstacle_vis is None or frontiers is None:
        return None
    image = obstacle_vis.copy()
    if len(frontiers) > 0:
        px = xy_to_px(frontiers.astype(np.float32), map_size, pixels_per_meter)
        for x, y in px:
            y_flipped = map_size - y
            if 0 <= x < map_size and 0 <= y_flipped < map_size:
                cv2.circle(image, (int(x), int(y_flipped)), 6, (255, 0, 0), 2)
    return image


def save_png(path: Path, image: np.ndarray, rgb_input: bool = False) -> None:
    if rgb_input:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def stats_for(values: np.ndarray) -> Dict[str, Any]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}


def merged_stats(arrays: Iterable[np.ndarray]) -> Dict[str, Any]:
    parts = [np.asarray(arr)[np.isfinite(arr)].reshape(-1) for arr in arrays]
    parts = [part for part in parts if part.size]
    if not parts:
        return {"min": None, "max": None, "mean": None}
    return stats_for(np.concatenate(parts))


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("step_*.npz", "step_*.png", "replay_metadata.json", "replay_summary.md"):
        for path in output_dir.glob(pattern):
            path.unlink()


def build_summary(metadata: Mapping[str, Any]) -> str:
    called = "\n".join(f"- {name}" for name in metadata["vlfm_functions_called"]) or "- none"
    not_called = "\n".join(
        f"- {name}: {reason}" for name, reason in metadata["vlfm_functions_not_called"].items()
    )
    frontier_counts = metadata["frontier_counts_per_step"]
    return (
        "# VLFM Geometry Replay Summary\n\n"
        f"## Command\n`{metadata['command']}`\n\n"
        f"## Frames Processed\n{metadata['frames_processed']}\n\n"
        "## VLFM Functions Actually Called\n"
        f"{called}\n\n"
        "## VLFM Functions Not Called\n"
        f"{not_called}\n\n"
        "## Coordinate Conversion\n"
        f"{metadata['coordinate_conversion']}\n\n"
        f"## Depth Scale\n{metadata['depth_scale']}\n\n"
        "## Trajectory Translation Stats\n"
        f"{metadata['trajectory_translation_stats']}\n\n"
        "## Depth Stats\n"
        f"{metadata['depth_stats']}\n\n"
        "## Candidate Frontiers Per Step\n"
        f"{frontier_counts}\n\n"
        "## Sanity Tests\n"
        + "\n".join(f"- {key}: {value}" for key, value in metadata["sanity_tests"].items())
        + "\n\n"
        "## Uncertainties\n"
        + "\n".join(f"- {item}" for item in metadata["uncertainties"])
        + "\n"
    )


def main() -> None:
    args = parse_args()
    if not args.geometry_only:
        raise ValueError("Stage 2a only supports --geometry_only true")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frame_paths = load_frames(input_dir, args.max_steps, args.stride)
    clean_output_dir(output_dir)

    obstacle_map = None
    vlfm_functions_called: List[str] = []
    vlfm_functions_not_called: Dict[str, str] = {
        "HabitatMixin._cache_observations": "Not called because it requires Habitat observation tensors and policy state.",
        "BaseITMPolicy._get_best_frontier": (
            "Not called because it requires semantic/value-map sorting and acyclic policy state."
        ),
        "ITMPolicyV2._sort_frontiers_by_value": (
            "Not called because it requires a populated ValueMap from BLIP2-ITM/VLM scoring."
        ),
        "PointNav/_pointnav": "Not called by design; geometry-only replay has no Habitat actions.",
        "Habitat environment step": "Not called by design; no Habitat environment is constructed.",
    }

    if ObstacleMap is not None:
        obstacle_map = ObstacleMap(
            min_height=args.min_obstacle_height,
            max_height=args.max_obstacle_height,
            agent_radius=args.agent_radius,
            area_thresh=args.obstacle_map_area_threshold,
            hole_area_thresh=args.hole_area_thresh,
            size=args.map_size,
            pixels_per_meter=args.pixels_per_meter,
        )
    else:
        vlfm_functions_not_called["ObstacleMap.update_map"] = (
            "Could not import vlfm.mapping.obstacle_map.ObstacleMap. Import traceback: "
            + str(VLFM_IMPORT_ERROR)
        )
        vlfm_functions_not_called["ObstacleMap._get_frontiers"] = (
            "Unavailable because ObstacleMap could not be imported."
        )

    first_frame = np.load(frame_paths[0])
    first_c2w = ensure_4x4(first_frame["extrinsic_c2w"], "extrinsic_c2w")
    first_w2c = ensure_4x4(first_frame["extrinsic_w2c"], "extrinsic_w2c")
    first_frame.close()

    step_results: List[StepResult] = []
    all_depths: List[np.ndarray] = []
    all_translations: List[np.ndarray] = []
    inverse_errors: List[float] = []
    frontier_counts: List[Optional[int]] = []
    maps_saved = False
    frontiers_saved = False

    for step_idx, frame_path in enumerate(frame_paths):
        with np.load(frame_path) as data:
            rgb = data["rgb"]
            depth_raw = squeeze_depth(data["depth"])
            depth_conf = data["depth_conf"] if "depth_conf" in data.files else None
            intrinsic = data["intrinsic"].astype(np.float32)
            c2w = ensure_4x4(data["extrinsic_c2w"], "extrinsic_c2w")
            w2c = ensure_4x4(data["extrinsic_w2c"], "extrinsic_w2c")

        if rgb.shape[:2] != depth_raw.shape:
            raise ValueError(f"Depth shape {depth_raw.shape} does not match RGB shape {rgb.shape}")
        inverse_error = float(np.max(np.abs(c2w @ w2c - np.eye(4, dtype=np.float32))))
        inverse_errors.append(inverse_error)

        depth_metric, depth_norm, valid_depth = normalize_depth_for_vlfm(
            depth_raw, args.depth_scale, args.min_depth, args.max_depth
        )
        all_depths.append(depth_metric[valid_depth])

        tf_camera_to_episodic = lingbot_pose_to_vlfm_episodic(c2w, first_w2c, args.axis_mode)
        robot_xy = tf_camera_to_episodic[:2, 3].astype(np.float32)
        robot_heading = local_extract_yaw(tf_camera_to_episodic)
        all_translations.append(tf_camera_to_episodic[:3, 3].astype(np.float64))

        points_camera = backproject_depth_to_vlfm_camera(depth_metric, intrinsic, valid_depth)
        points_episodic = local_transform_points(tf_camera_to_episodic, points_camera)
        finite_points = np.isfinite(points_episodic).all(axis=1)
        points_episodic = points_episodic[finite_points]
        projected_points_bev = points_episodic[:, :2].astype(np.float32)

        frontiers = None
        obstacle_vis = None
        if obstacle_map is not None:
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            topdown_fov = local_get_hfov(fx, rgb.shape[1])
            obstacle_map.update_map(
                depth_norm,
                tf_camera_to_episodic,
                args.min_depth,
                args.max_depth,
                fx,
                fy,
                topdown_fov,
            )
            obstacle_map.update_agent_traj(robot_xy, robot_heading)
            frontiers = obstacle_map.frontiers.astype(np.float32, copy=False)
            frontier_counts.append(int(len(frontiers)))
            vlfm_functions_called.extend(
                [
                    "ObstacleMap.update_map",
                    "ObstacleMap._get_frontiers (via update_map)",
                    "BaseMap.update_agent_traj",
                    "ObstacleMap.visualize",
                ]
            )
            obstacle_vis = obstacle_map.visualize()
        else:
            frontier_counts.append(None)

        step_prefix = f"step_{step_idx:06d}"
        save_png(output_dir / f"{step_prefix}_rgb.png", rgb, rgb_input=True)
        save_png(output_dir / f"{step_prefix}_depth.png", draw_depth(depth_metric, valid_depth, args.min_depth, args.max_depth))
        conf_vis = draw_confidence(depth_conf)
        if conf_vis is not None:
            save_png(output_dir / f"{step_prefix}_depth_conf.png", conf_vis)
        save_png(
            output_dir / f"{step_prefix}_projected_points_bev.png",
            draw_projected_points_bev(projected_points_bev, robot_xy, args.map_size, args.pixels_per_meter),
        )

        step_log: Dict[str, Any] = {
            "rgb": rgb,
            "depth": depth_metric.astype(np.float32),
            "depth_conf": depth_conf.astype(np.float32) if depth_conf is not None else np.array([]),
            "intrinsic": intrinsic,
            "extrinsic_c2w": c2w,
            "extrinsic_w2c": w2c,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "robot_xy": robot_xy,
            "robot_heading": np.array(robot_heading, dtype=np.float32),
            "projected_points_bev": projected_points_bev,
        }

        if obstacle_map is not None:
            step_log["obstacle_map"] = obstacle_map._map.copy()
            step_log["explored_map"] = obstacle_map.explored_area.copy()
            step_log["navigable_map"] = obstacle_map._navigable_map.copy()
            if frontiers is not None:
                step_log["candidate_frontiers"] = frontiers
            save_png(output_dir / f"{step_prefix}_obstacle_map.png", obstacle_vis)
            maps_saved = True
            frontiers_vis = draw_frontiers(obstacle_vis, frontiers, args.map_size, args.pixels_per_meter)
            if frontiers_vis is not None:
                save_png(output_dir / f"{step_prefix}_frontiers.png", frontiers_vis)
                frontiers_saved = True

        npz_path = output_dir / f"{step_prefix}.npz"
        np.savez_compressed(npz_path, **step_log)
        step_results.append(
            StepResult(
                frame_path=str(frame_path),
                output_npz=str(npz_path),
                rgb_shape=tuple(int(v) for v in rgb.shape),
                depth_shape=tuple(int(v) for v in depth_metric.shape),
                inverse_error=inverse_error,
                finite_projected_points=int(points_episodic.shape[0]),
                robot_xy=[float(robot_xy[0]), float(robot_xy[1])],
                robot_heading=robot_heading,
                frontier_count=frontier_counts[-1],
            )
        )

    translations = np.stack(all_translations, axis=0)
    translation_delta = float(np.linalg.norm(translations[-1] - translations[0])) if len(translations) > 1 else 0.0
    unique_called = list(dict.fromkeys(vlfm_functions_called))
    if obstacle_map is not None:
        vlfm_functions_not_called.pop("ObstacleMap.update_map", None)
        vlfm_functions_not_called.pop("ObstacleMap._get_frontiers", None)

    metadata: Dict[str, Any] = {
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frames_processed": len(step_results),
        "depth_scale": args.depth_scale,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "axis_mode": args.axis_mode,
        "coordinate_conversion": (
            "First LingBot camera is the episodic origin. For axis_mode=auto/opencv_first_camera, "
            "relative OpenCV camera transform inv(c2w_0)@c2w_i is converted from "
            "(right, down, forward) to VLFM (forward, left, up) with matrix "
            f"{OPENCV_TO_VLFM_ROT.tolist()}."
        ),
        "obstacle_height_filter": {
            "min_obstacle_height": args.min_obstacle_height,
            "max_obstacle_height": args.max_obstacle_height,
            "note": "Height is relative to first-camera VLFM episodic z, not calibrated ground height.",
        },
        "vlfm_functions_called": unique_called,
        "vlfm_functions_not_called": vlfm_functions_not_called,
        "frontier_counts_per_step": frontier_counts,
        "step_results": [result.__dict__ for result in step_results],
        "depth_stats": merged_stats(all_depths),
        "trajectory_translation_stats": {
            "translation_min": translations.min(axis=0).tolist(),
            "translation_max": translations.max(axis=0).tolist(),
            "translation_mean": translations.mean(axis=0).tolist(),
            "translation_delta_norm": translation_delta,
        },
        "inverse_consistency_max_abs_error": max(inverse_errors) if inverse_errors else None,
        "sanity_tests": {
            "c2w_w2c_inverse_consistency": bool(inverse_errors and max(inverse_errors) < 1e-4),
            "depth_shape_matches_rgb": all(tuple(r.depth_shape) == tuple(r.rgb_shape[:2]) for r in step_results),
            "projected_points_finite_after_filtering": all(r.finite_projected_points > 0 for r in step_results),
            "robot_trajectory_non_nan": bool(np.isfinite(translations).all()),
            "robot_trajectory_nonzero_motion": bool(translation_delta > 1e-6),
            "obstacle_maps_saved_without_habitat": bool(maps_saved),
            "frontier_maps_saved_without_habitat": bool(frontiers_saved),
            "pointnav_call_occurred": False,
            "habitat_env_step_occurred": False,
        },
        "uncertainties": [
            "LingBot depth and trajectory scale are model-predicted and not calibrated.",
            "Relative pose is anchored at the first LingBot camera; absolute LingBot world axes are not assumed.",
            "Obstacle height filtering is relative to camera-frame z because no ground height is exported.",
            "Frontier selection/value scoring is not performed in Stage 2a.",
        ],
    }

    (output_dir / "replay_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "replay_summary.md").write_text(build_summary(metadata))

    print(f"Processed {len(step_results)} frames")
    print(f"Wrote logs to {output_dir}")
    print(f"VLFM functions called: {unique_called}")
    print(f"Frontier counts: {frontier_counts}")
    print(f"Sanity tests: {metadata['sanity_tests']}")


if __name__ == "__main__":
    main()
