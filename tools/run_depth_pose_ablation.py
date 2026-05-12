#!/usr/bin/env python3
"""Offline depth/pose ablations for VLFM Habitat trace vs LingBot outputs.

This script is intentionally logging/analysis-only. It does not run Habitat,
does not call PointNav, does not call ValueMap/VLM servers, and does not modify
frontier selection. It reuses VLFM's ObstacleMap update path to diagnose whether
LingBot depth, LingBot pose, scalar scale, or confidence are responsible for the
current LingBot replay map collapse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_habitat_trace_vs_lingbot_replay import (  # noqa: E402
    colorize,
    panel,
    render_bev,
    rgb_to_bgr,
)
from tools.replay_lingbot_geometry_probe import (  # noqa: E402
    backproject_depth_to_vlfm_camera,
    draw_confidence,
    draw_depth,
    ensure_4x4,
    lingbot_pose_to_vlfm_episodic,
    local_extract_yaw,
    local_get_hfov,
    normalize_depth_for_vlfm,
    save_png,
    squeeze_depth,
    xy_to_px,
)
from vlfm.mapping.obstacle_map import ObstacleMap  # noqa: E402
from vlfm.utils.geometry_utils import transform_points  # noqa: E402


DEFAULT_HABITAT_TRACE = "/scratch/e1538633/lingbot_vlfm_probe/vlfm_habitat_trace_smoke/episode_45"
DEFAULT_LINGBOT_DIR = "/scratch/e1538633/lingbot_vlfm_probe/lingbot_npz_from_habitat_episode_45"
DEFAULT_OUTPUT = "/scratch/e1538633/lingbot_vlfm_probe/vlfm_depth_pose_ablation_episode_45"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    depth_source: str
    pose_source: str
    depth_scale: float = 1.0
    pose_scale: float = 1.0
    save_details: bool = False


def parse_float_list(spec: str) -> List[float]:
    values = [float(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_steps(spec: str, max_step: int) -> List[int]:
    result: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        step = max_step if token == "last" else int(token)
        if 0 <= step <= max_step:
            result.append(step)
    return sorted(set(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--habitat_trace_dir", default=DEFAULT_HABITAT_TRACE)
    parser.add_argument("--lingbot_dir", default=DEFAULT_LINGBOT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth_scales", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--pose_scales", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument(
        "--scale_sweep_stride",
        type=int,
        default=5,
        help="Evaluate scale sweep on every Nth paired frame, then rerun the best scale on full frames.",
    )
    parser.add_argument("--contact_steps", default="0,10,25,50,75,last")
    parser.add_argument("--map_size", type=int, default=1000)
    parser.add_argument("--pixels_per_meter", type=int, default=20)
    parser.add_argument("--habitat_min_depth", type=float, default=0.5)
    parser.add_argument("--habitat_max_depth", type=float, default=5.0)
    parser.add_argument("--habitat_hfov_deg", type=float, default=79.0)
    parser.add_argument("--habitat_camera_height", type=float, default=0.88)
    parser.add_argument("--lingbot_min_depth", type=float, default=0.05)
    parser.add_argument("--lingbot_max_depth", type=float, default=5.0)
    parser.add_argument("--lingbot_pose_camera_height", type=float, default=0.88)
    parser.add_argument("--min_obstacle_height", type=float, default=0.61)
    parser.add_argument("--max_obstacle_height", type=float, default=0.88)
    parser.add_argument("--agent_radius", type=float, default=0.18)
    parser.add_argument("--obstacle_map_area_threshold", type=float, default=1.5)
    parser.add_argument("--hole_area_thresh", type=int, default=100000)
    parser.add_argument("--confidence_radius_m", type=float, default=0.5)
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_step_pairs(
    habitat_trace_dir: Path,
    lingbot_dir: Path,
    max_steps: int,
    stride: int,
) -> Tuple[List[Path], List[Path]]:
    habitat_log_dir = habitat_trace_dir / "logs"
    habitat_paths = sorted(habitat_log_dir.glob("step_*.npz")) if habitat_log_dir.is_dir() else []
    lingbot_paths = sorted(lingbot_dir.glob("frame_*.npz"))
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    habitat_paths = habitat_paths[::stride]
    lingbot_paths = lingbot_paths[::stride]
    n = min(len(habitat_paths), len(lingbot_paths))
    if max_steps > 0:
        n = min(n, max_steps)
    if n <= 0:
        raise FileNotFoundError("No paired Habitat step logs and LingBot frame exports found")
    return habitat_paths[:n], lingbot_paths[:n]


def infer_habitat_intrinsics(width: int, hfov_deg: float) -> Tuple[float, float, float]:
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    return fx, fy, hfov_rad


def make_obstacle_map(args: argparse.Namespace) -> ObstacleMap:
    return ObstacleMap(
        min_height=args.min_obstacle_height,
        max_height=args.max_obstacle_height,
        agent_radius=args.agent_radius,
        area_thresh=args.obstacle_map_area_threshold,
        hole_area_thresh=args.hole_area_thresh,
        size=args.map_size,
        pixels_per_meter=args.pixels_per_meter,
    )


def scale_lingbot_pose(tf: np.ndarray, pose_scale: float, camera_height: float) -> np.ndarray:
    out = np.asarray(tf, dtype=np.float32).copy()
    out[:3, 3] *= np.float32(pose_scale)
    out[2, 3] += np.float32(camera_height)
    return out


def prepare_depth(
    spec: ExperimentSpec,
    habitat: Mapping[str, np.ndarray],
    lingbot: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, float, float, float, float, np.ndarray, np.ndarray]:
    if spec.depth_source == "habitat":
        depth_norm = np.asarray(habitat["depth"], dtype=np.float32)
        h, w = depth_norm.shape
        fx, fy, fov = infer_habitat_intrinsics(w, args.habitat_hfov_deg)
        depth_metric = depth_norm * (args.habitat_max_depth - args.habitat_min_depth) + args.habitat_min_depth
        valid = np.isfinite(depth_norm) & (depth_norm < 1.0)
        return depth_norm, args.habitat_min_depth, args.habitat_max_depth, fx, fy, fov, depth_metric, valid

    if spec.depth_source == "lingbot":
        intrinsic = np.asarray(lingbot["intrinsic"], dtype=np.float32)
        depth_raw = squeeze_depth(lingbot["depth"])
        depth_metric, depth_norm, valid = normalize_depth_for_vlfm(
            depth_raw,
            spec.depth_scale,
            args.lingbot_min_depth,
            args.lingbot_max_depth,
        )
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        fov = local_get_hfov(fx, depth_norm.shape[1])
        return depth_norm, args.lingbot_min_depth, args.lingbot_max_depth, fx, fy, fov, depth_metric, valid

    raise ValueError(f"Unsupported depth source: {spec.depth_source}")


def prepare_pose(
    spec: ExperimentSpec,
    habitat: Mapping[str, np.ndarray],
    lingbot: Mapping[str, np.ndarray],
    first_lingbot_w2c: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if spec.pose_source == "habitat":
        return np.asarray(habitat["tf_camera_to_episodic"], dtype=np.float32)
    if spec.pose_source == "lingbot":
        c2w = ensure_4x4(lingbot["extrinsic_c2w"], "extrinsic_c2w")
        tf = lingbot_pose_to_vlfm_episodic(c2w, first_lingbot_w2c, "auto")
        return scale_lingbot_pose(tf, spec.pose_scale, args.lingbot_pose_camera_height)
    raise ValueError(f"Unsupported pose source: {spec.pose_source}")


def bool_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a).astype(bool)
    b_bool = np.asarray(b).astype(bool)
    union = np.logical_or(a_bool, b_bool)
    if not np.any(union):
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum() / union.sum())


def summarize_series(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}


def render_map_from_arrays(step_data: Mapping[str, np.ndarray]) -> np.ndarray:
    return render_bev({key: np.asarray(value) for key, value in step_data.items()})


def save_experiment_contact_sheet(
    output_path: Path,
    step: int,
    habitat: Mapping[str, np.ndarray],
    experiments: Mapping[str, Optional[Mapping[str, np.ndarray]]],
) -> None:
    text = np.ones((240, 320, 3), dtype=np.uint8) * 255
    lines = [
        f"step: {step}",
        f"hab frontiers: {len(habitat.get('candidate_frontiers', []))}",
        f"hab selected: {np.array2string(habitat.get('selected_frontier', np.array([])), precision=2)[:22]}",
        f"action: {np.asarray(habitat.get('final_action', ['NA'])).reshape(-1)[0]}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(text, line, (10, 36 + idx * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    panels = [
        panel("RGB", rgb_to_bgr(habitat.get("rgb"))),
        panel("Habitat Depth", colorize(habitat.get("depth"))),
        panel("Original VLFM Map", render_bev(dict(habitat))),
        panel("Decision", text),
    ]
    for name in ["lingbot_depth_habitat_pose", "habitat_depth_lingbot_pose", "best_lingbot_scale"]:
        data = experiments.get(name)
        panels.append(panel(name, render_bev(dict(data)) if data is not None else None))
    while len(panels) < 8:
        panels.append(panel("unused", None))
    sheet = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:8])])
    cv2.imwrite(str(output_path), sheet)


def run_obstacle_experiment(
    spec: ExperimentSpec,
    habitat_paths: Sequence[Path],
    lingbot_paths: Sequence[Path],
    first_lingbot_w2c: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    contact_steps: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, np.ndarray]]]:
    obstacle_map = make_obstacle_map(args)
    exp_dir = output_dir / "experiments" / spec.name
    if spec.save_details:
        exp_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, Any]] = []
    saved_steps: Dict[int, Dict[str, np.ndarray]] = {}
    trajectory: List[np.ndarray] = []

    for idx, (habitat_path, lingbot_path) in enumerate(zip(habitat_paths, lingbot_paths)):
        habitat = load_npz(habitat_path)
        lingbot = load_npz(lingbot_path)
        depth_norm, min_depth, max_depth, fx, fy, fov, depth_metric, valid_depth = prepare_depth(
            spec,
            habitat,
            lingbot,
            args,
        )
        tf_camera_to_episodic = prepare_pose(spec, habitat, lingbot, first_lingbot_w2c, args)
        robot_xy = tf_camera_to_episodic[:2, 3].astype(np.float32)
        robot_heading = local_extract_yaw(tf_camera_to_episodic)

        obstacle_map.update_map(depth_norm, tf_camera_to_episodic, min_depth, max_depth, fx, fy, fov)
        obstacle_map.update_agent_traj(robot_xy, robot_heading)
        trajectory.append(robot_xy.copy())

        step_data: Dict[str, np.ndarray] = {
            "step": np.array(idx, dtype=np.int32),
            "depth": depth_metric.astype(np.float32),
            "tf_camera_to_episodic": tf_camera_to_episodic.astype(np.float32),
            "robot_xy": robot_xy.astype(np.float32),
            "robot_heading": np.array(robot_heading, dtype=np.float32),
            "obstacle_map": obstacle_map._map.copy(),
            "explored_map": obstacle_map.explored_area.copy(),
            "navigable_map": obstacle_map._navigable_map.copy(),
            "candidate_frontiers": obstacle_map.frontiers.astype(np.float32, copy=False),
        }
        if spec.depth_source == "lingbot" and "depth_conf" in lingbot:
            step_data["depth_conf"] = np.asarray(lingbot["depth_conf"], dtype=np.float32)

        original_obstacle = habitat.get("obstacle_map")
        original_explored = habitat.get("explored_map")
        metrics.append(
            {
                "experiment": spec.name,
                "step": idx,
                "depth_source": spec.depth_source,
                "pose_source": spec.pose_source,
                "depth_scale": spec.depth_scale,
                "pose_scale": spec.pose_scale,
                "frontier_count": int(len(obstacle_map.frontiers)),
                "explored_pixels": int(np.count_nonzero(obstacle_map.explored_area)),
                "obstacle_pixels": int(np.count_nonzero(obstacle_map._map)),
                "original_frontier_count": int(len(habitat.get("candidate_frontiers", []))),
                "original_explored_pixels": int(np.count_nonzero(original_explored)) if original_explored is not None else None,
                "original_obstacle_pixels": int(np.count_nonzero(original_obstacle)) if original_obstacle is not None else None,
                "explored_iou": bool_iou(obstacle_map.explored_area, original_explored) if original_explored is not None else None,
                "obstacle_iou": bool_iou(obstacle_map._map, original_obstacle) if original_obstacle is not None else None,
            }
        )

        if spec.save_details:
            if idx in contact_steps:
                save_png(exp_dir / f"step_{idx:06d}_depth.png", draw_depth(depth_metric, valid_depth, min_depth, max_depth))
                save_png(exp_dir / f"step_{idx:06d}_map.png", obstacle_map.visualize())
            np.savez_compressed(exp_dir / f"step_{idx:06d}.npz", **step_data)
            saved_steps[idx] = step_data

    frontier_counts = [row["frontier_count"] for row in metrics]
    explored_pixels = [row["explored_pixels"] for row in metrics]
    obstacle_pixels = [row["obstacle_pixels"] for row in metrics]
    explored_iou = [row["explored_iou"] for row in metrics if row["explored_iou"] is not None]
    obstacle_iou = [row["obstacle_iou"] for row in metrics if row["obstacle_iou"] is not None]
    translations = np.asarray(trajectory, dtype=np.float32)
    summary = {
        "name": spec.name,
        "depth_source": spec.depth_source,
        "pose_source": spec.pose_source,
        "depth_scale": spec.depth_scale,
        "pose_scale": spec.pose_scale,
        "frames": len(metrics),
        "frontier_count_stats": summarize_series(frontier_counts),
        "frontier_nonzero_steps": int(np.count_nonzero(frontier_counts)),
        "explored_pixel_stats": summarize_series(explored_pixels),
        "obstacle_pixel_stats": summarize_series(obstacle_pixels),
        "explored_iou_stats": summarize_series(explored_iou),
        "obstacle_iou_stats": summarize_series(obstacle_iou),
        "trajectory_delta_norm": float(np.linalg.norm(translations[-1] - translations[0])) if len(translations) > 1 else 0.0,
        "score_for_ranking": float(np.nanmean(explored_iou) + np.nanmean(obstacle_iou)) if explored_iou and obstacle_iou else 0.0,
        "per_step_metrics": metrics,
    }
    return summary, saved_steps


def write_metrics_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "experiment",
        "step",
        "depth_source",
        "pose_source",
        "depth_scale",
        "pose_scale",
        "frontier_count",
        "explored_pixels",
        "obstacle_pixels",
        "original_frontier_count",
        "original_explored_pixels",
        "original_obstacle_pixels",
        "explored_iou",
        "obstacle_iou",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for row in summary["per_step_metrics"]:
                writer.writerow({key: row.get(key) for key in fieldnames})


def plot_time_series(output_dir: Path, summaries: Sequence[Mapping[str, Any]], key: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))
    for summary in summaries:
        rows = summary["per_step_metrics"]
        steps = [row["step"] for row in rows]
        values = [row[key] for row in rows]
        plt.plot(steps, values, label=str(summary["name"]))
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{key}_over_time.png", dpi=160)
    plt.close()


def plot_scale_heatmap(output_dir: Path, scale_summaries: Sequence[Mapping[str, Any]]) -> None:
    if not scale_summaries:
        return
    depth_values = sorted({float(s["depth_scale"]) for s in scale_summaries})
    pose_values = sorted({float(s["pose_scale"]) for s in scale_summaries})
    grid = np.full((len(pose_values), len(depth_values)), np.nan, dtype=np.float32)
    for summary in scale_summaries:
        r = pose_values.index(float(summary["pose_scale"]))
        c = depth_values.index(float(summary["depth_scale"]))
        grid[r, c] = float(summary["score_for_ranking"])
    plt.figure(figsize=(7, 5))
    plt.imshow(grid, origin="lower", cmap="viridis")
    plt.colorbar(label="mean explored IoU + mean obstacle IoU")
    plt.xticks(range(len(depth_values)), [str(v) for v in depth_values])
    plt.yticks(range(len(pose_values)), [str(v) for v in pose_values])
    plt.xlabel("depth scale")
    plt.ylabel("pose scale")
    plt.title("LingBot depth+pose scale sweep")
    for r, pose in enumerate(pose_values):
        for c, depth in enumerate(depth_values):
            if np.isfinite(grid[r, c]):
                plt.text(c, r, f"{grid[r, c]:.3f}", ha="center", va="center", color="white", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "scale_sweep_similarity_heatmap.png", dpi=180)
    plt.close()


def rasterize_lingbot_confidence_with_habitat_pose(
    habitat: Mapping[str, np.ndarray],
    lingbot: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_raw = squeeze_depth(lingbot["depth"])
    depth_metric, _, valid = normalize_depth_for_vlfm(
        depth_raw,
        1.0,
        args.lingbot_min_depth,
        args.lingbot_max_depth,
    )
    conf = np.asarray(lingbot["depth_conf"], dtype=np.float32)
    intrinsic = np.asarray(lingbot["intrinsic"], dtype=np.float32)
    points_camera = backproject_depth_to_vlfm_camera(depth_metric, intrinsic, valid)
    points_episodic = transform_points(np.asarray(habitat["tf_camera_to_episodic"], dtype=np.float32), points_camera)
    conf_values = conf[valid].astype(np.float32)

    sums = np.zeros((args.map_size, args.map_size), dtype=np.float64)
    counts = np.zeros((args.map_size, args.map_size), dtype=np.uint16)
    px = xy_to_px(points_episodic[:, :2], args.map_size, args.pixels_per_meter)
    in_bounds = (px[:, 0] >= 0) & (px[:, 0] < args.map_size) & (px[:, 1] >= 0) & (px[:, 1] < args.map_size)
    px = px[in_bounds]
    vals = conf_values[in_bounds]
    np.add.at(sums, (px[:, 1], px[:, 0]), vals)
    np.add.at(counts, (px[:, 1], px[:, 0]), 1)
    mean_conf = np.full((args.map_size, args.map_size), np.nan, dtype=np.float32)
    mask = counts > 0
    mean_conf[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return mean_conf, counts


def confidence_stats_near_frontiers(
    habitat_paths: Sequence[Path],
    lingbot_paths: Sequence[Path],
    output_dir: Path,
    args: argparse.Namespace,
    contact_steps: Sequence[int],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    conf_dir = output_dir / "confidence"
    conf_dir.mkdir(parents=True, exist_ok=True)
    radius_px = max(1, int(round(args.confidence_radius_m * args.pixels_per_meter)))

    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    disk = (xx * xx + yy * yy) <= radius_px * radius_px

    for idx, (habitat_path, lingbot_path) in enumerate(zip(habitat_paths, lingbot_paths)):
        habitat = load_npz(habitat_path)
        lingbot = load_npz(lingbot_path)
        if "candidate_frontiers" not in habitat or "depth_conf" not in lingbot:
            continue
        conf_map, count_map = rasterize_lingbot_confidence_with_habitat_pose(habitat, lingbot, args)
        frontiers = np.asarray(habitat["candidate_frontiers"], dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(habitat["frontier_scores"], dtype=np.float32).reshape(-1) if "frontier_scores" in habitat else np.full(len(frontiers), np.nan)
        selected = np.asarray(habitat["selected_frontier"], dtype=np.float32).reshape(-1, 2) if "selected_frontier" in habitat else np.empty((0, 2), dtype=np.float32)
        robot_xy = np.asarray(habitat["robot_xy"], dtype=np.float32).reshape(2)
        frontier_px = xy_to_px(frontiers, args.map_size, args.pixels_per_meter) if len(frontiers) else np.empty((0, 2), dtype=np.int32)

        for frontier_id, (frontier, px) in enumerate(zip(frontiers, frontier_px)):
            x, y = int(px[0]), int(px[1])
            x0, x1 = max(0, x - radius_px), min(args.map_size, x + radius_px + 1)
            y0, y1 = max(0, y - radius_px), min(args.map_size, y + radius_px + 1)
            local = conf_map[y0:y1, x0:x1]
            local_disk = disk[(y0 - (y - radius_px)) : (y1 - (y - radius_px)), (x0 - (x - radius_px)) : (x1 - (x - radius_px))]
            values = local[local_disk & np.isfinite(local)]
            selected_by_vlfm = bool(len(selected) > 0 and np.linalg.norm(frontier - selected[0]) < 1e-3)
            row = {
                "step": idx,
                "frontier_id": frontier_id,
                "frontier_x": float(frontier[0]),
                "frontier_y": float(frontier[1]),
                "original_vlfm_score": float(scores[frontier_id]) if frontier_id < len(scores) and np.isfinite(scores[frontier_id]) else None,
                "selected_by_original_vlfm": selected_by_vlfm,
                "distance_to_robot": float(np.linalg.norm(frontier - robot_xy)),
                "num_conf_pixels": int(values.size),
                "mean_depth_conf_near_frontier": float(values.mean()) if values.size else None,
                "min_depth_conf_near_frontier": float(values.min()) if values.size else None,
                "max_depth_conf_near_frontier": float(values.max()) if values.size else None,
                "var_depth_conf_near_frontier": float(values.var()) if values.size else None,
            }
            rows.append(row)

        if idx in contact_steps:
            conf_vis = draw_confidence(conf_map)
            if conf_vis is not None:
                for px in frontier_px:
                    cv2.circle(conf_vis, (int(px[0]), int(px[1])), radius_px, (0, 0, 255), 2)
                save_png(conf_dir / f"step_{idx:06d}_confidence_frontiers.png", conf_vis)
            np.savez_compressed(conf_dir / f"step_{idx:06d}_confidence_bev.npz", confidence_map=conf_map, confidence_count=count_map)

    csv_path = conf_dir / "confidence_frontier_stats.csv"
    fieldnames = [
        "step",
        "frontier_id",
        "frontier_x",
        "frontier_y",
        "original_vlfm_score",
        "selected_by_original_vlfm",
        "distance_to_robot",
        "num_conf_pixels",
        "mean_depth_conf_near_frontier",
        "min_depth_conf_near_frontier",
        "max_depth_conf_near_frontier",
        "var_depth_conf_near_frontier",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected_means = [r["mean_depth_conf_near_frontier"] for r in rows if r["selected_by_original_vlfm"] and r["mean_depth_conf_near_frontier"] is not None]
    nonselected_means = [r["mean_depth_conf_near_frontier"] for r in rows if not r["selected_by_original_vlfm"] and r["mean_depth_conf_near_frontier"] is not None]
    if selected_means or nonselected_means:
        plt.figure(figsize=(6, 4))
        data = [selected_means, nonselected_means]
        plt.boxplot(data, labels=["selected", "non-selected"], showmeans=True)
        plt.ylabel("mean LingBot depth_conf near original frontier")
        plt.tight_layout()
        plt.savefig(conf_dir / "confidence_selected_vs_nonselected.png", dpi=160)
        plt.close()

    summary = {
        "rows": len(rows),
        "csv": str(csv_path),
        "radius_m": args.confidence_radius_m,
        "selected_frontier_conf_stats": summarize_series(selected_means),
        "nonselected_frontier_conf_stats": summarize_series(nonselected_means),
        "notes": [
            "Confidence is rasterized using LingBot depth and Habitat pose to keep frontier coordinates in the original VLFM episodic frame.",
            "This is logging-only; frontier selection is not modified.",
        ],
    }
    with (conf_dir / "confidence_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (conf_dir / "confidence_summary.md").open("w") as f:
        f.write("# Confidence Near Original VLFM Frontiers\n\n")
        f.write(f"- rows: `{summary['rows']}`\n")
        f.write(f"- csv: `{summary['csv']}`\n")
        f.write(f"- radius_m: `{summary['radius_m']}`\n")
        f.write(f"- selected_frontier_conf_stats: `{summary['selected_frontier_conf_stats']}`\n")
        f.write(f"- nonselected_frontier_conf_stats: `{summary['nonselected_frontier_conf_stats']}`\n")
        f.write("\n## Notes\n")
        for note in summary["notes"]:
            f.write(f"- {note}\n")
    return summary


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    summaries: Sequence[Mapping[str, Any]],
    scale_summaries: Sequence[Mapping[str, Any]],
    best_scale: Mapping[str, Any],
    confidence_summary: Mapping[str, Any],
) -> None:
    compact = []
    for summary in summaries:
        compact.append(
            {
                "name": summary["name"],
                "depth_source": summary["depth_source"],
                "pose_source": summary["pose_source"],
                "depth_scale": summary["depth_scale"],
                "pose_scale": summary["pose_scale"],
                "frames": summary["frames"],
                "frontier_count_stats": summary["frontier_count_stats"],
                "frontier_nonzero_steps": summary["frontier_nonzero_steps"],
                "explored_pixel_stats": summary["explored_pixel_stats"],
                "obstacle_pixel_stats": summary["obstacle_pixel_stats"],
                "explored_iou_stats": summary["explored_iou_stats"],
                "obstacle_iou_stats": summary["obstacle_iou_stats"],
                "trajectory_delta_norm": summary["trajectory_delta_norm"],
                "score_for_ranking": summary["score_for_ranking"],
            }
        )
    metadata = {
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "habitat_trace_dir": args.habitat_trace_dir,
        "lingbot_dir": args.lingbot_dir,
        "output_dir": args.output_dir,
        "experiments": compact,
        "scale_sweep_experiment_count": len(scale_summaries),
        "scale_sweep_stride": args.scale_sweep_stride,
        "best_scale_experiment": {
            "name": best_scale["name"],
            "depth_scale": best_scale["depth_scale"],
            "pose_scale": best_scale["pose_scale"],
            "score_for_ranking": best_scale["score_for_ranking"],
            "frontier_count_stats": best_scale["frontier_count_stats"],
            "explored_iou_stats": best_scale["explored_iou_stats"],
            "obstacle_iou_stats": best_scale["obstacle_iou_stats"],
        },
        "confidence_summary": confidence_summary,
        "uncertainties": [
            "Habitat intrinsics are inferred from image width and HFOV=79 because H1 logs did not store intrinsics directly.",
            "LingBot pose is anchored at the first LingBot frame and lifted by camera_height=0.88m for floor-frame compatibility.",
            "LingBot depth and pose scales remain model-predicted; the scale sweep is diagnostic, not a calibrated solution.",
            "Confidence is only logged near original VLFM frontiers; it is not used for selection.",
        ],
        "guardrails": [
            "No Habitat environment step is run.",
            "No PointNav call is run.",
            "No ValueMap/VLM server is run.",
            "No frontier selection behavior is modified.",
        ],
    }
    with (output_dir / "ablation_summary.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    lines = ["# Depth/Pose Ablation Summary\n"]
    lines.append(f"- habitat_trace_dir: `{args.habitat_trace_dir}`")
    lines.append(f"- lingbot_dir: `{args.lingbot_dir}`")
    lines.append(f"- output_dir: `{args.output_dir}`")
    lines.append("")
    lines.append("## Core Experiments")
    for summary in compact:
        if summary["name"].startswith("scale_"):
            continue
        lines.append(
            f"- `{summary['name']}`: frontier_mean={summary['frontier_count_stats']['mean']}, "
            f"frontier_nonzero_steps={summary['frontier_nonzero_steps']}, "
            f"explored_iou_mean={summary['explored_iou_stats']['mean']}, "
            f"obstacle_iou_mean={summary['obstacle_iou_stats']['mean']}, "
            f"trajectory_delta_norm={summary['trajectory_delta_norm']:.3f}"
        )
    lines.append("")
    lines.append("## Best Scale Sweep Result")
    lines.append(
        f"- `{best_scale['name']}`: depth_scale={best_scale['depth_scale']}, "
        f"pose_scale={best_scale['pose_scale']}, score={best_scale['score_for_ranking']:.6f}"
    )
    lines.append(f"- scale_sweep_stride: `{args.scale_sweep_stride}`")
    lines.append(f"- frontier_count_stats: `{best_scale['frontier_count_stats']}`")
    lines.append(f"- explored_iou_stats: `{best_scale['explored_iou_stats']}`")
    lines.append(f"- obstacle_iou_stats: `{best_scale['obstacle_iou_stats']}`")
    lines.append("")
    lines.append("## Confidence Logging")
    lines.append(f"- rows: `{confidence_summary.get('rows')}`")
    lines.append(f"- csv: `{confidence_summary.get('csv')}`")
    lines.append(f"- selected_frontier_conf_stats: `{confidence_summary.get('selected_frontier_conf_stats')}`")
    lines.append(f"- nonselected_frontier_conf_stats: `{confidence_summary.get('nonselected_frontier_conf_stats')}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- `ablation_metrics.csv`")
    lines.append("- `frontier_count_over_time.png`")
    lines.append("- `explored_pixels_over_time.png`")
    lines.append("- `obstacle_pixels_over_time.png`")
    lines.append("- `explored_iou_over_time.png`")
    lines.append("- `obstacle_iou_over_time.png`")
    lines.append("- `scale_sweep_similarity_heatmap.png`")
    lines.append("- `contact_sheets/`")
    lines.append("- `confidence/`")
    lines.append("")
    lines.append("## Notes")
    for note in metadata["uncertainties"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Guardrails")
    for guardrail in metadata["guardrails"]:
        lines.append(f"- {guardrail}")
    (output_dir / "ablation_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)

    habitat_paths, lingbot_paths = load_step_pairs(
        Path(args.habitat_trace_dir),
        Path(args.lingbot_dir),
        args.max_steps,
        args.stride,
    )
    contact_steps = parse_steps(args.contact_steps, len(habitat_paths) - 1)
    first_lingbot = load_npz(lingbot_paths[0])
    first_lingbot_w2c = ensure_4x4(first_lingbot["extrinsic_w2c"], "extrinsic_w2c")

    core_specs = [
        ExperimentSpec("lingbot_depth_habitat_pose", "lingbot", "habitat", 1.0, 1.0, True),
        ExperimentSpec("habitat_depth_lingbot_pose", "habitat", "lingbot", 1.0, 1.0, True),
    ]
    depth_scales = parse_float_list(args.depth_scales)
    pose_scales = parse_float_list(args.pose_scales)
    scale_specs = [
        ExperimentSpec(f"scale_d{depth_scale:g}_p{pose_scale:g}", "lingbot", "lingbot", depth_scale, pose_scale, False)
        for depth_scale in depth_scales
        for pose_scale in pose_scales
    ]
    if args.scale_sweep_stride < 1:
        raise ValueError("--scale_sweep_stride must be >= 1")
    scale_habitat_paths = habitat_paths[:: args.scale_sweep_stride]
    scale_lingbot_paths = lingbot_paths[:: args.scale_sweep_stride]

    all_summaries: List[Dict[str, Any]] = []
    saved_core_steps: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for spec in core_specs:
        summary, saved_steps = run_obstacle_experiment(
            spec,
            habitat_paths,
            lingbot_paths,
            first_lingbot_w2c,
            output_dir,
            args,
            contact_steps,
        )
        all_summaries.append(summary)
        saved_core_steps[spec.name] = saved_steps

    scale_summaries: List[Dict[str, Any]] = []
    for spec in scale_specs:
        summary, _ = run_obstacle_experiment(
            spec,
            scale_habitat_paths,
            scale_lingbot_paths,
            first_lingbot_w2c,
            output_dir,
            args,
            contact_steps,
        )
        scale_summaries.append(summary)
    best_scale = max(scale_summaries, key=lambda item: float(item["score_for_ranking"]))
    best_spec = ExperimentSpec(
        "best_lingbot_scale",
        "lingbot",
        "lingbot",
        float(best_scale["depth_scale"]),
        float(best_scale["pose_scale"]),
        True,
    )
    best_summary, best_saved_steps = run_obstacle_experiment(
        best_spec,
        habitat_paths,
        lingbot_paths,
        first_lingbot_w2c,
        output_dir,
        args,
        contact_steps,
    )
    all_summaries.append(best_summary)
    saved_core_steps["best_lingbot_scale"] = best_saved_steps

    write_metrics_csv(output_dir / "ablation_metrics.csv", [*all_summaries, *scale_summaries])
    plot_time_series(output_dir, all_summaries, "frontier_count", "candidate frontier count")
    plot_time_series(output_dir, all_summaries, "explored_pixels", "explored pixels")
    plot_time_series(output_dir, all_summaries, "obstacle_pixels", "obstacle pixels")
    plot_time_series(output_dir, all_summaries, "explored_iou", "explored IoU vs original trace")
    plot_time_series(output_dir, all_summaries, "obstacle_iou", "obstacle IoU vs original trace")
    plot_scale_heatmap(output_dir, scale_summaries)

    for step in contact_steps:
        habitat = load_npz(habitat_paths[step])
        sheet_inputs = {name: steps.get(step) for name, steps in saved_core_steps.items()}
        save_experiment_contact_sheet(
            output_dir / "contact_sheets" / f"ablation_contact_step_{step:06d}.png",
            step,
            habitat,
            sheet_inputs,
        )

    confidence_summary = confidence_stats_near_frontiers(
        habitat_paths,
        lingbot_paths,
        output_dir,
        args,
        contact_steps,
    )
    write_summary(output_dir, args, all_summaries, scale_summaries, best_summary, confidence_summary)
    print(f"Wrote ablation outputs to {output_dir}")
    print(f"Best scale: depth={best_summary['depth_scale']} pose={best_summary['pose_scale']}")


if __name__ == "__main__":
    main()
