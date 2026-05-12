#!/usr/bin/env python3
"""Analyze VLFM Habitat decision trace logs.

This is an additive offline analyzer for traces produced by
``VLFM_HABITAT_TRACE_DIR``.  It does not import or run Habitat.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_INPUT = "/scratch/e1538633/lingbot_vlfm_probe/vlfm_habitat_trace_smoke"
DEFAULT_OUTPUT = "/scratch/e1538633/lingbot_vlfm_probe/vlfm_habitat_trace_analysis"
PIXELS_PER_METER = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--contact_steps", default="0,10,25,50,100,last")
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def normalize_u8(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape[:2], dtype=np.uint8)
    lo = float(np.nanmin(image[finite]))
    hi = float(np.nanmax(image[finite]))
    if hi <= lo:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    out = (np.clip(image, lo, hi) - lo) / (hi - lo)
    return (out * 255).astype(np.uint8)


def colorize(arr: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    return cv2.applyColorMap(normalize_u8(arr), colormap)


def to_bgr_rgb(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    if image.ndim == 2:
        return cv2.cvtColor(normalize_u8(image), cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def xy_to_px(points: np.ndarray, shape: Tuple[int, int], ppm: int = PIXELS_PER_METER) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    origin = np.array([shape[0] // 2, shape[1] // 2], dtype=np.float32)
    px = np.rint(pts[:, ::-1] * ppm) + origin
    px[:, 0] = shape[0] - px[:, 0]
    return px.astype(int)


def render_bev(data: Dict[str, np.ndarray], trajectory: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    obstacle = data.get("obstacle_map")
    explored = data.get("explored_map")
    navigable = data.get("navigable_map")
    if obstacle is None and explored is None and navigable is None:
        return None

    shape = np.asarray(next(x for x in [obstacle, explored, navigable] if x is not None)).shape[:2]
    vis = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
    if explored is not None:
        vis[np.asarray(explored).astype(bool)] = (200, 255, 200)
    if navigable is not None:
        vis[np.asarray(navigable) == 0] = (110, 110, 110)
    if obstacle is not None:
        vis[np.asarray(obstacle).astype(bool)] = (0, 0, 0)

    frontiers = data.get("candidate_frontiers")
    if frontiers is not None and len(frontiers) > 0:
        for px in xy_to_px(frontiers, shape):
            cv2.circle(vis, (int(px[0]), int(px[1])), 5, (0, 0, 255), 2)

    selected = data.get("selected_frontier")
    if selected is not None and np.asarray(selected).size >= 2:
        px = xy_to_px(np.asarray(selected).reshape(1, 2), shape)[0]
        cv2.circle(vis, (int(px[0]), int(px[1])), 8, (0, 255, 255), 2)

    if trajectory is not None and len(trajectory) > 0:
        traj_px = xy_to_px(np.asarray(trajectory), shape)
        for p0, p1 in zip(traj_px[:-1], traj_px[1:]):
            cv2.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 2)
        cv2.circle(vis, (int(traj_px[-1, 0]), int(traj_px[-1, 1])), 7, (255, 0, 0), -1)

    return cv2.flip(vis, 0)


def panel(label: str, image: Optional[np.ndarray], target: Tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = target
    if image is None:
        image = np.ones((h, w, 3), dtype=np.uint8) * 245
        cv2.putText(image, "unavailable", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    else:
        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (w, 26), (255, 255, 255), -1)
    cv2.putText(image, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def make_contact_sheet(step_data: Dict[str, np.ndarray], trajectory: np.ndarray) -> np.ndarray:
    action = scalar(step_data.get("final_action"), "NA")
    mode = scalar(step_data.get("mode"), "NA")
    selected = step_data.get("selected_frontier")
    selected_text = "none" if selected is None else np.array2string(np.asarray(selected), precision=2)
    text_img = np.ones((240, 320, 3), dtype=np.uint8) * 255
    lines = [
        f"step: {scalar(step_data.get('step'), 'NA')}",
        f"mode: {mode}",
        f"action: {action}",
        f"frontiers: {len(step_data.get('candidate_frontiers', []))}",
        f"selected: {selected_text}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(text_img, line[:45], (10, 35 + idx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1)

    value = step_data.get("value_map")
    if value is not None and value.ndim == 3:
        value = np.max(value, axis=-1)

    panels = [
        panel("RGB", to_bgr_rgb(step_data["rgb"]) if "rgb" in step_data else None),
        panel("Depth", colorize(step_data["depth"]) if "depth" in step_data else None),
        panel("Obstacle/Frontiers", render_bev(step_data, trajectory)),
        panel("Explored/Navigable", render_bev({k: v for k, v in step_data.items() if k in {"explored_map", "navigable_map"}})),
        panel("Value Map", colorize(value) if value is not None else None),
        panel("Decision", text_img),
    ]
    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    return np.vstack([top, bottom])


def save_line_plot(path: Path, xs: Sequence[int], ys: Sequence[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", linewidth=1, markersize=2)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_contact_steps(spec: str, max_step: int) -> List[int]:
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "last":
            out.append(max_step)
        else:
            out.append(int(token))
    return sorted(set(s for s in out if 0 <= s <= max_step))


def summarize_episode(ep_dir: Path, output_dir: Path, contact_spec: str) -> Dict[str, Any]:
    log_paths = sorted((ep_dir / "logs").glob("step_*.npz"))
    if not log_paths:
        raise FileNotFoundError(f"No step logs found under {ep_dir / 'logs'}")

    steps: List[int] = []
    frontier_counts: List[int] = []
    selected_distances: List[float] = []
    selected_distance_steps: List[int] = []
    value_scores: List[float] = []
    value_score_steps: List[int] = []
    actions: List[int] = []
    trajectory: List[np.ndarray] = []
    field_presence: Counter[str] = Counter()
    sample_shapes: Dict[str, Dict[str, Any]] = {}

    all_step_data: Dict[int, Dict[str, np.ndarray]] = {}
    for path in log_paths:
        data = load_npz(path)
        step = int(scalar(data.get("step"), len(steps)))
        steps.append(step)
        all_step_data[step] = data
        for key, value in data.items():
            field_presence[key] += 1
            if key not in sample_shapes:
                sample_shapes[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}

        frontiers = data.get("candidate_frontiers")
        frontier_counts.append(0 if frontiers is None else int(len(frontiers)))

        action = data.get("final_action")
        if action is not None:
            actions.append(int(np.asarray(action).reshape(-1)[0]))

        robot_xy = data.get("robot_xy")
        if robot_xy is not None and np.asarray(robot_xy).size >= 2:
            trajectory.append(np.asarray(robot_xy).reshape(-1)[:2].astype(np.float32))

        selected = data.get("selected_frontier")
        if selected is not None and robot_xy is not None and np.asarray(selected).size >= 2:
            selected_distances.append(float(np.linalg.norm(np.asarray(selected).reshape(2) - np.asarray(robot_xy).reshape(2))))
            selected_distance_steps.append(step)

        selected_value = data.get("selected_frontier_value")
        if selected_value is not None:
            value_scores.append(float(scalar(selected_value)))
            value_score_steps.append(step)
        elif "frontier_scores" in data and len(data["frontier_scores"]) > 0:
            value_scores.append(float(np.max(data["frontier_scores"])))
            value_score_steps.append(step)

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / ep_dir.name / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    save_line_plot(output_dir / "frontier_count_over_time.png", steps, frontier_counts, "Frontier Count Over Time", "frontiers")
    if selected_distances:
        save_line_plot(
            output_dir / "selected_frontier_distance_over_time.png",
            selected_distance_steps,
            selected_distances,
            "Selected Frontier Distance Over Time",
            "meters",
        )
    if value_scores:
        save_line_plot(output_dir / "value_score_over_time.png", value_score_steps, value_scores, "Value Score Over Time", "score")

    if actions:
        counts = Counter(actions)
        plt.figure(figsize=(6, 4))
        plt.bar([str(k) for k in sorted(counts)], [counts[k] for k in sorted(counts)])
        plt.title("Action Histogram")
        plt.xlabel("action id")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / "action_histogram.png")
        plt.close()

    traj_arr = np.asarray(trajectory, dtype=np.float32)
    last_data = all_step_data[steps[-1]]
    traj_img = render_bev(last_data, traj_arr)
    if traj_img is not None:
        cv2.imwrite(str(output_dir / "trajectory_on_map.png"), traj_img)

    contact_paths = []
    for step in parse_contact_steps(contact_spec, steps[-1]):
        if step not in all_step_data:
            continue
        traj_until = traj_arr[: steps.index(step) + 1] if len(traj_arr) else traj_arr
        sheet = make_contact_sheet(all_step_data[step], traj_until)
        out_path = contact_dir / f"contact_sheet_step_{step:06d}.png"
        cv2.imwrite(str(out_path), sheet)
        contact_paths.append(str(out_path))

    expected_fields = [
        "rgb",
        "depth",
        "gps",
        "compass",
        "heading",
        "robot_xy",
        "robot_heading",
        "tf_camera_to_episodic",
        "obstacle_map",
        "explored_map",
        "navigable_map",
        "value_map",
        "value_confidence_map",
        "candidate_frontiers",
        "frontier_scores",
        "sorted_frontiers",
        "selected_frontier",
        "selected_local_goal",
        "final_action",
        "stop_called",
    ]
    unavailable = {}
    for field in expected_fields:
        if field_presence[field] == 0:
            unavailable[field] = "not present in any step log"
        elif field_presence[field] < len(log_paths):
            unavailable[field] = (
                f"present in {field_presence[field]} of {len(log_paths)} steps; "
                "typically absent outside explore/frontier-selection steps"
            )

    summary_path = ep_dir / "episode_summary.json"
    episode_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    action_hist = {str(k): int(v) for k, v in Counter(actions).items()}
    frontier_count_stats = {
        "min": int(np.min(frontier_counts)),
        "max": int(np.max(frontier_counts)),
        "mean": float(np.mean(frontier_counts)),
        "distribution": {str(k): int(v) for k, v in Counter(frontier_counts).items()},
    }
    trajectory_stats = {}
    if len(traj_arr):
        trajectory_stats = {
            "min_xy": traj_arr.min(axis=0).tolist(),
            "max_xy": traj_arr.max(axis=0).tolist(),
            "delta_norm": float(np.linalg.norm(traj_arr[-1] - traj_arr[0])),
            "num_points": int(len(traj_arr)),
        }

    return {
        "episode_dir": str(ep_dir),
        "num_steps": len(log_paths),
        "steps": steps,
        "episode_summary": episode_summary,
        "sample_shapes": sample_shapes,
        "available_fields": sorted(field_presence.keys()),
        "unavailable_or_partial_fields": unavailable,
        "frontier_count_stats": frontier_count_stats,
        "selected_frontier_distance_stats": {
            "available": bool(selected_distances),
            "min": float(np.min(selected_distances)) if selected_distances else None,
            "max": float(np.max(selected_distances)) if selected_distances else None,
            "mean": float(np.mean(selected_distances)) if selected_distances else None,
        },
        "value_score_stats": {
            "available": bool(value_scores),
            "min": float(np.min(value_scores)) if value_scores else None,
            "max": float(np.max(value_scores)) if value_scores else None,
            "mean": float(np.mean(value_scores)) if value_scores else None,
        },
        "action_histogram": action_hist,
        "trajectory_stats": trajectory_stats,
        "contact_sheets": contact_paths,
    }


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    lines = ["# VLFM Habitat Trace Analysis", ""]
    lines.append(f"- input_dir: `{summary['input_dir']}`")
    lines.append(f"- output_dir: `{summary['output_dir']}`")
    lines.append(f"- episodes_analyzed: `{len(summary['episodes'])}`")
    lines.append("")
    lines.append("## Outputs")
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    for ep in summary["episodes"]:
        lines.extend(
            [
                "",
                f"## {Path(ep['episode_dir']).name}",
                f"- steps: `{ep['num_steps']}`",
                f"- frontier_count_stats: `{ep['frontier_count_stats']}`",
                f"- action_histogram: `{ep['action_histogram']}`",
                f"- trajectory_stats: `{ep['trajectory_stats']}`",
                f"- selected_frontier_distance_stats: `{ep['selected_frontier_distance_stats']}`",
                f"- value_score_stats: `{ep['value_score_stats']}`",
                "",
                "### Available Fields",
            ]
        )
        lines.extend([f"- `{field}`" for field in ep["available_fields"]])
        lines.append("")
        lines.append("### Unavailable Or Partial Fields")
        if ep["unavailable_or_partial_fields"]:
            lines.extend([f"- `{k}`: {v}" for k, v in ep["unavailable_or_partial_fields"].items()])
        else:
            lines.append("- none")
        lines.append("")
        lines.append("### Contact Sheets")
        lines.extend([f"- `{path}`" for path in ep["contact_sheets"]])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    episode_dirs = sorted(p for p in input_dir.glob("episode_*") if (p / "logs").is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* trace directories found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [summarize_episode(ep_dir, output_dir, args.contact_steps) for ep_dir in episode_dirs]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "episodes": episodes,
        "outputs": {
            "trace_analysis_summary_md": str(output_dir / "trace_analysis_summary.md"),
            "trace_analysis_summary_json": str(output_dir / "trace_analysis_summary.json"),
            "frontier_count_over_time": str(output_dir / "frontier_count_over_time.png"),
            "selected_frontier_distance_over_time": str(output_dir / "selected_frontier_distance_over_time.png"),
            "value_score_over_time": str(output_dir / "value_score_over_time.png"),
            "action_histogram": str(output_dir / "action_histogram.png"),
            "trajectory_on_map": str(output_dir / "trajectory_on_map.png"),
        },
    }
    (output_dir / "trace_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "trace_analysis_summary.md")
    print(json.dumps({"output_dir": str(output_dir), "episodes": len(episodes)}, indent=2))


if __name__ == "__main__":
    main()
