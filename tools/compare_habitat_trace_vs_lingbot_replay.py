#!/usr/bin/env python3
"""Compare original VLFM Habitat trace logs with LingBot-driven geometry replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_TRACE = "/scratch/e1538633/lingbot_vlfm_probe/vlfm_habitat_trace_smoke/episode_45"
DEFAULT_REPLAY = "/scratch/e1538633/lingbot_vlfm_probe/vlfm_lingbot_replay_from_habitat_episode_45"
DEFAULT_OUTPUT = "/scratch/e1538633/lingbot_vlfm_probe/paired_habitat_lingbot_comparison_episode_45"
PIXELS_PER_METER = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--habitat_trace_dir", default=DEFAULT_TRACE)
    parser.add_argument("--lingbot_replay_dir", default=DEFAULT_REPLAY)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--contact_steps", default="0,10,25,50,75,99")
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
    return ((np.clip(image, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def colorize(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return cv2.applyColorMap(normalize_u8(arr), cv2.COLORMAP_INFERNO)


def rgb_to_bgr(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    image = np.asarray(arr)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def xy_to_px(points: np.ndarray, shape: Tuple[int, int], ppm: int = PIXELS_PER_METER) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    origin = np.array([shape[0] // 2, shape[1] // 2], dtype=np.float32)
    px = np.rint(pts[:, ::-1] * ppm) + origin
    px[:, 0] = shape[0] - px[:, 0]
    return px.astype(int)


def render_bev(data: Optional[Dict[str, np.ndarray]], trajectory: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if data is None:
        return None
    obstacle = data.get("obstacle_map")
    explored = data.get("explored_map")
    navigable = data.get("navigable_map")
    candidates = data.get("candidate_frontiers")
    if obstacle is None and explored is None and navigable is None:
        return None

    shape = np.asarray(next(v for v in [obstacle, explored, navigable] if v is not None)).shape[:2]
    vis = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
    if explored is not None:
        vis[np.asarray(explored).astype(bool)] = (200, 255, 200)
    if navigable is not None:
        vis[np.asarray(navigable) == 0] = (110, 110, 110)
    if obstacle is not None:
        vis[np.asarray(obstacle).astype(bool)] = (0, 0, 0)
    if candidates is not None and len(candidates) > 0:
        for px in xy_to_px(candidates, shape):
            cv2.circle(vis, (int(px[0]), int(px[1])), 5, (0, 0, 255), 2)
    selected = data.get("selected_frontier")
    if selected is not None and np.asarray(selected).size >= 2:
        px = xy_to_px(np.asarray(selected).reshape(1, 2), shape)[0]
        cv2.circle(vis, (int(px[0]), int(px[1])), 8, (0, 255, 255), 2)
    if trajectory is not None and len(trajectory) > 0:
        traj_px = xy_to_px(trajectory, shape)
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
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (w, 26), (255, 255, 255), -1)
    cv2.putText(image, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def step_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_steps(log_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    return {step_number(path): load_npz(path) for path in sorted(log_dir.glob("step_*.npz"))}


def resolve_log_dir(root: Path) -> Path:
    nested = root / "logs"
    if nested.is_dir() and any(nested.glob("step_*.npz")):
        return nested
    return root


def field_count(data: Optional[Dict[str, np.ndarray]], key: str) -> Optional[int]:
    if data is None or key not in data:
        return None
    return int(len(data[key]))


def pixel_count(data: Optional[Dict[str, np.ndarray]], key: str) -> Optional[int]:
    if data is None or key not in data:
        return None
    return int(np.count_nonzero(data[key]))


def fmt_array(arr: Optional[np.ndarray], max_items: int = 12) -> str:
    if arr is None:
        return ""
    flat = np.asarray(arr).reshape(-1)
    if flat.size > max_items:
        return np.array2string(flat[:max_items], precision=3) + f" ... ({flat.size} values)"
    return np.array2string(np.asarray(arr), precision=3)


def parse_contact_steps(spec: str, available_steps: Sequence[int]) -> List[int]:
    max_step = max(available_steps)
    result = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        step = max_step if token == "last" else int(token)
        if step in available_steps:
            result.append(step)
    return sorted(set(result))


def make_contact_sheet(
    step: int,
    habitat: Optional[Dict[str, np.ndarray]],
    replay: Optional[Dict[str, np.ndarray]],
    habitat_traj: np.ndarray,
    replay_traj: np.ndarray,
) -> np.ndarray:
    text = np.ones((240, 320, 3), dtype=np.uint8) * 255
    lines = [
        f"step: {step}",
        f"hab frontiers: {field_count(habitat, 'candidate_frontiers')}",
        f"lb frontiers: {field_count(replay, 'candidate_frontiers')}",
        f"hab action: {scalar(habitat.get('final_action') if habitat else None, 'NA')}",
        f"hab selected: {fmt_array(habitat.get('selected_frontier') if habitat else None, 4)[:26]}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(text, line, (10, 35 + idx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    trajectory_overlay = render_bev(habitat, habitat_traj)
    if trajectory_overlay is None:
        trajectory_overlay = render_bev(replay, replay_traj)

    panels = [
        panel("RGB", rgb_to_bgr(habitat.get("rgb") if habitat else replay.get("rgb") if replay else None)),
        panel("Habitat Depth", colorize(habitat.get("depth") if habitat else None)),
        panel("LingBot Depth", colorize(replay.get("depth") if replay else None)),
        panel("LingBot Depth Conf", colorize(replay.get("depth_conf") if replay else None)),
        panel("VLFM Habitat Map", render_bev(habitat, habitat_traj)),
        panel("LingBot Replay Map", render_bev(replay, replay_traj)),
        panel("Decision/Table", text),
        panel("Trajectory Overlay", trajectory_overlay),
    ]
    top = np.hstack(panels[:4])
    bottom = np.hstack(panels[4:])
    return np.vstack([top, bottom])


def write_markdown_table(rows: List[Dict[str, Any]], path: Path) -> None:
    headers = list(rows[0].keys()) if rows else []
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    habitat_dir = Path(args.habitat_trace_dir)
    replay_dir = Path(args.lingbot_replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    habitat_log_dir = resolve_log_dir(habitat_dir)
    replay_log_dir = resolve_log_dir(replay_dir)
    habitat_steps = load_steps(habitat_log_dir)
    replay_steps = load_steps(replay_log_dir)
    steps = sorted(set(habitat_steps) | set(replay_steps))
    if not steps:
        raise FileNotFoundError("No step_*.npz logs found in either input")

    habitat_traj_all = np.asarray(
        [habitat_steps[s]["robot_xy"] for s in sorted(habitat_steps) if "robot_xy" in habitat_steps[s]],
        dtype=np.float32,
    )
    replay_traj_all = np.asarray(
        [replay_steps[s]["robot_xy"] for s in sorted(replay_steps) if "robot_xy" in replay_steps[s]],
        dtype=np.float32,
    )

    rows: List[Dict[str, Any]] = []
    missing: Dict[str, str] = {}
    for step in steps:
        habitat = habitat_steps.get(step)
        replay = replay_steps.get(step)
        row = {
            "step": step,
            "original_frontier_count": field_count(habitat, "candidate_frontiers"),
            "lingbot_replay_frontier_count": field_count(replay, "candidate_frontiers"),
            "original_explored_area_pixel_count": pixel_count(habitat, "explored_map"),
            "lingbot_explored_area_pixel_count": pixel_count(replay, "explored_map"),
            "original_obstacle_pixel_count": pixel_count(habitat, "obstacle_map"),
            "lingbot_obstacle_pixel_count": pixel_count(replay, "obstacle_map"),
            "selected_frontier": fmt_array(habitat.get("selected_frontier") if habitat else None, 4),
            "lingbot_candidate_frontiers": fmt_array(replay.get("candidate_frontiers") if replay else None, 12),
        }
        rows.append(row)

    for name, collection in [("habitat", habitat_steps), ("lingbot_replay", replay_steps)]:
        for key in ["candidate_frontiers", "explored_map", "obstacle_map", "selected_frontier", "depth_conf"]:
            count = sum(1 for data in collection.values() if key in data)
            if count == 0:
                missing[f"{name}.{key}"] = "not present in any step log"
            elif count < len(collection):
                missing[f"{name}.{key}"] = f"present in {count} of {len(collection)} step logs"

    csv_path = output_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_table_path = output_dir / "comparison_table.md"
    write_markdown_table(rows, md_table_path)

    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    contact_paths = []
    for step in parse_contact_steps(args.contact_steps, steps):
        habitat = habitat_steps.get(step)
        replay = replay_steps.get(step)
        h_indices = [s for s in sorted(habitat_steps) if s <= step and "robot_xy" in habitat_steps[s]]
        r_indices = [s for s in sorted(replay_steps) if s <= step and "robot_xy" in replay_steps[s]]
        habitat_traj = np.asarray([habitat_steps[s]["robot_xy"] for s in h_indices], dtype=np.float32)
        replay_traj = np.asarray([replay_steps[s]["robot_xy"] for s in r_indices], dtype=np.float32)
        sheet = make_contact_sheet(step, habitat, replay, habitat_traj, replay_traj)
        out_path = contact_dir / f"paired_contact_step_{step:06d}.png"
        cv2.imwrite(str(out_path), sheet)
        contact_paths.append(str(out_path))

    summary = {
        "habitat_trace_dir": str(habitat_dir),
        "lingbot_replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "num_habitat_steps": len(habitat_steps),
        "num_lingbot_replay_steps": len(replay_steps),
        "num_compared_steps": len(steps),
        "comparison_table_csv": str(csv_path),
        "comparison_table_md": str(md_table_path),
        "contact_sheets": contact_paths,
        "missing_or_partial_fields": missing,
        "frontier_count_pairs": [
            {
                "step": row["step"],
                "habitat": row["original_frontier_count"],
                "lingbot_replay": row["lingbot_replay_frontier_count"],
            }
            for row in rows
        ],
        "habitat_trajectory_delta_norm": float(np.linalg.norm(habitat_traj_all[-1] - habitat_traj_all[0]))
        if len(habitat_traj_all) > 1
        else None,
        "lingbot_replay_trajectory_delta_norm": float(np.linalg.norm(replay_traj_all[-1] - replay_traj_all[0]))
        if len(replay_traj_all) > 1
        else None,
    }
    (output_dir / "paired_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Paired Habitat Trace vs LingBot Replay Comparison",
        "",
        f"- habitat_trace_dir: `{habitat_dir}`",
        f"- lingbot_replay_dir: `{replay_dir}`",
        f"- num_habitat_steps: `{len(habitat_steps)}`",
        f"- num_lingbot_replay_steps: `{len(replay_steps)}`",
        f"- comparison_table_csv: `{csv_path}`",
        f"- comparison_table_md: `{md_table_path}`",
        "",
        "## Missing Or Partial Fields",
    ]
    if missing:
        lines.extend([f"- `{k}`: {v}" for k, v in missing.items()])
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Contact Sheets")
    lines.extend([f"- `{p}`" for p in contact_paths])
    (output_dir / "paired_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "steps": len(steps)}, indent=2))


if __name__ == "__main__":
    main()
