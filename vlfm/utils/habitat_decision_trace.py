# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Optional Habitat decision trace logging for VLFM evaluation.

This module is intentionally environment-gated.  It should have no effect unless
``VLFM_HABITAT_TRACE_DIR`` is set.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


_LOGGER: Optional["HabitatDecisionTraceLogger"] = None


def trace_enabled() -> bool:
    return bool(os.environ.get("VLFM_HABITAT_TRACE_DIR"))


def get_trace_logger() -> "HabitatDecisionTraceLogger":
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = HabitatDecisionTraceLogger(os.environ.get("VLFM_HABITAT_TRACE_DIR", ""))
    return _LOGGER


def _sanitize(value: Any) -> str:
    text = str(value if value is not None else "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:160] or "unknown"


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            pass
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except ValueError:
            return value
    if isinstance(value, (str, bytes)):
        return value
    if np.isscalar(value):
        return np.asarray(value)
    return value


def _batch0(value: Any) -> Any:
    arr = _to_numpy(value)
    if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _as_scalar_action(action: Any) -> np.ndarray:
    arr = _to_numpy(action)
    if isinstance(arr, np.ndarray):
        return arr.reshape(-1).astype(np.int64)
    return np.asarray(arr)


def _normalize_float_image(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)
    lo = float(np.nanmin(image[finite]))
    hi = float(np.nanmax(image[finite]))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    out = (np.clip(image, lo, hi) - lo) / (hi - lo)
    return (out * 255).astype(np.uint8)


def _save_rgb(path: Path, rgb: Optional[np.ndarray]) -> None:
    if rgb is None:
        return
    image = np.asarray(rgb)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_gray_or_colormap(path: Path, arr: Optional[np.ndarray], colormap: bool = True) -> None:
    if arr is None:
        return
    gray = _normalize_float_image(np.asarray(arr))
    if colormap:
        image = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    else:
        image = gray
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _save_map_image(path: Path, arr: Optional[np.ndarray]) -> None:
    if arr is None:
        return
    image = np.asarray(arr)
    if image.ndim == 3 and image.shape[-1] > 1:
        reduced = np.max(image, axis=-1)
    else:
        reduced = image
    _save_gray_or_colormap(path, reduced, colormap=True)


def _value_reduce(value_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value_map is None:
        return None
    arr = np.asarray(value_map)
    if arr.ndim == 3:
        return np.max(arr, axis=-1)
    return arr


class HabitatDecisionTraceLogger:
    def __init__(self, root: str) -> None:
        self.enabled = bool(root)
        self.root = Path(root) if root else Path()
        self.context: Dict[str, Any] = {}
        self.episode_steps: Dict[str, int] = {}
        self.episode_fields: Dict[str, Dict[str, Any]] = {}
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def set_episode_context(self, episode: Any) -> None:
        if not self.enabled:
            return
        self.context = {
            "scene_id": getattr(episode, "scene_id", None),
            "episode_id": getattr(episode, "episode_id", None),
            "object_category": getattr(episode, "object_category", None),
        }

    def _episode_id(self) -> str:
        return _sanitize(self.context.get("episode_id", "unknown"))

    def _episode_dir(self) -> Path:
        ep_dir = self.root / f"episode_{self._episode_id()}"
        for subdir in ["rgb", "depth", "logs", "maps"]:
            (ep_dir / subdir).mkdir(parents=True, exist_ok=True)
        return ep_dir

    def log_policy_step(self, policy: Any, observations: Dict[str, Any], mode: str, action: Any) -> None:
        if not self.enabled:
            return
        try:
            self._log_policy_step(policy, observations, mode, action)
        except Exception as exc:  # pragma: no cover - defensive: tracing must not change policy behavior.
            err_dir = self.root / "trace_errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            err_path = err_dir / f"trace_error_{len(list(err_dir.glob('trace_error_*.txt'))):06d}.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            print(f"[VLFM trace] logging failed; policy action preserved. See {err_path}")

    def _log_policy_step(self, policy: Any, observations: Dict[str, Any], mode: str, action: Any) -> None:
        ep_dir = self._episode_dir()
        step = int(getattr(policy, "_num_steps", self.episode_steps.get(self._episode_id(), 0)))
        step_name = f"step_{step:06d}"

        cache = getattr(policy, "_observations_cache", {})
        rgb = _batch0(observations.get("rgb")) if isinstance(observations, dict) else None
        depth_obs = _batch0(observations.get("depth")) if isinstance(observations, dict) else None
        if isinstance(depth_obs, np.ndarray) and depth_obs.ndim == 3 and depth_obs.shape[-1] == 1:
            depth_for_log = depth_obs[..., 0]
        else:
            depth_for_log = depth_obs

        object_map_rgbd = cache.get("object_map_rgbd", [])
        tf_camera_to_episodic = None
        if len(object_map_rgbd) > 0:
            tf_camera_to_episodic = object_map_rgbd[0][2]

        obstacle_map_obj = getattr(policy, "_obstacle_map", None)
        value_map_obj = getattr(policy, "_value_map", None)

        obstacle_map = getattr(obstacle_map_obj, "_map", None)
        explored_map = getattr(obstacle_map_obj, "explored_area", None)
        navigable_map = getattr(obstacle_map_obj, "_navigable_map", None)
        candidate_frontiers = cache.get("frontier_sensor", None)
        value_map = getattr(value_map_obj, "_value_map", None)
        value_confidence_map = getattr(value_map_obj, "_map", None)

        sorted_frontiers = getattr(policy, "_vlfm_trace_sorted_frontiers", None)
        frontier_scores = getattr(policy, "_vlfm_trace_frontier_scores", None)
        selected_frontier = getattr(policy, "_vlfm_trace_selected_frontier", None)
        selected_frontier_value = getattr(policy, "_vlfm_trace_selected_frontier_value", None)

        robot_xy = cache.get("robot_xy", None)
        robot_heading = cache.get("robot_heading", None)
        final_action = _as_scalar_action(action)
        stop_called = np.asarray(bool(getattr(policy, "_called_stop", False)))

        selected_local_goal = getattr(policy, "_last_goal", None)
        rho_theta = getattr(policy, "_policy_info", {}).get("rho_theta", None)

        npz_payload = {
            "step": np.asarray(step, dtype=np.int64),
            "scene_id": np.asarray(str(self.context.get("scene_id", ""))),
            "episode_id": np.asarray(str(self.context.get("episode_id", ""))),
            "object_goal": np.asarray(str(getattr(policy, "_target_object", self.context.get("object_category", "")))),
            "mode": np.asarray(mode),
            "rgb": rgb,
            "depth": depth_for_log,
            "gps": _batch0(observations.get("gps")) if isinstance(observations, dict) and "gps" in observations else None,
            "compass": _batch0(observations.get("compass"))
            if isinstance(observations, dict) and "compass" in observations
            else None,
            "heading": _batch0(observations.get("heading"))
            if isinstance(observations, dict) and "heading" in observations
            else None,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "obstacle_map": obstacle_map,
            "explored_map": explored_map,
            "navigable_map": navigable_map,
            "value_map": value_map,
            "value_confidence_map": value_confidence_map,
            "candidate_frontiers": candidate_frontiers,
            "frontier_scores": frontier_scores,
            "sorted_frontiers": sorted_frontiers,
            "selected_frontier": selected_frontier,
            "selected_frontier_value": selected_frontier_value,
            "selected_local_goal": selected_local_goal,
            "selected_local_goal_rho_theta": rho_theta,
            "final_action": final_action,
            "stop_called": stop_called,
        }
        npz_payload = {k: _to_numpy(v) for k, v in npz_payload.items() if v is not None}
        np.savez_compressed(ep_dir / "logs" / f"{step_name}.npz", **npz_payload)

        _save_rgb(ep_dir / "rgb" / f"{step_name}.png", rgb)
        if depth_for_log is not None:
            np.save(ep_dir / "depth" / f"{step_name}.npy", np.asarray(depth_for_log))
            _save_gray_or_colormap(ep_dir / "depth" / f"{step_name}.png", np.asarray(depth_for_log), colormap=True)

        if obstacle_map_obj is not None:
            cv2.imwrite(str(ep_dir / "maps" / f"{step_name}_obstacle.png"), obstacle_map_obj.visualize())
            frontiers_img = obstacle_map_obj.visualize()
            if selected_frontier is not None and len(np.asarray(selected_frontier).reshape(-1)) >= 2:
                selected_px = obstacle_map_obj._xy_to_px(np.asarray(selected_frontier).reshape(1, 2))[0]
                selected_px = (int(selected_px[0]), int(frontiers_img.shape[0] - 1 - selected_px[1]))
                cv2.circle(frontiers_img, selected_px, 8, (0, 255, 255), 2)
            cv2.imwrite(str(ep_dir / "maps" / f"{step_name}_frontiers.png"), frontiers_img)

        _save_map_image(ep_dir / "maps" / f"{step_name}_value.png", _value_reduce(value_map))

        ep_key = self._episode_id()
        self.episode_steps[ep_key] = max(self.episode_steps.get(ep_key, 0), step + 1)
        self.episode_fields[ep_key] = {
            "last_step": step,
            "last_npz_keys": sorted(npz_payload.keys()),
            "scene_id": str(self.context.get("scene_id", "")),
            "episode_id": str(self.context.get("episode_id", "")),
            "object_goal": str(getattr(policy, "_target_object", self.context.get("object_category", ""))),
        }

    def end_episode(self, episode: Any, episode_stats: Dict[str, Any], info: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        old_context = self.context.copy()
        self.set_episode_context(episode)
        ep_key = self._episode_id()
        ep_dir = self._episode_dir()
        summary = {
            "scene_id": str(getattr(episode, "scene_id", "")),
            "episode_id": str(getattr(episode, "episode_id", "")),
            "object_category": str(getattr(episode, "object_category", "")),
            "num_steps_logged": int(self.episode_steps.get(ep_key, 0)),
            "episode_stats": {k: _json_safe(v) for k, v in episode_stats.items()},
            "available_last_step_fields": self.episode_fields.get(ep_key, {}).get("last_npz_keys", []),
            "top_level_info_keys": sorted([str(k) for k in info.keys()]),
            "trace_dir": str(ep_dir),
        }
        (ep_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lines = [
            "# VLFM Habitat Trace Episode Summary",
            "",
            f"- scene_id: `{summary['scene_id']}`",
            f"- episode_id: `{summary['episode_id']}`",
            f"- object_category: `{summary['object_category']}`",
            f"- num_steps_logged: `{summary['num_steps_logged']}`",
            f"- success: `{summary['episode_stats'].get('success', 'unavailable')}`",
            f"- spl: `{summary['episode_stats'].get('spl', 'unavailable')}`",
            "",
            "## Last Step Fields",
            "",
        ]
        lines.extend([f"- `{k}`" for k in summary["available_last_step_fields"]])
        (ep_dir / "episode_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.context = old_context


def _json_safe(value: Any) -> Any:
    arr = _to_numpy(value)
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            return arr.item()
        if arr.size <= 32:
            return arr.tolist()
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if isinstance(arr, (np.floating, np.integer, np.bool_)):
        return arr.item()
    try:
        json.dumps(arr)
        return arr
    except TypeError:
        return str(arr)
