# coding=utf-8
"""Utilities for saving per-sample evaluation outputs for visualization."""

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image


def _safe_token(value: Any, max_len: int = 96) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if text == "":
        text = "empty"
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.+-]", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    if text == "":
        text = "na"
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _sample_file_token(value: Any, max_len: int = 64, hash_len: int = 8) -> str:
    """Make a readable sample token while avoiding collisions after truncation."""
    raw = "none" if value is None else str(value)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[: max(1, int(hash_len))]
    suffix = f"_{digest}"
    keep_len = max(1, int(max_len) - len(suffix))
    prefix = _safe_token(value, max_len=keep_len)
    return f"{prefix}{suffix}"


def _short_ckpt(ckpt_path: Optional[str]) -> str:
    if not ckpt_path:
        return "none"
    norm = str(ckpt_path).rstrip("/")
    parent = os.path.basename(os.path.dirname(norm))
    base = os.path.basename(norm)
    if base:
        if base.startswith("checkpoint-") and parent:
            return f"{parent}_{base}"
        return base
    return parent or "none"


def _extract_load_step(ckpt_path: Optional[str]) -> str:
    """Extract training step from checkpoint path like checkpoint-120000."""
    if not ckpt_path:
        return "unknown"
    text = str(ckpt_path)
    match = re.search(r"checkpoint-(\d+)", text)
    if match:
        return match.group(1)
    return "unknown"


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _to_uint8_image(image_like: Any) -> np.ndarray:
    arr = _to_numpy(image_like)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim={arr.ndim}, shape={arr.shape}")

    arr = arr.astype(np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if max_v <= 1.0 and min_v >= -1.0:
        if min_v < 0.0:
            arr = (arr + 1.0) * 0.5
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


class SaveVisManager:
    def __init__(
        self,
        enabled: bool,
        root_dir: str,
        script_tag: str,
        tasks: Iterable[str],
        seed: int,
        quick: bool,
        batch_size: int,
        ckpt_path: Optional[str],
        run_timestamp: str,
        rank: int = 0,
        world_size: int = 1,
        is_main_process: bool = True,
        save_all_replications: bool = False,
    ):
        self.enabled = bool(enabled)
        self.root_dir = root_dir
        self.script_tag = script_tag
        self.tasks = [str(t) for t in tasks]
        self.seed = int(seed)
        self.quick = bool(quick)
        self.batch_size = int(batch_size)
        self.ckpt_path = ckpt_path
        self.run_timestamp = str(run_timestamp)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_main_process = bool(is_main_process)
        self.save_all_replications = bool(save_all_replications)

        self.run_dir = ""
        self._task_dirs: Dict[str, str] = {}

        if not self.enabled:
            return

        tasks_token = _safe_token("+".join(self.tasks), max_len=120)
        load_step = _safe_token(_extract_load_step(self.ckpt_path), max_len=32)
        dir_name = (
            f"{_safe_token(self.script_tag)}"
            f"__tasks-{tasks_token}"
            f"__loadstep-{load_step}"
        )
        self.run_dir = os.path.join(self.root_dir, dir_name)
        os.makedirs(self.run_dir, exist_ok=True)

    def should_save_replication(self, rep_i: int) -> bool:
        if not self.enabled:
            return False
        return self.save_all_replications or int(rep_i) == 0

    def write_run_config_snapshot(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or not self.is_main_process:
            return
        out_path = os.path.join(self.run_dir, "run_config_snapshot.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def _ensure_task_dir(self, task: str) -> str:
        task = str(task)
        if task in self._task_dirs:
            return self._task_dirs[task]
        task_dir = os.path.join(self.run_dir, _safe_token(task, max_len=64))
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(os.path.join(task_dir, "motions"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "images"), exist_ok=True)
        self._task_dirs[task] = task_dir
        return task_dir

    def _rank_manifest_path(self, task: str) -> str:
        task_dir = self._ensure_task_dir(task)
        return os.path.join(task_dir, f"manifest_rank{self.rank}.jsonl")

    def _append_record(self, task: str, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._rank_manifest_path(task)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_motion_sample(
        self,
        task: str,
        replication: int,
        batch_idx: int,
        in_batch_idx: int,
        sample_id: str,
        motion: Any,
        keep_dim: Optional[int] = None,
        name_prefix: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        arr = _to_numpy(motion).astype(np.float32)
        if keep_dim is not None and arr.shape[-1] > int(keep_dim):
            arr = arr[..., : int(keep_dim)]

        task_dir = self._ensure_task_dir(task)
        sample_token = _sample_file_token(sample_id, max_len=64)
        prefix = ""
        if name_prefix:
            prefix = f"{_safe_token(name_prefix, max_len=32)}__"
        fname = (
            f"{prefix}rep{int(replication):02d}_b{int(batch_idx):05d}_i{int(in_batch_idx):03d}"
            f"_{sample_token}.npy"
        )
        rel_path = os.path.join("motions", fname)
        abs_path = os.path.join(task_dir, rel_path)
        np.save(abs_path, arr)

        record = {
            "task": str(task),
            "output_type": "motion",
            "sample_id": str(sample_id),
            "replication": int(replication),
            "rank": int(self.rank),
            "batch_idx": int(batch_idx),
            "in_batch_idx": int(in_batch_idx),
            "data_path": rel_path,
            "motion_shape": list(arr.shape),
        }
        if extra:
            record.update(extra)
        self._append_record(task, record)

    def save_text_sample(
        self,
        task: str,
        replication: int,
        batch_idx: int,
        in_batch_idx: int,
        sample_id: str,
        text: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        record = {
            "task": str(task),
            "output_type": "text",
            "sample_id": str(sample_id),
            "replication": int(replication),
            "rank": int(self.rank),
            "batch_idx": int(batch_idx),
            "in_batch_idx": int(in_batch_idx),
            "text": "" if text is None else str(text),
        }
        if extra:
            record.update(extra)
        self._append_record(task, record)

    def save_image_sample(
        self,
        task: str,
        replication: int,
        batch_idx: int,
        in_batch_idx: int,
        sample_id: str,
        image: Any,
        name_prefix: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        img = _to_uint8_image(image)

        task_dir = self._ensure_task_dir(task)
        sample_token = _sample_file_token(sample_id, max_len=64)
        prefix = ""
        if name_prefix:
            prefix = f"{_safe_token(name_prefix, max_len=32)}__"
        fname = (
            f"{prefix}rep{int(replication):02d}_b{int(batch_idx):05d}_i{int(in_batch_idx):03d}"
            f"_{sample_token}.png"
        )
        rel_path = os.path.join("images", fname)
        abs_path = os.path.join(task_dir, rel_path)
        Image.fromarray(img).save(abs_path)

        record = {
            "task": str(task),
            "output_type": "image",
            "sample_id": str(sample_id),
            "replication": int(replication),
            "rank": int(self.rank),
            "batch_idx": int(batch_idx),
            "in_batch_idx": int(in_batch_idx),
            "data_path": rel_path,
            "image_shape": list(img.shape),
        }
        if extra:
            record.update(extra)
        self._append_record(task, record)

    def finalize_task(self, task: str) -> None:
        if not self.enabled or not self.is_main_process:
            return

        task_dir = self._ensure_task_dir(task)
        rank_files: List[str] = []
        for idx in range(self.world_size):
            path = os.path.join(task_dir, f"manifest_rank{idx}.jsonl")
            if os.path.exists(path):
                rank_files.append(path)

        records: List[Dict[str, Any]] = []
        for path in rank_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        manifest_path = os.path.join(task_dir, "manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        motion_dims = sorted(
            {
                int(r["motion_shape"][-1])
                for r in records
                if r.get("output_type") == "motion" and isinstance(r.get("motion_shape"), list) and len(r["motion_shape"]) > 0
            }
        )
        summary = {
            "task": str(task),
            "num_records": len(records),
            "num_motion_records": sum(1 for r in records if r.get("output_type") == "motion"),
            "num_text_records": sum(1 for r in records if r.get("output_type") == "text"),
            "num_image_records": sum(1 for r in records if r.get("output_type") == "image"),
            "replications_saved": sorted({int(r.get("replication", 0)) for r in records}),
            "motion_feature_dims": motion_dims,
            "rank_manifests": [os.path.basename(p) for p in rank_files],
        }
        summary_path = os.path.join(task_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def finalize_tasks(self, tasks: Iterable[str]) -> None:
        if not self.enabled or not self.is_main_process:
            return
        for task in tasks:
            self.finalize_task(str(task))
