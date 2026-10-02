"""Readable deployment traces with bounded buffering, no worker or command queue."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot log {type(value).__name__}")


class BasketballLog:
    def __init__(self, directory, cfg, arguments):
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        self.path = Path(directory) / f"{now:%Y%m%d-%H%M%S-%f}_{os.getpid()}"
        self.path.mkdir(parents=True)
        self.started_ns = time.perf_counter_ns()
        self.last_flush_ns = self.started_ns
        self.last_cycle_ns = None
        self.last_submit_ns = None
        self.last_tick = None
        self.tick_changed_ns = None
        self.samples = 0
        self.metrics = {}
        self.tick_summary = {"samples": 0, "repeated": 0, "backwards_or_wrap": 0, "max_unchanged_ms": 0.0}
        self.closed = False
        self.budget_ms = 1000 / cfg.policy.freq
        versions = {}
        for name in ("numpy", "onnxruntime", "mujoco", "unitree_cpp"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        cpp_spec = importlib.util.find_spec("unitree_cpp")
        metadata = {
            "schema": "robojudo.basketball.text.v1",
            "started_at": now.isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "package_versions": versions,
            "unitree_cpp_path": None if cpp_spec is None else cpp_spec.origin,
            "arguments": arguments,
            "config": cfg.to_dict(),
            "policy_sha256": hashlib.sha256(Path(cfg.policy.policy_file).read_bytes()).hexdigest(),
            "env_joint_names": cfg.env.dof.joint_names,
            "policy_joint_names": cfg.policy.action_dof.joint_names,
            "clock": "perf_counter_ns; only differences on this host are meaningful",
            "submission": "env.step return confirms API submission, not actuator execution or DDS delivery",
            "state_freshness": "tick changes indicate new robot states; unchanged_tick_ms is NOT absolute state age",
            "buffering": "1 MiB text buffer; flush at least once per second while recording; no fsync per step",
        }
        self._save_json("metadata.json", metadata)
        self.stream = (self.path / "trace.jsonl").open("w", encoding="utf-8", buffering=1024 * 1024)
        self.event("session_start")

    def _save_json(self, name, data):
        (self.path / name).write_text(
            json.dumps(data, default=json_value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write(self, record):
        now = time.perf_counter_ns()
        record = {"monotonic_ns": now, "elapsed_s": (now - self.started_ns) / 1e9, **record}
        self.stream.write(json.dumps(record, default=json_value, ensure_ascii=False, separators=(",", ":")) + "\n")
        if now - self.last_flush_ns >= 1_000_000_000:
            self.stream.flush()
            self.last_flush_ns = now

    def event(self, name, **details):
        self.write({"type": "event", "event": name, **details})

    def begin_cycle(self, stage):
        start = time.perf_counter_ns()
        interval = None if self.last_cycle_ns is None else (start - self.last_cycle_ns) / 1e6
        self.last_cycle_ns = start
        if interval is not None:
            self._metric(f"{stage}.start_interval_ms", interval)
        return start, interval

    def _metric(self, name, value):
        stats = self.metrics.setdefault(name, {"count": 0, "sum_ms": 0.0, "max_ms": 0.0, "over_budget": 0})
        stats["count"] += 1
        stats["sum_ms"] += value
        stats["max_ms"] = max(stats["max_ms"], value)
        stats["over_budget"] += int(value > self.budget_ms)

    def sample(self, stage, env_data, commands, target, timing, **details):
        start = time.perf_counter_ns()
        submit_interval = None
        if details.get("submitted"):
            submit_ns = details["submit_call_ns"]
            if self.last_submit_ns is not None:
                submit_interval = (submit_ns - self.last_submit_ns) / 1e6
                self._metric(f"{stage}.submit_interval_ms", submit_interval)
            self.last_submit_ns = submit_ns
        tick = env_data.get("state_tick")
        tick_delta = None
        unchanged_ms = None
        if tick is not None:
            tick_delta = None if self.last_tick is None else tick - self.last_tick
            if tick != self.last_tick:
                self.tick_changed_ns = start
            self.last_tick = tick
            unchanged_ms = (start - self.tick_changed_ns) / 1e6
            self.tick_summary["samples"] += 1
            self.tick_summary["repeated"] += int(tick_delta == 0)
            self.tick_summary["backwards_or_wrap"] += int(tick_delta is not None and tick_delta < 0)
            self.tick_summary["max_unchanged_ms"] = max(self.tick_summary["max_unchanged_ms"], unchanged_ms)
        self.write(
            {
                "type": "sample",
                "sample": self.samples,
                "stage": stage,
                "env": env_data,
                "commands": commands,
                "target": target,
                "tick_delta": tick_delta,
                "unchanged_tick_ms": unchanged_ms,
                "timing_ms": timing,
                "submit_interval_ms": submit_interval,
                **details,
            }
        )
        self.samples += 1
        for name, duration in timing.items():
            self._metric(f"{stage}.{name}", duration)
        return (time.perf_counter_ns() - start) / 1e6

    def finish_cycle(self, cycle, stage, log_ms):
        start, interval = cycle
        work_ms = (time.perf_counter_ns() - start) / 1e6
        self._metric(f"{stage}.work_ms", work_ms)
        self._metric(f"{stage}.sample_log_ms", log_ms)
        self.write(
            {
                "type": "cycle",
                "stage": stage,
                "cycle_start_ns": start,
                "start_interval_ms": interval,
                "work_ms": work_ms,
                "sample_log_ms": log_ms,
                "budget_ms": self.budget_ms,
                "work_overrun_ms": max(0.0, work_ms - self.budget_ms),
            }
        )

    def close(self, reason="normal"):
        if self.closed:
            return
        self.event("session_end", reason=reason)
        self.stream.close()
        summary = {}
        for name, stats in self.metrics.items():
            summary[name] = {**stats, "mean_ms": stats["sum_ms"] / stats["count"]}
        self._save_json(
            "summary.json",
            {
                "reason": reason,
                "samples": self.samples,
                "budget_ms": self.budget_ms,
                "robot_tick": self.tick_summary,
                "timing": summary,
            },
        )
        self.closed = True
