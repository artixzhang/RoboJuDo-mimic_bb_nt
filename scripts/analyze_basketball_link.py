"""Analyze the real receive/infer/submit test without importing any robot SDK."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.verify_basketball_history import verify


def stats(values, budget):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "over_budget": int(np.count_nonzero(values > budget)),
    }


def analyze(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    cfg = metadata["config"]
    student = cfg["policy"]["student"]
    initial = student["phase"]["motion_profiles"][0]["initial_joint_position_rad"]
    indices = [student["joint_names"].index(name) for name in metadata["env_joint_names"]]
    target = np.asarray(initial)[indices]
    budget = 1000 / student["policy_hz"]
    metrics = defaultdict(list)
    ready = running = submitted = ticks = repeats = target_errors = nonfinite = 0
    max_unchanged = 0.0
    previous_sample = previous_submit = None
    last_running = False
    with (directory / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "sample" and row.get("stage") == "control":
                last_running = row.get("inference_ran", False)
                ready += int(not last_running)
                running += int(last_running)
                submitted += int(row["submitted"])
                target_errors += int(not np.allclose(row["target"], target, atol=1e-6, rtol=0))
                for field in (row["observation"], row["target"], row.get("raw_action")):
                    if field is not None:
                        nonfinite += int(not np.isfinite(field).all())
                if not last_running:
                    continue
                for name, value in row["timing_ms"].items():
                    metrics[name].append(value)
                sampled = row["env"].get("policy_sample_monotonic_ns")
                if sampled is not None and previous_sample is not None:
                    metrics["sample_interval_ms"].append((sampled - previous_sample) / 1e6)
                previous_sample = sampled
                if row["submitted"]:
                    if previous_submit is not None:
                        metrics["submit_interval_ms"].append((row["submit_call_ns"] - previous_submit) / 1e6)
                    previous_submit = row["submit_call_ns"]
                    if sampled is not None:
                        metrics["read_to_submit_return_ms"].append((row["submit_return_ns"] - sampled) / 1e6)
                if row["env"].get("state_tick") is not None:
                    ticks += 1
                    repeats += int(row["tick_delta"] == 0)
                    max_unchanged = max(max_unchanged, row["unchanged_tick_ms"])
            elif row.get("type") == "cycle" and row.get("stage") == "control" and last_running:
                metrics["work_ms"].append(row["work_ms"])
                metrics["sample_log_ms"].append(row["sample_log_ms"])
    history = verify(directory) if ready + running else None
    findings = []
    if not cfg.get("link_test"):
        findings.append("This run was not configured as a link test")
    if target_errors:
        findings.append(f"{target_errors} targets differed from the fixed initial pose")
    if submitted != ready + running:
        findings.append("Some control samples were not submitted")
    if nonfinite:
        findings.append("Nonfinite observations, network outputs or targets")
    if history is not None and not history["history_ok"]:
        findings.append("Observation history verification failed")
    if not cfg["env"]["is_sim"] and ticks != running:
        findings.append("Missing real state tick telemetry")
    if max_unchanged >= 2 * budget:
        findings.append("State tick remained unchanged for at least two policy periods")
    if any(value > budget for value in metrics["work_ms"]):
        findings.append("Some control work exceeded the policy time budget")
    if any(value >= 2 * budget for value in metrics["submit_interval_ms"]):
        findings.append("Some submission intervals reached two policy periods")
    completed = summary["reason"] == "link_test_complete" and running > 0
    return {
        "status": "incomplete" if not completed else ("issues_found" if findings else "software_checks_passed"),
        "backend": cfg["env"]["env_type"],
        "is_sim": cfg["env"]["is_sim"],
        "completion_reason": summary["reason"],
        "ready_samples": ready,
        "inference_samples": running,
        "submitted_samples": submitted,
        "fixed_target_mismatches": target_errors,
        "timing": {name: stats(values, budget) for name, values in metrics.items()},
        "state_ticks": {"samples": ticks, "repeated": repeats, "max_unchanged_ms": max_unchanged},
        "history": history,
        "findings": findings,
        "thresholds": {"work_budget_ms": budget, "long_submit_interval_ms": 2 * budget},
        "scope": "Software receive/read/infer/submit timing under fixed PD commands; not shot validation",
        "not_measured": [
            "Sensor capture to C++ callback latency (no source/receive timestamps from installed SDK)",
            "DDS delivery acknowledgement and motor execution latency",
            "Independent protection against Python/C++ stalls",
        ],
    }


def write_report(directory):
    result = analyze(directory)
    (Path(directory) / "link_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = write_report(args.directory)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
