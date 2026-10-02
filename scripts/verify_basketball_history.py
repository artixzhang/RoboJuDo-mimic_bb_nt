"""Reconstruct observation histories independently from readable deployment logs."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def cadence(values, expected_ms):
    if not values:
        return None
    values = np.asarray(values)
    return {
        "count": len(values),
        "expected_ms": expected_ms,
        "mean_ms": float(values.mean()),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "max_abs_deviation_ms": float(np.max(np.abs(values - expected_ms))),
    }


def verify(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    cfg = metadata["config"]["policy"]
    student = cfg["student"]
    names = student["joint_names"]
    indices = [metadata["env_joint_names"].index(name) for name in names]
    nominal = np.asarray(student["nominal_joint_position"], dtype=np.float32)
    initial = np.asarray(student["phase"]["motion_profiles"][0]["initial_joint_position_rad"], dtype=np.float32)
    last_frame = student["phase"]["motion_profiles"][0]["frame_count"] - 1
    fields = student["observation"]["fields"]
    scales = {f["name"]: f["scale"] for f in fields}
    hoop = np.asarray(cfg["hoop_pos"], dtype=np.float32)
    resets = samples = failures = repeated_ticks = 0
    errors = []
    sensor_history = action_history = None
    previous_sample_ns = previous_sim_time = previous_tick = None
    sample_intervals, sim_intervals, cycle_intervals = [], [], []

    def compare(label, actual, expected, line):
        nonlocal failures
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=5e-6):
            failures += 1
            if len(errors) < 20:
                errors.append(
                    {
                        "line": line,
                        "field": label,
                        "actual_shape": list(actual.shape),
                        "expected_shape": list(expected.shape),
                    }
                )

    def sensors(data, order):
        return np.concatenate(
            (
                Rotation.from_quat(data["base_quat"]).inv().apply([0, 0, -1]),
                data["base_ang_vel"],
                np.asarray(data["dof_pos"])[order] - nominal,
                np.asarray(data["dof_vel"])[order],
            )
        ).astype(np.float32)

    with (directory / "trace.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("event") == "prepare_complete":
                # prepare_complete.env was adapted to policy order before history reset.
                seed = sensors(row["env"], list(range(29)))
                sensor_history = deque([seed.copy() for _ in range(5)], maxlen=5)
                action_history = deque([(initial - nominal).copy() for _ in range(5)], maxlen=5)
                compare("reset.sensor_history", row["sensor_history"], list(sensor_history), line_number)
                compare("reset.action_history", row["action_history"], list(action_history), line_number)
                compare("reset.hoop", row["hoop"], hoop * scales["initial_hoop_position_b"], line_number)
                resets += 1
                previous_sample_ns = previous_sim_time = previous_tick = None
            elif row.get("type") == "cycle" and row.get("stage") == "control":
                if row["start_interval_ms"] is not None:
                    cycle_intervals.append(row["start_interval_ms"])
            elif row.get("type") == "sample" and row.get("stage") == "control":
                if sensor_history is None:
                    raise ValueError(f"Line {line_number}: control sample has no preceding prepare_complete")
                data = row["env"]
                sensor_history.append(sensors(data, indices))
                h = np.asarray(sensor_history)
                expected = {
                    "phase": np.array([row["frame"] / last_frame]),
                    "projected_gravity_history": h[:, :3].ravel(),
                    "pelvis_angular_velocity_history": h[:, 3:6].ravel(),
                    "joint_position_offset_history": h[:, 6:35].ravel(),
                    "joint_velocity_history": h[:, 35:64].ravel(),
                    "applied_action_offset_history": np.asarray(action_history).ravel(),
                    "initial_hoop_position_b": hoop,
                }
                obs = np.asarray(row["observation"])
                if obs.shape != (469,):
                    raise ValueError(f"Line {line_number}: expected 469 observations")
                for field in fields:
                    compare(
                        field["name"],
                        obs[field["start"] : field["stop"]],
                        expected[field["name"]] * field["scale"],
                        line_number,
                    )
                # Current target may enter ONLY the NEXT observation's action window.
                if row["submitted"]:
                    action_history.append(np.asarray(row["target"], dtype=np.float32)[indices] - nominal)
                sampled_ns = data.get("policy_sample_monotonic_ns", data.get("state_read_monotonic_ns"))
                if sampled_ns is not None and previous_sample_ns is not None:
                    sample_intervals.append((sampled_ns - previous_sample_ns) / 1e6)
                previous_sample_ns = sampled_ns
                sim_time = data.get("sim_time_s")
                if sim_time is not None and previous_sim_time is not None:
                    sim_intervals.append((sim_time - previous_sim_time) * 1000)
                previous_sim_time = sim_time
                tick = data.get("state_tick")
                repeated_ticks += int(tick is not None and tick == previous_tick)
                previous_tick = tick
                samples += 1
    if not samples:
        raise ValueError("No control samples to verify")
    expected_ms = 1000 / student["policy_hz"]
    return {
        "history_ok": failures == 0,
        "samples_checked": samples,
        "resets_checked": resets,
        "failed_checks": failures,
        "first_errors": errors,
        "policy_sample_interval": cadence(sample_intervals, expected_ms),
        "simulation_sample_interval": cadence(sim_intervals, expected_ms),
        "control_start_interval": cadence(cycle_intervals, expected_ms),
        "repeated_state_ticks": repeated_ticks,
        "scope": "Checks logged histories, reset tiling, scales and frozen hoop; not DDS/motor latency",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Run directory containing metadata.json and trace.jsonl")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = verify(args.directory)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["history_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
