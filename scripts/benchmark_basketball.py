"""Offline CPU benchmark. No Environment, controller, SDK, or robot connection is created."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from box import Box

from robojudo.config.g1.policy.g1_mimic_bb_nt_policy_cfg import MODEL_DIR, load_student_config
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy.mimic_bb_policy import MimicBBPolicy
from robojudo.tools.basketball_log import BasketballLog


def statistics(values, budget_ms):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "over_budget": int(np.count_nonzero(values > budget_ms)),
    }


def initial_state(policy):
    return Box(
        base_quat=np.asarray(policy.profile["initial_root_quaternion_wxyz"], dtype=np.float32)[[1, 2, 3, 0]],
        base_ang_vel=np.zeros(3, dtype=np.float32),
        dof_pos=policy.initial_pos.copy(),
        dof_vel=np.zeros(29, dtype=np.float32),
    )


def benchmark(args):
    cfg = load_student_config(args.config, model_path=args.model, pre_hold=0)
    loaded = time.perf_counter_ns()
    policy = MimicBBPolicy(cfg)
    load_ms = (time.perf_counter_ns() - loaded) / 1e6
    state = initial_state(policy)
    policy.reset_from_state(state)
    # Sweep phase even for synthetic inputs; never interpret these as a physics rollout.
    observations = []
    if args.trace:
        with Path(args.trace).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("type") == "sample" and row.get("stage") == "control":
                    observations.append(np.asarray(row["observation"], dtype=np.float32))
        if not observations:
            raise ValueError("Trace contains no control observations")
    else:
        for frame in range(policy.profile["frame_count"]):
            policy.frame = frame
            observations.append(policy.get_observation(state, {})[0].copy())
    batches = [obs.reshape(1, 469) for obs in observations]
    session = policy.model

    def infer(batch):
        return session.run([policy.output_name], {policy.input_name: batch})

    start = time.perf_counter_ns()
    infer(batches[0])
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    for i in range(args.warmup):
        infer(batches[i % len(batches)])
    ort_ms = []
    for i in range(args.steps):
        start = time.perf_counter_ns()
        infer(batches[i % len(batches)])
        ort_ms.append((time.perf_counter_ns() - start) / 1e6)

    trace = None
    if args.with_log:
        # Configuration only. This is deliberately not a real/simulation Environment instance.
        run_cfg = RlPipelineCfg(
            robot="g1",
            policy=cfg,
            env=EnvCfg(env_type="OfflineNoSend", xml="not_loaded", dof=cfg.action_dof),
        )
        trace = BasketballLog(args.output.parent / "benchmark_traces", run_cfg, vars(args))
        trace.event("offline_benchmark", note="Synthetic states and virtual action history; no targets are sent")

    def compute():
        if policy.frame >= policy.profile["frame_count"] - 1:
            policy.reset_from_state(state)
        obs, _ = policy.get_observation(state, {})
        target = policy.get_action(obs) + policy.default_pos
        # Virtual feedback only, to exercise the deployment history code off robot.
        policy.record_applied_target(target)
        policy.post_step_callback()
        if trace:
            trace.sample(
                "offline",
                state,
                [],
                target,
                {},
                observation=obs,
                raw_action=policy.raw_action,
                submitted=False,
                frame_after=policy.frame,
            )
        return target

    try:
        policy.reset_from_state(state)
        for _ in range(args.warmup):
            compute()
        compute_ms, interval_ms = [], []
        previous = None
        for _ in range(args.steps):
            start = time.perf_counter_ns()
            if previous is not None:
                interval_ms.append((start - previous) / 1e6)
            previous = start
            target = compute()
            compute_ms.append((time.perf_counter_ns() - start) / 1e6)
            if not np.isfinite(target).all():
                raise ValueError("ONNX produced a nonfinite target")
            if args.paced:
                remaining = policy.dt - (time.perf_counter_ns() - start) / 1e9
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if trace:
            trace.close()
    return {
        "mode": "offline_no_hardware",
        "host": platform.node(),
        "platform": platform.platform(),
        "model": cfg.policy_file,
        "model_sha256": hashlib.sha256(Path(cfg.policy_file).read_bytes()).hexdigest(),
        "providers": session.get_providers(),
        "ort_threads": 1,
        "onnx_input_source": str(args.trace) if args.trace else "synthetic_initial_pose_phase_sweep",
        "compute_input_source": "synthetic_initial_pose_with_virtual_action_history",
        "warmup": args.warmup,
        "paced_compute": args.paced,
        "with_log": args.with_log,
        "session_load_ms": load_ms,
        "first_onnx_call_ms": cold_ms,
        "onnx_run": statistics(ort_ms, policy.dt * 1000),
        "policy_compute": statistics(compute_ms, policy.dt * 1000),
        "compute_start_interval": statistics(interval_ms, policy.dt * 1000),
        "trace_directory": str(trace.path) if trace else None,
        "excludes": "Sensor acquisition, Controller/DoF adapters, DDS submission, network, motor execution",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(MODEL_DIR / "student_config.json"))
    parser.add_argument("--model", default=str(MODEL_DIR / "student_policy.onnx"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--paced", action="store_true", help="Pace the policy-compute pass at policy_hz")
    parser.add_argument("--with_log", action="store_true", help="Include readable sample logging in compute timing")
    parser.add_argument("--trace", type=Path, help="Use recorded observations for the ONNX-only pass")
    parser.add_argument("--output", type=Path, default=Path("logs/basketball_benchmark.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup nonnegative")
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
