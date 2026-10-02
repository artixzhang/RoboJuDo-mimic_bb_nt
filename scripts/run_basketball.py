"""Basketball sim2sim preview and on-board Unitree C++ deployment."""

import argparse
import logging
import math
import os
import platform
import time
import traceback

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

from robojudo.config import ASSETS_DIR
from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.config.g1.policy.g1_mimic_bb_nt_policy_cfg import MODEL_DIR, load_student_config
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.basketball_pipeline import BasketballPipeline, BasketballPipelineCfg
from robojudo.tools.basketball_log import BasketballLog
from robojudo.tools.dof import merge_dof_cfgs

logger = logging.getLogger("robojudo.basketball")


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Expected a finite, nonnegative number")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="On-board only: use UnitreeCppEnv")
    parser.add_argument("--net_if", default="eth0")
    parser.add_argument("--hoop_pos", type=float, nargs=3, default=[3.0, 0.0, 1.8], metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--pre_hold",
        type=float,
        default=0.0,
        help="After Start: 0 advances phase immediately; -1 waits for Space/B; positive holds phase=0 for seconds",
    )
    parser.add_argument("--auto_start", action="store_true", help="Simulation only: skip the Ready button wait")
    parser.add_argument("--link_test", action="store_true", help="Run inference but submit only the initial PD pose")
    parser.add_argument(
        "--test_seconds", type=nonnegative, default=30, help="Link test duration after Start; then damping"
    )
    parser.add_argument("--prepare_seconds", type=nonnegative, help="Prepare duration; default: sim 0, real 3 seconds")
    parser.add_argument("--config", default=str(MODEL_DIR / "student_config.json"))
    parser.add_argument("--model", default=str(MODEL_DIR / "student_policy.onnx"))
    parser.add_argument("--no_ball", action="store_true", help="Preview without basketball")
    parser.add_argument("--joystick", action="store_true", help="Use a local gamepad in simulation")
    parser.add_argument("--no_keyboard", action="store_true", help="Use only the gamepad (no display session needed)")
    parser.add_argument("--headless", action="store_true", help="Simulation without viewer or keyboard")
    parser.add_argument("--steps", type=int, help="Stop after this many simulation steps (offline smoke test)")
    parser.add_argument(
        "--log_dir", default="logs/basketball", help="Directory for readable JSON/JSONL deployment logs"
    )
    parser.add_argument("--no_log", action="store_true", help="Disable detailed recording for timing comparisons")
    args = parser.parse_args(argv)
    if not math.isfinite(args.pre_hold) or (args.pre_hold < 0 and args.pre_hold != -1):
        parser.error("--pre_hold must be -1 (manual trigger), 0 (immediate), or a positive duration")
    if args.real and (args.headless or args.steps is not None):
        parser.error("--headless and --steps are simulation-only")
    if args.real and args.auto_start:
        parser.error("Real hardware requires Start after Prepare; --auto_start is simulation-only")
    if args.link_test and (args.no_log or args.test_seconds <= 0):
        parser.error("--link_test requires logging and a positive --test_seconds")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if not all(math.isfinite(x) for x in args.hoop_pos):
        parser.error("--hoop_pos must be finite")
    return args


def make_config(args):
    policy = load_student_config(args.config, model_path=args.model, hoop_pos=args.hoop_pos, pre_hold=args.pre_hold)
    if set(policy.action_dof.joint_names) != set(G1_29DoF().joint_names):
        raise ValueError("Student joint_names must contain exactly the 29 G1 joints")
    # Environment stays in SDK motor order; adapters map the JSON policy order by name.
    dof = merge_dof_cfgs(G1_29DoF(), policy.action_dof)
    xml = str(ASSETS_DIR / "robots/g1/g1_bb_sdf_mode16.xml")
    common = dict(xml=xml, dof=dof, forward_kinematic=None, update_with_fk=False, born_place_align=False)
    ctrl = (
        []
        if args.headless or args.no_keyboard
        else [
            KeyboardCtrlCfg(
                trigger_on_press=True,
                triggers_extra={"Key.enter": "[POLICY_START]", "Key.space": "[SHOT_TRIGGER]", "r": "[MOTION_RESET]"},
            )
        ]
    )
    if args.real:
        env = G1RealEnvCfg(
            **common,
            env_type="UnitreeCppEnv",
            odometry_type="DUMMY",
            unitree=G1UnitreeCfg(net_if=args.net_if, enable_odometry=False, control_dt=1 / policy.freq),
        )
        ctrl.append(UnitreeCtrlCfg(triggers_extra={"Start": "[POLICY_START]", "B": "[SHOT_TRIGGER]"}))
    else:
        from robojudo.environment.basketball_mujoco_env import BasketballMujocoEnvCfg

        sim_dt = 0.001
        decimation = round(1 / policy.freq / sim_dt)
        if decimation < 1 or not math.isclose(decimation * sim_dt, 1 / policy.freq):
            raise ValueError("policy_hz must divide the 1000 Hz MuJoCo simulation rate")
        env = BasketballMujocoEnvCfg(
            **common,
            sim_dt=sim_dt,
            sim_decimation=decimation,
            headless=args.headless,
            initial_profile=policy.student["phase"]["motion_profiles"][0],
            ball_xml=None if args.no_ball else str(ASSETS_DIR / "objects/basketball.xml"),
        )
        if args.joystick:
            ctrl.append(JoystickCtrlCfg(triggers_extra={"Start": "[POLICY_START]", "B": "[SHOT_TRIGGER]"}))
    return BasketballPipelineCfg(
        robot="g1", env=env, policy=policy, ctrl=ctrl, auto_start=args.auto_start, link_test=args.link_test
    )


def main():
    args = parse_args()
    cfg = make_config(args)
    trace = None if args.no_log else BasketballLog(args.log_dir, cfg, vars(args))
    pipeline = None
    reason = "normal"
    try:
        if trace:
            logger.info("Text deployment log: %s", trace.path)
        pipeline = BasketballPipeline(cfg, trace=trace)
        pipeline.prepare(prepare_seconds=args.prepare_seconds)
        step = 0
        while args.steps is None or step < args.steps:
            if args.link_test and pipeline.started_ns is not None:
                if (time.perf_counter_ns() - pipeline.started_ns) / 1e9 >= args.test_seconds:
                    reason = "link_test_complete"
                    break
            start = time.perf_counter()
            pipeline.step()
            step += 1
            if args.real and pipeline.stopped:
                reason = "emergency_stop"
                break
            remaining = pipeline.dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)  # Rate pacing only; no sensor/action delay queue.
        logger.info("Finished %d steps; reference frame %.1f", step, pipeline.policy.frame)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except BaseException as exc:
        reason = type(exc).__name__
        if trace:
            trace.event("exception", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if pipeline is not None:
                pipeline.env.shutdown()
                if not args.real:
                    pipeline.env.close()
                if trace:
                    trace.event("shutdown_complete")
        finally:
            if trace:
                trace.close(reason)
                if args.link_test:
                    from scripts.analyze_basketball_link import write_report

                    report = write_report(trace.path)
                    logger.info("Link test report: %s (status=%s)", trace.path / "link_report.json", report["status"])


if __name__ == "__main__":
    main()
