"""Prepare -> ready/pre-hold -> shot -> indefinite post-hold."""

import logging
import time

import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger(__name__)


class BasketballPipelineCfg(RlPipelineCfg):
    pipeline_type: str = "BasketballPipeline"
    auto_start: bool = False
    link_test: bool = False


@pipeline_registry.register
class BasketballPipeline(RlPipeline):
    def __init__(self, cfg, trace=None):
        self.trace = trace
        self._last_logged_mode = None
        super().__init__(cfg)

    def self_check(self):
        # Do not run policy dry steps before Prepare has provided a real s_0.
        self.env.self_check()

    def reset(self):
        super().reset()
        self.ready = False
        self.stopped = False
        self.started = False
        self.started_ns = None
        if self.trace:
            self.trace.event("pipeline_reset")

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        seconds = (0.0 if self.cfg.env.is_sim else 3.0) if prepare_seconds is None else prepare_seconds
        self._prepare_seconds = seconds
        if self.trace:
            self.trace.event("prepare_start", seconds=seconds)
        self._ramp_to_pose(self.policy.get_init_dof_pos(), max(1, round(seconds * self.freq)), trace=self.trace)
        if self.cfg.env.is_sim:
            self.env.place_ball()
        self.env.update()
        data = self.env.get_data()
        data.dof_pos = self.policy.obs_adapter.fit(data.dof_pos)
        data.dof_vel = self.policy.obs_adapter.fit(data.dof_vel)
        self.policy.reset_from_state(data)
        self.ctrl_manager.reset()
        self.ready = True
        self.started = False
        self.started_ns = None
        self.hold_target = self.policy.get_init_dof_pos()
        self._last_logged_mode = None
        if self.trace:
            self.trace.event(
                "prepare_complete",
                env=data,
                sensor_history=self.policy.sensor_history,
                action_history=self.policy.action_history,
                hoop=self.policy.hoop,
            )
        if self.cfg.auto_start:
            self.start_policy()
        else:
            logger.info("Ready: PD hold only, no inference. Press Start/Enter or Space/B to start policy")
        if self.cfg.link_test:
            logger.warning("LINK TEST: inference outputs are logged ONLY; every submitted target stays at initial pose")
        logger.info("Esc / A enters damping in every stage")

    def start_policy(self):
        self.started = True
        self.started_ns = time.perf_counter_ns()
        if self.trace:
            self.trace.event("policy_start", frame=self.policy.frame, link_test=self.cfg.link_test)
        pre_hold = self.cfg.policy.pre_hold
        if pre_hold == 0:
            logger.info("Policy started: advancing phase immediately, without pre-hold")
        elif pre_hold == -1:
            logger.info("Pre-hold: policy running at phase=0; waiting for Space / B to start the shot")
        else:
            logger.info("Pre-hold: policy running at phase=0 for %.2fs; Space / B can start the shot early", pre_hold)

    def step(self, dry_run=False):
        if not self.ready:
            raise RuntimeError("Call prepare() before running the basketball pipeline")
        cycle = self.trace.begin_cycle("control") if self.trace else None
        read_start = time.perf_counter_ns()
        self.env.update()
        env_data = self.env.get_data()
        read_end = time.perf_counter_ns()
        env_data.policy_sample_monotonic_ns = read_end
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        ctrl_end = time.perf_counter_ns()
        commands = ctrl_data.get("COMMANDS", [])
        # Emergency Stop has priority over every other command, before inference/send.
        if "[SHUTDOWN]" in commands:
            self.stopped = True
            self.env.shutdown()
            if self.trace:
                self.trace.event("emergency_stop", stage="control", commands=commands, env=env_data)
        if self.stopped:
            if self.cfg.env.is_sim:
                self.env.step(self.env.dof_pos)
            if self.trace:
                log_ms = self.trace.sample(
                    "damping",
                    env_data,
                    commands,
                    None,
                    {},
                    mode="damping",
                    submitted=False,
                    native_damping=True,
                )
                self.trace.finish_cycle(cycle, "damping", log_ms)
            return
        if "[MOTION_RESET]" in commands or "[SIM_REBORN]" in commands:
            if self.trace:
                self.trace.event("reset_requested", commands=commands, env=env_data)
            if self.cfg.env.is_sim:
                self.env.reborn()
            self.prepare(prepare_seconds=self._prepare_seconds)
            return
        if not self.started and ("[POLICY_START]" in commands or "[SHOT_TRIGGER]" in commands):
            self.start_policy()
        elif self.started and "[SHOT_TRIGGER]" in commands:
            self.policy.trigger()
            if self.trace:
                self.trace.event("shot_trigger", frame=self.policy.frame)

        frame = self.policy.frame
        mode = (
            "ready"
            if not self.started
            else (
                "post_hold"
                if frame >= self.policy.profile["frame_count"] - 1
                else ("active" if self.policy.active else "pre_hold")
            )
        )
        if self.trace and mode != self._last_logged_mode:
            self.trace.event("mode", mode=mode, frame=frame)
            self._last_logged_mode = mode
        obs_start = time.perf_counter_ns()
        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        obs_end = time.perf_counter_ns()
        policy_target = self.policy.get_pd_target(obs) if self.started else None
        target = self.hold_target if not self.started or self.cfg.link_test else policy_target
        infer_end = time.perf_counter_ns()
        submit_start = time.perf_counter_ns()
        if not dry_run:
            try:
                self.env.step(target)
            except Exception as exc:
                if self.trace:
                    self.trace.event("submission_error", error=repr(exc), env=env_data, target=target, observation=obs)
                raise
        submit_end = time.perf_counter_ns()
        if not dry_run:
            # Convert the submitted env-order target back to the policy joint order.
            applied = self.policy.obs_adapter.fit(np.asarray(target))
            self.policy.record_applied_target(applied)
            if self.started:
                self.policy.post_step_callback()
            self.timestep += 1
        self.ctrl_manager.post_step_callback(ctrl_data)
        if self.trace:
            log_ms = self.trace.sample(
                "control",
                env_data,
                commands,
                target,
                {
                    "read_ms": (read_end - read_start) / 1e6,
                    "controller_ms": (ctrl_end - read_end) / 1e6,
                    "observation_ms": (obs_end - obs_start) / 1e6,
                    "inference_ms": (infer_end - obs_end) / 1e6,
                    "submit_ms": (submit_end - submit_start) / 1e6,
                },
                mode=mode,
                frame=frame,
                frame_after=self.policy.frame,
                phase=frame / (self.policy.profile["frame_count"] - 1),
                observation=obs,
                raw_action=self.policy.raw_action if self.started else None,
                policy_target=policy_target,
                inference_ran=self.started,
                link_test=self.cfg.link_test,
                target_source="initial_pose" if not self.started or self.cfg.link_test else "policy",
                submitted=not dry_run,
                submit_call_ns=submit_start,
                submit_return_ns=submit_end,
                control=ctrl_data,
            )
            self.trace.finish_cycle(cycle, "control", log_ms)
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=target,
                timestep=self.timestep,
            )
