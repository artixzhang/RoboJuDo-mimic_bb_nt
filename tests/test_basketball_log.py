"""Text recording and timing tests; C++ telemetry uses a mock, never hardware."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from box import Box

from robojudo.environment.base_env import Environment
from robojudo.pipeline.basketball_pipeline import BasketballPipeline
from robojudo.tools.basketball_log import BasketballLog
from scripts.run_basketball import main, make_config, parse_args


class TextLogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cfg = make_config(parse_args(["--headless", "--auto_start", "--pre_hold", "-1"]))

    def rows(self, trace):
        return [json.loads(line) for line in (trace.path / "trace.jsonl").read_text().splitlines()]

    def test_entire_motion_reset_and_stop_are_readable(self):
        trace = BasketballLog(self.directory.name, self.cfg, {})
        pipeline = BasketballPipeline(self.cfg, trace=trace)
        try:
            pipeline.prepare(prepare_seconds=0.02)
            pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SHOT_TRIGGER]"]))
            pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
            for _ in range(166):
                pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=["[MOTION_RESET]"])
            pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=["[SHUTDOWN]"])
            pipeline.step()
        finally:
            pipeline.env.close()
            trace.close()
        rows = self.rows(trace)
        samples = [r for r in rows if r["type"] == "sample"]
        control = [r for r in samples if r["stage"] == "control"]
        self.assertEqual(len([r for r in samples if r["stage"] == "prepare"]), 4)
        self.assertEqual(len(control[0]["observation"]), 469)
        self.assertEqual(len(control[0]["raw_action"]), 29)
        self.assertEqual(len(control[0]["target"]), 29)
        self.assertEqual(len(control[0]["env"]["dof_pos"]), 29)
        self.assertEqual(control[0]["mode"], "pre_hold")
        self.assertEqual(control[-1]["mode"], "post_hold")
        self.assertEqual(control[-1]["frame"], 165)
        self.assertEqual(samples[-1]["stage"], "damping")
        self.assertIsNone(samples[-1]["target"])
        for row in control:
            self.assertTrue(row["submitted"])
            self.assertGreaterEqual(row["submit_return_ns"], row["submit_call_ns"])
            self.assertGreaterEqual(row["timing_ms"]["inference_ms"], 0)
        events = {r["event"] for r in rows if r["type"] == "event"}
        self.assertTrue(
            {"prepare_start", "prepare_complete", "shot_trigger", "reset_requested", "emergency_stop"} <= events
        )
        summary = json.loads((trace.path / "summary.json").read_text())
        self.assertEqual(summary["samples"], len(samples))
        self.assertIn("control.sample_log_ms", summary["timing"])
        self.assertIn("control.submit_interval_ms", summary["timing"])
        metadata = json.loads((trace.path / "metadata.json").read_text())
        self.assertEqual(len(metadata["policy_sha256"]), 64)
        self.assertEqual({f.suffix for f in trace.path.iterdir()}, {".json", ".jsonl"})

    def test_tick_repetition_and_snapshot_are_logged_without_a_queue(self):
        trace = BasketballLog(self.directory.name, self.cfg, {})
        target = np.ones(29)
        for tick in (100, 100, 110, 1):
            trace.sample("control", {"state_tick": tick}, [], target, {})
            target[:] += 1
        trace.close()
        samples = [r for r in self.rows(trace) if r["type"] == "sample"]
        self.assertEqual([r["tick_delta"] for r in samples], [None, 0, 10, -109])
        self.assertGreater(samples[1]["unchanged_tick_ms"], 0)
        self.assertEqual(samples[2]["unchanged_tick_ms"], 0)
        self.assertEqual(samples[0]["target"], [1] * 29)
        summary = json.loads((trace.path / "summary.json").read_text())
        self.assertEqual(summary["robot_tick"]["repeated"], 1)
        self.assertEqual(summary["robot_tick"]["backwards_or_wrap"], 1)

    def test_failed_submission_records_attempt_without_advancing(self):
        trace = BasketballLog(self.directory.name, self.cfg, {})
        pipeline = BasketballPipeline(self.cfg, trace=trace)
        try:
            pipeline.prepare(prepare_seconds=0)
            with patch.object(pipeline.env, "step", side_effect=RuntimeError("failed send")):
                with self.assertRaisesRegex(RuntimeError, "failed send"):
                    pipeline.step()
            self.assertEqual(pipeline.policy.frame, 0)
        finally:
            pipeline.env.close()
            trace.close("test_error")
        errors = [r for r in self.rows(trace) if r.get("event") == "submission_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0]["target"]), 29)

    def test_main_records_initialization_exception_and_closes(self):
        argv = ["run_basketball", "--headless", "--log_dir", self.directory.name]
        with (
            patch.object(sys, "argv", argv),
            patch("scripts.run_basketball.BasketballPipeline", side_effect=RuntimeError("init failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "init failed"):
                main()
        run = next(Path(self.directory.name).iterdir())
        rows = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["event"], "session_end")
        self.assertIn("init failed", next(r for r in rows if r.get("event") == "exception")["traceback"])
        self.assertEqual(json.loads((run / "summary.json").read_text())["reason"], "RuntimeError")


class CppTelemetryTests(unittest.TestCase):
    def test_latest_mock_state_exposes_tick_and_read_time(self):
        stub = SimpleNamespace(RobotState=object, SportState=object, UnitreeController=Mock(side_effect=AssertionError))
        with patch.dict(sys.modules, {"unitree_cpp": stub}):
            module = importlib.import_module("robojudo.environment.unitree_cpp_env")
        env = object.__new__(module.UnitreeCppEnv)
        cfg = make_config(parse_args(["--real", "--no_keyboard"])).env
        Environment.__init__(env, cfg)
        env.robot = "g1"
        env._dof_idx = None
        env._odometry_type = "DUMMY"
        env.RemoteControllerHandler = None
        state = SimpleNamespace(
            tick=42,
            motor_state=SimpleNamespace(q=[0.1] * 29, dq=[0.2] * 29, tau_est=[0.3] * 29),
            imu_state=SimpleNamespace(
                quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0], rpy=[0, 0, 0], accelerometer=[0, 0, 9.81]
            ),
        )
        env.unitree = Mock(get_robot_state=Mock(return_value=state))
        env.update()
        first = env.get_data()
        state.tick = 43
        state.motor_state.q = [0.5] * 29
        env.update()
        second = env.get_data()
        self.assertEqual(second.state_tick, 43)
        self.assertGreater(second.state_read_monotonic_ns, first.state_read_monotonic_ns)
        np.testing.assert_allclose(second.dof_pos, 0.5)
        np.testing.assert_allclose(second.motor_tau_est, 0.3)
        stub.UnitreeController.assert_not_called()


if __name__ == "__main__":
    unittest.main()
