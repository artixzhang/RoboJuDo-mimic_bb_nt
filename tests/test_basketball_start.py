"""Manual Start and fixed-command link testing, entirely offline."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from box import Box

from robojudo.pipeline.basketball_pipeline import BasketballPipeline
from robojudo.tools.basketball_log import BasketballLog
from scripts.analyze_basketball_link import write_report
from scripts.run_basketball import main, make_config, parse_args
from scripts.verify_basketball_history import verify


class ManualStartTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        cfg = make_config(parse_args(["--headless"]))
        self.trace = BasketballLog(self.directory.name, cfg, {})
        self.p = BasketballPipeline(cfg, trace=self.trace)
        self.addCleanup(self.trace.close)
        self.addCleanup(self.p.env.close)
        self.p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=[]))

    def press(self, command):
        self.p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[command])
        self.p.step()
        self.p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])

    def test_ready_never_infers_and_start_preserves_history(self):
        p = self.p
        with patch.object(p.policy.policy, "get_action", side_effect=AssertionError("Ready must not infer")):
            p.prepare(prepare_seconds=0.02)
            with patch.object(p.env, "step", wraps=p.env.step) as send:
                for _ in range(7):
                    p.step()
                for call in send.call_args_list:
                    np.testing.assert_array_equal(call.args[0], p.hold_target)
        self.assertFalse(p.started)
        self.assertEqual(p.policy.hold_steps, 0)
        old_sensors, old_actions = p.policy.sensor_history.copy(), p.policy.action_history.copy()
        self.press("[POLICY_START]")
        self.assertTrue(p.started)
        self.assertEqual(p.policy.frame, 1)
        self.assertEqual(p.policy.obs[0], 0)
        np.testing.assert_array_equal(p.policy.sensor_history[:-1], old_sensors[1:])
        np.testing.assert_allclose(p.policy.obs[321:466], old_actions.ravel())
        self.trace.close()
        self.assertTrue(verify(self.trace.path)["history_ok"])

    def test_prehold_only_counts_after_start_and_start_does_not_trigger_shot(self):
        p = self.p
        p.cfg.policy.pre_hold = 0.03
        p.prepare(prepare_seconds=0)
        for _ in range(10):
            p.step()
        self.assertEqual(p.policy.hold_steps, 0)
        self.press("[POLICY_START]")
        self.assertEqual(p.policy.hold_steps, 1)
        self.assertFalse(p.policy.active)
        p.step()
        p.step()
        self.assertTrue(p.policy.active)
        self.assertEqual(p.policy.frame, 0)
        p.step()
        self.assertEqual(p.policy.frame, 1)

    def test_manual_prehold_needs_second_shot_key_and_reset_returns_to_ready(self):
        p = self.p
        p.cfg.policy.pre_hold = -1
        p.prepare(prepare_seconds=0)
        self.press("[SHOT_TRIGGER]")
        self.assertTrue(p.started)
        self.assertFalse(p.policy.active)
        self.press("[POLICY_START]")
        self.assertEqual(p.policy.frame, 0)
        self.press("[SHOT_TRIGGER]")
        self.assertEqual(p.policy.frame, 1)
        self.press("[MOTION_RESET]")
        self.assertFalse(p.started)
        self.assertIsNone(p.started_ns)
        with patch.object(p.policy.policy, "get_action", side_effect=AssertionError):
            p.step()

    def test_ready_estop_preempts_start_and_latches(self):
        p = self.p
        p.prepare(prepare_seconds=0)
        p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=["[POLICY_START]", "[SHUTDOWN]"])
        with patch.object(p.policy.policy, "get_action", side_effect=AssertionError):
            p.step()
            self.press("[POLICY_START]")
        self.assertTrue(p.stopped)
        self.assertFalse(p.started)

    def test_link_test_never_sends_network_output_even_after_triggers(self):
        p = self.p
        p.cfg.link_test = True
        # Update metadata too: this test switches mode before Prepare.
        metadata_path = self.trace.path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["config"]["link_test"] = True
        metadata_path.write_text(json.dumps(metadata))
        p.prepare(prepare_seconds=0)
        p.step()
        with (
            patch.object(p.policy.policy, "get_action", return_value=np.ones(29)) as infer,
            patch.object(p.env, "step", wraps=p.env.step) as send,
        ):
            p.policy.policy.raw_action = np.ones(29)
            self.press("[POLICY_START]")
            self.press("[SHOT_TRIGGER]")
            for _ in range(8):
                p.step()
            self.assertEqual(infer.call_count, 10)
            for call in send.call_args_list:
                np.testing.assert_array_equal(call.args[0], p.hold_target)
        np.testing.assert_allclose(
            p.policy.action_history, np.tile(p.policy.initial_pos - p.policy.default_pos, (5, 1))
        )
        self.trace.close("link_test_complete")
        report = write_report(self.trace.path)
        self.assertEqual(report["fixed_target_mismatches"], 0)
        self.assertEqual(report["inference_samples"], 10)
        self.assertTrue(report["history"]["history_ok"])
        rows = [json.loads(line) for line in (self.trace.path / "trace.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row.get("inference_ran"))
        row["target"][0] += 0.1
        row["unchanged_tick_ms"] = 100
        row["env"]["state_tick"] = 5
        (self.trace.path / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = write_report(self.trace.path)
        self.assertEqual(report["status"], "issues_found")
        self.assertEqual(report["fixed_target_mismatches"], 1)
        self.assertGreaterEqual(len(report["findings"]), 2)


class EntryPointTests(unittest.TestCase):
    def test_keyboard_import_fails_before_backend_construction(self):
        cfg = make_config(parse_args(["--real"]))
        with (
            patch("robojudo.pipeline.basketball_pipeline.import_module", side_effect=ImportError("No X display")),
            patch("robojudo.pipeline.basketball_pipeline.RlPipeline.__init__") as initialize,
            self.assertRaisesRegex(RuntimeError, "--no_keyboard"),
        ):
            BasketballPipeline(cfg)
        initialize.assert_not_called()

    def test_partial_initialization_failure_shuts_down_created_backend(self):
        cfg = make_config(parse_args(["--real", "--no_keyboard"]))
        env, trace = Mock(), Mock()

        def fail_after_environment(pipeline, config):
            pipeline.env = env
            raise RuntimeError("Controller initialization failed")

        with (
            patch("robojudo.pipeline.basketball_pipeline.RlPipeline.__init__", new=fail_after_environment),
            self.assertRaisesRegex(RuntimeError, "Controller initialization failed"),
        ):
            BasketballPipeline(cfg, trace=trace)
        env.shutdown.assert_called_once_with()
        trace.event.assert_called_once_with("shutdown_complete", stage="initialization")

    def test_cpp_pipeline_with_fake_sdk_and_without_mujoco_or_display_packages(self):
        code = """
import sys, types, tempfile
from importlib.machinery import ModuleSpec
from unittest.mock import Mock
import numpy as np
from box import Box
sys.modules.update({name: None for name in ("mujoco", "mujoco_viewer", "glfw", "pynput", "pygame")})
sdk = types.ModuleType("unitree_cpp")
sdk.__spec__ = ModuleSpec("unitree_cpp", loader=None)
sdk.RobotState = sdk.SportState = object
class FakeCpp:
    def __init__(self, cfg):
        self.sent = []
        self.stopped = False
        self.state = types.SimpleNamespace(
            tick=0,
            motor_state=types.SimpleNamespace(q=[0.] * 29, dq=[0.] * 29, tau_est=[0.] * 29),
            imu_state=types.SimpleNamespace(
                quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0], rpy=[0, 0, 0], accelerometer=[0, 0, 9.81]))
    def self_check(self): return True
    def set_gains(self, *args): pass
    def get_robot_state(self):
        self.state.tick += 10
        return self.state
    def step(self, target):
        self.sent.append(target)
        self.state.motor_state.q = target.copy()
    def shutdown(self): self.stopped = True
sdk.UnitreeController = FakeCpp  # The installed extension is never imported.
sys.modules["unitree_cpp"] = sdk
from scripts.run_basketball import make_config, parse_args
from robojudo.pipeline.basketball_pipeline import BasketballPipeline
from robojudo.tools.basketball_log import BasketballLog
from scripts.analyze_basketball_link import write_report
cfg = make_config(parse_args(["--real", "--no_keyboard", "--link_test"]))
assert cfg.env.env_type == "UnitreeCppEnv" and not cfg.auto_start
assert cfg.ctrl[0].triggers_extra["Start"] == "[POLICY_START]"
cfg.ctrl = []
with tempfile.TemporaryDirectory() as directory:
    trace = BasketballLog(directory, cfg, {})
    p = BasketballPipeline(cfg, trace=trace)
    p.prepare(prepare_seconds=0)
    p.step()
    assert not p.started
    p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[POLICY_START]"]))
    p.step()
    p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
    for _ in range(5): p.step()
    for target in p.env.unitree.sent:
        np.testing.assert_allclose(target, p.hold_target)
    p.env.shutdown()
    assert p.env.unitree.stopped
    trace.close("link_test_complete")
    report = write_report(trace.path)
    assert report["backend"] == "UnitreeCppEnv" and not report["is_sim"]
    assert report["state_ticks"]["samples"] == 6
    assert report["state_ticks"]["repeated"] == 0
    assert report["history"]["history_ok"]
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_link_test_deadline_damps_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = [
                "run_basketball",
                "--headless",
                "--auto_start",
                "--link_test",
                "--test_seconds",
                "0.03",
                "--log_dir",
                directory,
            ]
            with patch.object(sys, "argv", args):
                main()
            run = next(Path(directory).iterdir())
            report = json.loads((run / "link_report.json").read_text())
            self.assertEqual(report["completion_reason"], "link_test_complete")
            self.assertGreater(report["inference_samples"], 0)
            rows = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
            self.assertIn("shutdown_complete", [r.get("event") for r in rows])

    def test_real_auto_start_is_rejected(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--real", "--auto_start"])


if __name__ == "__main__":
    unittest.main()
