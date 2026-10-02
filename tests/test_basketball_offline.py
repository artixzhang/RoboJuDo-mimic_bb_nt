"""Offline timing and independent observation audit; never create a hardware backend."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from box import Box

from robojudo.config.g1.policy.g1_mimic_bb_nt_policy_cfg import MODEL_DIR
from robojudo.environment.base_env import Environment
from robojudo.pipeline.basketball_pipeline import BasketballPipeline
from robojudo.tools.basketball_log import BasketballLog
from scripts.benchmark_basketball import benchmark
from scripts.run_basketball import make_config, parse_args
from scripts.verify_basketball_history import verify


class OfflineBenchmarkTests(unittest.TestCase):
    def test_benchmark_logs_without_creating_an_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                config=str(MODEL_DIR / "student_config.json"),
                model=str(MODEL_DIR / "student_policy.onnx"),
                steps=10,
                warmup=2,
                paced=False,
                with_log=True,
                trace=None,
                output=Path(directory) / "benchmark.json",
            )
            with patch.object(Environment, "__init__", side_effect=AssertionError("No environment allowed")):
                result = benchmark(args)
            self.assertEqual(result["mode"], "offline_no_hardware")
            self.assertEqual(result["onnx_run"]["count"], 10)
            self.assertEqual(result["policy_compute"]["count"], 10)
            rows = (Path(result["trace_directory"]) / "trace.jsonl").read_text().splitlines()
            samples = [json.loads(row) for row in rows if json.loads(row)["type"] == "sample"]
            self.assertEqual(len(samples), 12)
            self.assertTrue(all(row["stage"] == "offline" and not row["submitted"] for row in samples))


class HistoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        cfg = make_config(parse_args(["--headless", "--auto_start", "--pre_hold", "-1"]))
        trace = BasketballLog(self.directory.name, cfg, {})
        pipeline = BasketballPipeline(cfg, trace=trace)
        try:
            pipeline.prepare(prepare_seconds=0)
            for _ in range(6):
                pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SHOT_TRIGGER]"]))
            pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
            for _ in range(6):
                pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=["[MOTION_RESET]"])
            pipeline.step()
            pipeline.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
            pipeline.step()
        finally:
            pipeline.env.close()
            trace.close()
        self.path = trace.path
        self.rows = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.controls = [row for row in self.rows if row.get("stage") == "control" and row["type"] == "sample"]

    def write_rows(self):
        (self.path / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def test_reconstructs_histories_across_trigger_and_reset_at_10ms(self):
        report = verify(self.path)
        self.assertTrue(report["history_ok"], report["first_errors"])
        self.assertEqual(report["samples_checked"], 14)
        self.assertEqual(report["resets_checked"], 2)
        self.assertLess(report["simulation_sample_interval"]["max_abs_deviation_ms"], 1e-8)
        self.assertEqual(report["policy_sample_interval"]["count"], 12)

    def test_detects_reversed_sensor_window(self):
        obs = self.controls[7]["observation"]
        obs[31:176] = np.asarray(obs[31:176]).reshape(5, 29)[::-1].ravel().tolist()
        self.write_rows()
        report = verify(self.path)
        self.assertFalse(report["history_ok"])
        self.assertIn("joint_position_offset_history", [error["field"] for error in report["first_errors"]])

    def test_detects_current_action_leaking_into_its_own_observation(self):
        metadata = json.loads((self.path / "metadata.json").read_text())
        student = metadata["config"]["policy"]["student"]
        order = [metadata["env_joint_names"].index(name) for name in student["joint_names"]]
        scale = next(f["scale"] for f in student["observation"]["fields"] if f["start"] == 321)
        row = self.controls[7]
        current = (np.asarray(row["target"])[order] - student["nominal_joint_position"]) * scale
        row["observation"][321:466] = row["observation"][350:466] + current.tolist()
        self.write_rows()
        report = verify(self.path)
        self.assertFalse(report["history_ok"])
        self.assertIn("applied_action_offset_history", [error["field"] for error in report["first_errors"]])

    def test_detects_broken_reset_and_scale(self):
        reset = next(row for row in self.rows if row.get("event") == "prepare_complete")
        reset["sensor_history"] = np.zeros((5, 64)).tolist()
        self.controls[0]["observation"][466:469] = [3, 0, 1.8]
        self.write_rows()
        report = verify(self.path)
        self.assertFalse(report["history_ok"])
        fields = {error["field"] for error in report["first_errors"]}
        self.assertTrue({"reset.sensor_history", "initial_hoop_position_b"} <= fields)


if __name__ == "__main__":
    unittest.main()
