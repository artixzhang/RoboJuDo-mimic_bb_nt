"""Offline contract tests. Never instantiate UnitreeCppEnv or an SDK controller."""

import io
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from box import Box

from robojudo.config.g1.policy.g1_mimic_bb_nt_policy_cfg import load_student_config
from robojudo.pipeline.basketball_pipeline import BasketballPipeline
from robojudo.policy.mimic_bb_policy import MimicBBPolicy
from scripts.run_basketball import make_config, parse_args


class StudentContractTests(unittest.TestCase):
    def setUp(self):
        self.policy = MimicBBPolicy(load_student_config(pre_hold=-1))
        self.state = Box(
            base_quat=np.array([0, 0, 0, 1], dtype=np.float32),
            base_ang_vel=np.array([1, 2, 3], dtype=np.float32),
            dof_pos=self.policy.initial_pos.copy(),
            dof_vel=np.arange(29, dtype=np.float32),
        )
        self.policy.reset_from_state(self.state)

    def test_initial_tiles_and_all_field_scales(self):
        p = self.policy
        # Non-default scales catch fields accidentally left unscaled in implementation.
        p.scales[:1] = 2
        p.scales[1:16] = 3
        p.scales[31:176] = 4
        p.scales[321:466] = 5
        p.scales[466:] = 0.5
        p.reset_from_state(self.state)
        p.frame = 82.5
        obs, _ = p.get_observation(self.state, {})
        expected = np.concatenate(
            (
                [1],
                np.tile([0, 0, -3], 5),
                np.tile([0.25, 0.5, 0.75], 5),
                np.tile((self.state.dof_pos - p.default_pos) * 4, 5),
                np.tile(self.state.dof_vel * 0.05, 5),
                np.tile((p.initial_pos - p.default_pos) * 5, 5),
                [1.5, 0, 0.9],
            )
        )
        self.assertEqual(obs.shape, (469,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs, expected, atol=1e-7)

    def test_latest_sensor_and_causal_applied_action(self):
        p = self.policy
        old_actions = p.action_history.copy()
        self.state.base_ang_vel[:] = [8, 12, 16]
        obs, _ = p.get_observation(self.state, {})
        np.testing.assert_allclose(obs[16:31].reshape(5, 3)[-1], [2, 3, 4])
        p.get_action(obs)
        np.testing.assert_array_equal(p.action_history, old_actions)
        target = np.clip(p.default_pos + 0.2, p.lower, p.upper)
        p.record_applied_target(target)
        obs, _ = p.get_observation(self.state, {})
        np.testing.assert_allclose(obs[321:466].reshape(5, 29)[-1], target - p.default_pos)
        np.testing.assert_array_equal(p.action_history[:-1], old_actions[1:])

    def test_clipping_before_action_commit(self):
        p = self.policy
        p.model = Mock()
        raw = np.full((1, 29), 100, dtype=np.float32)
        raw[:, ::2] = -100
        p.model.run.return_value = [raw]
        offset = p.get_action(p.get_observation(self.state, {})[0])
        target = offset + p.default_pos
        np.testing.assert_allclose(target[::2], p.lower[::2], atol=3e-7)
        np.testing.assert_allclose(target[1::2], p.upper[1::2], atol=3e-7)
        p.record_applied_target(target)
        np.testing.assert_allclose(p.action_history[-1], offset)

    def test_trigger_preserves_histories_and_hoop(self):
        p = self.policy
        for _ in range(10):
            p.get_observation(self.state, {})
            p.post_step_callback()
        self.assertEqual(p.frame, 0)
        before = (p.sensor_history.copy(), p.action_history.copy(), p.hoop.copy())
        p.trigger()
        for got, want in zip((p.sensor_history, p.action_history, p.hoop), before, strict=True):
            np.testing.assert_array_equal(got, want)
        for _ in range(300):
            p.post_step_callback()
        self.assertEqual(p.frame, 165)
        self.assertEqual(p.get_observation(self.state, {})[0][0], 1)

    def test_timed_prehold_and_reference_rate(self):
        p = self.policy
        p.cfg_policy.pre_hold = 0.03
        for _ in range(3):
            p.post_step_callback()
        self.assertTrue(p.active)
        self.assertEqual(p.frame, 0)
        p.profile["reference_fps"] = 50
        p.post_step_callback()
        self.assertEqual(p.frame, 0.5)

    def test_hardware_config_is_cpp_only(self):
        cfg = make_config(parse_args(["--real"]))  # Configuration only: no hardware imports/init.
        self.assertEqual(cfg.env.env_type, "UnitreeCppEnv")
        self.assertEqual(cfg.env.unitree.control_dt, 0.01)
        self.assertEqual(cfg.ctrl[0].triggers["Key.esc"], "[SHUTDOWN]")
        self.assertEqual(cfg.ctrl[1].triggers["A"], "[SHUTDOWN]")
        self.assertEqual(cfg.ctrl[1].triggers_extra["B"], "[SHOT_TRIGGER]")
        np.testing.assert_allclose(cfg.env.dof.stiffness, self.policy.cfg_policy.student["pd_stiffness"])

    def test_prehold_cli_semantics(self):
        self.assertEqual(parse_args([]).pre_hold, 0)
        self.assertEqual(load_student_config().pre_hold, 0)
        for value in (-1, 0, 0.5, 2):
            self.assertEqual(parse_args(["--pre_hold", str(value)]).pre_hold, value)
        for value in ("-2", "-0.5", "nan", "inf"):
            with (
                self.subTest(value=value),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--pre_hold", value])


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = BasketballPipeline(make_config(parse_args(["--headless", "--pre_hold", "-1"])))

    def tearDown(self):
        self.pipeline.env.close()

    def test_prepare_never_infers_and_seeds_measured_state(self):
        p = self.pipeline
        with patch.object(p.policy.policy, "get_action", side_effect=AssertionError("Prepare ran policy")):
            p.prepare(prepare_seconds=0.02)
        expected = p.policy.policy._sensors(p.env.get_data())
        np.testing.assert_array_equal(p.policy.sensor_history, np.tile(expected, (5, 1)))
        np.testing.assert_array_equal(
            p.policy.action_history,
            np.tile(p.policy.initial_pos - p.policy.default_pos, (5, 1)),
        )

    def test_ball_freejoint_does_not_change_robot_observations_or_controls(self):
        e = self.pipeline.env
        robot_q = e.dof_pos.copy()
        ball = e.model.joint("basketball_free")
        q, v = ball.qposadr[0], ball.dofadr[0]
        e.data.qpos[q : q + 3] = [100, 200, 300]
        e.data.qvel[v : v + 6] = 99
        e.update()
        np.testing.assert_array_equal(e.dof_pos, robot_q)
        np.testing.assert_array_equal(e.dof_vel, np.zeros(29))
        for i, name in enumerate(e.joint_names):
            self.assertEqual(e.model.actuator_trnid[e._actuator_idx[i], 0], e.model.joint(name).id)

    def test_send_failure_does_not_commit_action_or_advance_phase(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.policy.trigger()
        old = p.policy.action_history.copy()
        with patch.object(p.env, "step", side_effect=RuntimeError("send failed")):
            with self.assertRaisesRegex(RuntimeError, "send failed"):
                p.step()
        np.testing.assert_array_equal(p.policy.action_history, old)
        self.assertEqual(p.policy.frame, 0)

    def test_emergency_stop_preempts_inference_and_latches(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SHOT_TRIGGER]", "[SHUTDOWN]"]))
        with patch.object(p.policy.policy, "get_action", side_effect=AssertionError("Inference after stop")):
            p.step()
            p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=["[MOTION_RESET]"])
            p.step()
        self.assertTrue(p.stopped)
        np.testing.assert_array_equal(p.env.stiffness, np.zeros(29))

    def test_emergency_stop_during_prepare(self):
        p = self.pipeline
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SHUTDOWN]"]))
        with patch.object(p.env, "step", side_effect=AssertionError("Command after stop")):
            with self.assertRaises(SystemExit):
                p.prepare(prepare_seconds=0.02)
        np.testing.assert_array_equal(p.env.stiffness, np.zeros(29))

    def test_zero_prehold_starts_immediately_after_prepare_without_trigger(self):
        p = self.pipeline
        p.cfg.policy.pre_hold = 0
        with patch.object(p.policy.policy, "get_action", side_effect=AssertionError("Prepare ran policy")):
            p.prepare(prepare_seconds=0.02)
        self.assertTrue(p.policy.active)
        self.assertEqual(p.policy.frame, 0)
        p.step()
        self.assertEqual(p.policy.obs[0], 0)
        self.assertEqual(p.policy.frame, 1)
        p.step()
        self.assertAlmostEqual(p.policy.obs[0], 1 / 165)
        self.assertEqual(p.policy.frame, 2)

    def test_manual_prehold_runs_policy_until_trigger(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        with patch.object(p.policy.policy, "get_action", wraps=p.policy.policy.get_action) as infer:
            for _ in range(5):
                p.step()
            self.assertEqual(infer.call_count, 5)
        self.assertFalse(p.policy.active)
        self.assertEqual(p.policy.frame, 0)
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SHOT_TRIGGER]"]))
        p.step()
        self.assertEqual(p.policy.frame, 1)

    def test_zero_prehold_reset_restarts_immediately(self):
        p = self.pipeline
        p.cfg.policy.pre_hold = 0
        p.prepare(prepare_seconds=0)
        p.step()
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[MOTION_RESET]"]))
        p.step()
        self.assertTrue(p.policy.active)
        self.assertEqual(p.policy.frame, 0)
        p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
        p.step()
        self.assertEqual(p.policy.frame, 1)

    def test_reset_returns_to_prehold(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.policy.trigger()
        p.step()
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[MOTION_RESET]"]))
        p.step()
        self.assertEqual(p.policy.frame, 0)
        self.assertFalse(p.policy.active)
        np.testing.assert_array_equal(p.policy.sensor_history[0], p.policy.sensor_history[-1])

    def test_no_ball_scene(self):
        from robojudo.environment.basketball_mujoco_env import BasketballMujocoEnv

        cfg = make_config(parse_args(["--headless", "--no_ball"]))
        env = BasketballMujocoEnv(cfg.env)
        try:
            self.assertEqual(env.model.nq, 36)
            env.step(env.dof_pos)
            self.assertEqual(env.dof_pos.shape, (29,))
        finally:
            env.close()

    def test_real_onnx_rollout_through_posthold(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        for _ in range(5):
            p.step()
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SHOT_TRIGGER]"]))
        p.step()
        p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
        for _ in range(180):
            p.step()
        self.assertEqual(p.policy.frame, 165)
        self.assertTrue(np.isfinite(p.env.data.qpos).all())
        self.assertTrue(np.isfinite(p.policy.obs).all())
        self.assertEqual(p.policy.obs[0], 1)
        mujoco.mj_forward(p.env.model, p.env.data)


if __name__ == "__main__":
    unittest.main()
