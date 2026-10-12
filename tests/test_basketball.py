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
        self.policy = MimicBBPolicy(load_student_config())
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
        p.scales[:3] = 2
        p.scales[3:18] = 3
        p.scales[33:178] = 4
        p.scales[323:468] = 5
        p.scales[468:] = 0.5
        p.reset_from_state(self.state)
        obs, _ = p.get_observation(self.state, {})
        expected = np.concatenate(
            (
                [2, 0, 0],
                np.tile([0, 0, -3], 5),
                np.tile([0.25, 0.5, 0.75], 5),
                np.tile((self.state.dof_pos - p.default_pos) * 4, 5),
                np.tile(self.state.dof_vel * 0.05, 5),
                np.tile((p.initial_pos - p.default_pos) * 5, 5),
                [1.5, 0, 0.9],
            )
        )
        self.assertEqual(obs.shape, (471,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs, expected, atol=1e-7)

    def test_latest_sensor_and_causal_applied_action(self):
        p = self.policy
        old_actions = p.action_history.copy()
        self.state.base_ang_vel[:] = [8, 12, 16]
        obs, _ = p.get_observation(self.state, {})
        np.testing.assert_allclose(obs[18:33].reshape(5, 3)[-1], [2, 3, 4])
        p.get_action(obs)
        np.testing.assert_array_equal(p.action_history, old_actions)
        target = np.clip(p.default_pos + 0.2, p.lower, p.upper)
        p.record_applied_target(target)
        obs, _ = p.get_observation(self.state, {})
        np.testing.assert_allclose(obs[323:468].reshape(5, 29)[-1], target - p.default_pos)
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

    def test_skill_and_hoop_updates_preserve_histories(self):
        p = self.policy
        before = p.sensor_history.copy(), p.action_history.copy()
        p.set_command(1)
        p.update_hoop([4, 0.25, 2])
        np.testing.assert_array_equal(p.sensor_history, before[0])
        np.testing.assert_array_equal(p.action_history, before[1])
        obs, _ = p.get_observation(self.state, {})
        np.testing.assert_array_equal(obs[:3], [0, 1, 0])
        np.testing.assert_allclose(obs[-3:], [1, 0.5, 0.5])
        p.reset_from_state(self.state)
        self.assertEqual(p.command_id, 0)
        np.testing.assert_allclose(p.hoop_pos, [4, 0.25, 2])

    def test_invalid_runtime_updates_leave_state_unchanged(self):
        p = self.policy
        for command in (-1, 3, 5):
            with self.assertRaises(ValueError):
                p.set_command(command)
        for hoop in ([1, 2], [-1, 0, 1], [1, float("nan"), 1]):
            with self.assertRaises(ValueError):
                p.update_hoop(hoop)
        self.assertEqual(p.command_id, 0)
        np.testing.assert_allclose(p.hoop_pos, [3, 0, 1.8])

    def test_hardware_config_is_cpp_only(self):
        cfg = make_config(parse_args(["--real"]))  # Configuration only: no hardware imports/init.
        self.assertEqual(cfg.env.env_type, "UnitreeCppEnv")
        self.assertEqual(cfg.env.unitree.control_dt, 0.01)
        self.assertEqual(cfg.ctrl[0].triggers["Key.esc"], "[SHUTDOWN]")
        self.assertEqual(cfg.ctrl[1].triggers["Select"], "[SHUTDOWN]")
        self.assertEqual(cfg.ctrl[1].triggers["R2+B"], "[SKILL_1]")
        np.testing.assert_allclose(cfg.env.dof.stiffness, self.policy.cfg_policy.student["pd_stiffness"])

    def test_removed_phase_cli_is_rejected(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            parse_args(["--pre_hold", "0"])


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = BasketballPipeline(make_config(parse_args(["--headless", "--auto_start"])))

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

    def test_send_failure_does_not_commit_action(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.policy.set_command(1)
        old = p.policy.action_history.copy()
        with patch.object(p.env, "step", side_effect=RuntimeError("send failed")):
            with self.assertRaisesRegex(RuntimeError, "send failed"):
                p.step()
        np.testing.assert_array_equal(p.policy.action_history, old)
        self.assertEqual(p.policy.command_id, 1)

    def test_emergency_stop_preempts_inference_and_latches(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SKILL_1]", "[SHUTDOWN]"]))
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

    def test_sim_auto_start_runs_default_skill(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        self.assertTrue(p.started)
        p.step()
        np.testing.assert_array_equal(p.policy.obs[:3], [1, 0, 0])

    def test_runtime_skill_switch_preserves_history(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.step()
        before = p.policy.action_history.copy()
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SKILL_2]"]))
        p.step()
        np.testing.assert_array_equal(p.policy.obs[:3], [0, 0, 1])
        np.testing.assert_array_equal(p.policy.obs[323:468], before.ravel())

    def test_sim_auto_start_reset_restarts_default_skill(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.policy.set_command(1)
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[MOTION_RESET]"]))
        p.step()
        self.assertTrue(p.started)
        self.assertEqual(p.policy.command_id, 0)

    def test_reset_tiles_histories(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        p.step()
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[MOTION_RESET]"]))
        p.step()
        np.testing.assert_array_equal(p.policy.sensor_history[0], p.policy.sensor_history[-1])

    def test_no_ball_scene(self):
        from robojudo.environment.basketball_mujoco_env import BasketballMujocoEnv

        cfg = make_config(parse_args(["--headless", "--auto_start", "--no_ball"]))
        env = BasketballMujocoEnv(cfg.env)
        try:
            self.assertEqual(env.model.nq, 36)
            env.step(env.dof_pos)
            self.assertEqual(env.dof_pos.shape, (29,))
        finally:
            env.close()

    def test_real_onnx_rollout_with_skill_switch(self):
        p = self.pipeline
        p.prepare(prepare_seconds=0)
        for _ in range(5):
            p.step()
        p.ctrl_manager.get_ctrl_data = Mock(return_value=Box(COMMANDS=["[SKILL_1]"]))
        p.step()
        p.ctrl_manager.get_ctrl_data.return_value = Box(COMMANDS=[])
        for _ in range(180):
            p.step()
        self.assertEqual(p.policy.command_id, 1)
        self.assertTrue(np.isfinite(p.env.data.qpos).all())
        self.assertTrue(np.isfinite(p.policy.obs).all())
        np.testing.assert_array_equal(p.policy.obs[:3], [0, 1, 0])
        mujoco.mj_forward(p.env.model, p.env.data)


if __name__ == "__main__":
    unittest.main()
