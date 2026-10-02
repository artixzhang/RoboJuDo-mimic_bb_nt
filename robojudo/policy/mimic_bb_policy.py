"""Causal, field-major basketball student inference without software delay."""

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.utils.util_func import get_gravity_orientation


@policy_registry.register
class MimicBBPolicy(Policy):
    def __init__(self, cfg_policy, device="cpu"):
        super().__init__(cfg_policy, device)
        student = cfg_policy.student
        self.profile = student["phase"]["motion_profiles"][0]
        self.default_pos = np.asarray(student["nominal_joint_position"], dtype=np.float32)
        self.initial_pos = np.asarray(self.profile["initial_joint_position_rad"], dtype=np.float32)
        self.lower = np.asarray(student["safe_lower_joint_position"], dtype=np.float32)
        self.upper = np.asarray(student["safe_upper_joint_position"], dtype=np.float32)
        self.scales = np.empty(469, dtype=np.float32)
        for field in student["observation"]["fields"]:
            self.scales[field["start"] : field["stop"]] = field["scale"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(cfg_policy.policy_file, options, providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        if self.model.get_inputs()[0].shape[-1] != 469 or self.model.get_outputs()[0].shape[-1] != 29:
            raise ValueError("Student ONNX must map 469 observations to 29 joint offsets")
        self.obs = np.empty(469, dtype=np.float32)
        self.reset()

    def reset(self):
        self.frame = 0.0
        self.hold_steps = 0
        self.active = False
        self.initialized = False

    def _sensors(self, data):
        return np.concatenate(
            (
                get_gravity_orientation(data.base_quat),
                data.base_ang_vel,
                data.dof_pos - self.default_pos,
                data.dof_vel,
            )
        ).astype(np.float32)

    def reset_from_state(self, env_data):
        """Called only after Prepare; trigger() deliberately does not reset history."""
        self.reset()
        self.sensor_history = np.tile(self._sensors(env_data), (5, 1))
        self.action_history = np.tile(self.initial_pos - self.default_pos, (5, 1))
        self.hoop = np.asarray(self.cfg_policy.hoop_pos, dtype=np.float32) * self.scales[466:469]
        self.active = self.cfg_policy.pre_hold == 0
        self.initialized = True

    def trigger(self):
        self.active = True

    def get_init_dof_pos(self):
        return self.initial_pos.copy()

    def get_observation(self, env_data, ctrl_data):
        if not self.initialized:
            raise RuntimeError("Basketball policy must be initialized from the measured state after Prepare")
        self.sensor_history[:-1] = self.sensor_history[1:]
        self.sensor_history[-1] = self._sensors(env_data)
        h = self.sensor_history
        self.obs[0] = self.frame / (self.profile["frame_count"] - 1)
        self.obs[1:16] = h[:, :3].ravel()
        self.obs[16:31] = h[:, 3:6].ravel()
        self.obs[31:176] = h[:, 6:35].ravel()
        self.obs[176:321] = h[:, 35:64].ravel()
        self.obs[321:466] = self.action_history.ravel()
        self.obs[:466] *= self.scales[:466]
        self.obs[466:469] = self.hoop
        return self.obs, {}

    def get_action(self, obs):
        offset = self.model.run([self.output_name], {self.input_name: obs[None]})[0][0]
        self.raw_action = offset
        return np.clip(self.default_pos + offset, self.lower, self.upper) - self.default_pos

    def record_applied_target(self, target):
        """Commit only AFTER env.step succeeds, for the following observation."""
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = target - self.default_pos

    def post_step_callback(self, commands=None):
        if self.active:
            self.frame = min(
                self.frame + self.profile["reference_fps"] / self.freq,
                self.profile["frame_count"] - 1,
            )
        elif self.cfg_policy.pre_hold > 0:
            self.hold_steps += 1
            if self.hold_steps / self.freq >= self.cfg_policy.pre_hold:
                self.trigger()
