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
        self.dimension = student["command_dimension"]
        self.labels = student["skills"]["command"]["labels"]
        self.fields = student["observation"]["fields"]
        self.update_hoop(cfg_policy.hoop_pos)
        self.default_pos = np.asarray(student["nominal_joint_position"], dtype=np.float32)
        self.initial_pos = np.asarray(student["initial_pose"], dtype=np.float32)
        self.lower = np.asarray(student["safe_lower_joint_position"], dtype=np.float32)
        self.upper = np.asarray(student["safe_upper_joint_position"], dtype=np.float32)
        self.scales = np.empty(student["observation_dim"], dtype=np.float32)
        for field in student["observation"]["fields"]:
            self.scales[field["start"] : field["stop"]] = (
                np.tile(field["scale"], field["history_length"]) if isinstance(field["scale"], list) else field["scale"]
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(cfg_policy.policy_file, options, providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        if (
            self.model.get_inputs()[0].shape[-1] != student["observation_dim"]
            or self.model.get_outputs()[0].shape[-1] != 29
        ):
            raise ValueError("Student ONNX dimensions must match the exported config")
        self.obs = np.empty(student["observation_dim"], dtype=np.float32)
        self.reset()

    def reset(self):
        self.command_id = self.labels.index(self.cfg_policy.student["default_skill"])
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
        """Called only after Prepare; runtime updates retain history."""
        self.reset()
        self.sensor_history = np.tile(self._sensors(env_data), (5, 1))
        self.action_history = np.tile(self.initial_pos - self.default_pos, (5, 1))

        self.initialized = True

    def set_command(self, command_id):
        if not isinstance(command_id, (int, np.integer)) or command_id not in range(self.dimension):
            raise ValueError(f"Invalid command ID {command_id}; valid IDs: 0..{self.dimension - 1}")
        self.command_id = command_id

    def update_hoop(self, position):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)) or position[0] < 0:
            raise ValueError("Hoop requires finite [r, theta, z] with r >= 0 (theta in radians)")
        scale = np.asarray(self.fields[-1]["scale"])
        limit = np.finfo(np.float32).max
        if np.any(np.abs(position) > limit) or np.any(np.abs(position * scale) > limit):
            raise ValueError("Hoop position exceeds float32 observation range")
        self.hoop_pos = position.astype(np.float32)

    @property
    def hoop(self):
        return self.hoop_pos * self.scales[-3:]

    def get_init_dof_pos(self):
        return self.initial_pos.copy()

    def get_observation(self, env_data, ctrl_data):
        if not self.initialized:
            raise RuntimeError("Basketball policy must be initialized from the measured state after Prepare")
        self.sensor_history[:-1] = self.sensor_history[1:]
        self.sensor_history[-1] = self._sensors(env_data)
        h = self.sensor_history
        values = {
            "skill_command": np.eye(self.dimension, dtype=np.float32)[self.command_id],
            "projected_gravity_history": h[:, :3].ravel(),
            "pelvis_angular_velocity_history": h[:, 3:6].ravel(),
            "joint_position_offset_history": h[:, 6:35].ravel(),
            "joint_velocity_history": h[:, 35:64].ravel(),
            "applied_action_offset_history": self.action_history.ravel(),
            "hoop_pos_cyl": self.hoop_pos,
        }
        for field in self.fields:
            self.obs[field["start"] : field["stop"]] = values[field["name"]]
        self.obs *= self.scales
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
        pass
