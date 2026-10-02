import inspect
import logging
import time

import mujoco
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viewer import MujocoViewer
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = self._load_model()
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # Free objects may occur before or after the robot in qpos/qvel.
        joint_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        self._qpos_idx = self.model.jnt_qposadr[joint_ids]
        self._qvel_idx = self.model.jnt_dofadr[joint_ids]
        root_body = self.model.body_rootid[self.model.jnt_bodyid[joint_ids[0]]]
        root_joint = self.model.body_jntadr[root_body]
        self._root_qpos_adr = self.model.jnt_qposadr[root_joint]
        self._root_qvel_adr = self.model.jnt_dofadr[root_joint]
        self._actuator_idx = np.array([
            np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)[0] for joint_id in joint_ids
        ])
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = None
        if not cfg_env.headless:
            # Both upstream and RoboJuDo's patched viewer are supported.
            viewer_kwargs = {}
            if "diable_key_callbacks" in inspect.signature(MujocoViewer).parameters:
                viewer_kwargs["diable_key_callbacks"] = True
            self.viewer = MujocoViewer(
                self.model, self.data, width=1200, height=900, hide_menus=True, **viewer_kwargs,
            )
            # Controller owns keys, viewer retains all mouse camera callbacks.
            import glfw

            glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -10.0
            self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras and self.viewer is not None:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        self.update()  # get initial state

    def _load_model(self):
        return mujoco.MjModel.from_xml_path(self.cfg_env.xml)

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        start = self._root_qpos_adr + 3
        q = self.data.qpos[start:start + 4].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[start:start + 4] = [
            c * q[0] - s * q[3], c * q[1] - s * q[2],
            c * q[2] + s * q[1], c * q[3] + s * q[0],
        ]

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            start = self._root_qpos_adr
            self.data.qpos[start:start + 7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos[self._qpos_idx].astype(np.float32)
        dof_vel = self.data.qvel[self._qvel_idx].astype(np.float32)

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        q, v = self._root_qpos_adr, self._root_qvel_adr
        quat = self.data.qpos[q + 3:q + 7].astype(np.float32)[[1, 2, 3, 0]]
        ang_vel = self.data.qvel[v + 3:v + 6].astype(np.float32)
        base_pos = self.data.qpos[q:q + 3].astype(np.float32)
        lin_vel = self.data.qvel[v:v + 3].astype(np.float32)

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def get_data(self):
        data = super().get_data()
        data.sim_time_s = float(self.data.time)
        return data

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        if self.viewer is not None:
            if not self.viewer.is_alive:
                raise SystemExit
            if self.cfg_env.camera_follow:
                start = self._root_qpos_adr
                self.viewer.cam.lookat = self.data.qpos[start:start + 3]
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl[self._actuator_idx] = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
