"""Basketball scene assembly; all simulation and viewer work stays in MujocoEnv."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from robojudo.environment import env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.mujoco_env import MujocoEnv


class BasketballMujocoEnvCfg(MujocoEnvCfg):
    env_type: str = "BasketballMujocoEnv"
    ball_xml: str | None = None
    initial_profile: dict
    camera_follow: bool = False


@env_registry.register
class BasketballMujocoEnv(MujocoEnv):
    def __init__(self, cfg_env, device="cpu"):
        super().__init__(cfg_env, device)
        if self.viewer is not None:
            import glfw

            glfw.swap_interval(0)  # Do not gate the 100 Hz loop on monitor refresh.
            self.viewer._render_every_frame = True
            self.viewer.cam.lookat[:] = [0.5, 0, 0.8]
            self.viewer.cam.distance = 3.5
        self.reborn()

    def _load_model(self):
        path = Path(self.cfg_env.xml)
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((path.parent / compiler.get("meshdir", "")).resolve()))
        if self.cfg_env.ball_xml:
            ball = ET.parse(self.cfg_env.ball_xml).getroot().find("worldbody")
            root.find("worldbody").extend(ball)
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def reborn(self, init_qpos=None):
        mujoco.mj_resetData(self.model, self.data)
        profile = self.cfg_env.initial_profile
        root = self.model.joint("floating_base_joint")
        start = root.qposadr[0]
        self.data.qpos[start : start + 7] = (
            profile["initial_root_position_m"] + profile["initial_root_quaternion_wxyz"]
            if init_qpos is None
            else init_qpos
        )
        self.data.qpos[self._qpos_idx] = profile["initial_joint_position_rad"]
        self.place_ball()
        mujoco.mj_forward(self.model, self.data)
        self.update()

    def place_ball(self):
        if self.cfg_env.ball_xml is None:
            return
        joint = self.model.joint("basketball_free")
        q, v = joint.qposadr[0], joint.dofadr[0]
        self.data.qpos[q : q + 3] = self.cfg_env.initial_profile["initial_ball_position_m"]
        self.data.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
        self.data.qvel[v : v + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def shutdown(self):
        # Match real Emergency Stop in simulation: zero Kp, retain damping.
        self.set_gains(np.zeros(self.num_dofs), self.damping)

    def close(self):
        super().shutdown()
