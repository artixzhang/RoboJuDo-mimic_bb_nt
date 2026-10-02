"""Configuration boundary for the basketball student export."""

import json
from pathlib import Path

import numpy as np

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

MODEL_DIR = ASSETS_DIR / "models/g1/mimic_bb_nt"
FIELDS = (
    ("phase", 0, 1),
    ("projected_gravity_history", 1, 16),
    ("pelvis_angular_velocity_history", 16, 31),
    ("joint_position_offset_history", 31, 176),
    ("joint_velocity_history", 176, 321),
    ("applied_action_offset_history", 321, 466),
    ("initial_hoop_position_b", 466, 469),
)


class MimicBBPolicyCfg(PolicyCfg):
    policy_type: str = "MimicBBPolicy"
    robot: str = "g1"
    disable_autoload: bool = True
    history_length: int = 5
    freq: float = 100.0
    model_path: str = str(MODEL_DIR / "student_policy.onnx")
    student: dict
    hoop_pos: tuple[float, float, float] = (3.0, 0.0, 1.8)
    pre_hold: float = 0.0

    @property
    def policy_file(self) -> str:
        return self.model_path


def load_student_config(path=MODEL_DIR / "student_config.json", **kwargs) -> MimicBBPolicyCfg:
    student = json.loads(Path(path).read_text())
    layout = [(f["name"], f["start"], f["stop"]) for f in student["observation"]["fields"]]
    if (student["observation_dim"], student["action_dim"], student["history_length"]) != (469, 29, 5):
        raise ValueError("Basketball student requires 469 observations, 29 actions and 5 history frames")
    if len(student["joint_names"]) != 29 or len(set(student["joint_names"])) != 29:
        raise ValueError("Student joint_names must contain 29 unique names")
    if layout != list(FIELDS) or student["observation"]["history_order"] != "oldest_to_newest":
        raise ValueError("Unsupported basketball observation layout")
    if student["observation"]["layout"] != "field_major" or student["action"]["delay_steps"] != 0:
        raise ValueError("Expected field-major observations and zero action delay")
    profile = student["phase"]["motion_profiles"][0]
    lower = np.asarray(student["safe_lower_joint_position"])
    upper = np.asarray(student["safe_upper_joint_position"])
    initial = np.asarray(profile["initial_joint_position_rad"])
    if initial.shape != (29,) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Initial pose must be within the exported safe joint limits")
    if student["policy_hz"] <= 0 or profile["reference_fps"] <= 0 or profile["frame_count"] < 2:
        raise ValueError("Policy/reference rates must be positive and motion must have at least two frames")
    dof = DoFConfig(
        joint_names=student["joint_names"],
        default_pos=student["nominal_joint_position"],
        stiffness=student["pd_stiffness"],
        damping=student["pd_damping"],
        position_limits=list(zip(lower, upper, strict=True)),
    )
    return MimicBBPolicyCfg(student=student, freq=student["policy_hz"], obs_dof=dof, action_dof=dof, **kwargs)
