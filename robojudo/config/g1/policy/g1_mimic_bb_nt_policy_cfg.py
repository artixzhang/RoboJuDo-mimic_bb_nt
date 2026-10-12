"""Configuration boundary for the basketball student export."""

import json
from pathlib import Path

import numpy as np

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

MODEL_DIR = ASSETS_DIR / "models/g1/mimic_bb_nt"


def observation_fields(dimension):
    sizes = [
        ("skill_command", dimension),
        ("projected_gravity_history", 15),
        ("pelvis_angular_velocity_history", 15),
        ("joint_position_offset_history", 145),
        ("joint_velocity_history", 145),
        ("applied_action_offset_history", 145),
        ("hoop_pos_cyl", 3),
    ]
    offset = 0
    fields = []
    for name, size in sizes:
        fields.append((name, offset, offset + size))
        offset += size
    return fields


class MimicBBPolicyCfg(PolicyCfg):
    policy_type: str = "MimicBBPolicy"
    robot: str = "g1"
    disable_autoload: bool = True
    history_length: int = 5
    freq: float = 100.0
    model_path: str = str(MODEL_DIR / "student_policy.onnx")
    student: dict
    hoop_pos: tuple[float, float, float] = (3.0, 0.0, 1.8)

    @property
    def policy_file(self) -> str:
        return self.model_path


def load_student_config(path=MODEL_DIR / "student_config.json", **kwargs) -> MimicBBPolicyCfg:
    student = json.loads(Path(path).read_text())
    dimension = student["command_dimension"]
    command = student["skills"]["command"]
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("command_dimension must be a positive integer")
    if command["dimension"] != dimension or command["encoding"] != "one_hot":
        raise ValueError("Expected consistent one-hot command dimensions")
    if command["valid_ids"] != list(range(dimension)) or len(command["labels"]) != dimension:
        raise ValueError("Command IDs must be contiguous and labels must match dimension")
    if len(set(command["labels"])) != dimension or student["default_skill"] not in command["labels"]:
        raise ValueError("Expected unique skill labels and a valid default_skill")
    layout = [(f["name"], f["start"], f["stop"]) for f in student["observation"]["fields"]]
    if (student["observation_dim"], student["action_dim"], student["history_length"]) != (dimension + 468, 29, 5):
        raise ValueError(
            "Basketball student requires command_dimension + 468 observations, 29 actions and 5 history frames"
        )
    if len(student["joint_names"]) != 29 or len(set(student["joint_names"])) != 29:
        raise ValueError("Student joint_names must contain 29 unique names")
    if layout != observation_fields(dimension) or student["observation"]["history_order"] != "oldest_to_newest":
        raise ValueError("Unsupported basketball observation layout")
    if student["observation"]["layout"] != "field_major" or student["action"].get("delay_steps", 0) != 0:
        raise ValueError("Expected field-major observations and zero action delay")
    for name in (
        "nominal_joint_position",
        "safe_lower_joint_position",
        "safe_upper_joint_position",
        "initial_pose",
        "pd_stiffness",
        "pd_damping",
    ):
        values = np.asarray(student[name])
        if values.shape != (29,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain 29 finite values")
    lower = np.asarray(student["safe_lower_joint_position"])
    upper = np.asarray(student["safe_upper_joint_position"])
    initial = np.asarray(student["initial_pose"])
    if np.any(lower > upper) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Initial pose must be within the exported safe joint limits")
    if not np.isfinite(student["policy_hz"]) or student["policy_hz"] <= 0:
        raise ValueError("Policy rate must be finite and positive")
    for field in student["observation"]["fields"]:
        frame_size = field["stop"] - field["start"]
        history = 1 if field["name"] in ("skill_command", "hoop_pos_cyl") else 5
        if field["history_length"] != history or field["frame_size"] * history != frame_size:
            raise ValueError("Unsupported field history shape")
        scale = np.asarray(field["scale"])
        if scale.shape not in ((), (field["frame_size"],)) or not np.all(np.isfinite(scale)):
            raise ValueError("Invalid observation scale")
    dof = DoFConfig(
        joint_names=student["joint_names"],
        default_pos=student["nominal_joint_position"],
        stiffness=student["pd_stiffness"],
        damping=student["pd_damping"],
        position_limits=list(zip(lower, upper, strict=True)),
    )
    return MimicBBPolicyCfg(student=student, freq=student["policy_hz"], obs_dof=dof, action_dof=dof, **kwargs)
