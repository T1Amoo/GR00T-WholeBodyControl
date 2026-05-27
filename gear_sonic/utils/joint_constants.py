"""Canonical joint indices for G1 hardware variants.

Single source of truth — sim/training/deploy all import from here.

The 6 joints absent on the 23-DoF G1 hardware variant. Indices are MJCF order,
matching `G1JointIndex` in `gear_sonic_deploy/.../include/robot_parameters.hpp`.
"""

MISSING_23DOF_INDICES_MJCF: list[int] = [13, 14, 20, 21, 27, 28]

MISSING_23DOF_JOINT_NAMES: list[str] = [
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

assert len(MISSING_23DOF_INDICES_MJCF) == len(MISSING_23DOF_JOINT_NAMES) == 6, \
    "MISSING_23DOF_INDICES_MJCF and MISSING_23DOF_JOINT_NAMES must stay aligned"


# IL (IsaacLab) idx for each joint in MISSING_23DOF_JOINT_NAMES order. IL pairs
# wrists L/R by joint type (25=L-pitch, 26=R-pitch, 27=L-yaw, 28=R-yaw), while
# the names list groups them per-side, so this is NOT sorted. Probed at runtime
# 2026-05-19 against G1_ISAACLab_ORDER in gear_sonic/envs/env_utils/joint_utils.py.
MISSING_23DOF_INDICES_IL: list[int] = [5, 8, 25, 27, 26, 28]
