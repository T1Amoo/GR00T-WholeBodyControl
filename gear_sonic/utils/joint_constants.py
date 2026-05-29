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


# ---------------------------------------------------------------------------
# Runtime-active missing-dof set.
#
# The constants above are the FULL 6-joint 23DoF-hardware set. But a training
# run may lock a different subset (e.g. the `23dof_hardware_unlock_waist`
# variant locks only 5, or a wrist-only diagnostic locks 4). Every training
# consumer that needs the missing set — the physics lock (MissingDofsLockEnv),
# the obs masking (observations.py), and the pre-step action mask
# (manager_env_wrapper) — MUST read these ACTIVE_* vars (by module attribute,
# NOT `from ... import`) so they stay coherent with whatever the
# `missing_dofs` config selected. `set_active_missing_dofs` is called once at
# env construction (train_agent_trl.create_manager_env) from that config.
#
# Default = the full 6-joint set, so any consumer that runs before the setter
# (or in a context without a missing_dofs config) sees the canonical hardware
# set rather than nothing.
ACTIVE_MISSING_INDICES_IL: list[int] = list(MISSING_23DOF_INDICES_IL)
ACTIVE_MISSING_INDICES_MJCF: list[int] = list(MISSING_23DOF_INDICES_MJCF)
ACTIVE_MISSING_JOINT_NAMES: list[str] = list(MISSING_23DOF_JOINT_NAMES)


def set_active_missing_dofs(indices_il, indices_mjcf, joint_names) -> None:
    """Set the runtime-active missing-dof set from a resolved missing_dofs config.

    All three lists must describe the SAME joints in the SAME order. Called once
    at env construction so physics/obs/action masking are coherent. Replaces the
    module attributes (consumers read them by attribute access, so they observe
    the new values).
    """
    il = [int(x) for x in indices_il]
    mjcf = [int(x) for x in indices_mjcf]
    names = [str(x) for x in joint_names]
    if not (len(il) == len(mjcf) == len(names)):
        raise ValueError(
            f"set_active_missing_dofs: length mismatch il={il} mjcf={mjcf} names={names}"
        )
    global ACTIVE_MISSING_INDICES_IL, ACTIVE_MISSING_INDICES_MJCF, ACTIVE_MISSING_JOINT_NAMES
    ACTIVE_MISSING_INDICES_IL = il
    ACTIVE_MISSING_INDICES_MJCF = mjcf
    ACTIVE_MISSING_JOINT_NAMES = names
