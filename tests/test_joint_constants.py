"""Single-source-of-truth check for the 6 missing-on-23DoF joint indices."""

def test_missing_23dof_indices_mjcf_canonical():
    from gear_sonic.utils.joint_constants import (
        MISSING_23DOF_INDICES_MJCF,
        MISSING_23DOF_JOINT_NAMES,
    )
    assert MISSING_23DOF_INDICES_MJCF == [13, 14, 20, 21, 27, 28]
    assert MISSING_23DOF_JOINT_NAMES == [
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    assert len(MISSING_23DOF_INDICES_MJCF) == len(MISSING_23DOF_JOINT_NAMES) == 6


def test_sim_bridge_uses_shared_constants():
    """The sim bridge must import from the shared module, not redefine.

    Identity (`is`) — not equality — is the load-bearing assertion: a future
    accidental local re-definition would equal the canonical list but be a
    different object, so `is` catches the regression `==` would silently miss.
    """
    from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge
    from gear_sonic.utils.joint_constants import MISSING_23DOF_INDICES_MJCF
    assert unitree_sdk2py_bridge.MISSING_23DOF_INDICES is MISSING_23DOF_INDICES_MJCF


def test_base_sim_uses_shared_constants():
    """base_sim must also import from shared module, not redefine the constants.

    Identity assertions for the same reason as test_sim_bridge_uses_shared_constants.
    """
    from gear_sonic.utils.mujoco_sim import base_sim
    from gear_sonic.utils.joint_constants import (
        MISSING_23DOF_INDICES_MJCF,
        MISSING_23DOF_JOINT_NAMES,
    )
    assert base_sim.MISSING_23DOF_INDICES is MISSING_23DOF_INDICES_MJCF
    assert base_sim.MISSING_23DOF_JOINT_NAMES is MISSING_23DOF_JOINT_NAMES
