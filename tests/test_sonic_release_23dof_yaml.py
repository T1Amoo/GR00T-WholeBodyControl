"""Drift-check: sonic_release_23dof.yaml must wire obs/reward overrides
correctly against the canonical missing-joint constants.

Guards against:
- An obs term forgetting to point to its `_masked` variant
- joint_limit's joint_names list including any of the 6 hardware-absent joints
- joint_limit's joint_names list missing any of the 23 expected joints
"""

from pathlib import Path

import yaml

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.utils.joint_constants import MISSING_23DOF_JOINT_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_YAML_PATH = (
    _REPO_ROOT
    / "gear_sonic"
    / "config"
    / "exp"
    / "manager"
    / "universal_token"
    / "all_modes"
    / "sonic_release_23dof.yaml"
)


def _load_yaml() -> dict:
    with _YAML_PATH.open() as f:
        return yaml.safe_load(f)


def test_policy_joint_obs_terms_use_masked_variants():
    cfg = _load_yaml()
    policy = cfg["manager_env"]["observations"]["policy"]
    assert (
        policy["joint_pos"]["func"]
        == "gear_sonic.envs.manager_env.mdp:joint_pos_rel_masked"
    )
    assert (
        policy["joint_vel"]["func"]
        == "gear_sonic.envs.manager_env.mdp:joint_vel_rel_masked"
    )


def test_critic_joint_obs_terms_use_masked_variants():
    cfg = _load_yaml()
    critic = cfg["manager_env"]["observations"]["critic"]
    assert (
        critic["joint_pos"]["func"]
        == "gear_sonic.envs.manager_env.mdp:joint_pos_rel_masked"
    )
    assert (
        critic["joint_vel"]["func"]
        == "gear_sonic.envs.manager_env.mdp:joint_vel_rel_masked"
    )


def test_tokenizer_wrist_term_uses_masked_variant():
    cfg = _load_yaml()
    tok = cfg["manager_env"]["observations"]["tokenizer"]
    assert (
        tok["joint_pos_multi_future_wrist_for_smpl"]["func"]
        == "gear_sonic.envs.manager_env.mdp:joint_pos_multi_future_select_joints_masked"
    )


def test_joint_limit_excludes_all_missing_joints():
    cfg = _load_yaml()
    names = cfg["manager_env"]["rewards"]["joint_limit"]["params"]["asset_cfg"][
        "joint_names"
    ]
    for missing in MISSING_23DOF_JOINT_NAMES:
        assert missing not in names, f"joint_limit must exclude {missing}"


def test_joint_limit_covers_all_23_present_joints():
    cfg = _load_yaml()
    names = cfg["manager_env"]["rewards"]["joint_limit"]["params"]["asset_cfg"][
        "joint_names"
    ]
    expected = [j for j in G1_ISAACLab_ORDER if j not in MISSING_23DOF_JOINT_NAMES]
    assert sorted(names) == sorted(expected)
    assert len(names) == 23


def test_motion_lib_cfg_wires_missing_dofs_mjcf():
    """The runtime FK fix in motion_lib_base.py reads missing_dofs_mjcf from
    motion_lib_cfg and zeros those axis-angles in pose_aa before fk_batch.
    Without this field the fix is a no-op and 23DoF training silently regresses
    to the old broken behavior (FK output reflects 29DoF wrist articulation).
    """
    cfg = _load_yaml()
    motion_lib_cfg = cfg["manager_env"]["commands"]["motion"]["motion_lib_cfg"]
    assert "missing_dofs_mjcf" in motion_lib_cfg, (
        "motion_lib_cfg must set missing_dofs_mjcf so pose_aa is zeroed before FK"
    )
    # Wired via Hydra interpolation to the single source of truth in
    # 23dof_hardware.yaml. yaml.safe_load surfaces the raw interpolation string.
    assert motion_lib_cfg["missing_dofs_mjcf"] == "${missing_dofs.indices_mjcf}", (
        f"missing_dofs_mjcf should interpolate from missing_dofs config, "
        f"got: {motion_lib_cfg['missing_dofs_mjcf']}"
    )
