"""Drift-check: missing_dofs YAML must mirror gear_sonic.utils.joint_constants.

The YAML duplicates MJCF/IL indices and joint names from the Python module so
that Hydra-driven training pipelines can read them without importing Python.
This test guards against the two diverging — exactly the bug that shipped in
commit dd04090, where the YAML's indices_il used the MJCF idx by mistake.
"""

from pathlib import Path

import yaml

from gear_sonic.utils.joint_constants import (
    MISSING_23DOF_INDICES_IL,
    MISSING_23DOF_INDICES_MJCF,
    MISSING_23DOF_JOINT_NAMES,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_YAML_PATH = _REPO_ROOT / "gear_sonic" / "config" / "missing_dofs" / "23dof_hardware.yaml"


def test_missing_dofs_yaml_matches_joint_constants():
    with _YAML_PATH.open() as f:
        cfg = yaml.safe_load(f)

    assert cfg["enabled"] is True
    assert cfg["variant"] == "23dof_hardware"
    assert cfg["indices_mjcf"] == MISSING_23DOF_INDICES_MJCF
    assert cfg["indices_il"] == MISSING_23DOF_INDICES_IL
    assert cfg["joint_names"] == MISSING_23DOF_JOINT_NAMES
