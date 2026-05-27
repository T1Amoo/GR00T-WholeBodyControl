"""Source-level guards for the masked observation variants.

Why source-level: ``gear_sonic.envs.manager_env.mdp.observations`` imports
``isaaclab.envs.mdp`` at module top, which transitively imports ``pxr``
(Isaac Sim runtime). On a dev machine without Isaac Sim, the module can't be
imported. These tests inspect the source AST to verify the variants exist
and apply the right mask.

Behavioral coverage of the underlying mask helper lives in
``test_action_mask.py`` (the pure-tensor surface ``apply_missing_dof_mask``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import torch

from gear_sonic.utils.joint_constants import MISSING_23DOF_INDICES_IL

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OBS_PATH = (
    _REPO_ROOT / "gear_sonic" / "envs" / "manager_env" / "mdp" / "observations.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(_OBS_PATH.read_text())


def _find_function(name: str) -> ast.FunctionDef:
    for node in _module_ast().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"observations.py does not define a top-level function {name!r}")


def test_joint_pos_rel_masked_defined():
    fn = _find_function("joint_pos_rel_masked")
    body_src = ast.unparse(fn)
    # Must clone so we don't mutate the simulator's joint buffer
    assert ".clone(" in body_src, "joint_pos_rel_masked must clone before mutating"
    # Must zero the missing-IL columns
    assert "MISSING_23DOF_INDICES_IL" in body_src, (
        "joint_pos_rel_masked must reference MISSING_23DOF_INDICES_IL"
    )
    assert re.search(r"\[\s*\.\.\.\s*,\s*MISSING_23DOF_INDICES_IL\s*\]\s*=\s*0", body_src), (
        "joint_pos_rel_masked must assign 0 to [..., MISSING_23DOF_INDICES_IL]"
    )


def test_joint_vel_rel_masked_defined():
    fn = _find_function("joint_vel_rel_masked")
    body_src = ast.unparse(fn)
    assert ".clone(" in body_src
    assert "MISSING_23DOF_INDICES_IL" in body_src
    assert re.search(r"\[\s*\.\.\.\s*,\s*MISSING_23DOF_INDICES_IL\s*\]\s*=\s*0", body_src)


def test_joint_pos_multi_future_select_joints_masked_defined():
    fn = _find_function("joint_pos_multi_future_select_joints_masked")
    body_src = ast.unparse(fn)
    assert ".clone(" in body_src
    # Local-position remap: must filter joints_idx by membership in MISSING_23DOF_INDICES_IL
    assert "MISSING_23DOF_INDICES_IL" in body_src
    assert "joints_idx" in body_src, (
        "select_joints_masked must consult joints_idx to compute local zero positions"
    )


def test_local_zero_remap_logic_for_wrist_term():
    """The wrist-future obs term selects IL idx [23, 24, 25, 26, 27, 28]; of those,
    [25, 26, 27, 28] are the 4 missing wrist joints, mapping to local positions
    [2, 3, 4, 5] in the output tensor.

    This pure-tensor test reproduces the local-position remap logic and
    confirms the math, so a future refactor of
    ``joint_pos_multi_future_select_joints_masked`` can be validated against
    the same expectation.
    """
    joints_idx = [23, 24, 25, 26, 27, 28]
    expected_local_zero = [2, 3, 4, 5]
    actual_local_zero = [
        i for i, j in enumerate(joints_idx) if j in MISSING_23DOF_INDICES_IL
    ]
    assert actual_local_zero == expected_local_zero

    # Apply that to a tensor and confirm exactly those local positions go to 0
    n_envs, n_future = 4, 7
    out = torch.ones(n_envs, n_future, len(joints_idx))
    out[..., actual_local_zero] = 0.0
    for i in range(len(joints_idx)):
        if i in expected_local_zero:
            assert torch.all(out[..., i] == 0.0)
        else:
            assert torch.all(out[..., i] == 1.0)


def test_yaml_func_targets_match_observations_module():
    """Drift check: every ``func: gear_sonic.envs.manager_env.mdp:NAME`` in
    sonic_release_23dof.yaml must point to a top-level function defined in
    observations.py.
    """
    yaml_path = (
        _REPO_ROOT
        / "gear_sonic"
        / "config"
        / "exp"
        / "manager"
        / "universal_token"
        / "all_modes"
        / "sonic_release_23dof.yaml"
    )
    yaml_src = yaml_path.read_text()
    referenced = set(
        re.findall(
            r"gear_sonic\.envs\.manager_env\.mdp:([A-Za-z_][A-Za-z0-9_]*)",
            yaml_src,
        )
    )
    assert referenced, "sonic_release_23dof.yaml referenced no mdp functions"

    defined = {
        node.name
        for node in _module_ast().body
        if isinstance(node, ast.FunctionDef)
    }

    missing = referenced - defined
    assert not missing, (
        f"sonic_release_23dof.yaml references undefined mdp functions: {missing}"
    )
