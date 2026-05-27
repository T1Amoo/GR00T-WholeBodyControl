"""Reward-side correctness for the D1 23DoF training pipeline.

The design spec (§13.3) names two reward terms that need to ignore the 6
hardware-absent joints:

1. ``joint_limit`` — handled in ``sonic_release_23dof.yaml`` by replacing the
   ``joint_names: [".*"]`` regex with an explicit list of the 23 present
   joints. Already covered by ``tests/test_sonic_release_23dof_yaml.py``.

2. ``action_rate_l2`` — the design suggested an ``action_rate_l2_masked``
   variant. The implementation took a different path: the action mask
   (``apply_missing_dof_mask``) is applied **before** ``env.step`` in
   ``ManagerEnvWrapper.step`` (see ``manager_env_wrapper.py:844-849``). That
   means ``action_manager.action`` (which ``action_rate_l2`` reads) already
   has the 6 missing IL indices zeroed, and so does ``prev_action`` after the
   first step. The diff at those indices is identically 0, so squaring and
   summing contributes 0 — equivalent to the spec's masked variant, without
   requiring a custom reward term.

This test verifies that property in pure tensor form: it does NOT need
IsaacLab. If ``apply_missing_dof_mask`` is ever removed from the
pre-``env.step`` path, this test still passes (it only tests the math), so
the real guard is the ``ManagerEnvWrapper.step`` source — covered by
``test_action_pre_step_mask_invariant`` below, which inspects the source.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from gear_sonic.utils.joint_constants import MISSING_23DOF_INDICES_IL
from gear_sonic.utils.joint_mask import apply_missing_dof_mask

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Reproduces ``isaaclab.envs.mdp.action_rate_l2`` math: sum of squared diff."""
    return torch.sum(torch.square(action - prev_action), dim=1)


def test_action_rate_l2_zero_contribution_at_missing_idx_after_pre_mask():
    """If both action and prev_action are pre-masked, missing idx contribute 0.

    The full sum equals the sum restricted to the 23 kept indices.
    """
    torch.manual_seed(0)
    raw_action = torch.randn(8, 29)
    raw_prev = torch.randn(8, 29)

    # Both go through the same pre-step mask the wrapper applies.
    action = apply_missing_dof_mask(raw_action.clone(), MISSING_23DOF_INDICES_IL)
    prev_action = apply_missing_dof_mask(raw_prev.clone(), MISSING_23DOF_INDICES_IL)

    full_sum = _action_rate_l2(action, prev_action)

    keep_idx = [i for i in range(29) if i not in MISSING_23DOF_INDICES_IL]
    kept_sum = torch.sum(torch.square(action[:, keep_idx] - prev_action[:, keep_idx]), dim=1)

    assert torch.allclose(full_sum, kept_sum), (
        "action_rate_l2 over masked inputs must equal the sum over only the 23 kept idx"
    )


def test_action_rate_l2_missing_idx_diff_squared_is_zero():
    """Per-idx assertion: at every missing IL idx, (a - prev_a)^2 is exactly 0."""
    torch.manual_seed(1)
    raw_action = torch.randn(4, 29)
    raw_prev = torch.randn(4, 29)

    action = apply_missing_dof_mask(raw_action.clone(), MISSING_23DOF_INDICES_IL)
    prev_action = apply_missing_dof_mask(raw_prev.clone(), MISSING_23DOF_INDICES_IL)

    diff_sq = torch.square(action - prev_action)
    for c in MISSING_23DOF_INDICES_IL:
        assert torch.all(diff_sq[:, c] == 0.0), f"diff² nonzero at IL idx {c}"


def test_action_rate_l2_kept_idx_unchanged_by_pre_mask():
    """Sanity: the 23 kept indices' contribution is the same before/after masking."""
    torch.manual_seed(2)
    raw_action = torch.randn(4, 29)
    raw_prev = torch.randn(4, 29)

    action = apply_missing_dof_mask(raw_action.clone(), MISSING_23DOF_INDICES_IL)
    prev_action = apply_missing_dof_mask(raw_prev.clone(), MISSING_23DOF_INDICES_IL)

    keep_idx = [i for i in range(29) if i not in MISSING_23DOF_INDICES_IL]
    kept_after = torch.sum(
        torch.square(action[:, keep_idx] - prev_action[:, keep_idx]), dim=1
    )
    kept_before = torch.sum(
        torch.square(raw_action[:, keep_idx] - raw_prev[:, keep_idx]), dim=1
    )
    assert torch.allclose(kept_after, kept_before)


def test_action_pre_step_mask_invariant():
    """Source-level guard: ``ManagerEnvWrapper.step`` must apply the mask
    before ``self.env.step(env_actions)``.

    This is the actual mechanism the action_rate_l2 property relies on. If
    someone refactors the wrapper and drops the pre-step mask, this test
    fires.
    """
    src = (
        _REPO_ROOT / "gear_sonic" / "envs" / "wrapper" / "manager_env_wrapper.py"
    ).read_text()

    mask_call = re.search(
        r"apply_missing_dof_mask\(\s*env_actions\s*,\s*MISSING_23DOF_INDICES_IL\s*\)",
        src,
    )
    step_call = re.search(r"self\.env\.step\(\s*env_actions\s*\)", src)

    assert mask_call, (
        "ManagerEnvWrapper.step must call apply_missing_dof_mask(env_actions, "
        "MISSING_23DOF_INDICES_IL) — pre-step mask is what gives action_rate_l2 "
        "zero contribution at the 6 missing idx"
    )
    assert step_call, "ManagerEnvWrapper.step must dispatch to self.env.step(env_actions)"
    assert mask_call.start() < step_call.start(), (
        "apply_missing_dof_mask must run BEFORE self.env.step — otherwise "
        "action_manager.action sees unmasked values and action_rate_l2 mis-penalizes"
    )
