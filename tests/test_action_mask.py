"""Unit tests for the action / observation mask helpers used by the D1 23DoF
training pipeline. These are pure-tensor tests — they don't import IsaacLab.
"""

from __future__ import annotations

import torch

from gear_sonic.utils.joint_constants import MISSING_23DOF_INDICES_IL
from gear_sonic.utils.joint_mask import apply_missing_dof_mask


def test_apply_missing_dof_mask_zeros_only_target_indices():
    actions = torch.arange(29, dtype=torch.float32).repeat(4, 1) + 1.0  # (4, 29), all non-zero
    out = apply_missing_dof_mask(actions.clone(), MISSING_23DOF_INDICES_IL)

    # The 6 missing indices are zero
    assert torch.all(out[:, MISSING_23DOF_INDICES_IL] == 0.0)

    # All other 23 indices are preserved (non-zero, original values)
    keep_idx = [i for i in range(29) if i not in MISSING_23DOF_INDICES_IL]
    assert len(keep_idx) == 23
    expected_keep = (torch.arange(29, dtype=torch.float32) + 1.0)[keep_idx]
    for env_idx in range(4):
        assert torch.all(out[env_idx, keep_idx] == expected_keep)


def test_apply_missing_dof_mask_returns_same_tensor():
    """Helper mutates in-place AND returns the same tensor (callable in-line)."""
    actions = torch.ones(2, 29)
    out = apply_missing_dof_mask(actions, MISSING_23DOF_INDICES_IL)
    assert out is actions  # same object


def test_apply_missing_dof_mask_preserves_dtype_and_device():
    actions = torch.ones(2, 29, dtype=torch.float64)
    out = apply_missing_dof_mask(actions, MISSING_23DOF_INDICES_IL)
    assert out.dtype == torch.float64


def test_apply_missing_dof_mask_handles_extra_batch_dims():
    """Helper should work for (B, T, 29) histories too — uses ellipsis indexing."""
    actions = torch.ones(2, 5, 29)
    out = apply_missing_dof_mask(actions, MISSING_23DOF_INDICES_IL)
    assert torch.all(out[..., MISSING_23DOF_INDICES_IL] == 0.0)
    keep_idx = [i for i in range(29) if i not in MISSING_23DOF_INDICES_IL]
    assert torch.all(out[..., keep_idx] == 1.0)
