"""Pure-tensor mask helpers for the D1 23DoF training pipeline.

Lives outside `gear_sonic.envs.manager_env.mdp` so unit tests can import it
without pulling in IsaacLab's pxr-dependent action/observation manager chain.
"""

from __future__ import annotations

from typing import Sequence

import torch


def apply_missing_dof_mask(actions: torch.Tensor, missing_idx: Sequence[int]) -> torch.Tensor:
    """Zero the columns at `missing_idx` (in-place) and return `actions`.

    Used by the 23DoF training pipeline to ensure the policy's commands for the
    6 hardware-absent joints are exactly zero before they reach the simulator
    AND before `action_manager.action` is captured for `action_rate_l2`.
    """
    actions[..., list(missing_idx)] = 0.0
    return actions
