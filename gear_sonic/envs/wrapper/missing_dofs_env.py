"""ManagerBasedRLEnv subclass that physics-locks the 6 hardware-absent joints.

Used by the D1 23DoF training pipeline. Pinning is done INSIDE the decimation
loop (after each ``sim.step``) so observations/rewards see exactly the state
real 23DoF G1 hardware reports — those motors don't exist, so qpos≡default
and qvel≡0.

Implementation note: instead of overriding ``step()`` (whose body would have
to be kept in lock-step with upstream IsaacLab), we wrap ``self.sim.step``
once at construction time so every physics step is followed by a pin. The
parent's decimation loop then becomes:

    sim.step → pin → record_post_physics_decimation_step → render → scene.update

``scene.update`` reads simulator state into buffers; pinning before that
means observations and rewards both see the pinned values.

Why not ``EventTermCfg(mode="interval")``: interval events fire AFTER reward
computation, so the reward terms would briefly see un-pinned joint states.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from gear_sonic.utils.joint_constants import (
    MISSING_23DOF_INDICES_IL,
    MISSING_23DOF_JOINT_NAMES,
)


class MissingDofsLockEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with 6 missing-DoF joints physics-locked each sim step.

    On first sim step, resolves the joint ids from articulation joint names and
    raises if they don't match the canonical IL ordering documented in
    ``gear_sonic.utils.joint_constants`` — this guards against silent drift if
    IsaacLab's joint ordering ever changes.
    """

    _ROBOT_ASSET_NAME = "robot"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._missing_joint_ids: list[int] | None = None
        self._missing_default_qpos: torch.Tensor | None = None
        self._missing_default_qvel: torch.Tensor | None = None

        orig_sim_step = self.sim.step

        def _sim_step_with_lock(*args, **kwargs):
            ret = orig_sim_step(*args, **kwargs)
            self._pin_missing_joints()
            return ret

        self.sim.step = _sim_step_with_lock

    def _resolve_missing_joint_state(self) -> None:
        articulation = self.scene[self._ROBOT_ASSET_NAME]
        joint_names = articulation.data.joint_names
        joint_ids: list[int] = []
        for name in MISSING_23DOF_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(
                    f"MissingDofsLockEnv: joint '{name}' not in articulation "
                    f"joint_names ({len(joint_names)} entries). Refusing to silently skip."
                )
            joint_ids.append(joint_names.index(name))

        if joint_ids != list(MISSING_23DOF_INDICES_IL):
            raise RuntimeError(
                f"MissingDofsLockEnv: resolved missing joint ids {joint_ids} != "
                f"MISSING_23DOF_INDICES_IL {MISSING_23DOF_INDICES_IL}. "
                "IsaacLab joint ordering changed; update joint_constants."
            )

        self._missing_joint_ids = joint_ids
        self._missing_default_qpos = articulation.data.default_joint_pos[
            :, joint_ids
        ].clone()
        self._missing_default_qvel = torch.zeros_like(self._missing_default_qpos)

    def _pin_missing_joints(self) -> None:
        if self._missing_joint_ids is None:
            self._resolve_missing_joint_state()
        articulation = self.scene[self._ROBOT_ASSET_NAME]
        articulation.write_joint_state_to_sim(
            position=self._missing_default_qpos,
            velocity=self._missing_default_qvel,
            joint_ids=self._missing_joint_ids,
        )
