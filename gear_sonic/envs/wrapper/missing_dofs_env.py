"""ManagerBasedRLEnv subclass that locks the hardware-absent joints (F2).

Used by the D1 23DoF training pipeline. The missing joints (e.g. waist
roll/pitch + L/R wrist pitch/yaw on the 23DoF G1) don't exist on real hardware,
so training must hold them rigidly at their default pose without disturbing the
rest of the body.

**Locking mechanism (F2, deploy-aligned):** at the first sim step we raise the
PD stiffness + damping of the missing joints to a high value via the implicit
actuator (PhysX gains). Their action is masked to 0 elsewhere (action mask in
the wrapper + actor), so the PD target stays at default and the stiff implicit
PD pins them near default. This is set ONCE (PhysX dof gains persist across
resets; the implicit actuator does not re-write stiffness per step), so there
is zero per-step intervention — legs/torso physics integrate normally.

This replaces the earlier per-substep ``write_joint_state_to_sim`` pin, which
pushed the FULL (stale, mid-decimation) DOF position+velocity array to PhysX
every substep and thereby teleported EVERY joint back to the previous frame,
destroying locomotion (verified 2026-05-29: foot termination 1% -> 82% with
only 4 wrists "locked"; reverting to no-pin restored 1%). High implicit
stiffness is the deploy-parity form (sim2sim ``--simulate-23dof`` locks via
damping) and is unconditionally stable.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from gear_sonic.utils import joint_constants


class MissingDofsLockEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that rigidly locks the missing-DoF joints via stiffness.

    On the first sim step, resolves the joint ids from articulation joint names
    and raises if they don't match the canonical IL ordering in
    ``gear_sonic.utils.joint_constants`` — guards against silent drift if
    IsaacLab's joint ordering ever changes.
    """

    _ROBOT_ASSET_NAME = "robot"

    # PD gains used to rigidly hold the missing joints at default. ~18-35x the
    # nominal G1 joint gains (STIFFNESS_4010≈16.8, waist 2*STIFFNESS_5020≈28.5),
    # heavily over-damped to avoid ringing. Implicit actuators are stable at high
    # stiffness. Steady-state deflection under load = torque/stiffness (a few mrad
    # for wrists, ~2deg worst case for waist_pitch under torso load) — and these
    # joints are masked out of the policy obs anyway, so any residual is invisible
    # to the network.
    _LOCK_STIFFNESS = 500.0
    _LOCK_DAMPING = 50.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._missing_joint_ids: list[int] | None = None
        self._lock_applied = False

        orig_sim_step = self.sim.step

        def _sim_step_then_lock(*args, **kwargs):
            ret = orig_sim_step(*args, **kwargs)
            if not self._lock_applied:
                self._apply_stiffness_lock()
            return ret

        self.sim.step = _sim_step_then_lock

    def _resolve_missing_joint_ids(self) -> list[int]:
        articulation = self.scene[self._ROBOT_ASSET_NAME]
        joint_names = articulation.data.joint_names
        joint_ids: list[int] = []
        for name in joint_constants.ACTIVE_MISSING_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(
                    f"MissingDofsLockEnv: joint '{name}' not in articulation "
                    f"joint_names ({len(joint_names)} entries). Refusing to silently skip."
                )
            joint_ids.append(joint_names.index(name))

        if joint_ids != list(joint_constants.ACTIVE_MISSING_INDICES_IL):
            raise RuntimeError(
                f"MissingDofsLockEnv: resolved missing joint ids {joint_ids} != "
                f"ACTIVE_MISSING_INDICES_IL {joint_constants.ACTIVE_MISSING_INDICES_IL}. "
                "IsaacLab joint ordering changed or missing_dofs config mismatched."
            )
        return joint_ids

    def _apply_stiffness_lock(self) -> None:
        articulation = self.scene[self._ROBOT_ASSET_NAME]
        self._missing_joint_ids = self._resolve_missing_joint_ids()
        n_env = articulation.num_instances
        n_missing = len(self._missing_joint_ids)
        device = articulation.device
        stiffness = torch.full(
            (n_env, n_missing), self._LOCK_STIFFNESS, device=device
        )
        damping = torch.full(
            (n_env, n_missing), self._LOCK_DAMPING, device=device
        )
        articulation.write_joint_stiffness_to_sim(
            stiffness, joint_ids=self._missing_joint_ids
        )
        articulation.write_joint_damping_to_sim(
            damping, joint_ids=self._missing_joint_ids
        )
        self._lock_applied = True
