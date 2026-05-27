#!/usr/bin/env python3
"""Convert a GMR-output G1 29DoF pkl into the deploy-format pkl consumed by convert_motions.py.

GMR pkl (from lgy/GMR/scripts/{smplx,bvh,fbx,...}_to_robot.py):
    fps:int, root_pos:(T,3), root_rot:(T,4 xyzw), dof_pos:(T,29 MJCF order)

Deploy pkl (input to convert_motions.py):
    {motion_name: {joint_pos, joint_vel,
                   body_pos_w, body_quat_w (wxyz),
                   body_lin_vel_w, body_ang_vel_w,
                   _body_indexes, time_step_total}}

Run with .venv_sim:
    source .venv_sim/bin/activate
    python gear_sonic_deploy/reference/gmr_to_deploy.py \
        --input penguin_29dof.pkl --motion_name penguin_walking
"""

import argparse
import os
import pickle

import numpy as np
import pinocchio as pin  # noqa: F401  (required so pinocchio can load the URDF)
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

TRACKED_LINKS = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

BODY_INDEXES = np.array(
    [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29], dtype=np.int64
)

MJCF_BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# MJ → IL DOF reorder (copied from gear_sonic/data_process/convert_soma_csv_to_motion_lib.py)
MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17,
     1, 4, 7, 10, 14, 18,
     2, 5, 8, 11, 15, 19,
     21, 23, 25, 27,
     12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., :3]], axis=-1)


def angular_velocity_world(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """ω_world from a (T,4) xyzw sequence using central diff on relative rotation."""
    rots = R.from_quat(quat_xyzw)
    T = quat_xyzw.shape[0]
    omega = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        if t == 0:
            dq = rots[1] * rots[0].inv()
            omega[t] = dq.as_rotvec() / dt
        elif t == T - 1:
            dq = rots[-1] * rots[-2].inv()
            omega[t] = dq.as_rotvec() / dt
        else:
            dq = rots[t + 1] * rots[t - 1].inv()
            omega[t] = dq.as_rotvec() / (2.0 * dt)
    return omega


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="GMR pkl file")
    ap.add_argument("--motion_name", default=None,
                    help="Motion name in output dict (default: input filename stem)")
    ap.add_argument("--output", default=None,
                    help="Output deploy pkl (default: <input>_deploy.pkl)")
    ap.add_argument("--fps", type=int, default=None,
                    help="Override fps (default: read from pkl)")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos_mj = np.asarray(data["dof_pos"], dtype=np.float64)
    fps = args.fps if args.fps else int(data["fps"])
    T = dof_pos_mj.shape[0]
    dt = 1.0 / fps
    print(f"[load] {args.input}: T={T}, fps={fps}")
    assert dof_pos_mj.shape[1] == 29, f"dof_pos has {dof_pos_mj.shape[1]} cols, expected 29"
    assert root_pos.shape == (T, 3) and root_rot_xyzw.shape == (T, 4)

    robot = instantiate_g1_robot_model()
    n_dofs = robot.num_dofs
    body_q_indices = np.array(
        [robot.dof_index(name) for name in MJCF_BODY_JOINTS], dtype=np.int64
    )
    print(f"[robot] G1 num_dofs={n_dofs}, body_q_indices range [{body_q_indices.min()},"
          f"{body_q_indices.max()}]")

    body_pos_w = np.zeros((T, 14, 3), dtype=np.float32)
    body_quat_xyzw = np.zeros((T, 14, 4), dtype=np.float32)
    R_root = R.from_quat(root_rot_xyzw).as_matrix()

    q = robot.default_body_pose.copy()
    for t in range(T):
        q[body_q_indices] = dof_pos_mj[t]
        robot.cache_forward_kinematics(q, auto_clip=True)
        for k, link_name in enumerate(TRACKED_LINKS):
            T_local = robot.frame_placement(link_name)
            p_local = T_local.translation
            R_local = T_local.rotation
            p_world = root_pos[t] + R_root[t] @ p_local
            R_world = R_root[t] @ R_local
            body_pos_w[t, k] = p_world.astype(np.float32)
            body_quat_xyzw[t, k] = R.from_matrix(R_world).as_quat().astype(np.float32)
    print(f"[fk ] body_pos_w x[{body_pos_w[:,:,0].min():.2f},{body_pos_w[:,:,0].max():.2f}] "
          f"y[{body_pos_w[:,:,1].min():.2f},{body_pos_w[:,:,1].max():.2f}] "
          f"z[{body_pos_w[:,:,2].min():.2f},{body_pos_w[:,:,2].max():.2f}]")

    body_quat_w = quat_xyzw_to_wxyz(body_quat_xyzw).astype(np.float32)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_ang_vel_w = np.zeros((T, 14, 3), dtype=np.float32)
    for k in range(14):
        body_ang_vel_w[:, k, :] = angular_velocity_world(body_quat_xyzw[:, k, :], dt)

    # MJ → IL: dof_il[il] = dof_mj[mj] where MJ_TO_IL[mj] = il
    il_to_mj = np.argsort(MJ_TO_IL)
    joint_pos_il = dof_pos_mj[:, il_to_mj].astype(np.float32)
    joint_vel_il = np.gradient(joint_pos_il, dt, axis=0).astype(np.float32)

    motion_name = args.motion_name or os.path.splitext(os.path.basename(args.input))[0]
    output = {
        motion_name: {
            "joint_pos": joint_pos_il,
            "joint_vel": joint_vel_il,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "_body_indexes": BODY_INDEXES,
            "time_step_total": np.int64(T),
        }
    }

    out_path = args.output or os.path.splitext(args.input)[0] + "_deploy.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"[save] {out_path}  motion='{motion_name}' T={T}")


if __name__ == "__main__":
    main()
