"""Test the 23DoF reference-motion zeroing preprocessor."""
import joblib
import numpy as np
from pathlib import Path

from gear_sonic.data_process.zero_23dof_motion_lib import zero_23dof_arrays_in_pkl
from gear_sonic.utils.joint_constants import MISSING_23DOF_INDICES_IL


def make_synthetic_pkl(tmp_path: Path) -> Path:
    """Create a synthetic motion_lib PKL matching the real schema produced by
    convert_soma_csv_to_motion_lib.py: joblib-compressed dict
    {motion_name: {root_trans_offset, pose_aa, dof, root_rot, smpl_joints, fps}}."""
    n_frames, n_joints = 10, 29
    rng = np.random.RandomState(0)
    entry = {
        "root_trans_offset": rng.randn(n_frames, 3).astype(np.float32),
        "pose_aa": rng.randn(n_frames, 30, 3).astype(np.float32),
        "dof": rng.randn(n_frames, n_joints).astype(np.float32),
        "root_rot": rng.randn(n_frames, 4).astype(np.float32),
        "smpl_joints": rng.randn(n_frames, 24, 3).astype(np.float32),
        "fps": 30.0,
    }
    data = {"synthetic_motion_001": entry}
    p = tmp_path / "motion.pkl"
    joblib.dump(data, p, compress=True)
    return p


def test_zero_23dof_arrays_zeros_only_target_columns(tmp_path):
    p_in = make_synthetic_pkl(tmp_path)
    p_out = tmp_path / "motion_23dof.pkl"

    zero_23dof_arrays_in_pkl(p_in, p_out)

    out = joblib.load(p_out)
    inp = joblib.load(p_in)
    name = "synthetic_motion_001"

    out_dof = out[name]["dof"]
    in_dof = inp[name]["dof"]

    # Target IL columns zeroed in dof
    for c in MISSING_23DOF_INDICES_IL:
        assert (out_dof[:, c] == 0).all(), f"col {c} of dof not zeroed"

    # Other columns preserved
    keep = [i for i in range(29) if i not in MISSING_23DOF_INDICES_IL]
    np.testing.assert_array_equal(out_dof[:, keep], in_dof[:, keep])

    # Other fields preserved (untouched by the preprocessor)
    for field in ("root_trans_offset", "pose_aa", "root_rot", "smpl_joints"):
        np.testing.assert_array_equal(out[name][field], inp[name][field])
    assert out[name]["fps"] == 30.0
