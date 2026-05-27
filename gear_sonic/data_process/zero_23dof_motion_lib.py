"""Generate 23-DoF-compatible reference motion library.

Zeroes the 6 IL columns listed in joint_constants.MISSING_23DOF_INDICES_IL in
the joint_pos / joint_vel arrays of every PKL under --source, writing to
--dest. Mirrors the deploy-side helper gear_sonic_deploy/reference/zero_23dof_columns.py
but operates on the training motion_lib PKLs instead of CSVs.

Usage:
    python -m gear_sonic.data_process.zero_23dof_motion_lib \\
        --source data/motion_lib_bones_seed/robot_filtered \\
        --dest   data/motion_lib_bones_seed/robot_filtered_23dof \\
        --workers 16
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from gear_sonic.utils.joint_constants import MISSING_23DOF_INDICES_IL


def zero_23dof_arrays_in_pkl(src: Path, dst: Path) -> None:
    data = joblib.load(src)
    for motion_name, entry in data.items():
        dof = entry.get("dof")
        if isinstance(dof, np.ndarray) and dof.ndim == 2:
            dof = dof.copy()
            dof[:, MISSING_23DOF_INDICES_IL] = 0
            entry["dof"] = dof
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, dst, compress=True)


def _worker(args):
    src, dst = args
    zero_23dof_arrays_in_pkl(src, dst)
    return src.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    pkls = sorted(args.source.rglob("*.pkl"))
    print(f"[zero_23dof] {len(pkls)} PKLs in {args.source}")
    args.dest.mkdir(parents=True, exist_ok=True)

    jobs = [(p, args.dest / p.relative_to(args.source)) for p in pkls]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, name in enumerate(ex.map(_worker, jobs)):
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)} {name}")
    print(f"[zero_23dof] done. Output: {args.dest}")


if __name__ == "__main__":
    main()
