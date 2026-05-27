#!/usr/bin/env python3
"""Zero the 6 joint columns absent on 23-DoF G1 hardware in reference motion CSVs.

Reads each motion subfolder under --input (default: example/), copies it to
--output (default: example_23dof/), and rewrites joint_pos.csv and joint_vel.csv
with the 6 IL-order columns set to 0. All other files (body_*.csv, info.txt,
metadata.txt) are copied verbatim.

The 6 missing joints in MJCF order are [13, 14, 20, 21, 27, 28]
(WaistRoll, WaistPitch, L/R WristPitch/WristYaw). The CSVs are stored in IL
order, which maps these to columns [5, 8, 25, 26, 27, 28] via MJ_TO_IL in
gmr_to_deploy.py.

Usage:
    python gear_sonic_deploy/reference/zero_23dof_columns.py
    python gear_sonic_deploy/reference/zero_23dof_columns.py \
        --input gear_sonic_deploy/reference/example \
        --output gear_sonic_deploy/reference/example_23dof
"""

import argparse
import csv
import shutil
from pathlib import Path

# IL-order columns corresponding to the 6 missing 23-DoF joints.
# Derivation: MJ_TO_IL[mj] for mj in [13, 14, 20, 21, 27, 28]
#   MJ 13 (waist_roll)        -> IL 5
#   MJ 14 (waist_pitch)       -> IL 8
#   MJ 20 (left_wrist_pitch)  -> IL 25
#   MJ 21 (left_wrist_yaw)    -> IL 27
#   MJ 27 (right_wrist_pitch) -> IL 26
#   MJ 28 (right_wrist_yaw)   -> IL 28
MISSING_23DOF_IL_COLUMNS = [5, 8, 25, 26, 27, 28]

JOINT_CSVS = ["joint_pos.csv", "joint_vel.csv"]


def zero_columns_in_csv(src: Path, dst: Path, columns_to_zero):
    with open(src, "r", newline="") as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)
    if not rows:
        shutil.copy2(src, dst)
        return
    header, body = rows[0], rows[1:]
    for row in body:
        for c in columns_to_zero:
            if c < len(row):
                row[c] = "0.000000"
    with open(dst, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(body)


def process_motion_dir(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        dst = dst_dir / child.name
        if child.is_dir():
            continue  # motion folders shouldn't have nested dirs; skip if any
        if child.name in JOINT_CSVS:
            zero_columns_in_csv(child, dst, MISSING_23DOF_IL_COLUMNS)
        else:
            shutil.copy2(child, dst)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--input", default=str(here / "example"),
                    help="Input directory containing motion subfolders")
    ap.add_argument("--output", default=str(here / "example_23dof"),
                    help="Output directory (will be created)")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    if not in_root.is_dir():
        raise SystemExit(f"input not a directory: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    motion_dirs = [p for p in sorted(in_root.iterdir()) if p.is_dir()]
    print(f"[zero_23dof] {len(motion_dirs)} motion folders under {in_root}")
    for md in motion_dirs:
        out_md = out_root / md.name
        process_motion_dir(md, out_md)
        print(f"  + {md.name} -> {out_md}")

    # Copy any top-level files (e.g. motion_summary.txt) verbatim.
    for child in in_root.iterdir():
        if child.is_file():
            shutil.copy2(child, out_root / child.name)
    print(f"[zero_23dof] done. Output: {out_root}")


if __name__ == "__main__":
    main()
