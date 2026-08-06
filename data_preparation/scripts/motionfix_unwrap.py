"""
Unwrap the converted MotionFix joblib bundle into per-sequence .npy files
=========================================================================

Reads the output of convert_motionfix_to_269.py (a joblib bundle whose
data[key]['motion_source'/'motion_target']['repr_269'] holds the (T, 269)
arrays) and writes one .npy per sequence:

  <out-dir>/source/<key>.npy   (T, 269)
  <out-dir>/target/<key>.npy   (T, 269)

These directories are then renamed/moved to
data/UNI_MOTION_269/MotionFix/new_joint_vecs/{source,target}/ (see the
data_preparation README).

Usage (run from the repository root):
  python data_preparation/scripts/motionfix_unwrap.py \
    --input data/MotionFix/motionfix-dataset/motionfix_20fps_269.pth.tar \
    --out-dir data/MotionFix/motionfix-dataset/new_joint_vecs_269_fps20
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unwrap a converted MotionFix 269 joblib bundle into per-sequence .npy files."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input joblib bundle produced by convert_motionfix_to_269.py (e.g. motionfix_20fps_269.pth.tar)"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory; source/ and target/ subdirectories are created inside it"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .npy files (by default an existing file raises an error)"
    )
    args = parser.parse_args()

    src_dir = os.path.join(args.out_dir, "source")
    tgt_dir = os.path.join(args.out_dir, "target")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    print(f"[motionfix_unwrap] Loading: {args.input}")
    bundle = joblib.load(args.input)
    data = bundle["data"] if isinstance(bundle, dict) and "data" in bundle else bundle

    n_written = 0
    for key, value in tqdm(data.items(), desc="Unwrapping", unit="sample"):
        src_269 = value["motion_source"]["repr_269"]    # (T, 269)
        tgt_269 = value["motion_target"]["repr_269"]    # (T, 269)

        src_path = os.path.join(src_dir, f"{key}.npy")
        tgt_path = os.path.join(tgt_dir, f"{key}.npy")
        if not args.overwrite:
            if os.path.exists(src_path):
                raise FileExistsError(f"File exists (use --overwrite): {src_path}")
            if os.path.exists(tgt_path):
                raise FileExistsError(f"File exists (use --overwrite): {tgt_path}")

        np.save(src_path, src_269)
        np.save(tgt_path, tgt_269)
        n_written += 1

    print(f"[motionfix_unwrap] Done. {n_written} sample pairs written to {args.out_dir}")


if __name__ == "__main__":
    main()
