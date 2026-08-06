# Data Preparation: the 269-dim Unified Motion Representation

This directory contains everything needed to build the `data/UNI_MOTION_269/` training data (HumanML3D / MotionFix / Human3.6M) used by UniMotion.

The core of the pipeline is the **269-dim unified representation** (`unified_representation_269.py`): the first 263 dims are bit-exact with the official HumanML3D representation, and 6 extra dims append the root joint rotation in continuous 6D form. See the paper (Appendix D.1) and `docs/269_consistency_analysis.md` for the exact layout and its cross-dataset semantics.

```
data_preparation/
├── unified_representation_269.py   # core library (repr construction / recovery / utils)
├── scripts/                        # dataset conversion + Mean/Std scripts
├── docs/269_consistency_analysis.md
├── precomputed_stats/              # ready-to-use Mean.npy / Std.npy per dataset
└── README.md                       # this file
```

All commands below are run **from the repository root**. The shell wrappers `cd` there automatically.

> **Note.** Building the three datasets from scratch takes some time and requires several third-party downloads (some, like Human3.6M, are license-restricted). The `precomputed_stats/` directory in this folder already contains the exact Mean/Std files used in our experiments, so you do not need to recompute the statistics.

---

## 0. Prerequisites

```bash
pip install numpy scipy torch tqdm joblib smplx
```

Raw data to download:

| dataset | what you need | source |
|---|---|---|
| HumanML3D | AMASS + HumanAct12, processed with the official HumanML3D repo into `joints/`, `HumanML3D/new_joint_vecs/` (263-dim), `Mean.npy`/`Std.npy`, `texts/`, `train/val/test/all.txt` (make sure `joints/000021.npy` exists) | https://github.com/EricGuo5513/HumanML3D |
| MotionFix | `motionfix.pth.tar` (30 fps, joblib dict) | official MotionFix dataset release |
| Human3.6M | WHAM parsed train data `human36m_train_vit.pth` + TCMR preprocessed test data `h36m_test_25fps_db.pt`; video frames must be extracted/cropped by yourself (fps50, 448x448) — Human3.6M requires an individual license | WHAM / TCMR releases + http://vision.imar.ro/human3.6m |
| SMPL-H body models | `smplh/*.pkl` | https://mano.is.tue.mpg.de |

Expected input locations (defaults used by the scripts; every path can be overridden via CLI arguments):

```
data/
├── HumanML3D/                       # official HumanML3D output (incl. joints/000021.npy)
├── MotionFix/motionfix-dataset/motionfix.pth.tar
├── _DATA_for_WHAM/human36m_train_vit.pth
├── _DATA_for_TCMR/h36m_test_25fps_db.pt
└── body_models/smplh/               # SMPLH .pkl files
```

Note: the target skeleton for **all three datasets** is HumanML3D's `joints/000021.npy` (`--target-offset-mode humanml3d_default` for MotionFix), so bone-length semantics are consistent across datasets. The default path is `data/HumanML3D/joints/000021.npy`; override with `unified_representation_269.set_target_skeleton_path()` or the `--target-skeleton` CLI options.

---

## 1. HumanML3D (text-to-motion / motion-to-text / prediction, 20 fps)

```bash
# 1) joints + official 263 -> new_joint_vecs_269 (T-1, 269)
#    The first 263 dims are taken verbatim from the official files (and
#    verified with allclose against our recomputation; 29228/29228 passed).
bash data_preparation/scripts/humanml3d_263_to_269.sh data/HumanML3D

# 2) Mean_269/Std_269 (first 263 dims copied from the official Mean/Std,
#    last 6 dims streamed from the data; --smooth-last6 was used for release)
bash data_preparation/scripts/humanml3d_cal_mean_var.sh data/HumanML3D
```

Assemble:

```
data/UNI_MOTION_269/HumanML3D/
├── new_joint_vecs/        # rename/move from HumanML3D/new_joint_vecs_269/
├── Mean.npy               # rename from Mean_269.npy (or copy precomputed_stats/HumanML3D/Mean.npy)
├── Std.npy                # rename from Std_269.npy  (or copy precomputed_stats/HumanML3D/Std.npy)
├── texts/                 # copied from the official HumanML3D texts/
├── train.txt / val.txt / test.txt / all.txt   # copied from official HumanML3D (all.txt optional — not read by this repo)
```

Instruction templates used at training time live in the repo's `instructions/` directory (e.g. `instructions/template_pretrain.json`) and do not need to be copied into the dataset directory.

## 2. MotionFix (motion editing, 30 fps -> 20 fps)

```bash
# 1+2) Resample SMPL params to 20fps, then convert to (T, 269).
#      Downsampling happens in SMPL space first, then the 269 features are
#      computed, so velocities/contacts are natively 20fps.
bash data_preparation/scripts/motionfix_30to20fps.sh data/MotionFix/motionfix-dataset

# 3) Unwrap the joblib bundle into per-sequence npy files
python data_preparation/scripts/motionfix_unwrap.py \
  --input data/MotionFix/motionfix-dataset/motionfix_20fps_269.pth.tar \
  --out-dir data/MotionFix/motionfix-dataset/new_joint_vecs_269_fps20

# 4) Mean/Std over all source+target frames (12800 sequences / 1416337 frames
#    in our run; requires train.txt/test.txt to be in place first, see below)
bash data_preparation/scripts/motionfix_cal_mean_var.sh data/UNI_MOTION_269/MotionFix
```

Assemble:

```
data/UNI_MOTION_269/MotionFix/
├── new_joint_vecs/
│   ├── source/*.npy       # rename/move from new_joint_vecs_269_fps20/source/
│   └── target/*.npy       # rename/move from new_joint_vecs_269_fps20/target/
├── Mean.npy / Std.npy     # rename from Mean_269/Std_269 (or copy precomputed_stats/MotionFix/)
├── train.txt / test.txt   # one entry per line, format: source/000362   (see gap G2)
└── texts/{id}.txt         # editing instruction per sample id             (see gap G2)
```

## 3. Human3.6M (image-to-motion / image-to-text, single-frame mode)

```bash
# 1) SMPL pose -> per-frame 269 (single-frame mode; betas/trans deliberately
#    ignored for pose-only fair comparison; ROTATION_MODE=full is what the
#    released data uses: SMPL global_orient + body_pose override the rotation dims)
bash data_preparation/scripts/h36m_smpl_to_269.sh train data
bash data_preparation/scripts/h36m_smpl_to_269.sh test  data

# 2) [KNOWN GAP G1 — see below] extract repr_269 from the .pth outputs into
#    h36m_train_269.npy / h36m_test_269.npy + image_paths/*.npy + stride-5 subsampling

# 3) Mean/Std over train+test frames (computed on the full-fps25 arrays)
bash data_preparation/scripts/h36m_cal_mean_var.sh data/UNI_MOTION_269/H36M
```

Assemble:

```
data/UNI_MOTION_269/H36M/
├── new_joint_vecs/
│   ├── h36m_train_269.npy   # (156008, 269) after stride-5 subsampling (see gap G1)
│   └── h36m_test_269.npy    # (54895, 269)  after stride-5 subsampling (see gap G1)
├── image_paths/*.npy        # per-frame image path arrays aligned with the above (see gap G1)
├── Mean.npy / Std.npy       # rename from Mean_269/Std_269 (or copy precomputed_stats/H36M/)
├── train.txt / test.txt     # contain the big-array file names
├── text/S*/...              # per-frame text annotations (see gap G3)
└── joints2d_aff448.pkl      # 2D-joint branch input for the CMA-VAE image pathway
```

The cropped image frames themselves are expected at `data/Human3.6M_MMPose/processed/images_fps50_cropped_448x448` (produced from the Human3.6M videos with an external MMPose-based preprocessing pipeline that is not part of this repository).

---

## Known reproduction gaps (please read)

The conversion core (269 construction, resampling, Mean/Std) is fully scripted above. Three glue steps of the original data build were ad-hoc and are **not** covered by scripts in this release; they are documented below so that you can reimplement them if needed.

- **G1 — H36M `.pth` -> `.npy` extraction + image paths + stride-5.** Extracting `repr_269` from the `h36m_smpl_to_269.sh` outputs into the `h36m_{train,test}_269.npy` arrays, generating the aligned `image_paths/*.npy` (mapping TCMR `img_name` entries to the `Human3.6M_MMPose` fps50 448x448 crop paths via a name map), and the stride-5 temporal subsampling (780040 -> 156008 train, 274471 -> 54895 test) were done in an ad-hoc notebook that lived next to the data and is not included here.

- **G2 — MotionFix `texts/` and `train.txt`/`test.txt`.** The editing-instruction text files and the split lists come from the official MotionFix annotations (each sample's instruction text and the official split), but the small script that dumped them into the `texts/{id}.txt` + `source/{id}` line format was ad-hoc and is not included.

- **G3 — H36M per-frame text annotations (`text/S*/...`).** The per-frame textual descriptions used by the image-to-text task were generated by a separate annotation pipeline that is not part of this repository.

Additionally, the Human3.6M video -> fps50 448x448 crop image preprocessing is external to this repository (and redistribution of Human3.6M imagery is not permitted by its license).

---

## Sanity checks

```bash
# Self-test of the core library (T-pose round trip, mode checks)
python data_preparation/unified_representation_269.py

# Cross-dataset semantic consistency analysis (synthetic + optional real data)
python data_preparation/scripts/verify_269_consistency.py \
  --target-skeleton data/HumanML3D/joints/000021.npy
```

`docs/269_consistency_analysis.md` documents the known, intentional semantic differences between the sequence mode (HumanML3D/MotionFix) and the single-frame mode (H36M) — in particular the meaning of `global_orient_6d`.
