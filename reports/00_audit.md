# 00 — Audit of `skeleton-har-framework` (main @ `463f076`)

Scope: every file tracked at `463f076`, plus files deleted earlier in history (`git log --stat`), since the
"78.75%" number and the checkpoint were produced by code that no longer exists at HEAD.
Every statement below was checked against the code or by running it; items that could not be checked
are marked **unverified**.

## File tree at HEAD

```
.gitignore  README.md  config.yaml  requirements.txt
checkpoints/best_model.pt
src/pose_estimation/{__init__,config,dataset,model,training,test,utils}.py
src/pose_estimation/preprocessing/{__init__,common,extraction,le2i,urfall}.py
tests/{eval_combinations,live_pose,test_loading,test_pipeline}.py
```

No `scripts/`, no `models/`, no dataset, no logs. `tests/test_pipeline.py` passes (synthetic tensors only).

## 1. Which pose estimator produces the `.npy` features?

**YOLOv8-pose (Ultralytics), default weights `models/yolov8n-pose.pt`, `imgsz=320`, `conf=0.25`.**
Both dataset processors (`preprocessing/urfall.py`, `preprocessing/le2i.py`) call `ultralytics.YOLO(...).predict`
and write 34 values per frame. MediaPipe appears in two places, neither of which can produce training data:

| file | estimator | output | status |
|---|---|---|---|
| `preprocessing/urfall.py`, `preprocessing/le2i.py` | YOLOv8-pose | 17 × (x, y) = 34 | **the training-data path** |
| `preprocessing/extraction.py` | MediaPipe Pose | 33 × (x, y) = 66 | orphan; incompatible with `input_dim: 34` |
| `test.py` (inference CLI) | MediaPipe Pose → 17-joint subset | 34 | **train/serve skew**: model trained on YOLO keypoints, served MediaPipe keypoints (different joint definitions, esp. hips) |

`config.yaml` `input_dim: 34` and the README (`models/yolov8n-pose.pt`) agree with YOLOv8-pose.
Any CV text saying "MediaPipe 3D" is wrong: features are **2D** (x, y); no depth/z is used anywhere.

**Keypoint layout**: COCO-17 order (0 nose, 1–2 eyes, 3–4 ears, 5–6 shoulders, 7–8 elbows, 9–10 wrists,
11–12 hips, 13–14 knees, 15–16 ankles, left before right), flattened joint-major: `[x0, y0, x1, y1, …, x16, y16]`.

**Normalization** (per frame): `x/W`, `y/H` clipped to [0, 1] → subtract mid-hip (mean of joints 11, 12) →
divide by the neck–mid-hip distance (neck = mean of shoulders 5, 6; `eps = 1e-6`). Keypoint confidences are
discarded. Person selection: highest box confidence (UR-Fall); LE2I prefers the box with the largest IoU with the
annotated box.

Defects found:
* Frames with no detected person are written as **all-zero vectors** and are not masked — a zero skeleton is a
  valid-looking input. A fall is exactly when detection tends to drop, so zeros can correlate with the label.
* Nothing bounds the output when the torso is foreshortened (scale → eps).
* `utils.compute_inference_time` calls `model(x)` without the required `masks` argument (TypeError);
  `compute_model_flops` needs `fvcore` (not a dependency); `utils.py` imports `matplotlib` at module level.
* `SkeletonLSTM` docstring says dropout is "applied to LSTM outputs"; it is not — with `num_layers: 1` the
  `dropout` setting has no effect at all.

## 2. How are labels assigned and windows built?

**Per video (per file).** `training._infer_label_from_filename`: name contains `_fall` → 1; `_adl`, `_preadl`,
`_postadl` → 0; otherwise contains `fall` or starts with `f-` → 1; else 0. No frame-level annotation is used.

**Windows.** `SkeletonDataset.__getitem__` returns exactly **one** 32-frame window per file: a random crop for
training, the centre crop for val/test (shorter videos are zero-padded with a mask). A UR-Fall fall recording is
mostly standing/walking before the fall and lying after it, so a 32-frame (~1.07 s) crop labelled "fall" frequently
contains no falling motion → label noise; the model can instead learn "person lying down".

**LE2I labels are broken.** `le2i.py` writes one file per video named after its path (e.g.
`Coffee_room_01_Videos_video (1).npy`). None of the label rules match such names, so every LE2I video — most of
which contain a fall — would be labelled 0 (or all 1 if a parent folder name contains "fall"). The segmented
`_fall/_adl/_preadl/_postadl` files that `training.py` expects are not produced by any code in the repository.
`fall_start/fall_end` are saved to `metadata.json` but never read.

## 3. How is the split done? Leakage

| code path | split unit | leakage |
|---|---|---|
| `tests/eval_combinations.py` (**source of 78.75%**) | `torch.utils.data.random_split` over all `.npy` files, 320/80, **unseeded** | mirrored copies (`*_mirror.npy`) of a recording are independent items → the same recording (mirrored) can be in train and validation. No test set. |
| `training.py` @ `0f9e5a3` (commit that added the checkpoint) | none | **train and validation datasets were built from the same directory** → "validation accuracy" was training accuracy on a different random crop |
| `training.py` @ HEAD | group = file stem without `_mirror`, stratified 80/15/5 | video-level for UR-Fall, but cam0 and cam1 recordings of the same fall are different groups → the same fall event (other viewpoint) in train and test; LE2I segments of one video are deliberately separate groups (code comment) → adjacent segments leak; test = 5% of groups (~5 videos) |

Not a *random window-level* split (each file yields one window), but the effect is the same: near-duplicates of test
items were in training. None of the paths had a fixed held-out test set that was untouched by model selection.

## 4. Scripts referenced in the README that are missing

| README reference | fate | replacement in this branch |
|---|---|---|
| `scripts/pose_extraction.py` | deleted in `96b14af`, became `preprocessing/extraction.py` (MediaPipe) | `scripts/pose_extraction.py` → YOLOv8n-pose |
| `scripts/urfall_process.py` | moved to `preprocessing/urfall.py`; expected `.mp4` files in `fall-XX/`, `adl-XX/` folders, which is **not** the official layout (official = `*-cam0-rgb.zip` PNG frames) | rewritten: official download + zip reader + raw `.npz` + per-frame labels |
| `scripts/le2i_process.py` | moved to `preprocessing/le2i.py` | kept as-is (LE2I not reachable, see results.md) |
| `scripts/mirror_augment_pose_npy.py` | deleted in `d43b9a7` (function kept as `common.mirror_coco17_sequence`) | `scripts/mirror_augment_pose_npy.py` restored as a thin CLI |
| `models/` | never committed (`*.pt` gitignored) | `make models` downloads the official Ultralytics release asset |

Also rewritten/added: dataset download with SHA-256 manifest, per-frame label parsing, windowing with per-window
labels, video-level fixed split + 5-fold CV, seeded multi-run training with per-run logs, window- and event-level
metrics, rule baseline, GRU / TCN / ST-GCN, benchmarks, report generation, Makefile.

## 5. Does `checkpoints/` contain a usable model?

`checkpoints/best_model.pt` (105,468 B, sha256 `8c78bf73…40ae`) is a bare `state_dict` for `SkeletonLSTM`:
`input_dim 34`, `hidden 64`, `1 layer`, unidirectional, 2 classes, **25,730 parameters**. It loads with strict key
matching. There is no metadata: no training data list, split, seed, normalization or config. Weight std is 0.08 vs
0.072 for the PyTorch initialization, so it was trained only lightly.

It is **not usable for evaluation**: its training videos are unknown, so any test video might have been seen.
It was not used. The deleted `checkpoints/model_config.txt` (from `0f9e5a3`) logged a test result of accuracy
0.6667, precision 0.8108, recall 0.7895, F1 0.8000. Solving for integer confusion matrices that reproduce all
three rounded metrics gives only `(P, N, TP, FP, FN, TN) = k × (38, 7, 30, 7, 8, 0)`: that test set was ~84% falls
and **every non-fall sample was classified as a fall (TN = 0)**. The F1 of 0.80 is essentially the fall prevalence.

## 6. The "78.75% validation accuracy" claim

Reproduced from the deleted `logs/eval_results.txt`: it is the best entry (`hidden 64, lr 5e-4, 1 layer,
dropout 0.5`) of `tests/eval_combinations.py`, defined as the **maximum over 15 epochs** of accuracy on 80
validation files, **maximized again over 24 configurations** selected on the same files; the same run's final-epoch
accuracy was 72.50%. Because dropout is a no-op with one layer, configurations 13 and 14 are the same model; they
scored 76.25% and 78.75% — the spread is noise. Validation windows were random crops, so the score also changed
between evaluations of identical weights. With 400 files (320/80) — consistent with 100 UR-Fall recordings
× {original, mirror} × {raw, `_unit`} from the old scripts (**unverified**; the data are not in the repo) — the
number measures memorization of mirrored duplicates plus selection on the validation set. It is not comparable
to anything measured on unseen videos.

## 7. Dependencies (`requirements.txt`)

Old file: no `torch`, no `scikit-learn` (imported by `utils.py`), no `matplotlib` (imported by `utils.py`);
`mediapipe==0.10.14` listed twice; `opencv-contrib-python==4.8.1.78` collides with the `opencv-python` that
Ultralytics installs (both own the `cv2` package). New file pins the exact versions installed with `uv` for these
runs (Python 3.11.15): torch 2.14.0, numpy 1.26.4, opencv-python 4.11.0.86, ultralytics 8.4.161,
scikit-learn 1.9.1, pyyaml 6.0.3, tqdm 4.70.1, matplotlib 3.11.2, onnx 1.23.0, onnxruntime 1.30.0, pytest 9.1.1.
MediaPipe is removed (no longer imported). The full transitive set is frozen in `requirements.lock.txt`.

## 8. Fixes applied in this branch (summary)

* Missing-detection frames interpolated, features clipped to ±10; confidence kept in the raw `.npz`.
* Inference CLI (`test.py`) and `extraction.py` switched to YOLOv8n-pose → no train/serve skew.
* Labels from the official UR-Fall per-frame annotation, window-level; video-level splits; test set untouched by
  early stopping / threshold / model selection; early stopping on validation PR-AUC instead of accuracy.
* `utils.compute_inference_time` fixed (passes masks).
