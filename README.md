# skeleton-har-framework

Skeleton-based fall detection: video → 2D pose keypoints → 32-frame windows → binary classifier (fall / non-fall),
with a leak-free evaluation protocol. Audit of the original code: `reports/00_audit.md`.
Results (only numbers from logged runs): `reports/results.md`, machine-readable `reports/results.json`.

## True pipeline

| stage | what actually runs |
|---|---|
| Pose estimator | **YOLOv8n-pose** (Ultralytics, `models/yolov8n-pose.pt`, `imgsz=320`, `conf=0.25`), highest-confidence person per frame. Not MediaPipe. |
| Keypoints | **COCO-17, 2D (x, y)** → 34 values per frame, joint-major `[x0, y0, …, x16, y16]` (0 nose … 11/12 hips … 15/16 ankles). No depth. |
| Normalization | x/W, y/H → centre on mid-hip → divide by neck–mid-hip distance; frames without a detection are interpolated; clipped to ±10 (`src/pose_estimation/features.py`). |
| Windows | 32 frames (~1.07 s at 30 fps); stride 4 for training, 8 for validation/test, 16 for live inference. |
| Labels | Official UR-Fall per-frame annotation (−1 not lying, 0 falling, 1 lying). Positive window = contains ≥ min(8, fall length) *falling* frames; ADL windows negative; partial/post-fall-only windows ignored. |
| Models | rule baseline (hip velocity + box aspect ratio), LSTM (repo default: 64 hidden, 1 layer), LSTM + attention, GRU, 1D-CNN/TCN, small ST-GCN (`src/pose_estimation/model.py`). |

## Datasets

* **UR Fall Detection** (University of Rzeszów, official source http://fenix.ur.edu.pl/~mkepski/ds/uf.html):
  cam0 RGB frame archives for 30 falls + 40 ADLs and the official per-frame label CSVs. cam1 is not used
  (it exists for falls only, so it would leak the label through the viewpoint). `make download` records
  URL + SHA-256 of every file in `dataset/urfall/raw/manifest.json`.
* **LE2I**: processing code exists (`scripts/le2i_process.py`) but it is not part of the reported evaluation;
  see `reports/results.md` → Limitations.

## Protocol (non-negotiable parts)

* Split by **video** (UR-Fall publishes no subject IDs); never by window. All windows of a video share a partition.
* Fixed split 60/20/20 (train/val/test videos, stratified, `split_seed=0`) **plus** 5-fold video-level CV.
* Test videos are never used for early stopping, threshold choice, model selection or rule tuning.
* Early stopping on validation PR-AUC; decision threshold = max F1 on validation windows.
* Mirror augmentation on training windows only. Seeds 0, 1, 2; mean ± std reported.
* Metrics (positive = fall): precision, recall, F1, PR-AUC; ROC-AUC, balanced accuracy, accuracy, confusion
  matrix; event recall (fall video detected if any window crosses the threshold); false alarms per ADL video.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
make setup                                          # uv venv (Python 3.11) + pinned requirements
make models                                         # official yolov8n-pose.pt from the Ultralytics GitHub release
```

`requirements.txt` pins the versions used for the reported runs; `requirements.lock.txt` freezes the full set.

## Reproduce every number

```bash
make data                  # download UR-Fall, YOLOv8n-pose extraction, windows + splits, reports/01_data.md
make train MODEL=lstm      # one model; MODEL=all (default) runs rule, lstm, gru, lstm_attention, tcn, stgcn
make eval                  # consolidate runs/**/metrics.json -> runs/runs.csv
make bench                 # latency / FPS on this machine -> reports/bench.json
make report                # reports/results.json + reports/results.md (+ CV bullet text, auto-checked)
make test                  # protocol unit tests
make smoke                 # end-to-end code check on synthetic stand-in data (never reported)
```

Each run writes `runs/<model>/<fixed0|cv0..cv4>/seed<k>/{metrics.json,preds_test.npz,model.pt}`; `metrics.json`
holds the config, seed, split, git commit, per-epoch history, thresholds, validation and test metrics.

## Other entry points

```bash
uv run scripts/pose_extraction.py --video path/to/video.mp4          # YOLOv8n-pose keypoints (.npz raw + .npy features)
uv run scripts/urfall_process.py download|extract                    # same as make download / make extract
uv run scripts/mirror_augment_pose_npy.py --folder dataset/pose_npy  # legacy offline mirroring (not used by the protocol)
PYTHONPATH=src python src/pose_estimation/test.py --source video.mp4 --model-path runs/lstm/fixed0/seed0/model.pt
```

`checkpoints/best_model.pt` is the original repo checkpoint (LSTM, unknown training split); it is kept for
reference only and is not evaluated (see the audit).

## Layout

```
src/pose_estimation/
  preprocessing/   urfall.py (download + extraction), le2i.py, extraction.py (generic video), common.py
  experiments/     windows.py, splits.py, metrics.py, baseline.py, run.py, bench.py, data_stats.py, report*.py, synthetic.py
  features.py  model.py  dataset.py  training.py (legacy trainer)  test.py (inference CLI)
scripts/           thin CLIs referenced above
reports/           00_audit.md, 01_data.md, results.md, results.json, bench.json, splits.json
runs/              per-run logs
```

## License
MIT (or your choice)
