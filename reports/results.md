# Skeleton-based fall detection — verified evaluation

Generated 2026-09-24T18:08:09+00:00 from commit `b05f48f3389869cb3df39e1d810d784b6a9f0111` by `make report`. Every number below is read from `reports/results.json`, which is built only from logged runs (`runs/**/metrics.json`) and measured benchmarks (`reports/bench.json`).

**Status: complete**

## 1. Audit summary (details: `reports/00_audit.md`)

* Pose estimator actually used for training data: **YOLOv8n-pose (Ultralytics), 2D COCO-17 keypoints**, 34-d per frame (x, y), normalized by image size, centred on the mid-hip and divided by the neck–mid-hip distance. MediaPipe appeared only in an orphan extractor (33 landmarks, 66-d, incompatible with `input_dim: 34`) and in the old inference script (train/serve skew); both now use YOLOv8n-pose.
* Original labels were per video (whole fall video = 1) and each video contributed one random 32-frame crop per epoch; the crop often did not contain the fall. Replaced with per-window labels from the official per-frame annotation.
* The 78.75% figure came from `tests/eval_combinations.py`: file-level `random_split` (mirrored copies of the same video could sit on both sides), no test set, accuracy only, maximum over 15 epochs × 24 configurations on the same 80-file validation set. At that commit `training.py` validated on the training files themselves.
* Scripts listed in the README (`scripts/*.py`) had been deleted/moved; LE2I segment files expected by `training.py` (`_fall/_adl/_preadl/_postadl`) were never produced by any code.
* `checkpoints/best_model.pt`: plain LSTM state_dict (34→64, 1 layer, 25,730 params) with no record of its data or split; not usable for a leak-free evaluation and not used here.

## 2. Protocol decisions

* Dataset: UR-Fall cam0 only (30 falls + 40 ADLs). cam1 exists only for falls, so using it would let the camera identify the class. LE2I: not used (see Limitations).
* Split unit: **video** (UR-Fall publishes no per-sequence subject IDs, so a subject-level split is impossible). Splits are stratified by video class and depend only on `split_seed=0`; all models and seeds share them.
* Fixed split: 60% train / 20% validation / 20% test videos. The test videos are never used for early stopping, threshold selection, model selection or the rule's parameters.
* 5-fold video-level cross-validation (each fold: 1/5 test, the rest split 75/25 into train/validation). Out-of-fold test predictions are pooled per seed.
* Seeds 0, 1, 2 for every learned model; mean ± sample std (ddof=1) over seeds. The rule baseline is deterministic (std 0).
* Windows: 32 frames; training stride 4, validation/test stride 8. Positive window = contains ≥ min(8, |fall interval|) annotated *falling* frames; negative = ADL window or pre-fall window; windows with partial overlap or post-fall lying only are ignored for training and window metrics but scored for event metrics.
* Mirror augmentation: training partition only.
* Early stopping: validation PR-AUC (patience 10, max 50 epochs); the repo's validation accuracy is not used because the classes are imbalanced.
* **Threshold rule: maximize F1 on the validation windows** (ties → lower threshold). The recall ≥ 0.90 operating point is also stored in `results.json` (`fixed_recall90_rule`).
* Event level: a fall video is detected if any window crosses the threshold; false alarms = runs of consecutive alarming windows on ADL videos, divided by the number of ADL videos. Localized recall additionally requires the alarm window to overlap the annotated fall ±1 s.

## 3. Data

70 videos (30 fall / 40 ADL), 11936 frames; validation/test windows (stride 8): 147 positive, 1046 negative, 56 ignored. Full statistics: `reports/01_data.md`.

## 4. Main results — fixed held-out test videos (mean ± std over seeds)

| model | precision | recall | F1 | PR-AUC | event recall | false alarms / ADL video | params | size MB (fp32) | CPU ms/window (1 thr, median) |
|---|---|---|---|---|---|---|---|---|---|
| Rule (hip velocity + box aspect) | 0.65 ± 0.00 | 0.43 ± 0.00 | 0.52 ± 0.00 | 0.60 ± 0.00 | 1.00 ± 0.00 | 0.25 ± 0.00 | 0 | 0.000 | — |
| LSTM (repo default) | 0.58 ± 0.09 | 0.82 ± 0.07 | 0.68 ± 0.04 | 0.84 ± 0.00 | 0.94 ± 0.10 | 0.62 ± 0.33 | 25,730 | 0.106 | 0.489 |
| GRU | 0.54 ± 0.04 | 0.86 ± 0.02 | 0.66 ± 0.03 | 0.84 ± 0.07 | 1.00 ± 0.00 | 0.75 ± 0.22 | 19,330 | 0.080 | 1.093 |
| LSTM + attention | 0.49 ± 0.01 | 0.89 ± 0.02 | 0.63 ± 0.02 | 0.82 ± 0.05 | 1.00 ± 0.00 | 0.96 ± 0.19 | 29,955 | 0.123 | 0.567 |
| 1D-CNN / TCN | 0.81 ± 0.13 | 0.90 ± 0.00 | 0.85 ± 0.07 | 0.94 ± 0.05 | 1.00 ± 0.00 | 0.21 ± 0.14 | 27,986 | 0.122 | 0.657 |
| ST-GCN (small) | 0.80 ± 0.05 | 0.84 ± 0.12 | 0.82 ± 0.05 | 0.91 ± 0.04 | 1.00 ± 0.00 | 0.29 ± 0.07 | 62,473 | 0.272 | 2.050 |

Secondary metrics (same runs):

| model | ROC-AUC | balanced acc. | accuracy | localized event recall | ADL videos with ≥1 alarm | false alarms / hour |
|---|---|---|---|---|---|---|
| Rule (hip velocity + box aspect) | 0.89 ± 0.00 | 0.70 ± 0.00 | 0.89 ± 0.00 | 1.00 ± 0.00 | 0.12 ± 0.00 | 132.5 ± 0.0 |
| LSTM (repo default) | 0.95 ± 0.01 | 0.86 ± 0.01 | 0.89 ± 0.03 | 0.94 ± 0.10 | 0.50 ± 0.22 | 331.3 ± 175.3 |
| GRU | 0.94 ± 0.04 | 0.87 ± 0.02 | 0.88 ± 0.02 | 1.00 ± 0.00 | 0.58 ± 0.07 | 397.5 ± 114.8 |
| LSTM + attention | 0.94 ± 0.01 | 0.87 ± 0.01 | 0.86 ± 0.01 | 1.00 ± 0.00 | 0.75 ± 0.00 | 508.0 ± 101.2 |
| 1D-CNN / TCN | 0.98 ± 0.01 | 0.93 ± 0.01 | 0.96 ± 0.02 | 1.00 ± 0.00 | 0.21 ± 0.14 | 110.4 ± 76.5 |
| ST-GCN (small) | 0.98 ± 0.02 | 0.91 ± 0.05 | 0.95 ± 0.01 | 1.00 ± 0.00 | 0.25 ± 0.00 | 154.6 ± 38.3 |

## 5. 5-fold video-level cross-validation (pooled out-of-fold, mean ± std over seeds)

| model | precision | recall | F1 | PR-AUC | event recall | false alarms / ADL video | params | size MB (fp32) | CPU ms/window (1 thr, median) |
|---|---|---|---|---|---|---|---|---|---|
| Rule (hip velocity + box aspect) | 0.74 ± 0.00 | 0.65 ± 0.00 | 0.69 ± 0.00 | 0.76 ± 0.00 | 1.00 ± 0.00 | 0.38 ± 0.00 | 0 | 0.000 | — |
| LSTM (repo default) | 0.53 ± 0.04 | 0.68 ± 0.06 | 0.60 ± 0.05 | 0.53 ± 0.03 | 0.88 ± 0.02 | 0.48 ± 0.04 | 25,730 | 0.106 | 0.489 |
| GRU | 0.56 ± 0.07 | 0.73 ± 0.03 | 0.63 ± 0.03 | 0.67 ± 0.10 | 0.92 ± 0.04 | 0.53 ± 0.13 | 19,330 | 0.080 | 1.093 |
| LSTM + attention | 0.51 ± 0.02 | 0.70 ± 0.05 | 0.59 ± 0.03 | 0.59 ± 0.02 | 0.92 ± 0.02 | 0.57 ± 0.02 | 29,955 | 0.123 | 0.567 |
| 1D-CNN / TCN | 0.66 ± 0.04 | 0.73 ± 0.03 | 0.69 ± 0.01 | 0.69 ± 0.08 | 0.94 ± 0.02 | 0.30 ± 0.07 | 27,986 | 0.122 | 0.657 |
| ST-GCN (small) | 0.68 ± 0.02 | 0.68 ± 0.01 | 0.68 ± 0.01 | 0.73 ± 0.04 | 0.94 ± 0.02 | 0.51 ± 0.13 | 62,473 | 0.272 | 2.050 |


| model | ROC-AUC | balanced acc. | accuracy | localized event recall | ADL videos with ≥1 alarm | false alarms / hour |
|---|---|---|---|---|---|---|
| Rule (hip velocity + box aspect) | 0.92 ± 0.00 | 0.81 ± 0.00 | 0.93 ± 0.00 | 1.00 ± 0.00 | 0.33 ± 0.00 | 181.2 ± 0.0 |
| LSTM (repo default) | 0.88 ± 0.00 | 0.80 ± 0.03 | 0.89 ± 0.01 | 0.88 ± 0.02 | 0.32 ± 0.03 | 233.5 ± 18.5 |
| GRU | 0.90 ± 0.02 | 0.82 ± 0.00 | 0.89 ± 0.02 | 0.92 ± 0.04 | 0.39 ± 0.11 | 253.7 ± 63.9 |
| LSTM + attention | 0.88 ± 0.01 | 0.80 ± 0.02 | 0.88 ± 0.01 | 0.92 ± 0.02 | 0.39 ± 0.03 | 277.8 ± 12.1 |
| 1D-CNN / TCN | 0.95 ± 0.01 | 0.84 ± 0.01 | 0.92 ± 0.01 | 0.94 ± 0.02 | 0.23 ± 0.02 | 145.0 ± 32.0 |
| ST-GCN (small) | 0.91 ± 0.01 | 0.82 ± 0.00 | 0.92 ± 0.00 | 0.94 ± 0.02 | 0.38 ± 0.04 | 245.6 ± 60.8 |

### Model cost and training time

Training on 1 CPU thread per run (4 runs in parallel on the 4 vCPUs); fixed split, mean ± std over seeds.

| model | params | size MB (fp32 state_dict) | train time s | epochs run |
|---|---|---|---|---|
| Rule (hip velocity + box aspect) | 0 | 0.000 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| LSTM (repo default) | 25,730 | 0.106 | 20.1 ± 3.3 | 14.3 ± 2.3 |
| GRU | 19,330 | 0.080 | 112.6 ± 75.0 | 27.3 ± 17.9 |
| LSTM + attention | 29,955 | 0.123 | 38.0 ± 19.3 | 18.3 ± 9.2 |
| 1D-CNN / TCN | 27,986 | 0.122 | 78.1 ± 37.5 | 31.3 ± 15.1 |
| ST-GCN (small) | 62,473 | 0.272 | 481.3 ± 154.5 | 25.0 ± 7.9 |

### Reading the two tables

* Under 5-fold CV the highest PR-AUC is **Rule (hip velocity + box aspect)** (0.76 ± 0.00) and the highest F1 is **Rule (hip velocity + box aspect)** (0.69 ± 0.00). The non-learned rule is not beaten by any learned model on PR-AUC. Differences between the top models are within about one standard deviation.
* The fixed test split (6 fall videos) scores clearly higher than CV for most models; it is one easy draw of 14 videos. The CV numbers are the more reliable estimate.

## 6. Confusion matrix of the best model (fixed test windows)

Best model: **ST-GCN (small)** (highest mean validation F1 on the fixed split). Summed over 3 seeds (rows = true, columns = predicted):

| | pred non-fall | pred fall |
|---|---|---|
| true non-fall | 536 | 19 |
| true fall | 14 | 76 |

Per seed: `[[[179, 6], [8, 22]], [[181, 4], [5, 25]], [[176, 9], [1, 29]]]`.

## 7. Efficiency

Machine: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs (hypervisor KVM, container docker), 15.7 GB RAM, GPU: none, Ubuntu 24.04.4 LTS (kernel 6.18.44-fc-v37), Python 3.11.15, torch 2.14.0+cu130, ultralytics 8.4.161, onnxruntime 1.30.0.

Pose estimation (yolov8n-pose.pt, CPU, batch 1, input: UR-Fall fall-01-cam0-rgb.zip (640x480)):

| imgsz | median ms/frame | p95 ms/frame | inference-only median ms |
|---|---|---|---|
| 320 | 21.0 | 25.6 | 19.1 |
| 640 | 50.4 | 66.0 | 46.9 |

GPU pose latency: no GPU on this machine: not measured.

Classifier latency (CPU, batch 1, 1×32×34 window, 20 warm-up + 300 timed runs):

| model | params | median ms (1 thread) | p95 ms (1 thread) | median ms (all threads) | p95 ms (all threads) |
|---|---|---|---|---|---|
| LSTM (repo default) | 25,730 | 0.489 | 0.692 | 0.645 | 1.101 |
| GRU | 19,330 | 1.093 | 1.519 | 1.056 | 1.583 |
| LSTM + attention | 29,955 | 0.567 | 0.898 | 0.667 | 0.847 |
| 1D-CNN / TCN | 27,986 | 0.657 | 0.827 | 0.726 | 1.046 |
| ST-GCN (small) | 62,473 | 2.050 | 3.807 | 1.564 | 1.916 |

ONNX (ST-GCN (small), weights: runs/stgcn/fixed0/seed0/model.pt, 0.259 MB, max |Δ| vs torch 1.19e-07): onnxruntime CPU median 0.706 ms (1 thread), p95 0.901 ms.

End-to-end (CPU only, UR-Fall fall-01-cam0-rgb.zip (PNG decode from zip), 160 frames at 640x480, pose imgsz 320, classifier ST-GCN (small) every 16 frames): **31.8 FPS, 31.4 ms/frame** (decode 7.0, pose 24.3, features+classifier 0.17 ms/frame).

No edge-device numbers are claimed; nothing was run on an edge device.

## 8. Comparison with the old 78.75% number

Not comparable, and the old number should not be quoted:

1. **Leakage.** It came from `tests/eval_combinations.py`, which applied `torch.utils.data.random_split` to all `.npy` files. Mirrored copies (`*_mirror.npy`) and, if present, unit-scaled duplicates of the same recording were independent items, so the same fall could be in train and validation. The split was unseeded.
2. **Selection bias.** 78.75% is the *maximum* validation accuracy over 15 epochs for each of 24 hyper-parameter combinations, all selected on the same 80 validation files; there was no held-out test set. With `num_layers=1` the `dropout` setting is a no-op, so pairs of 'different' configurations are the same model: their best accuracies differ by up to 2.5 points (e.g. 76.25% vs 78.75% for hidden 64, lr 5e-4), which is pure run-to-run noise.
3. **Noisy target.** Each video was one sample labelled by its file name, and the validation window was a random 32-frame crop, so the metric changed between evaluations of the same weights and a 'fall' sample often showed no fall.
4. **Metric.** Accuracy only, on a set whose class ratio was never logged. If the files were UR-Fall cam0+cam1 recordings (60 fall / 40 ADL), predicting 'fall' for everything already scores 60%.
5. **Task definition differs.** This evaluation scores 32-frame windows against the annotated fall interval and whole videos (event recall, false alarms), on unseen videos only.

The new numbers answer a different, harder question; a lower number here is not a regression.

For reference, the repo's own LSTM configuration under this protocol: window accuracy 0.89 ± 0.03 (fixed test) / 0.89 ± 0.01 (CV), but fall-class F1 0.68 ± 0.04 / 0.60 ± 0.05. Accuracy looks high only because ~88% of windows are non-fall.

## 9. Limitations

* Small dataset: 70 cam0 videos, 30 falls; the fixed test split holds only 6 fall videos, so one video moves event recall by 1/that number. Treat the CV table as the more stable estimate.
* Staged (simulated) falls and ADLs performed by volunteers indoors; no real-world or elderly falls.
* Single fixed camera per sequence (cam0), one person per scene; no occlusion or multi-person handling beyond 'highest-confidence box'.
* No subject IDs → subject overlap between train and test is likely; generalization to unseen people is not measured.
* LE2I was not evaluated: the official host was not reachable from this environment, and no mirror of verifiable provenance was available.
* Latency measured on a shared cloud VM; numbers on another CPU will differ.

## 10. CV bullet text (auto-filled from results.json)

* Compared LSTM against baselines (rule-based, GRU, LSTM+attention, 1D-CNN/TCN, ST-GCN) on UR-Fall (cam0), 70 videos / 1193 labelled evaluation windows under a video-level split; best model (ST-GCN) reaches fall-class recall 0.84 and F1 0.82.
* (Same claim under 5-fold video-level CV, the more reliable estimate:) Compared LSTM against baselines (rule-based, GRU, LSTM+attention, 1D-CNN/TCN, ST-GCN) on UR-Fall (cam0), 70 videos under 5-fold video-level cross-validation; best model (ST-GCN) reaches fall-class recall 0.68 and F1 0.68.
* Measured end-to-end latency of 31.4 ms/frame (31.8 FPS, YOLOv8n-pose + ST-GCN) on Intel(R) Xeon(R) Processor @ 2.80GHz (4 vCPU) with a classifier of 62.5 K parameters / 0.27 MB.
* Pose estimator: YOLOv8n-pose (Ultralytics), 2D COCO-17 keypoints (x, y) = 34-d per frame; not MediaPipe 3D.

Automated check of every number in the bullets against results.json: PASSED (no mismatches).

## 11. Reproduce

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt
make models      # official YOLOv8n-pose weights (GitHub release)
make data        # download UR-Fall (official), YOLOv8n-pose extraction, windows, splits, reports/01_data.md
make train MODEL=lstm   # or rule | gru | lstm_attention | tcn | stgcn | all
make eval        # collect runs/ -> runs/runs.csv
make bench       # reports/bench.json
make report      # reports/results.json + reports/results.md
```
