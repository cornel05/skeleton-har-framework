# Skeleton-based fall detection — verified evaluation

Generated 2026-09-24T16:18:15+00:00 from commit `463f0765a37d1452e2fec039c84e2b7e3d9df448` by `make report`. Every number below is read from `reports/results.json`, which is built only from logged runs (`runs/**/metrics.json`) and measured benchmarks (`reports/bench.json`).

**Status: BLOCKED: no dataset -> no metrics were measured**

### What was not measured, and why

The UR-Fall official host (`fenix.ur.edu.pl`) and the LE2I host (`imvia.u-bourgogne.fr`) are denied by the network egress policy of the environment these runs were executed in (HTTP 403 "Host not in allowlist"). No copy of verifiable provenance was reachable, so **no classification metric, no data statistic, no end-to-end FPS on a dataset video and no pose latency on dataset frames exists yet**. Nothing below is estimated to fill those gaps. What *was* measured: classifier latency/size (architecture-only, independent of the data) and YOLOv8n-pose latency on proxy images (labelled PROXY).

To produce every missing number once the host is reachable: `make data train eval bench report`.

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

**not measured** — the dataset could not be downloaded in this environment (see Limitations).

## 4. Main results — fixed held-out test videos (mean ± std over seeds)

**not measured** — no runs exist (dataset unavailable).

## 5. 5-fold video-level cross-validation (pooled out-of-fold, mean ± std over seeds)

**not measured**.

## 6. Confusion matrix of the best model (fixed test windows)

**not measured**.

## 7. Efficiency

Machine: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs (hypervisor KVM, container docker), 15.7 GB RAM, GPU: none, Ubuntu 24.04.4 LTS (kernel 6.18.44-fc-v37), Python 3.11.15, torch 2.14.0+cu130, ultralytics 8.4.161, onnxruntime 1.30.0.

Pose estimation (yolov8n-pose.pt, CPU, batch 1, input: PROXY: ultralytics sample images resized to 640x480 (dataset frames unavailable)):

| imgsz | median ms/frame | p95 ms/frame | inference-only median ms |
|---|---|---|---|
| 320 | 24.5 | 34.5 | 22.4 |
| 640 | 57.6 | 76.8 | 54.4 |

GPU pose latency: no GPU on this machine: not measured.

Classifier latency (CPU, batch 1, 1×32×34 window, 20 warm-up + 300 timed runs):

| model | params | median ms (1 thread) | p95 ms (1 thread) | median ms (all threads) | p95 ms (all threads) |
|---|---|---|---|---|---|
| LSTM (repo default) | 25,730 | 0.487 | 0.603 | 0.585 | 0.936 |
| GRU | 19,330 | 1.022 | 1.606 | 1.126 | 1.834 |
| LSTM + attention | 29,955 | 0.601 | 0.954 | 0.765 | 1.204 |
| 1D-CNN / TCN | 27,986 | 0.681 | 0.855 | 0.738 | 0.957 |
| ST-GCN (small) | 62,473 | 2.094 | 2.609 | 1.624 | 2.293 |

ONNX (LSTM (repo default), weights: random init, 0.106 MB, max |Δ| vs torch 0.00e+00): onnxruntime CPU median 0.077 ms (1 thread), p95 0.141 ms.

End-to-end FPS on a dataset video: **not measured** — dataset video unavailable: not measured.

No edge-device numbers are claimed; nothing was run on an edge device.

## 8. Comparison with the old 78.75% number

Not comparable, and the old number should not be quoted:

1. **Leakage.** It came from `tests/eval_combinations.py`, which applied `torch.utils.data.random_split` to all `.npy` files. Mirrored copies (`*_mirror.npy`) and, if present, unit-scaled duplicates of the same recording were independent items, so the same fall could be in train and validation. The split was unseeded.
2. **Selection bias.** 78.75% is the *maximum* validation accuracy over 15 epochs for each of 24 hyper-parameter combinations, all selected on the same 80 validation files; there was no held-out test set. With `num_layers=1` the `dropout` setting is a no-op, so pairs of 'different' configurations are the same model: their best accuracies differ by up to 2.5 points (e.g. 76.25% vs 78.75% for hidden 64, lr 5e-4), which is pure run-to-run noise.
3. **Noisy target.** Each video was one sample labelled by its file name, and the validation window was a random 32-frame crop, so the metric changed between evaluations of the same weights and a 'fall' sample often showed no fall.
4. **Metric.** Accuracy only, on a set whose class ratio was never logged. If the files were UR-Fall cam0+cam1 recordings (60 fall / 40 ADL), predicting 'fall' for everything already scores 60%.
5. **Task definition differs.** This evaluation scores 32-frame windows against the annotated fall interval and whole videos (event recall, false alarms), on unseen videos only.

The new numbers answer a different, harder question; a lower number here is not a regression.

## 9. Limitations

* Small dataset: 70 cam0 videos, 30 falls; the fixed test split holds only 6 fall videos, so one video moves event recall by 1/that number. Treat the CV table as the more stable estimate.
* Staged (simulated) falls and ADLs performed by volunteers indoors; no real-world or elderly falls.
* Single fixed camera per sequence (cam0), one person per scene; no occlusion or multi-person handling beyond 'highest-confidence box'.
* No subject IDs → subject overlap between train and test is likely; generalization to unseen people is not measured.
* LE2I was not evaluated: the official host was not reachable from this environment, and no mirror of verifiable provenance was available.
* Latency measured on a shared cloud VM; numbers on another CPU will differ.

## 10. CV bullet text (auto-filled from results.json)

**Not filled** — no verified numbers exist yet. Pose estimator: YOLOv8n-pose 2D COCO-17 keypoints (not MediaPipe 3D).

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
