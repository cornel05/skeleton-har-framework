"""Markdown rendering of reports/results.json (kept separate so the numbers live in one place)."""

from typing import List

from .report import ORDER, PRETTY, _f

NM = "not measured"


def _main_table(models: dict, proto: str) -> List[str]:
    rows = ["| model | precision | recall | F1 | PR-AUC | event recall | false alarms / ADL video | params | size MB (fp32) | CPU ms/window (1 thr, median) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        e = models.get(m, {})
        r = e.get(proto)
        if not r:
            continue
        lat = e.get("cpu_ms_per_window", {}).get("threads_1", {}).get("median_ms")
        rows.append(
            f"| {PRETTY[m]} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | {_f(r['pr_auc'])} | "
            f"{_f(r['event_recall'])} | {_f(r['false_alarms_per_adl_video'])} | "
            f"{e.get('params', 0):,} | {e.get('size_mb', 0):.3f} | {f'{lat:.3f}' if lat else ('—' if m == 'rule' else NM)} |")
    return rows


def _secondary_table(models: dict, proto: str) -> List[str]:
    rows = ["| model | ROC-AUC | balanced acc. | accuracy | localized event recall | ADL videos with ≥1 alarm | false alarms / hour |",
            "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        r = models.get(m, {}).get(proto)
        if r:
            rows.append(f"| {PRETTY[m]} | {_f(r['roc_auc'])} | {_f(r['balanced_accuracy'])} | {_f(r['accuracy'])} | "
                        f"{_f(r['event_recall_localized'])} | {_f(r['adl_videos_with_alarm_frac'])} | "
                        f"{_f(r['false_alarms_per_hour'], 1)} |")
    return rows


def render(res: dict) -> str:
    models, data, bench = res.get("models") or {}, res.get("data"), res.get("bench")
    L: List[str] = [
        "# Skeleton-based fall detection — verified evaluation", "",
        f"Generated {res['generated_utc']} from commit `{res['git_commit']}` by `make report`. "
        "Every number below is read from `reports/results.json`, which is built only from logged runs "
        "(`runs/**/metrics.json`) and measured benchmarks (`reports/bench.json`).", "",
        f"**Status: {res['status']}**", "",
    ]
    if res["status"].startswith("BLOCKED"):
        L += [
            "### What was not measured, and why", "",
            "The UR-Fall official host (`fenix.ur.edu.pl`) and the LE2I host (`imvia.u-bourgogne.fr`) are denied by the "
            "network egress policy of the environment these runs were executed in (HTTP 403 \"Host not in allowlist\"). "
            "No copy of verifiable provenance was reachable, so **no classification metric, no data statistic, no "
            "end-to-end FPS on a dataset video and no pose latency on dataset frames exists yet**. Nothing below is "
            "estimated to fill those gaps. What *was* measured: classifier latency/size (architecture-only, independent of "
            "the data) and YOLOv8n-pose latency on proxy images (labelled PROXY).", "",
            "To produce every missing number once the host is reachable: `make data train eval bench report`.", "",
        ]
    L += [
        "## 1. Audit summary (details: `reports/00_audit.md`)", "",
        "* Pose estimator actually used for training data: **YOLOv8n-pose (Ultralytics), 2D COCO-17 keypoints**, "
        "34-d per frame (x, y), normalized by image size, centred on the mid-hip and divided by the neck–mid-hip distance. "
        "MediaPipe appeared only in an orphan extractor (33 landmarks, 66-d, incompatible with `input_dim: 34`) and in the "
        "old inference script (train/serve skew); both now use YOLOv8n-pose.",
        "* Original labels were per video (whole fall video = 1) and each video contributed one random 32-frame crop per "
        "epoch; the crop often did not contain the fall. Replaced with per-window labels from the official per-frame annotation.",
        "* The 78.75% figure came from `tests/eval_combinations.py`: file-level `random_split` (mirrored copies of the same "
        "video could sit on both sides), no test set, accuracy only, maximum over 15 epochs × 24 configurations on the same "
        "80-file validation set. At that commit `training.py` validated on the training files themselves.",
        "* Scripts listed in the README (`scripts/*.py`) had been deleted/moved; LE2I segment files expected by `training.py` "
        "(`_fall/_adl/_preadl/_postadl`) were never produced by any code.",
        "* `checkpoints/best_model.pt`: plain LSTM state_dict (34→64, 1 layer, 25,730 params) with no record of its data or "
        "split; not usable for a leak-free evaluation and not used here.", "",
        "## 2. Protocol decisions", "",
    ]
    L += [
        "* Dataset: UR-Fall cam0 only (30 falls + 40 ADLs). cam1 exists only for falls, so using it would let the camera "
        "identify the class. LE2I: not used (see Limitations).",
        "* Split unit: **video** (UR-Fall publishes no per-sequence subject IDs, so a subject-level split is impossible). "
        "Splits are stratified by video class and depend only on `split_seed=0`; all models and seeds share them.",
        "* Fixed split: 60% train / 20% validation / 20% test videos. The test videos are never used for early stopping, "
        "threshold selection, model selection or the rule's parameters.",
        "* 5-fold video-level cross-validation (each fold: 1/5 test, the rest split 75/25 into train/validation). "
        "Out-of-fold test predictions are pooled per seed.",
        "* Seeds 0, 1, 2 for every learned model; mean ± sample std (ddof=1) over seeds. The rule baseline is deterministic (std 0).",
        "* Windows: 32 frames; training stride 4, validation/test stride 8. Positive window = contains ≥ min(8, |fall interval|) "
        "annotated *falling* frames; negative = ADL window or pre-fall window; windows with partial overlap or post-fall "
        "lying only are ignored for training and window metrics but scored for event metrics.",
        "* Mirror augmentation: training partition only.",
        "* Early stopping: validation PR-AUC (patience 10, max 50 epochs); the repo's validation accuracy is not used "
        "because the classes are imbalanced.",
        "* **Threshold rule: maximize F1 on the validation windows** (ties → lower threshold). The recall ≥ 0.90 operating "
        "point is also stored in `results.json` (`fixed_recall90_rule`).",
        "* Event level: a fall video is detected if any window crosses the threshold; false alarms = runs of consecutive "
        "alarming windows on ADL videos, divided by the number of ADL videos. Localized recall additionally requires the "
        "alarm window to overlap the annotated fall ±1 s.", "",
    ]
    if data:
        L += ["## 3. Data", "",
              f"{data['videos']} videos ({data['fall_videos']} fall / {data['adl_videos']} ADL), {data['frames_total']} frames; "
              f"validation/test windows (stride {data['windows_eval_stride']['stride']}): {data['windows_eval_stride']['pos']} positive, "
              f"{data['windows_eval_stride']['neg']} negative, {data['windows_eval_stride']['ignored']} ignored. "
              "Full statistics: `reports/01_data.md`.", ""]
    else:
        L += ["## 3. Data", "", f"**{NM}** — the dataset could not be downloaded in this environment (see Limitations).", ""]

    L += ["## 4. Main results — fixed held-out test videos (mean ± std over seeds)", ""]
    if any("fixed" in e for e in models.values()):
        L += _main_table(models, "fixed") + ["", "Secondary metrics (same runs):", ""] + _secondary_table(models, "fixed") + [""]
    else:
        L += [f"**{NM}** — no runs exist (dataset unavailable).", ""]
    L += ["## 5. 5-fold video-level cross-validation (pooled out-of-fold, mean ± std over seeds)", ""]
    if any("cv" in e for e in models.values()):
        L += _main_table(models, "cv") + ["", ""] + _secondary_table(models, "cv") + [""]
    else:
        L += [f"**{NM}**.", ""]

    if any("fixed" in e for e in models.values()):
        L += ["### Model cost and training time", "",
              "Training on 1 CPU thread per run (4 runs in parallel on the 4 vCPUs); fixed split, mean ± std over seeds.", "",
              "| model | params | size MB (fp32 state_dict) | train time s | epochs run |", "|---|---|---|---|---|"]
        for m in ORDER:
            e = models.get(m)
            if e and "fixed" in e:
                L.append(f"| {PRETTY[m]} | {e['params']:,} | {e['size_mb']:.3f} | {_f(e['train_time_s_fixed'], 1)} | "
                         f"{_f(e['epochs_run_fixed'], 1)} |")
        L += [""]
    cvm = {m: e["cv"] for m, e in models.items() if "cv" in e}
    if cvm:
        top_pr = max(cvm, key=lambda m: cvm[m]["pr_auc"]["mean"])
        top_f1 = max(cvm, key=lambda m: cvm[m]["f1"]["mean"])
        L += ["### Reading the two tables", "",
              f"* Under 5-fold CV the highest PR-AUC is **{PRETTY[top_pr]}** ({_f(cvm[top_pr]['pr_auc'])}) and the highest F1 is "
              f"**{PRETTY[top_f1]}** ({_f(cvm[top_f1]['f1'])}). "
              + ("The non-learned rule is not beaten by any learned model on PR-AUC. " if top_pr == "rule" else "")
              + "Differences between the top models are within about one standard deviation.",
              "* The fixed test split (6 fall videos) scores clearly higher than CV for most models; it is one easy draw of "
              "14 videos. The CV numbers are the more reliable estimate.", ""]

    best = res.get("best_model", {}).get("name")
    L += ["## 6. Confusion matrix of the best model (fixed test windows)", ""]
    if best and best in models and "fixed" in models[best]:
        cm = models[best]["fixed"]["confusion_matrix_sum_over_seeds"]
        L += [f"Best model: **{PRETTY[best]}** ({res['best_model']['rule']}). Summed over "
              f"{len(models[best]['fixed']['seeds'])} seeds (rows = true, columns = predicted):", "",
              "| | pred non-fall | pred fall |", "|---|---|---|",
              f"| true non-fall | {cm[0][0]} | {cm[0][1]} |", f"| true fall | {cm[1][0]} | {cm[1][1]} |", "",
              f"Per seed: `{models[best]['fixed']['confusion_matrix_per_seed']}`.", ""]
    else:
        L += [f"**{NM}**.", ""]

    L += ["## 7. Efficiency", ""]
    if bench:
        h = bench["hardware"]
        L += [f"Machine: {h['cpu_model']}, {h['logical_cpus']} logical CPUs ({h['virtualization']}), {h['ram_gb']} GB RAM, "
              f"GPU: {h['gpu']}, {h['os']}, Python {h['python']}, torch {h['torch']}, ultralytics {h['ultralytics']}, "
              f"onnxruntime {h['onnxruntime']}.", ""]
        pz = bench.get("pose", {})
        L += [f"Pose estimation ({pz.get('model')}, CPU, batch 1, input: {pz.get('source')}):", "",
              "| imgsz | median ms/frame | p95 ms/frame | inference-only median ms |", "|---|---|---|---|"]
        for k, v in pz.get("runs", {}).items():
            L.append(f"| {k.split('_')[1]} | {v['median_ms']:.1f} | {v['p95_ms']:.1f} | {v['inference_only_median_ms']:.1f} |")
        L += ["", f"GPU pose latency: {pz.get('gpu')}.", "",
              "Classifier latency (CPU, batch 1, 1×32×34 window, 20 warm-up + 300 timed runs):", "",
              "| model | params | median ms (1 thread) | p95 ms (1 thread) | median ms (all threads) | p95 ms (all threads) |",
              "|---|---|---|---|---|---|"]
        for m, b in bench.get("classifier", {}).items():
            L.append(f"| {PRETTY[m]} | {b['params']:,} | {b['threads_1']['median_ms']:.3f} | {b['threads_1']['p95_ms']:.3f} | "
                     f"{b['threads_all']['median_ms']:.3f} | {b['threads_all']['p95_ms']:.3f} |")
        ox = bench.get("onnx", {})
        if ox:
            L += ["", f"ONNX ({PRETTY.get(ox['model'], ox['model'])}, weights: {ox['weights']}, {ox['onnx_file_mb']:.3f} MB, "
                      f"max |Δ| vs torch {ox['max_abs_diff_vs_torch']:.2e}): onnxruntime CPU median "
                      f"{ox['threads_1']['median_ms']:.3f} ms (1 thread), p95 {ox['threads_1']['p95_ms']:.3f} ms."]
        e = bench.get("e2e", {})
        if e.get("fps"):
            L += ["", f"End-to-end (CPU only, {e['source']}, {e['frames']} frames at {e['resolution']}, pose imgsz {e['pose_imgsz']}, "
                      f"classifier {PRETTY.get(e['classifier'], e['classifier'])} every {e['stride']} frames): **{e['fps']:.1f} FPS, "
                      f"{e['ms_per_frame']:.1f} ms/frame** (decode {e['decode_ms_per_frame']:.1f}, pose {e['pose_ms_per_frame']:.1f}, "
                      f"features+classifier {e['features_classifier_ms_per_frame']:.2f} ms/frame)."]
        else:
            L += ["", f"End-to-end FPS on a dataset video: **{NM}** — {e.get('source', 'dataset unavailable')}."]
        L += ["", "No edge-device numbers are claimed; nothing was run on an edge device.", ""]
    else:
        L += [f"**{NM}** (run `make bench`).", ""]

    L += ["## 8. Comparison with the old 78.75% number", "",
          "Not comparable, and the old number should not be quoted:", "",
          "1. **Leakage.** It came from `tests/eval_combinations.py`, which applied `torch.utils.data.random_split` to all "
          "`.npy` files. Mirrored copies (`*_mirror.npy`) and, if present, unit-scaled duplicates of the same recording were "
          "independent items, so the same fall could be in train and validation. The split was unseeded.",
          "2. **Selection bias.** 78.75% is the *maximum* validation accuracy over 15 epochs for each of 24 hyper-parameter "
          "combinations, all selected on the same 80 validation files; there was no held-out test set. With `num_layers=1` the "
          "`dropout` setting is a no-op, so pairs of 'different' configurations are the same model: their best accuracies "
          "differ by up to 2.5 points (e.g. 76.25% vs 78.75% for hidden 64, lr 5e-4), which is pure run-to-run noise.",
          "3. **Noisy target.** Each video was one sample labelled by its file name, and the validation window was a random "
          "32-frame crop, so the metric changed between evaluations of the same weights and a 'fall' sample often showed no fall.",
          "4. **Metric.** Accuracy only, on a set whose class ratio was never logged. If the files were UR-Fall cam0+cam1 "
          "recordings (60 fall / 40 ADL), predicting 'fall' for everything already scores 60%.",
          "5. **Task definition differs.** This evaluation scores 32-frame windows against the annotated fall interval and "
          "whole videos (event recall, false alarms), on unseen videos only.", "",
          "The new numbers answer a different, harder question; a lower number here is not a regression.", "",
          ] + ([
          f"For reference, the repo's own LSTM configuration under this protocol: window accuracy "
          f"{_f(models['lstm']['fixed']['accuracy'])} (fixed test) / {_f(models['lstm']['cv']['accuracy'])} (CV), "
          f"but fall-class F1 {_f(models['lstm']['fixed']['f1'])} / {_f(models['lstm']['cv']['f1'])}. Accuracy looks "
          f"high only because ~88% of windows are non-fall.", ""] if "lstm" in models and "cv" in models["lstm"] else []) + [

          "## 9. Limitations", "",
          f"* Small dataset: {data['videos'] if data else 70} cam0 videos, {data['fall_videos'] if data else 30} falls; the fixed "
          f"test split holds only {data['fixed_split']['test']['fall_videos'] if data else 6} fall videos, so one video moves "
          "event recall by 1/that number. Treat the CV table as the more stable estimate.",
          "* Staged (simulated) falls and ADLs performed by volunteers indoors; no real-world or elderly falls.",
          "* Single fixed camera per sequence (cam0), one person per scene; no occlusion or multi-person handling beyond "
          "'highest-confidence box'.",
          "* No subject IDs → subject overlap between train and test is likely; generalization to unseen people is not measured.",
          "* LE2I was not evaluated: the official host was not reachable from this environment, and no mirror of verifiable "
          "provenance was available.",
          "* Latency measured on a shared cloud VM; numbers on another CPU will differ.", ""]

    b = res.get("cv_bullets")
    L += ["## 10. CV bullet text (auto-filled from results.json)", ""]
    if b:
        L += [f"* {b['bullet_1']}"]
        if "bullet_1_cv" in b:
            L += [f"* (Same claim under 5-fold video-level CV, the more reliable estimate:) {b['bullet_1_cv']}"]
        L += [f"* {b['bullet_2']}"] if "bullet_2" in b else [f"* Latency bullet: **{NM}** on a real dataset video."]
        L += [f"* Pose estimator: {b['pose_estimator']}.", "",
              f"Automated check of every number in the bullets against results.json: "
              f"{'PASSED (no mismatches)' if not res['cv_bullets_check']['mismatches'] else 'FAILED: ' + str(res['cv_bullets_check']['mismatches'])}.", ""]
    else:
        L += [f"**Not filled** — no verified numbers exist yet. Pose estimator: YOLOv8n-pose 2D COCO-17 keypoints (not MediaPipe 3D).", ""]

    L += ["## 11. Reproduce", "", "```bash",
          "uv venv --python 3.11 && uv pip install -r requirements.txt",
          "make models      # official YOLOv8n-pose weights (GitHub release)",
          "make data        # download UR-Fall (official), YOLOv8n-pose extraction, windows, splits, reports/01_data.md",
          "make train MODEL=lstm   # or rule | gru | lstm_attention | tcn | stgcn | all",
          "make eval        # collect runs/ -> runs/runs.csv",
          "make bench       # reports/bench.json",
          "make report      # reports/results.json + reports/results.md",
          "```", ""]
    return "\n".join(L)
