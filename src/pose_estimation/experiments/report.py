"""
Aggregate runs/ + reports/bench.json + reports/data_stats.json into
reports/results.json (every number) and reports/results.md (tables + narrative).

Aggregation:
  fixed split : one value per seed on the fixed held-out test videos -> mean +- std (ddof=1) over seeds
  5-fold CV   : per seed, out-of-fold test predictions of the 5 folds are pooled (each fold uses its own
                validation-selected threshold) -> one value per seed -> mean +- std over seeds
Best model: highest mean validation F1 on the fixed split (test numbers are never used to pick it).
"""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from ..config import PROJECT_ROOT

LEARNED = ["lstm", "gru", "lstm_attention", "tcn", "stgcn"]
ORDER = ["rule"] + LEARNED
SHORT = {"rule": "rule-based", "lstm": "LSTM", "gru": "GRU", "lstm_attention": "LSTM+attention", "tcn": "1D-CNN/TCN",
         "stgcn": "ST-GCN"}
PRETTY = {"rule": "Rule (hip velocity + box aspect)", "lstm": "LSTM (repo default)", "gru": "GRU",
          "lstm_attention": "LSTM + attention", "tcn": "1D-CNN / TCN", "stgcn": "ST-GCN (small)"}
WINDOW_KEYS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "balanced_accuracy", "accuracy"]
EVENT_KEYS = ["event_recall", "event_recall_localized", "false_alarms_per_adl_video",
              "adl_videos_with_alarm_frac", "false_alarms_per_hour"]


def _load_runs(runs_dir: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/*/seed*/metrics.json"))]


def _ms(values: List[float]) -> Dict[str, object]:
    a = np.asarray([v for v in values if v is not None and not np.isnan(v)], float)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0, "values": list(values)}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size),
            "values": [float(v) for v in values]}


def best_model_name(runs_dir: Path, learned_only: bool = True) -> Tuple[str, str]:
    runs = [r for r in _load_runs(runs_dir) if r["protocol"] == "fixed"]
    cands = {}
    for r in runs:
        if learned_only and r["model"] == "rule":
            continue
        cands.setdefault(r["model"], []).append(r["results"]["max_f1"]["val"]["f1"])
    if not cands:
        return "lstm", "no runs found: defaulted to the repo's LSTM"
    name = max(cands, key=lambda k: np.mean(cands[k]))
    return name, "highest mean validation F1 on the fixed split" + (" (learned models)" if learned_only else "")


def _pooled_cv(runs: List[dict], runs_dir: Path) -> Dict[str, float]:
    """Pool out-of-fold test predictions of one (model, seed) across CV folds."""
    ys, ss, preds = [], [], []
    ev = {"det": 0, "nf": 0, "loc": 0, "fa": 0, "na": 0, "adl_alarm": 0, "hours": 0.0}
    for r in runs:
        d = np.load(runs_dir / r["model"] / f"cv{r['fold']}" / f"seed{r['seed']}" / "preds_test.npz")
        lab = d["y"] != -1
        thr = r["thresholds"]["max_f1"]
        ys.append(d["y"][lab])
        ss.append(d["scores"][lab])
        preds.append(d["scores"][lab] >= thr)
        e = r["results"]["max_f1"]["test_event"]
        ev["det"] += e["fall_videos_detected"]
        ev["nf"] += e["n_fall_videos"]
        ev["loc"] += round(e.get("event_recall_localized", 0) * e["n_fall_videos"])
        ev["fa"] += e["false_alarms"]
        ev["na"] += e["n_adl_videos"]
        ev["adl_alarm"] += round(e["adl_videos_with_alarm_frac"] * e["n_adl_videos"])
        ev["hours"] += e["adl_hours"]
    y, s, p = np.concatenate(ys), np.concatenate(ss), np.concatenate(preds)
    tp, fp = int(np.sum(p & (y == 1))), int(np.sum(p & (y == 0)))
    fn, tn = int(np.sum(~p & (y == 1))), int(np.sum(~p & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)), "accuracy": (tp + tn) / len(y),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "event_recall": ev["det"] / ev["nf"], "event_recall_localized": ev["loc"] / ev["nf"],
            "false_alarms_per_adl_video": ev["fa"] / ev["na"], "adl_videos_with_alarm_frac": ev["adl_alarm"] / ev["na"],
            "false_alarms_per_hour": ev["fa"] / ev["hours"] if ev["hours"] else float("nan"),
            "n_fall_videos": ev["nf"], "n_adl_videos": ev["na"]}


def aggregate(runs_dir: Path) -> Dict[str, dict]:
    runs = _load_runs(runs_dir)
    out: Dict[str, dict] = {}
    for model in ORDER:
        mr = [r for r in runs if r["model"] == model]
        if not mr:
            continue
        entry: Dict[str, object] = {"n_runs": len(mr)}
        fixed = sorted([r for r in mr if r["protocol"] == "fixed"], key=lambda r: r["seed"])
        if fixed:
            res = [r["results"]["max_f1"] for r in fixed]
            entry["fixed"] = {
                "seeds": [r["seed"] for r in fixed],
                **{k: _ms([x["test_window"][k] for x in res]) for k in WINDOW_KEYS},
                **{k: _ms([x["test_event"].get(k) for x in res]) for k in EVENT_KEYS},
                "val_f1": _ms([x["val"]["f1"] for x in res]),
                "threshold": _ms([r["thresholds"]["max_f1"] for r in fixed]),
                "confusion_matrix_per_seed": [[[x["test_window"]["tn"], x["test_window"]["fp"]],
                                               [x["test_window"]["fn"], x["test_window"]["tp"]]] for x in res],
                "n_test_videos": fixed[0]["n_test_videos"],
                "n_test_pos_windows": res[0]["test_window"]["n_pos"],
                "n_test_neg_windows": res[0]["test_window"]["n_neg"],
            }
            if "recall_0.90" in fixed[0]["results"]:
                r90 = [r["results"]["recall_0.90"] for r in fixed]
                entry["fixed_recall90_rule"] = {k: _ms([x["test_window"][k] for x in r90])
                                                for k in ("precision", "recall", "f1")}
            cms = np.array(entry["fixed"]["confusion_matrix_per_seed"])
            entry["fixed"]["confusion_matrix_sum_over_seeds"] = cms.sum(axis=0).tolist()
            entry["params"] = fixed[0]["params"]
            entry["size_mb"] = fixed[0]["size_mb"]
            entry["train_time_s_fixed"] = _ms([r["train_time_s"] for r in fixed])
            entry["epochs_run_fixed"] = _ms([r["epochs_run"] for r in fixed])
        cv = [r for r in mr if r["protocol"] == "cv"]
        if cv:
            by_seed: Dict[int, List[dict]] = {}
            for r in cv:
                by_seed.setdefault(r["seed"], []).append(r)
            complete = {s: rs for s, rs in by_seed.items() if len(rs) == max(len(v) for v in by_seed.values())}
            pooled = [_pooled_cv(rs, runs_dir) for _, rs in sorted(complete.items())]
            entry["cv"] = {"seeds": sorted(complete), "folds": len(next(iter(complete.values()))),
                           **{k: _ms([p[k] for p in pooled]) for k in WINDOW_KEYS + EVENT_KEYS},
                           "per_fold_f1": _ms([r["results"]["max_f1"]["test_window"]["f1"] for r in cv]),
                           "n_fall_videos": pooled[0]["n_fall_videos"], "n_adl_videos": pooled[0]["n_adl_videos"],
                           "train_time_s": _ms([r["train_time_s"] for r in cv])}
        out[model] = entry
    return out


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _f(ms: Optional[dict], nd: int = 2) -> str:
    if not ms or ms.get("mean") is None:
        return "n/a"
    return f"{ms['mean']:.{nd}f} ± {ms['std']:.{nd}f}"


def cv_bullets(res: dict) -> Optional[dict]:
    """Fill the CV templates only from numbers present in results.json."""
    models, stats, bench = res.get("models", {}), res.get("data"), res.get("bench")
    best = res.get("best_model", {}).get("name")
    if not models or not stats or best not in models or "fixed" not in models[best]:
        return None
    fx = models[best]["fixed"]
    names = [SHORT[m] for m in ORDER if m in models and m != "lstm"]
    b1 = (f"Compared LSTM against baselines ({', '.join(names)}) on UR-Fall (cam0), "
          f"{stats['videos']} videos / {stats['windows_eval_stride']['pos'] + stats['windows_eval_stride']['neg']} "
          f"labelled evaluation windows under a video-level split; best model ({SHORT[best]}) reaches fall-class "
          f"recall {fx['recall']['mean']:.2f} and F1 {fx['f1']['mean']:.2f}.")
    out = {"bullet_1": b1, "values_1": {"recall": round(fx["recall"]["mean"], 2), "f1": round(fx["f1"]["mean"], 2),
                                        "videos": stats["videos"]}}
    if "cv" in models[best]:
        cv = models[best]["cv"]
        out["bullet_1_cv"] = (f"Compared LSTM against baselines ({', '.join(names)}) on UR-Fall (cam0), {stats['videos']} videos "
                              f"under 5-fold video-level cross-validation; best model ({SHORT[best]}) reaches fall-class "
                              f"recall {cv['recall']['mean']:.2f} and F1 {cv['f1']['mean']:.2f}.")
        out["values_1_cv"] = {"recall": round(cv["recall"]["mean"], 2), "f1": round(cv["f1"]["mean"], 2)}
    e2e = (bench or {}).get("e2e", {})
    lb = (bench or {}).get("best_model")  # e2e always runs the best *learned* classifier
    if e2e.get("fps") and "unavailable" not in e2e.get("source", "") and "PROXY" not in e2e.get("source", "") \
            and lb in models:
        params = models[lb].get("params")
        size = models[lb].get("size_mb")
        out["bullet_2"] = (f"Measured end-to-end latency of {e2e['ms_per_frame']:.1f} ms/frame "
                           f"({e2e['fps']:.1f} FPS, YOLOv8n-pose + {SHORT[lb]}) on {bench['hardware']['cpu_model']} "
                           f"({bench['hardware']['logical_cpus']} vCPU) with a classifier of "
                           f"{params / 1e3:.1f} K parameters / {size:.2f} MB.")
        out["values_2"] = {"ms_per_frame": round(e2e["ms_per_frame"], 1), "fps": round(e2e["fps"], 1),
                           "params_k": round(params / 1e3, 1), "size_mb": round(size, 2)}
    out["pose_estimator"] = "YOLOv8n-pose (Ultralytics), 2D COCO-17 keypoints (x, y) = 34-d per frame; not MediaPipe 3D"
    return out


def verify_bullets(res: dict) -> List[str]:
    """Re-derive every number in the bullets from results.json; returns a list of mismatches."""
    b = res.get("cv_bullets")
    if not b:
        return []
    errs = []
    fx = res["models"][res["best_model"]["name"]]["fixed"]
    for key, src in (("recall", fx["recall"]["mean"]), ("f1", fx["f1"]["mean"])):
        if b["values_1"][key] != round(src, 2) or f"{round(src, 2):.2f}" not in b["bullet_1"]:
            errs.append(f"bullet_1 {key}")
    if "bullet_1_cv" in b:
        cv = res["models"][res["best_model"]["name"]]["cv"]
        for key in ("recall", "f1"):
            src = round(cv[key]["mean"], 2)
            if b["values_1_cv"][key] != src or f"{key if key == 'recall' else 'F1'} {src:.2f}" not in b["bullet_1_cv"]:
                errs.append(f"bullet_1_cv {key}")
    if "bullet_2" in b:
        e2e = res["bench"]["e2e"]
        for key, src in (("ms_per_frame", e2e["ms_per_frame"]), ("fps", e2e["fps"])):
            if b["values_2"][key] != round(src, 1) or f"{round(src, 1):.1f}" not in b["bullet_2"]:
                errs.append(f"bullet_2 {key}")
    return errs


def build(runs_dir: Path, reports: Path) -> dict:
    data = json.loads((reports / "data_stats.json").read_text()) if (reports / "data_stats.json").exists() else None
    bench = json.loads((reports / "bench.json").read_text()) if (reports / "bench.json").exists() else None
    models = aggregate(runs_dir)
    if bench:
        for m, b in bench.get("classifier", {}).items():
            if m in models:
                models[m]["cpu_ms_per_window"] = {"threads_1": b["threads_1"], "threads_all": b["threads_all"]}
    best, why = best_model_name(runs_dir, learned_only=False) if models else (None, "no runs")
    res = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git_commit": _git(),
           "data": data, "models": models, "best_model": {"name": best, "rule": why}, "bench": bench,
           "status": "complete" if models and data else "BLOCKED: no dataset -> no metrics were measured"}
    res["cv_bullets"] = cv_bullets(res)
    res["cv_bullets_check"] = {"mismatches": verify_bullets(res), "checked": res["cv_bullets"] is not None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    ap.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    a = ap.parse_args()
    from .report_md import render
    res = build(a.runs_dir, a.reports)
    (a.reports / "results.json").write_text(json.dumps(res, indent=2))
    (a.reports / "results.md").write_text(render(res))
    print(f"wrote {a.reports / 'results.json'} and results.md | status: {res['status']} | "
          f"bullet check: {res['cv_bullets_check']}")


if __name__ == "__main__":
    main()
