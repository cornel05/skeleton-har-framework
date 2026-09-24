"""
Evaluation protocol runner: every (model, protocol, fold, seed) job trains on the train
videos, early-stops and picks the decision threshold on the validation videos, and is
scored once on the held-out test videos. One JSON record + test predictions per run
under runs/, consolidated into runs/runs.csv.

    python -m pose_estimation.experiments.run prepare          # windows + splits cache
    python -m pose_estimation.experiments.run train --models lstm gru --workers 4
"""

import argparse
import csv
import json
import os
import pickle
import random
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import cfg as CONFIG, PROJECT_ROOT
from ..model import build_model
from ..preprocessing.common import mirror_coco17_sequence
from .baseline import fit_rule, rule_scores
from .metrics import event_metrics, select_threshold, window_metrics
from .splits import check_disjoint, cv_splits, fixed_split
from .windows import Prepared, build_windows, load_urfall_videos

ALL_MODELS = ("rule", "lstm", "gru", "lstm_attention", "tcn", "stgcn")


def protocol_cfg() -> dict:
    p = dict(CONFIG.get("protocol", {}))
    p.setdefault("window_length", CONFIG.get("dataset.sequence_length", 32))
    p.setdefault("train_stride", 4)
    p.setdefault("eval_stride", 8)
    p.setdefault("min_fall_frames", 8)
    p.setdefault("split_seed", 0)
    p.setdefault("seeds", [0, 1, 2])
    p.setdefault("cv_folds", 5)
    p.setdefault("test_frac", 0.2)
    p.setdefault("val_frac", 0.2)
    p.setdefault("mirror_train", True)
    p.setdefault("early_stop_metric", "val_pr_auc")
    p.setdefault("threshold_rule", "max_f1")
    return p


def git_state() -> Dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "config.yaml"],
            cwd=PROJECT_ROOT, text=True).strip()
        return {"git_commit": commit, "git_dirty": bool(dirty)}
    except Exception:  # noqa: BLE001
        return {"git_commit": "unknown", "git_dirty": None}


def prepare(pose_dir: Path, raw_dir: Path, cache: Path, labels_override: Optional[dict] = None) -> Prepared:
    from ..preprocessing.urfall import load_frame_labels

    p = protocol_cfg()
    frame_labels = labels_override if labels_override is not None else load_frame_labels(raw_dir)
    videos = load_urfall_videos(pose_dir, frame_labels)
    if not videos:
        raise FileNotFoundError(f"No pose .npz files in {pose_dir}; run `make data` first")
    L = p["window_length"]
    train_ws = build_windows(videos, L, p["train_stride"], p["min_fall_frames"])
    eval_ws = build_windows(videos, L, p["eval_stride"], p["min_fall_frames"])
    meta = {}
    for v in videos:
        span = None
        if v.label == 1 and v.fall_interval.any():
            idx = np.flatnonzero(v.fall_interval)
            span = (int(idx[0]), int(idx[-1]) + 1)
        meta[v.vid] = {"label": v.label, "group": v.group, "frames": int(v.feats.shape[0]), "fps": v.fps,
                       "fall_span": span, "missing_rate": v.missing_rate,
                       "n_unknown_frames": int((v.frame_labels == -9).sum()),
                       "n_falling_frames": int((v.frame_labels == 0).sum()),
                       "n_lying_frames": int((v.frame_labels == 1).sum())}
    ids = [v.vid for v in videos]
    groups = [v.group for v in videos]
    labels = [v.label for v in videos]
    splits = {"fixed": [fixed_split(ids, groups, labels, p["split_seed"], p["test_frac"], p["val_frac"])],
              "cv": cv_splits(ids, groups, labels, p["cv_folds"], p["split_seed"])}
    for s in splits["fixed"] + splits["cv"]:
        check_disjoint(s)
    prep = Prepared(train_ws, eval_ws, meta, splits, p)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(prep, f)
    return prep


def load_prepared(cache: Path) -> Prepared:
    with cache.open("rb") as f:
        return pickle.load(f)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _mirror_windows(X: np.ndarray) -> np.ndarray:
    return np.stack([mirror_coco17_sequence(w) for w in X]).astype(np.float32)


@torch.no_grad()
def _scores(model: nn.Module, X: np.ndarray, M: np.ndarray, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        logits = model(torch.from_numpy(X[i:i + bs]), torch.from_numpy(M[i:i + bs]))
        out.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def _evaluate(prep: Prepared, split: dict, s_val: np.ndarray, y_val: np.ndarray,
              s_test_all: np.ndarray, idx_test_all: np.ndarray, thr_rules: Dict[str, float]) -> dict:
    ev = prep.eval_ws
    y_all = ev.y[idx_test_all]
    lab = y_all != -1
    names = ev.videos
    vlabels = {n: prep.meta[n]["label"] for n in names}
    vframes = {n: prep.meta[n]["frames"] for n in names}
    spans = {n: tuple(prep.meta[n]["fall_span"]) for n in names if prep.meta[n]["fall_span"]}
    fps = float(np.median([prep.meta[n]["fps"] for n in names]))
    out = {}
    for rule, thr in thr_rules.items():
        out[rule] = {
            "val": window_metrics(y_val, s_val, thr),
            "test_window": window_metrics(y_all[lab], s_test_all[lab], thr),
            "test_event": event_metrics(s_test_all, ev.vid[idx_test_all], ev.start[idx_test_all], thr,
                                        names, vlabels, vframes, fps, spans, ev.length,
                                        tolerance=int(fps)),
        }
    return out


def run_job(job: dict) -> dict:
    torch.set_num_threads(job.get("threads", 1))
    prep = load_prepared(Path(job["cache"]))
    p = prep.protocol
    name, protocol, fold, seed = job["model"], job["protocol"], job["fold"], job["seed"]
    split = prep.splits[protocol][fold]
    out_dir = Path(job["runs_dir"]) / name / f"{protocol}{fold}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    tr, ev = prep.train_ws, prep.eval_ws
    itr = tr.select(split["train"], labeled_only=True)
    iva = ev.select(split["val"], labeled_only=True)
    ite = ev.select(split["test"], labeled_only=False)
    y_val = ev.y[iva]
    record = {"model": name, "protocol": protocol, "fold": fold, "seed": seed,
              "split_seed": p["split_seed"], **git_state(), "protocol_cfg": p,
              "n_train_videos": len(split["train"]), "n_val_videos": len(split["val"]),
              "n_test_videos": len(split["test"]), "n_train_windows": int(len(itr)),
              "n_train_pos": int(tr.y[itr].sum()), "n_val_windows": int(len(iva)),
              "n_test_windows_all": int(len(ite)), "threads": job.get("threads", 1)}

    t0 = time.perf_counter()
    if name == "rule":
        taus, val_f1 = fit_rule(ev.rule[iva], y_val)
        s_val = rule_scores(ev.rule[iva], **taus)
        s_test = rule_scores(ev.rule[ite], **taus)
        thr = {"max_f1": 1.0}
        record.update(rule_params=taus, params=0, size_mb=0.0, epochs_run=0, best_epoch=0, history=[])
    else:
        tcfg = dict(CONFIG.get("training", {}))
        epochs = job.get("epochs") or tcfg.get("num_epochs", 50)
        lr = tcfg.get("learning_rate", 5e-4)
        patience = tcfg.get("patience", 10)
        bs = CONFIG.get("dataset.batch_size", 8)
        X, M, y = tr.X[itr], tr.mask[itr], tr.y[itr]
        if p["mirror_train"]:  # augmentation on the training partition only
            X, M, y = np.concatenate([X, _mirror_windows(X)]), np.concatenate([M, M]), np.concatenate([y, y])
        record["n_train_windows_after_aug"] = int(len(X))
        model = build_model(name, input_dim=X.shape[2], cfg=CONFIG.get("model", {}))
        counts = np.bincount(y, minlength=2).astype(np.float64)
        weights = torch.tensor(len(y) / (2 * np.maximum(counts, 1)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        rng = np.random.default_rng(seed)
        Xt, Mt, yt = torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)
        best_score, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.from_numpy(rng.permutation(len(y)))
            tot = 0.0
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                if len(b) < 2 and name in ("tcn", "stgcn"):
                    continue  # BatchNorm needs >1 sample
                loss = loss_fn(model(Xt[b], Mt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                tot += loss.item() * len(b)
            sched.step()
            s_val_ep = _scores(model, ev.X[iva], ev.mask[iva])
            vm = window_metrics(y_val, s_val_ep, 0.5)
            history.append({"epoch": epoch, "train_loss": tot / len(y), "val_pr_auc": vm["pr_auc"],
                            "val_f1_at_0.5": vm["f1"]})
            if vm["pr_auc"] > best_score:
                best_score, best_epoch, bad = vm["pr_auc"], epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is None:  # validation PR-AUC undefined (no positive windows): keep the last epoch
            record["warning"] = "validation PR-AUC undefined; last-epoch weights used"
        else:
            model.load_state_dict(best_state)
        ckpt = out_dir / "model.pt"
        torch.save(model.state_dict(), ckpt)
        s_val = _scores(model, ev.X[iva], ev.mask[iva])
        s_test = _scores(model, ev.X[ite], ev.mask[ite])
        thr = {"max_f1": select_threshold(y_val, s_val, "max_f1"),
               "recall_0.90": select_threshold(y_val, s_val, "recall_0.90")}
        record.update(params=int(sum(q.numel() for q in model.parameters())),
                      size_mb=ckpt.stat().st_size / 1e6, epochs_run=len(history), best_epoch=best_epoch,
                      history=history, model_cfg=dict(CONFIG.get("model", {})),
                      training_cfg={"epochs": epochs, "lr": lr, "patience": patience, "batch_size": bs,
                                    "optimizer": "adam", "scheduler": "steplr(10,0.5)",
                                    "loss": "class-weighted CE", "grad_clip": 1.0})
    record["train_time_s"] = time.perf_counter() - t0
    record["thresholds"] = thr
    record["results"] = _evaluate(prep, split, s_val, y_val, s_test, ite, thr)
    np.savez_compressed(out_dir / "preds_test.npz", scores=s_test, y=ev.y[ite],
                        video=np.array([ev.videos[i] for i in ev.vid[ite]]), start=ev.start[ite])
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=float))
    return {"run": str(out_dir), "f1": record["results"]["max_f1"]["test_window"]["f1"]}


CSV_FIELDS = ["model", "protocol", "fold", "seed", "git_commit", "git_dirty", "params", "size_mb",
              "train_time_s", "epochs_run", "best_epoch", "threshold", "val_f1", "precision", "recall", "f1",
              "pr_auc", "roc_auc", "balanced_accuracy", "accuracy", "tn", "fp", "fn", "tp", "event_recall",
              "event_recall_localized", "false_alarms_per_adl_video", "false_alarms_per_hour"]


def collect(runs_dir: Path) -> Path:
    rows = []
    for mpath in sorted(runs_dir.glob("*/*/seed*/metrics.json")):
        r = json.loads(mpath.read_text())
        res = r["results"]["max_f1"]
        w, e = res["test_window"], res["test_event"]
        rows.append({"model": r["model"], "protocol": r["protocol"], "fold": r["fold"], "seed": r["seed"],
                     "git_commit": r["git_commit"], "git_dirty": r["git_dirty"], "params": r["params"],
                     "size_mb": r["size_mb"], "train_time_s": r["train_time_s"], "epochs_run": r["epochs_run"],
                     "best_epoch": r["best_epoch"], "threshold": r["thresholds"]["max_f1"],
                     "val_f1": res["val"]["f1"], **{k: w[k] for k in CSV_FIELDS if k in w},
                     **{k: e.get(k) for k in ("event_recall", "event_recall_localized",
                                              "false_alarms_per_adl_video", "false_alarms_per_hour")}})
    out = runs_dir / "runs.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"Collected {len(rows)} runs -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fall-detection evaluation protocol")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--pose-dir", type=Path, default=PROJECT_ROOT / "dataset/urfall/pose")
    pr.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "dataset/urfall/raw")
    pr.add_argument("--cache", type=Path, default=PROJECT_ROOT / "dataset/urfall/cache/prepared.pkl")
    pr.add_argument("--splits-out", type=Path, default=PROJECT_ROOT / "reports/splits.json")
    tr = sub.add_parser("train")
    tr.add_argument("--cache", type=Path, default=PROJECT_ROOT / "dataset/urfall/cache/prepared.pkl")
    tr.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    tr.add_argument("--models", nargs="+", default=list(ALL_MODELS))
    tr.add_argument("--protocols", nargs="+", default=["fixed", "cv"])
    tr.add_argument("--seeds", nargs="+", type=int, default=None)
    tr.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    tr.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    tr.add_argument("--force", action="store_true")
    co = sub.add_parser("collect")
    co.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    args = ap.parse_args()

    if args.cmd == "prepare":
        prep = prepare(args.pose_dir, args.raw_dir, args.cache)
        args.splits_out.parent.mkdir(parents=True, exist_ok=True)
        args.splits_out.write_text(json.dumps(prep.splits, indent=1))
        print(f"{len(prep.meta)} videos | train windows {len(prep.train_ws.y)} (stride {prep.train_ws.stride}) | "
              f"eval windows {len(prep.eval_ws.y)} (stride {prep.eval_ws.stride}) -> {args.cache}")
    elif args.cmd == "train":
        prep = load_prepared(args.cache)
        seeds = args.seeds or prep.protocol["seeds"]
        jobs = []
        for m in args.models:
            for proto in args.protocols:
                for fold in range(len(prep.splits[proto])):
                    for s in seeds:
                        done = args.runs_dir / m / f"{proto}{fold}" / f"seed{s}" / "metrics.json"
                        if done.exists() and not args.force:
                            continue
                        jobs.append({"model": m, "protocol": proto, "fold": fold, "seed": s, "cache": str(args.cache),
                                     "runs_dir": str(args.runs_dir), "epochs": args.epochs, "threads": 1})
        print(f"{len(jobs)} job(s) on {args.workers} worker(s)")
        with get_context("spawn").Pool(args.workers) as pool:
            for res in pool.imap_unordered(run_job, jobs):
                print(f"done {res['run']} test F1={res['f1']:.3f}", flush=True)
        collect(args.runs_dir)
    else:
        collect(args.runs_dir)


if __name__ == "__main__":
    main()
