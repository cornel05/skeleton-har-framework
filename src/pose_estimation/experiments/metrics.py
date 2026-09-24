"""Window-level and event-level metrics. Positive class = fall."""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def select_threshold(y: np.ndarray, scores: np.ndarray, rule: str = "max_f1",
                     target_recall: float = 0.90) -> float:
    """
    'max_f1'      : threshold maximizing F1 on (validation) windows; ties -> lowest threshold (higher recall).
    'recall_0.90' : highest threshold whose recall is still >= target_recall.
    Decisions are `score >= threshold`.
    """
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    prec, rec, thr = precision_recall_curve(y, scores)
    prec, rec = prec[:-1], rec[:-1]  # align with thresholds
    if rule == "max_f1":
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        return float(thr[int(np.argmax(f1))])
    if rule == "recall_0.90":
        ok = np.flatnonzero(rec >= target_recall)
        return float(thr[ok.max()]) if ok.size else float(thr.min())
    raise ValueError(rule)


def window_metrics(y: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(y)) == 2
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        "accuracy": float((tp + tn) / max(1, len(y))),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "threshold": float(thr),
    }


def count_alarms(flags: np.ndarray) -> int:
    """Number of maximal runs of consecutive above-threshold windows (one run = one alarm)."""
    flags = np.asarray(flags, bool)
    if flags.size == 0:
        return 0
    return int(flags[0]) + int(np.sum(flags[1:] & ~flags[:-1]))


def event_metrics(scores: np.ndarray, vid_idx: np.ndarray, starts: np.ndarray, thr: float,
                  video_names: Sequence[str], video_labels: Dict[str, int],
                  video_frames: Dict[str, int], fps: float,
                  fall_spans: Dict[str, tuple] = None, window_len: int = 32,
                  tolerance: int = 0) -> Dict[str, float]:
    """
    Event recall: a fall video is detected if any of its windows has score >= thr.
    Localized recall (stricter, supplementary): the alarm window must overlap the annotated
    fall interval extended by `tolerance` frames.
    False alarms: runs of consecutive alarming windows on ADL videos.
    """
    detected = located = n_fall = 0
    alarms = adl_with_alarm = n_adl = 0
    adl_frames = 0
    for vi in np.unique(vid_idx):
        name = video_names[vi]
        sel = np.flatnonzero(vid_idx == vi)
        order = sel[np.argsort(starts[sel])]
        flags = scores[order] >= thr
        if video_labels[name] == 1:
            n_fall += 1
            detected += int(flags.any())
            if fall_spans is not None and name in fall_spans and flags.any():
                a, b = fall_spans[name]
                s = starts[order][flags]
                located += int(np.any((s < b + tolerance) & (s + window_len > a - tolerance)))
        else:
            n_adl += 1
            k = count_alarms(flags)
            alarms += k
            adl_with_alarm += int(k > 0)
            adl_frames += video_frames[name]
    hours = adl_frames / fps / 3600.0
    out = {
        "event_recall": detected / n_fall if n_fall else float("nan"),
        "n_fall_videos": n_fall, "fall_videos_detected": detected,
        "false_alarms_per_adl_video": alarms / n_adl if n_adl else float("nan"),
        "adl_videos_with_alarm_frac": adl_with_alarm / n_adl if n_adl else float("nan"),
        "false_alarms_per_hour": alarms / hours if hours > 0 else float("nan"),
        "n_adl_videos": n_adl, "false_alarms": alarms, "adl_hours": hours,
    }
    if fall_spans is not None:
        out["event_recall_localized"] = located / n_fall if n_fall else float("nan")
    return out
