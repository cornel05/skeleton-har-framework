"""
Non-learned heuristic: a fall = fast downward motion of the hip centre AND the person
box becoming wider relative to its height.

Per window: v   = peak downward hip speed (body heights per second, box height at window start)
            dAR = mean box aspect (w/h) over the last 8 frames - over the first 8 frames
Score s = min(v / tau_v, dAR / tau_ar); alarm iff s >= 1. (tau_v, tau_ar) is picked on the
validation windows by grid search maximizing F1 (grid = 1st..99th percentiles of each
validation feature, so it adapts to the feature scale); nothing is fit on test data.
"""

from typing import Dict, Tuple

import numpy as np

GRID_QUANTILES = np.linspace(0.01, 0.99, 50)
MIN_TAU = 1e-3


def rule_scores(feats: np.ndarray, tau_v: float, tau_ar: float) -> np.ndarray:
    return np.minimum(feats[:, 0] / tau_v, feats[:, 1] / tau_ar).astype(np.float32)


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum(pred & (y == 1)))
    denom = int(pred.sum()) + int((y == 1).sum())
    return 2 * tp / denom if denom else 0.0


def fit_rule(feats_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, float], float]:
    tv_grid = np.unique(np.maximum(np.quantile(feats_val[:, 0], GRID_QUANTILES), MIN_TAU))
    ta_grid = np.unique(np.maximum(np.quantile(feats_val[:, 1], GRID_QUANTILES), MIN_TAU))
    best, best_f1 = {"tau_v": float(tv_grid[0]), "tau_ar": float(ta_grid[0])}, -1.0
    for tv in tv_grid:
        for ta in ta_grid:
            f1 = _f1(y_val, rule_scores(feats_val, tv, ta) >= 1.0)
            if f1 > best_f1 + 1e-12:  # strict: ties keep the earlier (lower, higher-recall) taus
                best, best_f1 = {"tau_v": float(tv), "tau_ar": float(ta)}, f1
    return best, best_f1
