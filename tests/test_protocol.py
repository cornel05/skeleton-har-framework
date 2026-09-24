"""Protocol invariants: run with `make test` (pytest)."""

import os
import sys

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pose_estimation.experiments.baseline import fit_rule, rule_scores  # noqa: E402
from pose_estimation.experiments.metrics import (count_alarms, event_metrics, select_threshold,  # noqa: E402
                                                 window_metrics)
from pose_estimation.experiments.splits import check_disjoint, cv_splits, fixed_split  # noqa: E402
from pose_estimation.experiments.windows import (FALLING, IGNORE, LYING, NEG, NOT_LYING, POS, Video,  # noqa: E402
                                                 build_windows, label_window)
from pose_estimation.features import normalize_repo_features  # noqa: E402
from pose_estimation.model import MODEL_NAMES, FullWindowExport, build_model  # noqa: E402
from pose_estimation.preprocessing.common import mirror_coco17_sequence  # noqa: E402


def _videos(n_fall=30, n_adl=40):
    ids = [f"fall-{i:02d}-cam0-rgb" for i in range(n_fall)] + [f"adl-{i:02d}-cam0-rgb" for i in range(n_adl)]
    return ids, [v[:7] for v in ids], [1] * n_fall + [0] * n_adl


def test_fixed_split_is_video_level_disjoint_and_stratified():
    ids, groups, labels = _videos()
    s = fixed_split(ids, groups, labels, split_seed=0)
    check_disjoint(s)
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(ids)
    for part in ("val", "test"):
        falls = sum(v.startswith("fall") for v in s[part])
        assert falls == 6 and len(s[part]) == 14


def test_groups_never_straddle_partitions():
    ids = [f"fall-{i:02d}-{c}-rgb" for i in range(10) for c in ("cam0", "cam1")] + [f"adl-{i:02d}-cam0-rgb" for i in range(10)]
    groups = [v[:7] for v in ids]
    labels = [1 if v.startswith("fall") else 0 for v in ids]
    for s in [fixed_split(ids, groups, labels)] + cv_splits(ids, groups, labels, k=5):
        where = {}
        for part, vids in s.items():
            for v in vids:
                where.setdefault(v[:7], set()).add(part)
        assert all(len(p) == 1 for p in where.values())


def test_cv_covers_each_video_once_in_test():
    ids, groups, labels = _videos()
    folds = cv_splits(ids, groups, labels, k=5)
    tested = [v for f in folds for v in f["test"]]
    assert sorted(tested) == sorted(ids)
    for f in folds:
        check_disjoint(f)


def test_splits_do_not_depend_on_training_seed():
    ids, groups, labels = _videos()
    assert fixed_split(ids, groups, labels, 0) == fixed_split(ids, groups, labels, 0)


def _fall_video(T=120, fall=(50, 62)):
    fl = np.full(T, NOT_LYING)
    fl[fall[0]:fall[1]] = FALLING
    fl[fall[1]:] = LYING
    feats = np.zeros((T, 34), np.float32)
    sig = {"hip_y": np.zeros(T, np.float32), "box_h": np.ones(T, np.float32), "aspect": np.ones(T, np.float32)}
    return Video("fall-01-cam0-rgb", "fall-01", 1, feats, sig, fl, 30.0, 0.0)


def test_window_labels():
    v = _fall_video()
    assert label_window(v, 0, 32, 8) == NEG           # pre-fall only
    assert label_window(v, 40, 32, 8) == POS          # contains all 12 falling frames
    assert label_window(v, 18, 32, 8) == NEG          # frames 18..49 end right before the fall
    assert label_window(v, 80, 32, 8) == IGNORE       # post-fall lying only
    assert label_window(v, 25, 32, 8) == IGNORE       # 7 falling frames < 8
    adl = Video("adl-01-cam0-rgb", "adl-01", 0, v.feats, v.signals, np.full(120, LYING), 30.0, 0.0)
    assert label_window(adl, 80, 32, 8) == NEG        # ADL windows are always negative


def test_short_fall_interval_uses_its_full_length():
    v = _fall_video(fall=(50, 55))                    # 5 falling frames < min_fall_frames
    assert label_window(v, 30, 32, 8) == POS          # holds all 5


def test_build_windows_pads_short_videos():
    v = _fall_video(T=20, fall=(5, 10))
    ws = build_windows([v], 32, 8, 8)
    assert ws.X.shape == (1, 32, 34) and ws.mask[0].sum() == 20


def test_mirror_is_an_involution_and_swaps_sides():
    x = np.random.default_rng(0).normal(size=(32, 34)).astype(np.float32)
    m = mirror_coco17_sequence(x)
    assert np.allclose(mirror_coco17_sequence(m), x)
    assert np.allclose(m.reshape(32, 17, 2)[:, 5, 0], -x.reshape(32, 17, 2)[:, 6, 0])


def test_normalization_interpolates_missing_frames_and_is_hip_centred():
    k = np.random.default_rng(1).uniform(100, 400, size=(10, 17, 3)).astype(np.float32)
    k[4] = np.nan
    f = normalize_repo_features(k, 640, 480).reshape(10, 17, 2)
    assert np.isfinite(f).all()
    assert np.allclose((f[:, 11] + f[:, 12]) / 2, 0, atol=1e-5)
    assert np.abs(f).max() <= 10.0


def test_threshold_rules_and_window_metrics():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(30), np.zeros(170)].astype(int)
    s = np.clip(y * 0.4 + rng.uniform(0, 0.7, 200), 0, 1)
    thr = select_threshold(y, s, "max_f1")
    m = window_metrics(y, s, thr)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))
    assert m["tp"] + m["fn"] == 30 and m["tn"] + m["fp"] == 170
    thr90 = select_threshold(y, s, "recall_0.90")
    assert window_metrics(y, s, thr90)["recall"] >= 0.90


def test_alarm_counting_and_event_metrics():
    assert count_alarms(np.array([0, 1, 1, 0, 1, 0, 0, 1])) == 3
    assert count_alarms(np.array([], bool)) == 0
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.95])
    vid = np.array([0, 0, 0, 1, 1, 1, 1])
    start = np.array([0, 8, 16, 0, 8, 16, 24])
    e = event_metrics(scores, vid, start, 0.5, ["fall-01", "adl-01"], {"fall-01": 1, "adl-01": 0},
                      {"fall-01": 48, "adl-01": 56}, 30.0, {"fall-01": (10, 20)}, 32)
    assert e["event_recall"] == 1.0 and e["event_recall_localized"] == 1.0
    assert e["false_alarms"] == 2 and e["false_alarms_per_adl_video"] == 2.0


def test_rule_baseline_is_fit_on_validation_only():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    feats = np.c_[rng.normal(0.3, 0.2, 100) + 1.5 * y, rng.normal(0.0, 0.1, 100) + 0.6 * y].astype(np.float32)
    taus, f1 = fit_rule(feats, y)
    assert f1 > 0.8 and window_metrics(y, rule_scores(feats, **taus), 1.0)["f1"] == pytest.approx(f1)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_forward_and_export_wrapper(name):
    torch.manual_seed(0)
    net = build_model(name).eval()
    x, m = torch.randn(3, 32, 34), torch.ones(3, 32)
    with torch.no_grad():
        logits = net(x, m)
        assert logits.shape == (3, 2)
        assert torch.allclose(torch.softmax(logits, 1), FullWindowExport(net)(x), atol=1e-6)
        m2 = m.clone()
        m2[0, 20:] = 0
        assert torch.isfinite(net(x, m2)).all()
