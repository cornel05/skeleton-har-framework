"""
Video/group-level splits. Windows are never split independently: every window of a
video (and every video of a group, e.g. two cameras of one recorded fall) lands in the
same partition. Splits depend only on `split_seed`, never on the training seed, so all
models and seeds see identical partitions.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def _groups(video_ids: Sequence[str], groups: Sequence[str], labels: Sequence[int]):
    g2v: Dict[str, List[str]] = {}
    g2y: Dict[str, int] = {}
    for v, g, y in zip(video_ids, groups, labels):
        g2v.setdefault(g, []).append(v)
        if g2y.setdefault(g, y) != y:
            raise ValueError(f"group {g} mixes labels")
    names = sorted(g2v)
    return names, np.array([g2y[g] for g in names]), g2v


def _expand(gnames, g2v) -> List[str]:
    return sorted(v for g in gnames for v in g2v[g])


def fixed_split(video_ids, groups, labels, split_seed: int = 0,
                test_frac: float = 0.2, val_frac: float = 0.2) -> Dict[str, List[str]]:
    names, y, g2v = _groups(video_ids, groups, labels)
    rest, test = train_test_split(names, test_size=test_frac, stratify=y, random_state=split_seed)
    y_rest = np.array([y[names.index(g)] for g in rest])
    train, val = train_test_split(rest, test_size=val_frac / (1 - test_frac), stratify=y_rest,
                                  random_state=split_seed)
    return {"train": _expand(train, g2v), "val": _expand(val, g2v), "test": _expand(test, g2v)}


def cv_splits(video_ids, groups, labels, k: int = 5, split_seed: int = 0,
              val_frac_of_rest: float = 0.25) -> List[Dict[str, List[str]]]:
    names, y, g2v = _groups(video_ids, groups, labels)
    folds = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=split_seed)
    for fold, (rest_idx, test_idx) in enumerate(skf.split(names, y)):
        rest = [names[i] for i in rest_idx]
        train, val = train_test_split(rest, test_size=val_frac_of_rest, stratify=y[rest_idx],
                                      random_state=split_seed + fold)
        folds.append({"train": _expand(train, g2v), "val": _expand(val, g2v),
                      "test": _expand([names[i] for i in test_idx], g2v)})
    return folds


def check_disjoint(split: Dict[str, List[str]]) -> None:
    a, b, c = (set(split[k]) for k in ("train", "val", "test"))
    if a & b or a & c or b & c:
        raise AssertionError("video overlap between partitions")
