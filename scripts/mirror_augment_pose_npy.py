"""
Write a horizontally mirrored copy (`<stem>_mirror.npy`) of every 34-d feature file in a folder.
The evaluation protocol does NOT use these files: it mirrors training windows in memory so that
mirrored copies can never reach validation or test.
"""
import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from pose_estimation.preprocessing.common import mirror_coco17_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    a = ap.parse_args()
    n = 0
    for path in sorted(a.folder.rglob("*.npy")):
        if path.stem.endswith("_mirror"):
            continue
        np.save(path.with_name(f"{path.stem}_mirror.npy"), mirror_coco17_sequence(np.load(path)))
        n += 1
    print(f"Mirrored {n} file(s) in {a.folder}")


if __name__ == "__main__":
    main()
