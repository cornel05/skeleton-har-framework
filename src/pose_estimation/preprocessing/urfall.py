"""
UR Fall Detection dataset (University of Rzeszow) -> YOLOv8-pose keypoints.

Pipeline (one command each, see Makefile `make data`):
    download : fetch the official cam0 RGB frame archives + per-frame label CSVs
    extract  : run YOLOv8n-pose on every frame, keep the highest-confidence person,
               write one raw `.npz` per sequence (pixel keypoints + confidences + box)
               and the legacy 34-d normalized `.npy` used by the original repo.

Official source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
Only cam0 is used for the evaluation protocol: cam1 exists for falls only, so
including it would make "camera" a proxy for the label.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ..utils import resolve_device
    from .common import mirror_coco17_sequence
    from ..features import normalize_repo_features
except ImportError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from pose_estimation.utils import resolve_device
    from pose_estimation.preprocessing.common import mirror_coco17_sequence
    from pose_estimation.features import normalize_repo_features


BASE_URL = "http://fenix.ur.edu.pl/~mkepski/ds/data/"
NUM_FALLS = 30
NUM_ADLS = 40
LABEL_CSVS = ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")
FPS = 30.0


def sequence_ids(cams: Tuple[str, ...] = ("cam0",)) -> List[str]:
    """Return archive stems, e.g. 'fall-01-cam0-rgb'. ADLs were recorded with cam0 only."""
    ids = []
    for cam in cams:
        ids += [f"fall-{i:02d}-{cam}-rgb" for i in range(1, NUM_FALLS + 1)]
        if cam == "cam0":
            ids += [f"adl-{i:02d}-{cam}-rgb" for i in range(1, NUM_ADLS + 1)]
    return ids


def sequence_name(archive_stem: str) -> str:
    """'fall-01-cam0-rgb' -> 'fall-01' (the key used by the official label CSVs)."""
    m = re.match(r"^((?:fall|adl)-\d+)", archive_stem)
    if not m:
        raise ValueError(f"Unexpected UR-Fall archive name: {archive_stem}")
    return m.group(1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 4) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.rename(dest)
            return
        except Exception as exc:  # noqa: BLE001 - network errors are reported and retried
            wait = 2 ** (attempt + 1)
            print(f"[WARN] {url}: {exc} (retry in {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def download(raw_dir: Path, base_url: str, cams: Tuple[str, ...], with_video: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{s}.zip" for s in sequence_ids(cams)] + list(LABEL_CSVS)
    if with_video:
        # One real video for the end-to-end throughput benchmark.
        names.append("fall-01-cam0.mp4")
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in tqdm(names, desc="UR-Fall download", unit="file"):
        dest = raw_dir / name
        if not dest.exists():
            _download(base_url + name, dest)
        if name not in manifest:
            manifest[name] = {"url": base_url + name, "sha256": _sha256(dest), "bytes": dest.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Downloaded {len(names)} files to {raw_dir}; provenance in {manifest_path}")


def iter_zip_frames(zip_path: Path) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_number, BGR image) in frame order from an official *-rgb.zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        entries = []
        for info in zf.infolist():
            m = re.search(r"-(\d+)\.png$", info.filename)
            if m and not info.is_dir():
                entries.append((int(m.group(1)), info.filename))
        entries.sort()
        for frame_no, name in entries:
            buf = np.frombuffer(zf.read(name), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                yield frame_no, img


def pick_pose_from_result(result) -> Optional[np.ndarray]:
    """
    Select one person's keypoints from a YOLOv8-pose result (highest box confidence,
    as in the original repo). Returns (17, 3) [x_px, y_px, conf] and (5,) box [x1, y1, x2, y2, conf].
    """
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    data = result.keypoints.data.detach().cpu().numpy()  # (N, 17, 3)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    best = int(np.argmax(conf))
    return data[best].astype(np.float32), np.concatenate([boxes[best], conf[best:best + 1]]).astype(np.float32)


def extract_sequence(frames: Iterator[Tuple[int, np.ndarray]], model, device: str, imgsz: int,
                     conf_thres: float, batch: int = 16) -> Dict[str, np.ndarray]:
    kpts, boxes, frame_ids = [], [], []
    width = height = 0
    buf: List[Tuple[int, np.ndarray]] = []

    def flush():
        results = model.predict(source=[img for _, img in buf], device=device, imgsz=imgsz,
                                conf=conf_thres, verbose=False)
        for (fno, _), res in zip(buf, results):
            picked = pick_pose_from_result(res)
            frame_ids.append(fno)
            if picked is None:
                kpts.append(np.full((17, 3), np.nan, np.float32))
                boxes.append(np.full((5,), np.nan, np.float32))
            else:
                kpts.append(picked[0])
                boxes.append(picked[1])
        buf.clear()

    for fno, img in frames:
        height, width = img.shape[:2]
        buf.append((fno, img))
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    return {
        "kpts": np.stack(kpts) if kpts else np.zeros((0, 17, 3), np.float32),
        "box": np.stack(boxes) if boxes else np.zeros((0, 5), np.float32),
        "frame_id": np.asarray(frame_ids, dtype=np.int32),
        "width": np.int32(width),
        "height": np.int32(height),
        "fps": np.float32(FPS),
    }


def extract(raw_dir: Path, pose_dir: Path, legacy_dir: Optional[Path], model_path: str, device: str,
            imgsz: int, conf_thres: float, cams: Tuple[str, ...], mirror_aug: bool) -> None:
    from ultralytics import YOLO

    pose_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    stems = sequence_ids(cams)
    for stem in tqdm(stems, desc="YOLOv8-pose extraction", unit="seq"):
        out = pose_dir / f"{stem}.npz"
        if out.exists():
            continue
        zip_path = raw_dir / f"{stem}.zip"
        if not zip_path.exists():
            print(f"[WARN] missing archive {zip_path}; run the download step first")
            continue
        rec = extract_sequence(iter_zip_frames(zip_path), model, device, imgsz, conf_thres)
        rec.update(pose_model=np.array(Path(model_path).name), imgsz=np.int32(imgsz),
                   conf_thres=np.float32(conf_thres), sequence=np.array(sequence_name(stem)))
        np.savez_compressed(out, **rec)
        if legacy_dir is not None:
            legacy_dir.mkdir(parents=True, exist_ok=True)
            feats = normalize_repo_features(rec["kpts"], int(rec["width"]), int(rec["height"]))
            np.save(legacy_dir / f"{stem}_skeleton.npy", feats)
            if mirror_aug:
                np.save(legacy_dir / f"{stem}_skeleton_mirror.npy", mirror_coco17_sequence(feats))


def load_frame_labels(raw_dir: Path) -> Dict[str, Dict[int, int]]:
    """
    Parse the official per-frame label CSVs (cam0). Column 0 = sequence name ('fall-01'),
    column 1 = frame number, column 2 = label: -1 not lying, 0 falling (transition), 1 lying on ground.
    """
    labels: Dict[str, Dict[int, int]] = {}
    for name in LABEL_CSVS:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing official label file {path}")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not re.match(r"^(fall|adl)-\d+$", parts[0]):
                continue  # header or malformed row
            try:
                frame, lab = int(float(parts[1])), int(float(parts[2]))
            except ValueError:
                continue
            labels.setdefault(parts[0], {})[frame] = lab
    return labels


def main():
    parser = argparse.ArgumentParser(description="UR-Fall dataset download + YOLOv8-pose extraction.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--raw-dir", type=Path, default=Path("dataset/urfall/raw"))
    d.add_argument("--base-url", type=str, default=BASE_URL)
    d.add_argument("--cams", nargs="+", default=["cam0"])
    d.add_argument("--no-video", action="store_true")

    e = sub.add_parser("extract")
    e.add_argument("--raw-dir", type=Path, default=Path("dataset/urfall/raw"))
    e.add_argument("--pose-dir", type=Path, default=Path("dataset/urfall/pose"))
    e.add_argument("--legacy-dir", type=Path, default=Path("dataset/pose_npy"))
    e.add_argument("--model", type=str, default="models/yolov8n-pose.pt")
    e.add_argument("--device", type=str, default="cpu")
    e.add_argument("--imgsz", type=int, default=320)
    e.add_argument("--conf", type=float, default=0.25)
    e.add_argument("--cams", nargs="+", default=["cam0"])
    e.add_argument("--mirror-aug", action="store_true",
                   help="Also write mirrored legacy .npy files (the protocol mirrors train windows itself).")
    args = parser.parse_args()

    if args.cmd == "download":
        download(args.raw_dir, args.base_url, tuple(args.cams), with_video=not args.no_video)
    else:
        extract(args.raw_dir, args.pose_dir, args.legacy_dir, args.model, str(resolve_device(args.device)),
                args.imgsz, args.conf, tuple(args.cams), args.mirror_aug)


if __name__ == "__main__":
    main()
