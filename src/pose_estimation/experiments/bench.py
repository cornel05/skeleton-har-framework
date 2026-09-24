"""
Efficiency benchmarks on the machine this runs on (CPU unless a GPU is present).

  pose      : YOLOv8n-pose ms/frame (batch 1, wall clock around model.predict incl. pre/post-processing)
  classifier: ms/window per model, batch 1, 20 warm-up runs, >=200 timed runs, median + p95
  onnx      : best model exported to ONNX, onnxruntime CPU latency (same protocol)
  e2e       : full pipeline FPS on a real dataset video, CPU only
              (decode -> pose -> features -> classifier every `stride` frames)
Every entry records its input source; nothing is measured on an edge device.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from ..config import PROJECT_ROOT, cfg as CONFIG
from ..features import normalize_repo_features
from ..model import FullWindowExport, build_model
from ..preprocessing.urfall import iter_zip_frames, pick_pose_from_result

WARMUP, RUNS, BLOCKS = 20, 300, 3


def hardware() -> Dict[str, str]:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    mem_kb = 0
    try:
        mem_kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    except (OSError, AttributeError):
        pass
    os_name = platform.platform()
    try:
        os_name = re.search(r'PRETTY_NAME="([^"]+)"', Path("/etc/os-release").read_text()).group(1) + \
            f" (kernel {platform.release()})"
    except (OSError, AttributeError):
        pass
    virt = []
    try:
        hv = re.search(r"Hypervisor vendor:\s+(\S+)", subprocess.check_output(["lscpu"], text=True))
        virt.append(f"hypervisor {hv.group(1)}" if hv else "no hypervisor reported")
    except Exception:  # noqa: BLE001
        pass
    try:
        virt.append("container " + subprocess.check_output(["systemd-detect-virt", "--container"], text=True).strip())
    except Exception:  # noqa: BLE001
        pass
    virt = ", ".join(virt)
    import ultralytics
    import onnxruntime
    return {
        "cpu_model": cpu, "logical_cpus": str(os.cpu_count()), "ram_gb": f"{mem_kb / 1024 ** 2:.1f}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "os": os_name, "virtualization": virt or "unknown", "python": platform.python_version(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "onnxruntime": onnxruntime.__version__, "numpy": np.__version__,
    }


def _stats(ms: List[float]) -> Dict[str, float]:
    a = np.asarray(ms)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n_runs": int(len(a))}


def _frames(raw_dir: Path, n: int) -> Tuple[List[np.ndarray], str]:
    """Real dataset frames if available, else ultralytics' bundled sample images at 640x480."""
    zips = sorted(raw_dir.glob("fall-*-cam0-rgb.zip"))
    if zips:
        frames = [img for _, img in zip(range(n), iter_zip_frames(zips[0]))]
        return frames, f"UR-Fall {zips[0].name} ({frames[0].shape[1]}x{frames[0].shape[0]})"
    import cv2
    import ultralytics
    assets = Path(ultralytics.__file__).parent / "assets"
    imgs = [cv2.resize(cv2.imread(str(p)), (640, 480)) for p in sorted(assets.glob("*.jpg"))]
    return [imgs[i % len(imgs)] for i in range(n)], \
        "PROXY: ultralytics sample images resized to 640x480 (dataset frames unavailable)"


def bench_pose(model_path: str, raw_dir: Path, imgsz_list=(320, 640)) -> dict:
    from ultralytics import YOLO
    frames, source = _frames(raw_dir, WARMUP + 200)
    out = {"model": Path(model_path).name, "source": source, "device": "cpu",
           "torch_threads": torch.get_num_threads(), "runs": {}}
    model = YOLO(model_path)
    for imgsz in imgsz_list:
        for f in frames[:WARMUP]:
            model.predict(f, imgsz=imgsz, device="cpu", verbose=False)
        ms, inf, st = [], [], _cpu_steal()
        for f in frames[WARMUP:]:
            t = time.perf_counter()
            r = model.predict(f, imgsz=imgsz, device="cpu", verbose=False)[0]
            ms.append((time.perf_counter() - t) * 1e3)
            inf.append(r.speed["inference"])
        out["runs"][f"imgsz_{imgsz}"] = {**_stats(ms), "inference_only_median_ms": float(np.median(inf)),
                                         "steal_jiffies": _cpu_steal() - st}
    if torch.cuda.is_available():
        out["gpu"] = "available but not benchmarked in this script"
    else:
        out["gpu"] = "no GPU on this machine: not measured"
    return out


def _cpu_steal() -> int:
    """Cumulative steal jiffies (time the hypervisor ran something else on our vCPUs)."""
    try:
        return int(Path("/proc/stat").read_text().splitlines()[0].split()[8])
    except (OSError, IndexError, ValueError):
        return -1


def _classifier_latency(model: torch.nn.Module, threads: int, L: int = 32, D: int = 34) -> dict:
    """RUNS timed calls after WARMUP, repeated in BLOCKS blocks; reports the block with the median median."""
    torch.set_num_threads(threads)
    model.eval()
    x, m = torch.randn(1, L, D), torch.ones(1, L)
    blocks = []
    with torch.no_grad():
        for _ in range(BLOCKS):
            for _ in range(WARMUP):
                model(x, m)
            ms, st = [], _cpu_steal()
            for _ in range(RUNS):
                t = time.perf_counter()
                model(x, m)
                ms.append((time.perf_counter() - t) * 1e3)
            blocks.append({**_stats(ms), "steal_jiffies": _cpu_steal() - st})
    order = np.argsort([b["median_ms"] for b in blocks])
    return {**blocks[int(order[len(order) // 2])], "block_medians_ms": [b["median_ms"] for b in blocks],
            "threads": threads}


def _checkpoint(runs_dir: Path, name: str) -> Optional[Path]:
    c = sorted(runs_dir.glob(f"{name}/fixed0/seed0/model.pt"))
    return c[0] if c else None


def bench_classifiers(runs_dir: Path, models: List[str]) -> dict:
    out = {}
    for name in models:
        model = build_model(name, cfg=CONFIG.get("model", {}))
        ckpt = _checkpoint(runs_dir, name)
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        out[name] = {"weights": str(ckpt.relative_to(PROJECT_ROOT)) if ckpt else "random init (latency only)",
                     "params": int(sum(p.numel() for p in model.parameters())),
                     "threads_1": _classifier_latency(model, 1), "threads_all": _classifier_latency(model, os.cpu_count())}
    torch.set_num_threads(os.cpu_count())
    return out


def bench_onnx(runs_dir: Path, name: str, out_dir: Path) -> dict:
    import onnxruntime as ort
    model = build_model(name, cfg=CONFIG.get("model", {}))
    ckpt = _checkpoint(runs_dir, name)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    wrapper = FullWindowExport(model).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.onnx"
    x = torch.randn(1, 32, 34)
    torch.onnx.export(wrapper, (x,), str(path), input_names=["window"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    res = {"model": name, "weights": str(ckpt.relative_to(PROJECT_ROOT)) if ckpt else "random init",
           "onnx_file_mb": path.stat().st_size / 1e6}
    with torch.no_grad():
        ref = wrapper(x).numpy()
    for threads in (1, os.cpu_count()):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        feed = {"window": x.numpy()}
        res["max_abs_diff_vs_torch"] = float(np.abs(sess.run(None, feed)[0] - ref).max())
        for _ in range(WARMUP):
            sess.run(None, feed)
        ms = []
        for _ in range(RUNS):
            t = time.perf_counter()
            sess.run(None, feed)
            ms.append((time.perf_counter() - t) * 1e3)
        res[f"threads_{threads}"] = _stats(ms)
    return res


def _video_frames(raw_dir: Path) -> Tuple[Optional[Iterator[np.ndarray]], str]:
    import cv2
    mp4 = raw_dir / "fall-01-cam0.mp4"
    if mp4.exists():
        cap = cv2.VideoCapture(str(mp4))

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
        return gen(), f"UR-Fall {mp4.name} (mp4 decode)"
    zips = sorted(raw_dir.glob("fall-01-cam0-rgb.zip"))
    if zips:
        return (img for _, img in iter_zip_frames(zips[0])), f"UR-Fall {zips[0].name} (PNG decode from zip)"
    return None, "dataset video unavailable: not measured"


def bench_e2e(model_path: str, raw_dir: Path, runs_dir: Path, name: str, imgsz: int = 320) -> dict:
    from ultralytics import YOLO
    torch.set_num_threads(os.cpu_count())
    frames, source = _video_frames(raw_dir)
    if frames is None:
        return {"source": source}
    L = CONFIG.get("dataset.sequence_length", 32)
    stride = CONFIG.get("inference.stride", 16)
    pose = YOLO(model_path)
    clf = build_model(name, cfg=CONFIG.get("model", {}))
    ckpt = _checkpoint(runs_dir, name)
    if ckpt:
        clf.load_state_dict(torch.load(ckpt, map_location="cpu"))
    clf.eval()
    buf_k, buf_b = [], []
    t_dec = t_pose = t_clf = 0.0
    n = n_clf = 0
    h = w = 0
    steal0 = _cpu_steal()
    t_start = time.perf_counter()
    t = time.perf_counter()
    for frame in frames:
        t_dec += time.perf_counter() - t
        h, w = frame.shape[:2]
        t = time.perf_counter()
        picked = pick_pose_from_result(pose.predict(frame, imgsz=imgsz, device="cpu", verbose=False)[0])
        buf_k.append(picked[0] if picked else np.full((17, 3), np.nan, np.float32))
        t_pose += time.perf_counter() - t
        n += 1
        t = time.perf_counter()
        if len(buf_k) >= L and (n - L) % stride == 0:
            feats = normalize_repo_features(np.stack(buf_k[-L:]), w, h)
            with torch.no_grad():
                torch.softmax(clf(torch.from_numpy(feats)[None], torch.ones(1, L)), dim=1)
            n_clf += 1
        t_clf += time.perf_counter() - t
        t = time.perf_counter()
    total = time.perf_counter() - t_start
    return {"source": source, "frames": n, "resolution": f"{w}x{h}", "pose_imgsz": imgsz,
            "classifier": name, "classifier_weights": str(ckpt.relative_to(PROJECT_ROOT)) if ckpt else "random init",
            "window": L, "stride": stride, "classifier_calls": n_clf, "total_s": total, "fps": n / total,
            "ms_per_frame": 1e3 * total / n, "decode_ms_per_frame": 1e3 * t_dec / n,
            "pose_ms_per_frame": 1e3 * t_pose / n, "features_classifier_ms_per_frame": 1e3 * t_clf / n,
            "device": "cpu", "torch_threads": torch.get_num_threads(), "steal_jiffies": _cpu_steal() - steal0}


def main():
    from .report import best_model_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-model", default=str(PROJECT_ROOT / "models/yolov8n-pose.pt"))
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "dataset/urfall/raw")
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    ap.add_argument("--models", nargs="+", default=["lstm", "gru", "lstm_attention", "tcn", "stgcn"])
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports/bench.json")
    a = ap.parse_args()
    a.runs_dir, a.raw_dir = a.runs_dir.resolve(), a.raw_dir.resolve()
    best, why = best_model_name(a.runs_dir)
    res = {"hardware": hardware(), "best_model": best, "best_model_rule": why}
    print("hardware:", res["hardware"])
    res["classifier"] = bench_classifiers(a.runs_dir, a.models)
    print("classifier done")
    res["onnx"] = bench_onnx(a.runs_dir, best, PROJECT_ROOT / "reports/onnx")
    print("onnx done")
    res["pose"] = bench_pose(a.pose_model, a.raw_dir)
    print("pose done")
    res["e2e"] = bench_e2e(a.pose_model, a.raw_dir, a.runs_dir, best)
    print("e2e done")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
