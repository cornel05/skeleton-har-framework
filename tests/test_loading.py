from pathlib import Path
import numpy as np

# Path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "dataset/pose_npy/adl-10-cam0-rgb_skeleton.npy"

if data_path.exists():
    data = np.load(data_path)
    print(f"Loaded: {data_path}")
    print(f"Shape: {data.shape}")
    print(data[:10])
else:
    print(f"File not found: {data_path}")