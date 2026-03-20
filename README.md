# skeleton-har-framework

A production-ready project for E2E Skeleton-based Human Action Recognition (HAR), focusing on fall detection and pose extraction.

## Project Structure

- `src/pose_estimation/`: Core logic for pose extraction and processing.
- `scripts/`: Standalone scripts for dataset processing (LE2I, UR-Fall).
- `models/`: Folder to store pre-trained weights (e.g., `yolov8n-pose.pt`).
- `tests/`: Basic scripts to verify functionality.
- `dataset/`: (Local only) Folder to store video datasets and extracted keypoints.

## Setup

1.  **Install uv (Package Manager):**
    If you haven't installed `uv` yet, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/):
    ```bash
    # For Linux and macOS
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Environment Setup:**
    ```bash
    # Create and sync the virtual environment using uv
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

3.  **Models:**
    Place pre-trained YOLO weights in the `models/` folder.

## Usage

- **Pose Extraction:**
  ```bash
  uv run scripts/pose_extraction.py --video path/to/video.mp4
  ```

- **Dataset Processing:**
  Check the `scripts/` directory for dataset-specific processing scripts. Use `uv run` to execute them:
  ```bash
  uv run scripts/urfall_process.py
  ```

- **Mirror Augmentation (Integrated During Processing):**
  Use `--mirror-aug` to also generate mirrored skeleton files while processing:
  ```bash
  uv run scripts/urfall_process.py --mirror-aug
  uv run scripts/le2i_process.py --mirror-aug
  ```

- **Mirror Augmentation (One-Pass Over Existing .npy Files):**
  Run one-time augmentation on extracted skeleton files:
  ```bash
  uv run scripts/mirror_augment_pose_npy.py --folder dataset/pose_npy
  ```

## License
MIT (or your choice)
