# Pose Estimation Project

A production-ready project for pose extraction from videos using Mediapipe and YOLOv8.

## Project Structure

- `src/pose_estimation/`: Core logic for pose extraction and processing.
- `scripts/`: Standalone scripts for dataset processing (LE2I, UR-Fall).
- `models/`: Folder to store pre-trained weights (e.g., `yolov8n-pose.pt`).
- `tests/`: Basic scripts to verify functionality.
- `dataset/`: (Local only) Folder to store video datasets and extracted keypoints.

## Setup

1.  **Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Models:**
    Place pre-trained YOLO weights in the `models/` folder.

## Usage

- **Pose Extraction:**
  ```bash
  python scripts/pose_extraction.py --video path/to/video.mp4
  ```

- **Dataset Processing:**
  Check the `scripts/` directory for dataset-specific processing scripts.

## License
MIT (or your choice)
