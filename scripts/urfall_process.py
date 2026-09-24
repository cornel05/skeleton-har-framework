"""UR-Fall: `download` (official files + SHA-256 manifest) and `extract` (YOLOv8n-pose). See preprocessing.urfall."""
import _bootstrap  # noqa: F401
from pose_estimation.preprocessing.urfall import main

if __name__ == "__main__":
    main()
