import argparse
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Live YOLOv8-Pose Detection")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/yolov8n-pose.pt", 
        help="Path to YOLOv8-pose model (e.g., models/yolov8n-pose.pt)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Source for video (0 for webcam, or path to video file)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Image size for inference"
    )
    args = parser.parse_args()

    # Load the model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Convert source to int if it's a digit (for webcam index)
    source = args.source
    if source.isdigit():
        source = int(source)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return

    print("Starting live preview. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished or failed to read frame.")
            break

        # Run inference
        results = model.predict(
            source=frame, 
            conf=args.conf, 
            imgsz=args.imgsz, 
            verbose=False
        )

        # Plot results on frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLOv8 Live Pose Detection", annotated_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
