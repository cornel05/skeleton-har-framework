import argparse
import logging
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

if not hasattr(mp, "solutions"):
    raise ImportError(
        "Installed Mediapipe does not expose 'mediapipe.solutions'. "
        "Install a compatible version, for example: pip install mediapipe==0.10.14"
    )

mp_pose = mp.solutions.pose


def extract_keypoints(video_path):
    pose = mp_pose.Pose()
    video_path = str(video_path)
    logging.info("Starting keypoint extraction: %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video file: %s", video_path)
        return np.array([])

    seq = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x, lm.y])

            seq.append(np.array(keypoints).flatten())

    cap.release()
    logging.info(
        "Completed: %s | frames read: %d | frames with pose: %d",
        video_path,
        frame_count,
        len(seq),
    )
    return np.array(seq)


def find_mp4_videos_in_subfolders(parent_folder):
    parent = Path(parent_folder)
    if not parent.exists() or not parent.is_dir():
        logging.error("Invalid folder path: %s", parent_folder)
        return []

    videos = []
    for subfolder in sorted(parent.iterdir()):
        if subfolder.is_dir():
            subfolder_videos = sorted(subfolder.glob("*.mp4"))
            videos.extend(subfolder_videos)

    logging.info("Discovered %d .mp4 file(s) under subfolders of %s", len(videos), parent_folder)
    return videos


def build_output_path(video_path, output_dir=None, batch_parent=None):
    video_path = Path(video_path)

    if output_dir:
        output_base = Path(output_dir)
        if batch_parent:
            relative_parent = video_path.parent.relative_to(Path(batch_parent))
            output_base = output_base / relative_parent
    else:
        output_base = video_path.parent

    output_base.mkdir(parents=True, exist_ok=True)
    return output_base / f"{video_path.stem}_keypoints.npy"


def save_keypoints(keypoints, output_path):
    np.save(output_path, keypoints)
    logging.info("Saved keypoints to %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="MediaPipe pose extraction.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str)
    group.add_argument("--folder", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.folder:
        videos = find_mp4_videos_in_subfolders(args.folder)
        for idx, video_path in enumerate(videos, start=1):
            logging.info("[%d/%d] Processing %s", idx, len(videos), video_path)
            keypoints = extract_keypoints(video_path)
            out_path = build_output_path(video_path, output_dir=args.output, batch_parent=args.folder)
            save_keypoints(keypoints, out_path)
    else:
        keypoints = extract_keypoints(args.video)
        out_path = build_output_path(args.video, output_dir=args.output)
        save_keypoints(keypoints, out_path)


if __name__ == "__main__":
    main()
