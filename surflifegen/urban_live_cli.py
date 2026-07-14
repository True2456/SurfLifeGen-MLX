# surflifegen/urban_live_cli.py
"""
Command-Line Interface for UrbanLiveVideoPlayer (`surflifegen-urban-live`).
Interactive live video player where pressing keys 1-9 toggles specific class layers ON/OFF in real time!
"""

import os
import glob
import argparse
from .urban_live import UrbanLiveVideoPlayer


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Interactive Live Video Segmenter with Number Key Class Toggling (`surflifegen-urban-live`)"
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="Path to video file (.mp4, .mov, .avi). If omitted, picks the latest video inside `./videos/` directory."
    )
    parser.add_argument(
        "--classes", "-c",
        type=str,
        default="building, road, grass, car, person, tree, edge of road, sidewalk",
        help="Comma-separated list of target urban classes (default: building, road, grass, car, person, tree, edge of road, sidewalk)"
    )
    parser.add_argument(
        "--engine", "-e",
        type=str,
        choices=["sam", "yolo"],
        default="sam",
        help="Segmentation engine: 'sam' (Zero-shot Grounded-SAM) or 'yolo' (Trained Ultralytics YOLOv8-Seg model) (default: sam)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to trained YOLO segmentation weights (.pt file) when using --engine yolo"
    )
    parser.add_argument(
        "--stride", "-s",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1, increase to e.g. 5 or 10 for faster video navigation)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=800,
        help="Resize long edge of frame to max-size before processing for faster speed (default: 800)"
    )
    parser.add_argument(
        "--box-threshold", "--thresh",
        type=float,
        default=0.20,
        help="Detection confidence threshold (default: 0.20)"
    )
    parser.add_argument(
        "--output-video", "-o",
        type=str,
        default=None,
        help="Optional path to save the processed video output with overlays"
    )

    args = parser.parse_args()

    video_path = args.video
    if video_path is None:
        # Check ./videos directory for latest video
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_dir = os.path.join(repo_dir, "videos")
        v_files = []
        if os.path.exists(video_dir):
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"):
                v_files.extend(glob.glob(os.path.join(video_dir, ext)))
        if not v_files:
            print(f"[Error] No `--video` path specified, and no video files found inside '{video_dir}'!")
            print(f"👉 Please place your video files inside '{video_dir}' or pass `--video /path/to/video.mp4`.")
            return
        # Pick most recently modified video
        video_path = max(v_files, key=os.path.getmtime)
        print(f"[surflifegen-urban-live] Auto-selected latest video from ./videos: '{os.path.basename(video_path)}'")

    classes_list = [c.strip() for c in args.classes.split(",") if c.strip()]

    player = UrbanLiveVideoPlayer(
        classes=classes_list,
        engine=args.engine,
        model_path=args.model,
        box_threshold=args.box_threshold,
        max_size=args.max_size,
        stride=args.stride
    )

    player.play_video(
        video_path=video_path,
        output_video_path=args.output_video
    )


if __name__ == "__main__":
    main()
