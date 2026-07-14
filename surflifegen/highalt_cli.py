"""
Command Line Interface for High-Altitude (`120m-400m`) Active Swimmer Synthetic Generation & Grounding Engine.
"""

import os
import argparse
from .highalt_pipeline import HighAltitudeSwimmerPipeline


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: High-Altitude (`120m-400m`) Active Swimmer Synthetic Generator & Grounding DINO Annotator"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./highalt_swimmer_dataset",
        help="Path to save generated high-altitude active swimmer images and DINO annotations"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of high-altitude synthetic scenes to generate (default: 5)"
    )
    parser.add_argument(
        "--altitude", "-a",
        type=int,
        default=100,
        help="Simulated altitude in meters (e.g. 80, 100, 120, 135). Default: 100"
    )
    parser.add_argument(
        "--swimmers", "-s",
        type=int,
        default=3,
        help="Number of active swimmers per scene (default: 3)"
    )
    parser.add_argument("--box-threshold", "--box-thresh", type=float, default=0.18, help="Detection sensitivity box threshold for Grounding DINO (lower = more sensitive, default: 0.18)")
    parser.add_argument("--text-threshold", "--text-thresh", type=float, default=0.18, help="Detection sensitivity text prompt threshold for Grounding DINO (default: 0.18)")
    parser.add_argument("--nms-threshold", "--iou-threshold", type=float, default=0.30, help="NMS IoU threshold for overlapping box removal (default: 0.30)")
    parser.add_argument("--detection-prompt", "--dino-prompt", "-d", type=str, default=None, help="Custom dot-separated text queries for Grounding DINO zero-shot detection (e.g. 'swimmer . person splashing .')")

    args = parser.parse_args()

    pipeline = HighAltitudeSwimmerPipeline(
        output_dir=args.output_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=args.nms_threshold
    )
    print(f"[HighAlt Generator] Starting generation of {args.count} high-altitude (`alt{args.altitude}m`) active swimmer scenes (BoxThresh={args.box_threshold}, TextThresh={args.text_threshold}, NMS={args.nms_threshold})...")

    # Strict anti-overwrite guarantee: find next starting index
    import glob, re
    pattern = os.path.join(args.output_dir, "highalt_*.png")
    files = glob.glob(pattern)
    start_idx = 1
    for fp in files:
        base = os.path.basename(fp)
        match = re.search(r"highalt_(\d+)", base)
        if match:
            try:
                idx = int(match.group(1))
                if idx >= start_idx:
                    start_idx = idx + 1
            except ValueError:
                pass

    for i in range(start_idx, start_idx + args.count):
        clean_p, ann_p, meta = pipeline.generate_highalt_scene(
            altitude_m=args.altitude,
            swimmer_count=args.swimmers,
            filename_prefix=f"highalt_{i:04d}",
            detection_prompt=args.detection_prompt
        )
        print(f"[{i - start_idx + 1}/{args.count}] Generated {os.path.basename(clean_p)} -> {meta.get('dino_summary')}")

    print(f"\n[HighAlt Generator] Completed! Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
