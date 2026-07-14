# surflifegen/highway_cli.py
"""
Command-Line Interface for High-Altitude & Surface Inspection Highway Defect Generator (`surflifegen-highway`).
Synthesizes cracks, potholes, alligatoring, rutting, and degraded road markings with Grounding DINO zero-shot annotation.
"""

import os
import argparse
import time
from typing import List

from .highway_pipeline import HighwayWearPipeline
from .highway_prompt_builder import DEFECT_TYPES, ASPHALT_SURFACES


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Synthetic Highway Defect & Asphalt Wear Generator & Grounding DINO Zero-Shot Annotator"
    )
    parser.add_argument(
        "--defect-type", "-t",
        type=str,
        default="random",
        choices=list(DEFECT_TYPES.keys()) + ["random"],
        help="Specific pavement defect class to synthesize (default: random)"
    )
    parser.add_argument(
        "--asphalt", "-a",
        type=str,
        default="random",
        help="Asphalt/surface type description or 'random' (e.g. 'weathered grey asphalt', 'dark newly paved')"
    )
    parser.add_argument(
        "--perspective",
        type=str,
        default="random",
        choices=["nadir_drone", "low_nadir", "vehicle_surface", "random"],
        help="Inspection camera perspective (default: random)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./highway_defect_dataset",
        help="Path to save generated highway inspection images and DINO annotations"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of highway defect scenes to generate (default: 5)"
    )
    parser.add_argument(
        "--bulk-count", "-n",
        type=int,
        default=None,
        help="Generate X amount of images using modular randomized prompt builder (alias for --count)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Custom text prompt for generation (overrides modular template)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=25,
        help="Number of MLX inference steps per image (default: 25)"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Path to local Cosmos3-Nano-MLX-8bit folder"
    )
    parser.add_argument(
        "--box-threshold", "--box-thresh",
        type=float,
        default=0.18,
        help="Detection sensitivity box threshold for Grounding DINO (lower = more sensitive, default: 0.18)"
    )
    parser.add_argument(
        "--text-threshold", "--text-thresh",
        type=float,
        default=0.18,
        help="Detection sensitivity text prompt threshold for Grounding DINO (default: 0.18)"
    )
    parser.add_argument(
        "--nms-threshold", "--iou-threshold",
        type=float,
        default=0.30,
        help="NMS IoU threshold for overlapping box removal (default: 0.30)"
    )
    parser.add_argument(
        "--detection-prompt", "--dino-prompt", "-d",
        type=str,
        default=None,
        help="Custom dot-separated text queries for Grounding DINO zero-shot detection (e.g. 'pothole . road crack .')"
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip auto-annotation/segmentation pass"
    )
    parser.add_argument(
        "--boxes-only", "--no-sam",
        action="store_true",
        help="Use coarse Grounding DINO rectangular bounding boxes instead of SAM precision polygon masks"
    )

    args = parser.parse_args()
    total_count = args.bulk_count if args.bulk_count is not None else args.count

    pipeline = HighwayWearPipeline(
        output_dir=args.output_dir,
        model_path=args.model_path,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=args.nms_threshold,
        no_annotate=args.no_annotate,
        use_sam=not args.boxes_only
    )

    print(f"[HighwayWear Generator] Starting batch synthesis of {total_count} highway defect & surface wear scenes...")
    t_start = time.time()

    for i in range(1, total_count + 1):
        print(f"\n=======================================================")
        print(f"[HighwayWear Generator] Processing Image {i} / {total_count}")
        print(f"=======================================================")
        clean_p, ann_p, meta = pipeline.generate_scene(
            defect_type=args.defect_type,
            asphalt_type=args.asphalt,
            perspective=args.perspective,
            custom_prompt=args.prompt,
            steps=args.steps,
            detection_prompt=args.detection_prompt
        )

    total_time = round(time.time() - t_start, 2)
    print(f"\n[HighwayWear Generator] Completed! {total_count} images generated across {total_time}s.")
    print(f"[HighwayWear Generator] Dataset & annotations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
