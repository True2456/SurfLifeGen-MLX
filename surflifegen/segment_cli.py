# surflifegen/segment_cli.py
"""
Command-Line Interface for Grounded-SAM Zero-Shot Pavement Defect & Highway Segmentation (`surflifegen-segment`).
Produces precision polygon segmentation masks (`_mask.png`) and translucent color overlays (`_segmented.png`).
"""

import argparse
from .segmenter import GroundedSamSegmenter


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Grounded-SAM Zero-Shot Pavement Defect Segmenter (Apple Silicon MPS)"
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Path to directory containing highway inspection images to segment"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="crack in asphalt . pothole . road edge .",
        help="Dot-separated text queries for localization before SAM mask generation (default: 'crack in asphalt . pothole . road edge .')"
    )
    parser.add_argument(
        "--box-threshold", "--box-thresh",
        type=float,
        default=0.22,
        help="Detection confidence threshold before SAM segmentation (default: 0.22)"
    )
    parser.add_argument(
        "--dino-model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO HuggingFace model ID"
    )
    parser.add_argument(
        "--sam-model-id",
        type=str,
        default="facebook/sam-vit-base",
        help="Segment Anything (SAM) HuggingFace model ID"
    )

    args = parser.parse_args()

    segmenter = GroundedSamSegmenter(
        dino_model_id=args.dino_model_id,
        sam_model_id=args.sam_model_id,
        box_threshold=args.box_threshold
    )
    segmenter.segment_dataset(args.dataset_dir, detection_prompt=args.prompt, box_threshold=args.box_threshold)


if __name__ == "__main__":
    main()
