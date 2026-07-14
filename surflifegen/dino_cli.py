"""
Command Line Interface for Grounding DINO Zero-Shot Auto-Annotation.
Runs natively on Apple Silicon PyTorch MPS.
"""

import argparse
from .dino_annotator import GroundingDinoAnnotator


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Grounding DINO Open-Vocabulary Zero-Shot Auto-Annotator (Apple Silicon MPS)"
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Path to dataset directory containing PNG images"
    )
    parser.add_argument(
        "--model-id", "-m",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model ID (e.g. IDEA-Research/grounding-dino-base or IDEA-Research/grounding-dino-tiny)"
    )
    parser.add_argument(
        "--target", "-t",
        choices=["swimmer", "shark"],
        default="swimmer",
        help="Target class to detect: 'swimmer' or 'shark'"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.22,
        help="Confidence threshold for bounding box detection (default: 0.22)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.22,
        help="Confidence threshold for text grounding (default: 0.22)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.30,
        help="Non-Maximum Suppression IoU threshold (default: 0.30)"
    )

    args = parser.parse_args()

    annotator = GroundingDinoAnnotator(
        model_id=args.model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=args.nms_iou
    )
    annotator.annotate_dataset(args.dataset_dir, target_type=args.target)


if __name__ == "__main__":
    main()
