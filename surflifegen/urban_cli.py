# surflifegen/urban_cli.py
"""
Command-Line Interface for UrbanSceneSegmenter (`surflifegen-urban`).
Multi-class zero-shot segmentation for aerial, drone, and urban photography.
Produces `_urban_segmented.png` (with legend), `_urban_mask.png`, `_urban_mask_vis.png`, and `urban_segmentations.json`.
"""

import os
import argparse
from .urban_segmenter import UrbanSceneSegmenter


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Multi-Class Urban & Aerial Scene Grounded-SAM Segmenter (`surflifegen-urban`)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an individual image or a directory of images to segment"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory where output segmentation masks, overlays, and JSON metadata will be written (default: same directory as input)"
    )
    parser.add_argument(
        "--classes", "-c",
        type=str,
        default="building, road, grass, car, person, tree, edge of road, sidewalk",
        help="Comma-separated list of target urban classes (default: building, road, grass, car, person, tree, edge of road, sidewalk)"
    )
    parser.add_argument(
        "--box-threshold", "--thresh",
        type=float,
        default=0.20,
        help="Grounding DINO bounding box confidence threshold (default: 0.20)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO text label matching threshold (default: 0.20)"
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

    classes_list = [c.strip() for c in args.classes.split(",") if c.strip()]

    segmenter = UrbanSceneSegmenter(
        dino_model_id=args.dino_model_id,
        sam_model_id=args.sam_model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        classes=classes_list
    )

    if os.path.isfile(args.input):
        out_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.input))
        os.makedirs(out_dir, exist_ok=True)
        instances, summary = segmenter.segment_image(args.input, output_dir=out_dir)
        print(f"\n[surflifegen-urban] {summary}")
        
        # Save single image metadata
        meta_path = os.path.join(out_dir, "urban_segmentations.json")
        fname = os.path.basename(args.input)
        record = [{
            "clean_file": fname,
            "mask_file": f"{os.path.splitext(fname)[0]}_urban_mask.png",
            "segmented_file": f"{os.path.splitext(fname)[0]}_urban_segmented.png",
            "instances": instances
        }]
        import json
        with open(meta_path, "w") as f:
            json.dump({
                "classes": segmenter.classes,
                "class_colors_bgr": {k: list(v) for k, v in segmenter.class_colors.items()},
                "records": record
            }, f, indent=2)
        print(f"  -> Metadata saved to: {meta_path}")
    elif os.path.isdir(args.input):
        segmenter.segment_dataset(args.input, output_dir=args.output_dir)
    else:
        print(f"[Error] Input path '{args.input}' does not exist as a file or directory!")


if __name__ == "__main__":
    main()
