"""
Command Line Interface for SurfLifeGen-MLX
"""

import os
import argparse
import time
from .generator import SurfLifeGenPipeline
from .annotator import PrecisionSwimmerAnnotator

DEFAULT_PROMPTS = [
    ("nadir_90m_midday_red_vest.png", "Direct overhead nadir photograph looking straight down from 90 meters altitude over open coastal ocean water at a single human swimmer treading water. Swimmer wears a high-visibility red-and-yellow lifeguard rash vest. Clean distinct swimmer silhouette visible against turquoise ocean water, realistic seafoam and swell ripples, high optical resolution aerial photography", 1),
    ("nadir_100m_two_swimmers_orange.png", "Straight-down overhead photograph from 100 meters altitude above sea level showing two swimmers floating side by side in deep azure ocean swell outside breaking sandbar waves. Both swimmers wear high-contrast orange safety rash vests, clear head and shoulder silhouettes separated from white water", 2),
    ("nadir_85m_golden_hour_swimmer.png", "Overhead high-altitude photograph taken from 85 meters height capturing a single person swimming in gentle coastal swell during morning golden hour. Crisp human silhouette treading water against warm solar light reflections on blue seawater", 1),
]

def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Native Apple Silicon MLX 8-Bit Swimmer Dataset Generator & YOLO Annotator"
    )
    parser.add_argument("--output-dir", "-o", default="./surflife_dataset", help="Output directory for generated images and YOLO labels")
    parser.add_argument("--model-path", "-m", default=None, help="Path to local Cosmos3-Nano-MLX-8bit folder")
    parser.add_argument("--auto-download", action="store_true", default=True, help="Automatically download model from Hugging Face Hub if not found locally")
    parser.add_argument("--steps", "-s", type=int, default=25, help="Number of MLX inference steps per image (default: 25)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Custom prompt to generate a single image")
    parser.add_argument("--swimmer-count", "-c", type=int, default=1, help="Expected swimmer count for custom prompt annotation")
    parser.add_argument("--no-annotate", action="store_true", help="Skip automatic YOLO bounding box annotation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = SurfLifeGenPipeline(model_path=args.model_path, auto_download=args.auto_download)

    annotator = None if args.no_annotate else PrecisionSwimmerAnnotator(args.output_dir)
    annotations = []

    if args.prompt:
        prompts = [("custom_generation.png", args.prompt, args.swimmer_count)]
    else:
        prompts = DEFAULT_PROMPTS

    print(f"\n[SurfLifeGen-MLX] Starting generation of {len(prompts)} image(s) -> {args.output_dir}")
    for i, (filename, text, count) in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating {filename}...")
        t0 = time.time()
        img = pipe.generate(prompt=text, steps=args.steps)
        out_path = os.path.join(args.output_dir, filename)
        img.save(out_path)
        print(f"  -> Saved {out_path} ({time.time()-t0:.1f}s)")

        if annotator:
            ann = annotator.annotate_image(out_path, target_count=count)
            annotations.append(ann)
            print(f"  -> Annotated {len(ann['detections'])} swimmer(s): {ann['yolo_label_file']}")

    if annotator and annotations:
        html_file = annotator.export_html_gallery(annotations)
        print(f"\n[SUCCESS] Inspection gallery exported to: {html_file}")

if __name__ == "__main__":
    main()
