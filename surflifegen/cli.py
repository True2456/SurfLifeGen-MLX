"""
Command Line Interface for SurfLifeGen-MLX
Supports safe auto-incrementing filenames and non-destructive metadata/annotation merging.
"""

import os
import glob
import re
import argparse
import time
import json
from .generator import SurfLifeGenPipeline
from .annotator import PrecisionSwimmerAnnotator
from .prompt_builder import generate_modular_prompt

DEFAULT_PROMPTS = [
    ("nadir_90m_midday_red_vest.png", "Direct overhead nadir photograph looking straight down from 90 meters altitude over open coastal ocean water at a single human swimmer treading water. Swimmer wears a high-visibility red-and-yellow lifeguard rash vest. Clean distinct swimmer silhouette visible against turquoise ocean water, realistic seafoam and swell ripples, high optical resolution aerial photography", 1),
    ("nadir_100m_two_swimmers_orange.png", "Straight-down overhead photograph from 100 meters altitude above sea level showing two swimmers floating side by side in deep azure ocean swell outside breaking sandbar waves. Both swimmers wear high-contrast orange safety rash vests, clear head and shoulder silhouettes separated from white water", 2),
    ("nadir_85m_golden_hour_swimmer.png", "Overhead high-altitude photograph taken from 85 meters height capturing a single person swimming in gentle coastal swell during morning golden hour. Crisp human silhouette treading water against warm solar light reflections on blue seawater", 1),
]

def get_next_start_index(output_dir: str, prefix: str = "bulk_") -> int:
    """
    Scans the output directory for existing files like bulk_0042_alt90m.png
    and returns the next index (e.g. 43) so previous files are never overwritten.
    """
    pattern = os.path.join(output_dir, f"{prefix}*.png")
    files = glob.glob(pattern)
    max_idx = 0
    for fp in files:
        base = os.path.basename(fp)
        match = re.search(r"_(\d{4})_", base)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1

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
    parser.add_argument("--bulk-count", "-n", type=int, default=None, help="Generate X amount of images using modular randomized prompt builder")
    parser.add_argument("--no-annotate", action="store_true", help="Skip automatic YOLO bounding box annotation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = SurfLifeGenPipeline(model_path=args.model_path, auto_download=args.auto_download)

    annotator = None if args.no_annotate else PrecisionSwimmerAnnotator(args.output_dir)

    # Load existing metadata & annotations if present so we never overwrite past work
    meta_file = os.path.join(args.output_dir, "dataset_metadata.json")
    coco_file = os.path.join(args.output_dir, "bounding_boxes.json")

    metadata = []
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = []

    annotations = []
    if os.path.exists(coco_file):
        try:
            with open(coco_file, "r") as f:
                annotations = json.load(f)
        except Exception:
            annotations = []

    if args.bulk_count:
        start_idx = get_next_start_index(args.output_dir, prefix="bulk_")
        end_idx = start_idx + args.bulk_count - 1
        print(f"\n[SurfLifeGen-MLX] Safe Bulk Mode: Generating images #{start_idx:04d} to #{end_idx:04d} (No overwrites)")
        prompts = []
        for idx in range(start_idx, start_idx + args.bulk_count):
            mod = generate_modular_prompt()
            filename = f"bulk_{idx:04d}_alt{mod['altitude_m']}m.png"
            prompts.append((filename, mod["prompt"], mod["swimmer_count"], mod))
    elif args.prompt:
        prompts = [("custom_generation.png", args.prompt, args.swimmer_count, None)]
    else:
        prompts = [(fn, txt, c, None) for fn, txt, c in DEFAULT_PROMPTS]

    for i, (filename, text, count, mod_meta) in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating {filename}...")
        t0 = time.time()
        img = pipe.generate(prompt=text, steps=args.steps)
        out_path = os.path.join(args.output_dir, filename)
        img.save(out_path)
        elapsed = round(time.time() - t0, 2)
        print(f"  -> Saved {out_path} ({elapsed}s)")

        if mod_meta:
            mod_meta["filename"] = filename
            mod_meta["generation_time_sec"] = elapsed
            metadata.append(mod_meta)

        if annotator:
            ann = annotator.annotate_image(out_path, target_count=count)
            annotations.append(ann)
            print(f"  -> Annotated {len(ann['detections'])} swimmer(s): {ann['yolo_label_file']}")

    if metadata:
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

    if annotator and annotations:
        with open(coco_file, "w") as f:
            json.dump(annotations, f, indent=2)
        html_file = annotator.export_html_gallery(annotations)
        print(f"\n[SUCCESS] Combined inspection gallery exported to: {html_file}")

if __name__ == "__main__":
    main()
