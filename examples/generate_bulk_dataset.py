#!/usr/bin/env python3
"""
Bulk Synthetic Swimmer Dataset Generator & Automated Annotator
Generates X amount of images using modular prompts (varied altitude, lighting, water, attire)
and automatically outputs YOLO labels + inspection gallery.
Includes safe auto-incrementing filenames and non-destructive metadata merging.
"""

import os
import glob
import re
import time
import json
import argparse
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator, generate_modular_prompt

def get_next_start_index(output_dir: str, prefix: str = "bulk_") -> int:
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

def generate_bulk(num_images: int, output_dir: str, steps: int = 25, model_path: str = None, auto_download: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    start_idx = get_next_start_index(output_dir, prefix="bulk_")
    end_idx = start_idx + num_images - 1

    print("=========================================================================")
    print("  SURFLIFEGEN-MLX: SAFE BULK MODULAR SYNTHETIC DATASET GENERATOR")
    print(f"  Batch Range  : #{start_idx:04d} to #{end_idx:04d} ({num_images} Images • No Overwrites)")
    print(f"  Output Folder: {output_dir}")
    print(f"  MLX Steps    : {steps} steps/image")
    print("=========================================================================")

    pipe = SurfLifeGenPipeline(model_path=model_path, auto_download=auto_download)
    annotator = PrecisionSwimmerAnnotator(output_dir=output_dir)

    meta_file = os.path.join(output_dir, "dataset_metadata.json")
    coco_file = os.path.join(output_dir, "bounding_boxes.json")

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

    total_t0 = time.time()

    for idx in range(start_idx, start_idx + num_images):
        mod = generate_modular_prompt()
        filename = f"bulk_{idx:04d}_alt{mod['altitude_m']}m.png"
        filepath = os.path.join(output_dir, filename)

        print(f"\n---> [Index #{idx:04d}] Generating: {filename}")
        print(f"     Altitude: {mod['altitude_m']}m | Swimmers: {mod['swimmer_count']} | Attire: {mod['attire'][:30]}...")

        t0 = time.time()
        img = pipe.generate(prompt=mod["prompt"], width=1024, height=768, steps=steps)
        img.save(filepath)
        elapsed = round(time.time() - t0, 2)

        ann = annotator.annotate_image(filepath, target_count=mod["swimmer_count"])
        print(f"     [Saved & Annotated in {elapsed:.1f}s] -> {len(ann['detections'])} box(es)")

        mod["filename"] = filename
        mod["generation_time_sec"] = elapsed
        metadata.append(mod)
        annotations.append(ann)

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(coco_file, "w") as f:
        json.dump(annotations, f, indent=2)

    html_file = annotator.export_html_gallery(annotations)

    print("\n=========================================================================")
    print(f"[SUCCESS] Safe bulk generation of {num_images} images completed in {time.time()-total_t0:.1f}s!")
    print(f"          Total Dataset Size : {len(annotations)} Images")
    print(f"          YOLO Labels Folder : {os.path.join(output_dir, 'labels')}")
    print(f"          Combined Gallery   : {html_file}")
    print("=========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Modular Swimmer Dataset Generator")
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--output-dir", "-o", default="./bulk_swimmer_dataset", help="Output directory")
    parser.add_argument("--steps", "-s", type=int, default=25, help="MLX inference steps per image")
    parser.add_argument("--model-path", "-m", default=None, help="Local path to Cosmos3-Nano-MLX-8bit folder")
    parser.add_argument("--no-auto-download", action="store_true", help="Disable auto-downloading model")

    args = parser.parse_args()
    generate_bulk(
        num_images=args.count,
        output_dir=args.output_dir,
        steps=args.steps,
        model_path=args.model_path,
        auto_download=not args.no_auto_download if hasattr(args, "no_auto_download") else True
    )
