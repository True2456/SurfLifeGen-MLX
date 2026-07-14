#!/usr/bin/env python3
"""
Bulk Synthetic Swimmer Dataset Generator & Automated Annotator
Generates X amount of images using modular prompts (varied altitude, lighting, water, attire)
and automatically outputs YOLO labels + inspection gallery.
Designed so downstream AI models or QA validators can filter/inspect the output batch.
"""

import os
import time
import json
import argparse
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator, generate_modular_prompt

def generate_bulk(num_images: int, output_dir: str, steps: int = 25, model_path: str = None, auto_download: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    print("=========================================================================")
    print("  SURFLIFEGEN-MLX: BULK MODULAR SYNTHETIC DATASET GENERATOR")
    print(f"  Target Count : {num_images} Images")
    print(f"  Output Folder: {output_dir}")
    print(f"  MLX Steps    : {steps} steps/image")
    print("=========================================================================")

    pipe = SurfLifeGenPipeline(model_path=model_path, auto_download=auto_download)
    annotator = PrecisionSwimmerAnnotator(output_dir=output_dir)

    metadata = []
    annotations = []
    total_t0 = time.time()

    for i in range(1, num_images + 1):
        mod = generate_modular_prompt()
        filename = f"bulk_{i:04d}_alt{mod['altitude_m']}m.png"
        filepath = os.path.join(output_dir, filename)

        print(f"\n---> [{i:04d}/{num_images:04d}] Generating: {filename}")
        print(f"     Altitude: {mod['altitude_m']}m | Swimmers: {mod['swimmer_count']} | Attire: {mod['attire'][:30]}...")

        t0 = time.time()
        img = pipe.generate(prompt=mod["prompt"], width=1024, height=768, steps=steps)
        img.save(filepath)
        elapsed = time.time() - t0

        ann = annotator.annotate_image(filepath, target_count=mod["swimmer_count"])
        print(f"     [Saved & Annotated in {elapsed:.1f}s] -> {len(ann['detections'])} box(es)")

        mod["filename"] = filename
        mod["generation_time_sec"] = round(elapsed, 2)
        metadata.append(mod)
        annotations.append(ann)

    meta_file = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    coco_file = os.path.join(output_dir, "bounding_boxes.json")
    with open(coco_file, "w") as f:
        json.dump(annotations, f, indent=2)

    html_file = annotator.export_html_gallery(annotations)

    print("\n=========================================================================")
    print(f"[SUCCESS] Bulk generation of {num_images} images completed in {time.time()-total_t0:.1f}s!")
    print(f"          YOLO Labels  : {os.path.join(output_dir, 'labels')}")
    print(f"          Gallery HTML : {html_file}")
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
        auto_download=not args.no_auto-download if hasattr(args, "no_auto_download") else True
    )
