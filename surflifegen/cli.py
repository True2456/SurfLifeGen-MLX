"""
Command Line Interface for SurfLifeGen-MLX
Supports safe auto-incrementing filenames, real-time per-image metadata saving,
and exhaustive multi-target annotation (--target swimmer OR --target shark).
"""

import os
import glob
import re
import argparse
import time
import json
from .generator import SurfLifeGenPipeline
from .dino_annotator import GroundingDinoAnnotator, DINO_AVAILABLE
from .prompt_builder import generate_modular_prompt
from .shark_prompt_builder import generate_shark_prompt

DEFAULT_PROMPTS = [
    ("nadir_100m_active_swimmers_default.png", "Direct overhead nadir photograph from 100 meters altitude above sea level showing multiple human swimmers actively swimming or splashing across coastal ocean water. Each swimmer is partially submerged. Make sure humans have no extra limbs. Clean distinct human silhouettes visible against coastal ocean swell and seafoam, high optical resolution aerial photography", 3),
    ("nadir_100m_two_swimmers_orange.png", "Straight-down overhead photograph from 100 meters altitude above sea level showing two swimmers floating side by side in deep azure ocean swell outside breaking sandbar waves. Both swimmers wear high-contrast orange safety rash vests, clear head and shoulder silhouettes separated from white water", 2),
    ("nadir_85m_golden_hour_swimmer.png", "Overhead high-altitude photograph taken from 85 meters height capturing a single person swimming in gentle coastal swell during morning golden hour. Crisp human silhouette treading water against warm solar light reflections on blue seawater", 1),
]

def get_next_start_index(output_dir: str, prefix: str = "bulk_") -> int:
    pattern = os.path.join(output_dir, f"{prefix}*.png")
    files = glob.glob(pattern)
    max_idx = 0
    for fp in files:
        base = os.path.basename(fp)
        match = re.search(r"_(\d+)", base) or re.search(r"(\d+)", base)
        if match:
            try:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return max_idx + 1

def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Native Apple Silicon MLX 8-Bit Swimmer & Submerged Shark Dataset Generator"
    )
    parser.add_argument("--target", "-t", choices=["swimmer", "shark"], default="swimmer", help="Target class to synthesize: 'swimmer' or 'shark'")
    parser.add_argument("--output-dir", "-o", default="./surflife_dataset", help="Output directory for generated images and YOLO labels")
    parser.add_argument("--model-path", "-m", default=None, help="Path to local Cosmos3-Nano-MLX-8bit folder")
    parser.add_argument("--auto-download", action="store_true", default=True, help="Automatically download model from Hugging Face Hub if not found locally")
    parser.add_argument("--steps", "-s", type=int, default=25, help="Number of MLX inference steps per image (default: 25)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Custom prompt to generate a single image")
    parser.add_argument("--count", "-c", type=int, default=100, help="Max target count for custom prompt annotation")
    parser.add_argument("--bulk-count", "-n", type=int, default=None, help="Generate X amount of images using modular randomized prompt builder")
    parser.add_argument("--no-annotate", action="store_true", help="Skip automatic YOLO bounding box annotation")
    parser.add_argument("--box-threshold", "--box-thresh", type=float, default=0.22, help="Detection sensitivity box threshold for Grounding DINO (lower = more sensitive, default: 0.22)")
    parser.add_argument("--text-threshold", "--text-thresh", type=float, default=0.22, help="Detection sensitivity text prompt threshold for Grounding DINO (default: 0.22)")
    parser.add_argument("--nms-threshold", "--iou-threshold", type=float, default=0.30, help="NMS IoU threshold for overlapping box removal (default: 0.30)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = SurfLifeGenPipeline(model_path=args.model_path, auto_download=args.auto_download)

    annotator = None
    if not args.no_annotate and DINO_AVAILABLE:
        try:
            print(f"\n[SurfLifeGen-MLX] Initializing Option A: Grounding DINO Zero-Shot Auto-Annotator (BoxThresh={args.box_threshold}, TextThresh={args.text_threshold}, NMS={args.nms_threshold})...")
            annotator = GroundingDinoAnnotator(box_threshold=args.box_threshold, text_threshold=args.text_threshold, nms_iou_threshold=args.nms_threshold)
        except Exception as e:
            print(f"[SurfLifeGen-MLX] Warning: Grounding DINO could not be initialized: {e}")

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
        prefix = f"bulk_{args.target}_"
        start_idx = get_next_start_index(args.output_dir, prefix=prefix)
        end_idx = start_idx + args.bulk_count - 1
        print(f"\n[SurfLifeGen-MLX] Safe Bulk Mode ({args.target.upper()}S): Generating images #{start_idx:04d} to #{end_idx:04d}")
        prompts = []
        for idx in range(start_idx, start_idx + args.bulk_count):
            if args.target == "shark":
                mod = generate_shark_prompt()
            else:
                mod = generate_modular_prompt()
            
            prompt_text = args.prompt if args.prompt else mod["prompt"]
            filename = f"{prefix}{idx:04d}_alt{mod['altitude_m']}m.png"
            target_cnt = mod.get("shark_count", mod.get("swimmer_count", 100))
            prompts.append((filename, prompt_text, target_cnt, mod))
    elif args.prompt:
        idx = get_next_start_index(args.output_dir, prefix="custom_")
        prompts = [(f"custom_{idx:04d}.png", args.prompt, args.count, None)]
    else:
        prompts = []
        for fn, txt, c in DEFAULT_PROMPTS:
            base_prefix = fn.split(".")[0] + "_"
            idx = get_next_start_index(args.output_dir, prefix=base_prefix)
            prompts.append((f"{base_prefix}{idx:04d}.png", txt, c, None))

    for i, (filename, text, count, mod_meta) in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating {filename} ({args.target.upper()})...")
        t0 = time.time()
        img = pipe.generate(prompt=text, steps=args.steps)
        out_path = os.path.join(args.output_dir, filename)
        
        # Strict anti-overwrite guarantee: increment counter if file exists
        counter = 1
        while os.path.exists(out_path) or os.path.exists(out_path.replace(".png", "_annotated.png")):
            base, ext = os.path.splitext(filename)
            match = re.search(r"_(\d+)$", base)
            if match:
                new_num = int(match.group(1)) + 1
                base = re.sub(r"_(\d+)$", f"_{new_num:04d}", base)
            else:
                base = f"{base}_{counter:04d}"
            filename = f"{base}{ext}"
            out_path = os.path.join(args.output_dir, filename)
            counter += 1

        img.save(out_path)
        elapsed = round(time.time() - t0, 2)
        print(f"  -> Saved {out_path} ({elapsed}s)")

        if mod_meta:
            mod_meta["filename"] = filename
            mod_meta["generation_time_sec"] = elapsed
            metadata.append(mod_meta)

        if annotator:
            annotated_path = out_path.replace(".png", "_annotated.png")
            detections, summary = annotator.annotate_image(out_path, output_path=annotated_path, target_type=args.target)
            ann_record = {
                "image_file": filename,
                "annotated_file": os.path.basename(annotated_path),
                "target_type": args.target,
                "detections": detections,
                "summary": summary
            }
            annotations.append(ann_record)
            print(f"  -> {summary} ({os.path.basename(annotated_path)})")

        if metadata:
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)
        if annotator and annotations:
            with open(coco_file, "w") as f:
                json.dump(annotations, f, indent=2)

if __name__ == "__main__":
    main()
