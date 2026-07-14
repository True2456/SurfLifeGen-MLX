# SurfLifeGen-MLX

High-Altitude Synthetic Aerial Maritime & Search and Rescue (SAR) Dataset Generator with **Zero-Shot Option A Grounding DINO Bounding Box Annotation**, **Strict Zero-Overwrite Safeguards**, and **Native Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni / Qwen3-VL Verification**.

---

## Overview

SurfLifeGen-MLX is a state-of-the-art synthetic data generation and automated annotation pipeline designed for Search and Rescue (SAR), Lifeguard operations, and Shark Patrol drone vision models. It runs natively on Apple Silicon (M1 through M5 Max/Ultra) using optimized `mlx` kernels to synthesize photorealistic aerial nadir maritime imagery at patrol altitudes (**80 to 400 meters**) for both **Active Swimmers** and **Submerged Sharks**.

### ✨ New Upgrades & Capabilities

- **Option A Grounding DINO Zero-Shot Auto-Annotation:** Replaced fragile color heuristics with open-vocabulary `Grounding DINO` (`IDEA-Research/grounding-dino-base`) running natively on PyTorch `MPS`. Automatically outputs both exact YOLO annotations (`bounding_boxes.json`) and visual debug frames (`_annotated.png` / `_dino.png`) for every generated image.
- **Strict Zero-Overwrite Guarantee:** All generation modes automatically scan your output directory (`./custom_highalt_swimmers`), identify existing filenames (`_0001.png`, `_0002.png`), and dynamically increment counters right before saving so no dataset file or annotation is ever accidentally overwritten.
- **Upgraded Active Swimming & Anatomical Defaults:** Default swimming prompts (`--target swimmer`) are weighted toward **100 meters nadir altitude** with strict anatomical and submersion safety rules injected into every scene: *"Each swimmer is partially submerged. Make sure humans have no extra limbs."*
- **High-Altitude Active Swimmer Simulation (`surflifegen-highalt`):** Dedicated CLI and physics scaling engine for simulating dynamic swimming kinematics (`freestyle with arm splashing`, `breaststroke kicks`, `treading water`, `water ripples`) scaled to exact field-of-view trigonometric pixel ratios for `100m–400m` altitudes.
- **Custom Prompt + Bulk Generation (`--bulk-count X --prompt "..."`):** Seamlessly pass custom target prompts into bulk mode to generate 50+ variations of your exact prompt while automatically auto-annotating and incrementing filenames.
- **Custom Grounding DINO Queries (`--detection-prompt` / `-d`):** Full open-vocabulary control allowing you to specify custom dot-separated detection targets on the fly (e.g., `'swimmer . person splashing . lifeguard vest .'`).
- **Adjustable Detection Sensitivity:** Fine-tune confidence thresholds (`--box-thresh`, `--text-thresh`, `--nms-thresh`) right from the command line to reliably localize faint, tiny, or deeply submerged swimmers.

---

## System Requirements

- macOS running on Apple Silicon (M1/M2/M3/M4/M5 architecture, 32GB+ Unified Memory recommended)
- Python 3.10 or later
- Apple MLX framework (`mlx >= 0.31.0`)
- PyTorch with `mps` backend (`torch`, `torchvision`) for Grounding DINO zero-shot localization
- Transformers & Hugging Face Hub (`transformers >= 4.48.0`)

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/True2456/SurfLifeGen-MLX.git
cd SurfLifeGen-MLX
pip install -e .
```

This installs three global command-line tools into your environment:
1. `surflifegen`: Main Cosmos 3 MLX dataset generator with Grounding DINO annotation.
2. `surflifegen-highalt`: Specialized high-altitude (`100m-400m`) active swimmer scene synthesizer.
3. `surflifegen-verify`: Apple Silicon MLX Qwen3-VL automated bounding box audit & correction tool.

---

## Command-Line Usage

### 1. Custom Prompt Bulk Generation with High-Sensitivity DINO Detection

Generate `50` custom active swimming scenes at `100m` altitude using an exact prompt, while tuning Grounding DINO to detect faint submerged swimmers and custom concepts (`swimmer`, `person splashing`, `lifeguard vest`):

```bash
surflifegen \
  --prompt "Direct overhead nadir photograph from 100 meters altitude above sea level showing multiple human swimmers actively swimming or splashing across coastal ocean water. Each swimmer is partially submerged. Make sure humans have no extra limbs. Clean distinct human silhouettes visible against coastal ocean swell and seafoam, high optical resolution aerial photography" \
  --steps 25 \
  --bulk-count 50 \
  --box-thresh 0.16 \
  --nms-thresh 0.30 \
  --detection-prompt "swimmer . person splashing . human silhouette in water . lifeguard vest ." \
  --output-dir ./custom_highalt_swimmers
```

### 2. Bulk Swimmer Dataset Generation (Default Modular Engine)

Generate `100` randomized active swimmer scenes (freestyle, breaststroke, splashing, rip currents) using the upgraded modular prompt engine (defaulting around `100m` altitude and strict anatomical rules):

```bash
surflifegen --target swimmer --bulk-count 100 --output-dir ./bulk_swimmer_dataset --steps 25
```

### 3. Bulk Submerged Shark Patrol Dataset (`--target shark`)

Generate synthetic aerial patrol photographs of submerged marine predators cruising `1–3m` below turquoise water over sandy sea floors or drop-offs:

```bash
surflifegen --target shark --bulk-count 50 --output-dir ./shark_aerial_dataset --steps 25
```

### 4. High-Altitude Active Swimmer Synthesis (`surflifegen-highalt`)

Simulate high-altitude aerial drone passes (`100m–400m`) with exact trigonometric swimmer pixel scaling, active arm/leg stroke sprites, and automatic zero-shot Grounding DINO localization:

```bash
surflifegen-highalt \
  --altitude 120 \
  --swimmers 4 \
  --count 20 \
  --box-thresh 0.18 \
  --detection-prompt "swimmer . person floating . splash ." \
  --output-dir ./highalt_120m_dataset
```

---

## CLI Options & Detection Sensitivity Tuning

### `surflifegen` Options:

| Flag | Shortcut | Default | Description |
| :--- | :--- | :--- | :--- |
| `--target` | `-t` | `swimmer` | Target class to synthesize: `swimmer` or `shark` |
| `--output-dir` | `-o` | `./surflife_dataset` | Output directory for generated images and bounding box JSON |
| `--steps` | `-s` | `25` | Number of MLX inference steps per image |
| `--prompt` | `-p` | `None` | Custom text prompt for generation (overrides modular template) |
| `--bulk-count` | `-n` | `None` | Generate `X` images automatically. Works with or without `--prompt` |
| `--box-threshold` | `--box-thresh` | `0.22` | Grounding DINO box confidence threshold (lower = more sensitive) |
| `--text-threshold` | `--text-thresh` | `0.22` | Grounding DINO text prompt matching threshold |
| `--nms-threshold` | `--iou-thresh` | `0.30` | Non-Maximum Suppression (NMS) IoU threshold for overlapping boxes |
| `--detection-prompt` | `-d` | `None` | Custom dot-separated queries for Grounding DINO (e.g. `'swimmer . splash .'`) |
| `--no-annotate` | | `False` | Skip Grounding DINO auto-annotation pass |

---

## Python API Usage

You can integrate `SurfLifeGen-MLX` directly into your custom Python training scripts:

```python
from surflifegen import SurfLifeGenPipeline, GroundingDinoAnnotator, generate_modular_prompt

# 1. Initialize MLX Quantized Cosmos 3 Pipeline & Grounding DINO on Apple Silicon MPS
pipeline = SurfLifeGenPipeline(auto_download=True)
annotator = GroundingDinoAnnotator(
    box_threshold=0.16,
    text_threshold=0.18,
    nms_iou_threshold=0.30
)

# 2. Generate a 100m Active Swimming Scene
config = generate_modular_prompt()
print("Generated Prompt:", config["prompt"])

image = pipeline.generate(prompt=config["prompt"], width=1024, height=768, steps=25)
image_path = "./output/swimmer_scene.png"
image.save(image_path)

# 3. Zero-Shot Open-Vocabulary Localization
detections, summary = annotator.annotate_image(
    image_path,
    output_path="./output/swimmer_scene_annotated.png",
    target_type="swimmer",
    detection_prompt="swimmer . person splashing . human silhouette ."
)
print("Annotation Summary:", summary)
print("Boxes ([x1, y1, x2, y2]):", [d["box"] for d in detections])
```

---

## License

MIT License
