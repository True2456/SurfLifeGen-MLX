# SurfLifeGen-MLX

High-Altitude Synthetic Aerial Maritime & Search and Rescue (SAR) Dataset Generator with **Zero-Shot Option A Grounding DINO Bounding Box Annotation**, **Strict Zero-Overwrite Safeguards**, and **Native Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni / Qwen3-VL Verification**.

---

## Overview

SurfLifeGen-MLX is a state-of-the-art synthetic data generation and automated annotation pipeline designed for Search and Rescue (SAR), Lifeguard operations, and Shark Patrol drone vision models. It runs natively on Apple Silicon (M1 through M5 Max/Ultra) using optimized `mlx` kernels to synthesize photorealistic aerial nadir maritime imagery at patrol altitudes (**80 to 400 meters**) for both **Active Swimmers** and **Submerged Sharks**.

### ✨ New Upgrades & Capabilities

- **Option A Grounding DINO Zero-Shot Auto-Annotation:** Replaced fragile color heuristics with open-vocabulary `Grounding DINO` (`IDEA-Research/grounding-dino-base`) running natively on PyTorch `MPS`. Automatically outputs both exact YOLO annotations (`bounding_boxes.json`) and visual debug frames (`_annotated.png` / `_dino.png`) for every generated image.
- **Strict Zero-Overwrite Guarantee:** All generation modes automatically scan your output directory (`./custom_highalt_swimmers`, `./High_Alt_Shark`), identify existing filenames (`_0001.png`, `_0002.png`), and dynamically increment counters right before saving so no dataset file or annotation is ever accidentally overwritten.
- **Upgraded Active Swimming & Anatomical Defaults:** Default swimming prompts (`--target swimmer`) are weighted toward **100 meters nadir altitude** with strict anatomical and submersion safety rules injected into every scene: *"Each swimmer is partially submerged. Make sure humans have no extra limbs."*
- **High-Altitude Active Swimmer Simulation (`surflifegen-highalt`):** Dedicated CLI and physics scaling engine for simulating dynamic swimming kinematics (`freestyle with arm splashing`, `breaststroke kicks`, `treading water`, `water ripples`) scaled to exact field-of-view trigonometric pixel ratios for `100m–400m` altitudes.
- **Custom Prompt + Bulk Generation (`--bulk-count X --prompt "..."`):** Seamlessly pass custom target prompts into bulk mode to generate 50+ variations of your exact prompt while automatically auto-annotating and incrementing filenames across both Swimmer and Shark modes.
- **Custom Grounding DINO Queries (`--detection-prompt` / `-d`):** Full open-vocabulary control allowing you to specify custom dot-separated detection targets on the fly (`'submerged shark . shark in water . marine predator silhouette .'`).
- **Adjustable Detection Sensitivity:** Fine-tune confidence thresholds (`--box-thresh`, `--text-thresh`, `--nms-thresh`) right from the command line to reliably localize faint, tiny, or deeply submerged targets.

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

This installs six global command-line tools into your environment:
1. `surflifegen`: Main Cosmos 3 MLX dataset generator with Grounding DINO annotation (`--target swimmer` | `--target shark`).
2. `surflifegen-highalt`: Specialized high-altitude (`100m-400m`) active swimmer scene synthesizer.
3. `surflifegen-highway`: Dedicated highway pavement defect generator with **Grounded-SAM** precision polygon mask segmentation.
4. `surflifegen-segment`: Standalone Grounded-SAM (`Grounding DINO + Segment Anything`) zero-shot polygon mask segmenter for any existing road/defect dataset.
5. `surflifegen-yolo-export`: One-click dataset exporter converting synthetic/real annotations into Ultralytics `YOLOv8`/`YOLOv11` ready formats (`images/`, `labels/`, `data.yaml`) for both real-time Object Detection and Instance Segmentation.
6. `surflifegen-verify`: Apple Silicon MLX Qwen3-VL automated bounding box audit & correction tool.

---

## Command-Line Usage & Recipes

### 1. Highway Defect & Surface Wear Inspection Dataset (`surflifegen-highway`)

Generate synthetic high-resolution pavement inspection datasets (cracks, potholes, alligatoring, rutting, and degraded road paint) with automatic zero-shot Grounding DINO localization:

#### Alligator Fatigue Cracking (Nadir Drone Perspective):
```bash
surflifegen-highway \
  --defect-type alligator_crack \
  --asphalt "weathered grey oxidized asphalt" \
  --perspective nadir_drone \
  --count 50 \
  --box-thresh 0.18 \
  --output-dir ./highway_alligator_dataset \
  --steps 25
```

#### Potholes & Structural Cavities (Vehicle-Mounted Surface Perspective):
```bash
surflifegen-highway \
  --defect-type pothole \
  --perspective vehicle_surface \
  --count 100 \
  --detection-prompt "pothole . road crater . pavement hole . broken asphalt ." \
  --output-dir ./highway_pothole_dataset
```

---

### 2. Custom Prompt Submerged Shark Generation (`--target shark` + `--detection-prompt`)

Generate custom high-altitude aerial patrol photographs of submerged marine predators using your exact custom prompt (`--prompt`), while explicitly setting `--target shark` and passing a matching Grounding DINO open-vocabulary query (`--detection-prompt`):

```bash
surflifegen \
  --target shark \
  --prompt "Direct overhead flying nadir photograph looking straight down from high altitude showing 1-3 submerged sharks cruising underwater beneath coastal ocean water. Distinct shark silhouette visible below the waters surface. Water clarity should be random. Random Time of day but not night time" \
  --steps 40 \
  --bulk-count 1 \
  --box-thresh 0.16 \
  --nms-thresh 0.30 \
  --detection-prompt "submerged shark . shark in water . marine predator silhouette ." \
  --output-dir ./High_Alt_Shark
```

> [!IMPORTANT]
> **Why `--target` and `--detection-prompt` must match:**
> If you pass a custom `--prompt` describing sharks, you MUST also specify `--target shark` (otherwise `surflifegen` defaults to `--target swimmer`). Furthermore, your `--detection-prompt` must query for sharks (`"submerged shark . marine predator silhouette ."`). If `--target swimmer` or a swimmer detection query is used on a shark photograph, Grounding DINO will search the shark photo for human swimmers and return `0 detections`.

---

### 2. Custom Prompt Swimmer Generation (`--target swimmer` + `--detection-prompt`)

Generate `50` custom active swimming scenes at `100m` altitude using an exact prompt, while tuning Grounding DINO to detect faint submerged swimmers and custom concepts (`swimmer`, `person splashing`, `lifeguard vest`):

```bash
surflifegen \
  --target swimmer \
  --prompt "Direct overhead nadir photograph from 100 meters altitude above sea level showing multiple human swimmers actively swimming or splashing across coastal ocean water. Each swimmer is partially submerged. Make sure humans have no extra limbs. Clean distinct human silhouettes visible against coastal ocean swell and seafoam, high optical resolution aerial photography" \
  --steps 25 \
  --bulk-count 50 \
  --box-thresh 0.16 \
  --nms-thresh 0.30 \
  --detection-prompt "swimmer . person splashing . human silhouette in water . lifeguard vest ." \
  --output-dir ./custom_highalt_swimmers
```

---

### 3. Fully Automated Randomized Dataset Generation (Omitting `--prompt`)

When you omit `--prompt` and simply specify `--bulk-count X`, `surflifegen` activates its **Modular Randomized Prompt Engine**. Before generating each individual image, Python dynamically constructs a unique, concrete scene with randomized quantities:
- **Randomized Target Counts:** e.g., 1, 2, 3, or 4 sharks/swimmers per shot.
- **Randomized Altitudes:** e.g., 60m, 80m, 100m, or 120m above sea level.
- **Randomized Lighting & Time of Day:** e.g., midday bright sun, morning golden light, afternoon glare.
- **Randomized Water Clarity:** e.g., crystal clear turquoise, slight coastal turbidity, deep oceanic blue.

#### Bulk Randomized Swimmer Dataset:
```bash
surflifegen --target swimmer --bulk-count 100 --output-dir ./bulk_swimmer_dataset --steps 25
```

#### Bulk Randomized Submerged Shark Dataset:
```bash
surflifegen --target shark --bulk-count 50 --output-dir ./shark_aerial_dataset --steps 25
```

> [!TIP]
> **Prompt Wording vs. Automatic Randomization:**
> Text-to-image AI vision models (like Cosmos 3) do not have a built-in random number generator. If you type meta-instructions like `"Water clarity should be random. Random Time of day"` into `--prompt`, the neural network literally tries to visually paint what abstract words look like—which can lead to weird fog or text artifacts.
> - **If using `--prompt`:** Provide exact, concrete visual descriptions (`"two submerged sharks beneath clear turquoise water under midday sun"`).
> - **If you want randomization across dozens of images:** Omit `--prompt` and let `surflifegen --bulk-count X` automatically generate distinct, concrete prompt variations for each frame!

---

### 4. High-Altitude Active Swimmer Simulation (`surflifegen-highalt`)

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
| `--box-threshold` | `--box-thresh` | `0.22` | Grounding DINO box confidence threshold (lower = more sensitive, e.g. `0.16`) |
| `--text-threshold` | `--text-thresh` | `0.22` | Grounding DINO text prompt matching threshold |
| `--nms-threshold` | `--iou-thresh` | `0.30` | Non-Maximum Suppression (NMS) IoU threshold for overlapping boxes |
| `--detection-prompt` | `-d` | `None` | Custom dot-separated queries for Grounding DINO (e.g. `'submerged shark . shark in water .'`) |
| `--no-annotate` | | `False` | Skip Grounding DINO auto-annotation pass |

### `surflifegen-highway` Options:

| Flag | Shortcut | Default | Description |
| :--- | :--- | :--- | :--- |
| `--defect-type` | `-t` | `random` | Defect class: `alligator_crack`, `longitudinal_crack`, `transverse_crack`, `pothole`, `rutting`, `faded_lane_marking`, `edge_break`, `mixed`, `random` |
| `--asphalt` | `-a` | `random` | Surface type description (e.g. `'weathered grey asphalt'`, `'dark newly paved'`) |
| `--perspective` | | `random` | Camera inspection view: `nadir_drone`, `low_nadir`, `vehicle_surface`, `random` |
| `--count` | `-c` | `5` | Number of highway defect scenes to generate |
| `--bulk-count` | `-n` | `None` | Alias for `--count` when running automated generation loops |
| `--box-threshold` | `--box-thresh` | `0.22` | Grounded-SAM / Grounding DINO localization confidence threshold |
| `--detection-prompt` | `-d` | `None` | Custom dot-separated queries (`'crack in asphalt . pothole . road edge .'`) |
| `--boxes-only` | `--no-sam` | `False` | Use coarse Grounding DINO rectangular bounding boxes instead of SAM precision polygon masks |
| `--no-annotate` | | `False` | Skip auto-annotation/segmentation pass |

### `surflifegen-segment` Options (Grounded-SAM Standalone Segmenter):

| Flag | Shortcut | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset-dir` | `-d` | Required | Path to directory containing road/defect images to segment |
| `--prompt` | `-p` | `'crack in asphalt . pothole . road edge .'` | Dot-separated localization queries before SAM mask generation |
| `--box-threshold` | `--box-thresh` | `0.22` | Detection confidence threshold before mask creation |
| `--sam-model-id` | | `'facebook/sam-vit-base'` | HuggingFace model ID for Segment Anything (running natively on Apple Silicon MPS) |

### `surflifegen-yolo-export` Options (Ultralytics YOLOv8/YOLOv11 Dataset Exporter):

| Flag | Shortcut | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset-dir` | `-d` | Required | Path to input directory (`surflifegen`, `surflifegen-highalt`, or `surflifegen-highway`) |
| `--output-dir` | `-o` | `./yolo_dataset` | Destination folder for `images/`, `labels/`, and `data.yaml` |
| `--split-ratio` | `-s` | `0.8` | Train vs Validation split ratio (e.g. `0.8` = 80% train / 20% val) |
| `--mode` | `-m` | `auto` | `detect` (boxes), `segment` (polygon masks), or `auto` |
| `--symlink` | | `False` | Use symlinks instead of copying `.png` images to conserve disk space |

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

# 2. Generate a Submerged Shark Scene
config = generate_modular_prompt(target_type="shark")
print("Generated Prompt:", config["prompt"])

image = pipeline.generate(prompt=config["prompt"], width=1024, height=768, steps=40)
image_path = "./output/shark_scene.png"
image.save(image_path)

# 3. Zero-Shot Open-Vocabulary Localization
detections, summary = annotator.annotate_image(
    image_path,
    output_path="./output/shark_scene_annotated.png",
    target_type="shark",
    detection_prompt="submerged shark . shark in water . marine predator silhouette ."
)
print("Annotation Summary:", summary)
print("Boxes ([x1, y1, x2, y2]):", [d["box"] for d in detections])
```

---

## License

MIT License
