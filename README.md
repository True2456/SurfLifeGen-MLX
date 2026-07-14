# SurfLifeGen-MLX

High-Altitude Synthetic Aerial Maritime & Surf Life Saving Dataset Generator with Precision Automated YOLO Bounding Box Annotation, powered by native Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni.

---

## Overview

SurfLifeGen-MLX is a specialized synthetic data generation and automated annotation pipeline designed for Search and Rescue (SAR) drone vision models. It runs natively on Apple Silicon (M1 through M5 Max/Ultra) using optimized `mlx` attention kernels to synthesize realistic aerial nadir maritime imagery at patrol altitudes (40 to 100 meters) with low unified memory consumption (~12 GB).

### Key Features

- **Apple Silicon Native Inference:** Built on Apple MLX 8-bit quantization for high-speed local execution (~25 seconds per 1024x768 image on M-series Max chips).
- **Modular Prompt Engine:** Dynamically generates high-variance coastal search-and-rescue scenarios (altitude, lighting, water conditions, rescue attire) while eliminating aircraft noun hallucinations.
- **Precision Automated Annotation:** Segment swimmers from complex ocean backgrounds using chromatic saliency and seafoam suppression, outputting normalized YOLO format bounding boxes (`labels/*.txt`).
- **Interactive Inspection Tools:** Generates visual bounding box previews and a standalone browser inspection gallery (`annotated_gallery.html`).
- **Flexible Model Resolution:** Accepts an explicit local path to pre-downloaded weights or automatically fetches and caches model files from Hugging Face Hub.

---

## System Requirements

- macOS running on Apple Silicon (M1/M2/M3/M4/M5 architecture)
- Python 3.10 or later
- Apple MLX framework (`mlx >= 0.22.0`)

---

## Installation

### Method 1: Python Package Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/True2456/SurfLifeGen-MLX.git
cd SurfLifeGen-MLX
pip install -e .
```

### Method 2: Homebrew Installation

Install directly using the provided Homebrew formula:

```bash
brew install --build-from-source https://raw.githubusercontent.com/True2456/SurfLifeGen-MLX/main/Formula/surflifegen.rb
```

---

## Command-Line Usage

Once installed, the `surflifegen` command is globally available.

### 1. Bulk Modular Dataset Generation

Generate an arbitrary number of synthetic images with varied altitudes (40m to 100m), water states, and rescue attire, automatically exporting YOLO label files:

```bash
surflifegen --bulk-count 100 --output-dir ./bulk_swimmer_dataset --steps 25
```

### 2. Specifying Local Model Path vs. Automatic Download

By default, `surflifegen` checks local search paths and automatically fetches model weights from Hugging Face Hub (`True2456/Cosmos3-Nano-MLX-8bit`) if not found.

To specify an explicit local model path:

```bash
surflifegen \
  --model-path /path/to/Cosmos3-Nano-MLX-8bit \
  --bulk-count 50 \
  --output-dir ./dataset_output
```

### 3. Single Custom Prompt Generation

Generate and annotate a single specific prompt:

```bash
surflifegen \
  --prompt "Direct overhead nadir photograph looking straight down from 90 meters altitude over coastal ocean swell showing a lone swimmer wearing a high-visibility red lifeguard vest" \
  --swimmer-count 1 \
  --output-dir ./custom_output
```

---

## Python API Usage

You can integrate SurfLifeGen-MLX directly into Python data pipelines or notebooks:

```python
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator, generate_modular_prompt

# Initialize pipeline (resolves local weights or downloads automatically)
pipeline = SurfLifeGenPipeline(auto_download=True)
annotator = PrecisionSwimmerAnnotator(output_dir="./output_dataset")

# Generate a randomized modular search-and-rescue prompt configuration
config = generate_modular_prompt()
print("Generated Prompt:", config["prompt"])

# Generate image
image = pipeline.generate(prompt=config["prompt"], width=1024, height=768, steps=25)
image_path = "./output_dataset/sample_swimmer.png"
image.save(image_path)

# Perform precision swimmer bounding box detection
annotation = annotator.annotate_image(image_path, target_count=config["swimmer_count"])
print("YOLO Annotation File:", annotation["yolo_label_file"])
```

---

## Dataset Output Format

Each dataset run produces a structured directory ready for YOLO model training or automated validation:

```text
dataset_output/
├── *.png                     # High-resolution aerial nadir photographs
├── labels/                   # YOLO format coordinate files (.txt)
│   └── sample_swimmer.txt    # Format: class_id x_center y_center width height
├── annotated_previews/       # Visual inspection PNGs with overlaid bounding boxes
├── dataset_metadata.json     # Complete prompt and generation parameter metadata
└── annotated_gallery.html    # Standalone HTML gallery for visual batch review
```

---

## License

MIT License
