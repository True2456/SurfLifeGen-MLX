# SurfLifeGen-MLX

High-Altitude Synthetic Aerial Maritime & Surf Life Saving Dataset Generator with Precision Automated YOLO Bounding Box Annotation and Native Vision-Language Model (VLM) Tag Verification, powered by native Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni and Qwen2.5-VL.

---

## Overview

SurfLifeGen-MLX is a specialized synthetic data generation, automated annotation, and Vision-Language verification pipeline designed for Search and Rescue (SAR) drone vision models. It runs natively on Apple Silicon (M1 through M5 Max/Ultra) using optimized `mlx` kernels to synthesize realistic aerial nadir maritime imagery at patrol altitudes (40 to 100 meters) and verify spatial bounding box tags.

### Key Features

- **Apple Silicon Native Inference:** Built on Apple MLX 8-bit quantization for high-speed local generation (~25 seconds per 1024x768 image on M-series Max chips).
- **Qwen2.5-VL Native Tag Verifier:** Integrated Vision-Language Model verification (`surflifegen-verify`) running natively via `mlx-vlm` to visually inspect generated photos, verify swimmer counts, and correct YOLO bounding box coordinates.
- **Modular Prompt Engine:** Dynamically generates high-variance coastal search-and-rescue scenarios (altitude, lighting, water conditions, rescue attire) while eliminating aircraft noun hallucinations.
- **Precision Automated Annotation:** Size-normalized chromatic background subtraction and morphological filtering to localize swimmers and suppress seafoam.
- **Real-Time Checkpointing:** Saves and appends dataset metadata and YOLO annotations after every single image so generation can be safely paused and resumed anytime.

---

## System Requirements

- macOS running on Apple Silicon (M1/M2/M3/M4/M5 architecture)
- Python 3.10 or later
- Apple MLX framework (`mlx >= 0.31.0`)
- Apple MLX VLM framework (`mlx-vlm >= 0.6.0` for VLM tag verification)

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

Once installed, two global CLI commands are available: `surflifegen` and `surflifegen-verify`.

### 1. Bulk Modular Dataset Generation

Generate an arbitrary number of synthetic images with varied altitudes (40m to 100m), water states, and rescue attire, automatically exporting YOLO label files:

```bash
surflifegen --bulk-count 100 --output-dir ./bulk_swimmer_dataset --steps 25
```

### 2. VLM Automated Tag Verification & Correction (`surflifegen-verify`)

Use native Apple Silicon MLX Qwen2.5-VL (`Qwen2.5-VL-7B-Instruct-4bit`) to inspect every image in your dataset folder, verify/correct swimmer bounding box tags, and generate an HTML verification report:

```bash
surflifegen-verify --dataset-dir ./bulk_swimmer_dataset
```

### 3. Specifying Local Model Path vs. Automatic Download

By default, `surflifegen` checks local search paths and automatically fetches model weights from Hugging Face Hub (`True2456/Cosmos3-Nano-MLX-8bit`) if not found.

To specify an explicit local model path:

```bash
surflifegen \
  --model-path /path/to/Cosmos3-Nano-MLX-8bit \
  --bulk-count 50 \
  --output-dir ./dataset_output
```

---

## Python API Usage

You can integrate SurfLifeGen-MLX and the VLM Verifier directly into Python pipelines:

```python
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator, VLMTagVerifier, generate_modular_prompt

# 1. Generate & Annotate
pipeline = SurfLifeGenPipeline(auto_download=True)
annotator = PrecisionSwimmerAnnotator(output_dir="./output_dataset")

config = generate_modular_prompt()
image = pipeline.generate(prompt=config["prompt"], width=1024, height=768, steps=25)
image_path = "./output_dataset/sample_swimmer.png"
image.save(image_path)
annotation = annotator.annotate_image(image_path, target_count=config["swimmer_count"])

# 2. Verify & Correct Tags with Qwen2.5-VL Native VLM
verifier = VLMTagVerifier()
audit = verifier.verify_and_correct_dataset("./output_dataset")
print("VLM Audit HTML Report:", audit["report_file"])
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
├── vlm_verified_previews/    # VLM audited & corrected bounding box previews
├── dataset_metadata.json     # Complete prompt and generation parameter metadata
├── annotated_gallery.html    # Standalone HTML gallery for visual batch review
└── vlm_verification_report.html # Qwen2.5-VL visual audit & tag verification report
```

---

## License

MIT License
