# SurfLifeGen-MLX

High-Altitude Synthetic Aerial Maritime & Surf Life Saving Dataset Generator with Precision Automated YOLO Bounding Box Annotation and Native Vision-Language Model (VLM) Tag Verification, powered by native Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni and Qwen3-VL.

---

## Overview

SurfLifeGen-MLX is a specialized synthetic data generation, automated annotation, and Vision-Language verification pipeline designed for Search and Rescue (SAR) and Shark Patrol drone vision models. It runs natively on Apple Silicon (M1 through M5 Max/Ultra) using optimized `mlx` kernels to synthesize realistic aerial nadir maritime imagery at patrol altitudes (40 to 100 meters) for both **Swimmers** and **Submerged Sharks**.

### Key Features

- **Apple Silicon Native Inference:** Built on Apple MLX 8-bit quantization for high-speed local generation (~25 seconds per 1024x768 image on M-series Max chips).
- **Dual Target Modes (`--target swimmer` | `--target shark`):** Specialized prompt engines for high-visibility rescue swimmers and submerged coastal marine predators (Great White, Bull Shark, Tiger Shark, Bronze Whaler) under transparent coastal water.
- **Qwen3-VL Native Tag Verifier:** Integrated Vision-Language Model verification (`surflifegen-verify`) running natively via `mlx-vlm` to visually inspect generated photos, verify target counts, and correct YOLO bounding box coordinates.
- **Modular Prompt Engine:** Dynamically generates high-variance coastal search-and-rescue scenarios (altitude, lighting, water transparency, seabed contrast) while eliminating aircraft noun hallucinations.
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

### 1. Bulk Submerged Shark Dataset Generation (`--target shark`)

Generate synthetic aerial patrol photographs of submerged marine predators (sharks cruising 1-3m below turquoise water over sandy sea floors or drop-offs):

```bash
surflifegen --target shark --bulk-count 50 --output-dir ./shark_aerial_dataset --steps 25
```

### 2. Bulk Swimmer Dataset Generation (`--target swimmer`)

Generate synthetic aerial Search and Rescue images of swimmers wearing high-visibility lifeguard rash vests:

```bash
surflifegen --target swimmer --bulk-count 100 --output-dir ./bulk_swimmer_dataset --steps 25
```

### 3. VLM Automated Tag Verification & Correction (`surflifegen-verify`)

Use native Apple Silicon MLX Qwen3-VL (`Qwen3-VL-30B-A3B-Instruct-4bit`) to inspect every image in your dataset folder, verify/correct bounding box tags, and generate an HTML verification report:

```bash
surflifegen-verify --dataset-dir ./shark_aerial_dataset
```

---

## Python API Usage

You can integrate SurfLifeGen-MLX directly into Python pipelines:

```python
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator, VLMTagVerifier, generate_shark_prompt

# 1. Generate & Annotate Submerged Shark Photo
pipeline = SurfLifeGenPipeline(auto_download=True)
annotator = PrecisionSwimmerAnnotator(output_dir="./shark_dataset")

config = generate_shark_prompt(altitude_m=75)
print("Shark Prompt:", config["prompt"])

image = pipeline.generate(prompt=config["prompt"], width=1024, height=768, steps=25)
image_path = "./shark_dataset/sample_shark.png"
image.save(image_path)

annotation = annotator.annotate_image(image_path, target_count=config["shark_count"])

# 2. Verify with Qwen3-VL Native VLM
verifier = VLMTagVerifier()
audit = verifier.verify_and_correct_dataset("./shark_dataset")
```

---

## License

MIT License
