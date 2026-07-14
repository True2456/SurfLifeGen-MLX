# 🚁 SurfLifeGen-MLX

**High-Altitude Synthetic Aerial Maritime & Surf Life Saving Dataset Generator + Precision Automated YOLO Bounding Box Annotator** powered by native **Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni**.

---

## ✨ Features

- **⚡ Apple Silicon Native MLX 8-Bit Inference:** Runs directly on Apple Silicon (M1/M2/M3/M4/M5 Max/Ultra) using native `mlx` attention kernels, utilizing ~12GB memory while generating 1024×768 aerial photographs in ~25 seconds.
- **🎲 Modular Bulk Prompt Builder:** Dynamically creates varied, high-standard maritime prompts spanning altitudes (40m–100m), water conditions (rip currents, reefs, swell), lighting (golden hour, noon, overcast), and rescue attire—guaranteed **zero drone noun hallucinations**.
- **🎯 Precision Automated YOLO Annotator:** Automatically detects swimmers, suppresses low-saturation seafoam/whitecaps, matches expected swimmer counts, and outputs normalized YOLO `0 xc yc w h` label files (`labels/*.txt`).
- **👀 Visual Inspection Gallery:** Automatically generates green bounding box preview images and an interactive HTML inspection sheet (`annotated_gallery.html`).
- **📂 Flexible Model Location:** Point directly to your local Cosmos 3 MLX folder (`--model-path`) **or** let the pipeline automatically fetch/download weights (`--auto-download`) from Hugging Face Hub.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/True2456/SurfLifeGen-MLX.git
   cd SurfLifeGen-MLX
   ```

2. **Install package:**
   ```bash
   pip install -e .
   ```

---

## 🚀 Usage

### 1. Generate Bulk Modular Dataset (`--bulk-count`)
Generate any number `X` of varied aerial dataset images using the modular prompt builder (perfect for generating batches to filter/validate with an AI QA checker):

```bash
surflifegen \
  --bulk-count 100 \
  --output-dir ./bulk_swimmer_dataset \
  --steps 25
```

Or using the standalone Python script:
```bash
python examples/generate_bulk_dataset.py --count 100 --output-dir ./bulk_swimmer_dataset
```

---

### 2. Standard Usage Options

#### Option A: Point to an existing local Cosmos 3 MLX install directory
```bash
surflifegen \
  --model-path /Users/true/Documents/Mati_Train/models/Cosmos3-Nano-MLX-8bit \
  --output-dir ./my_swimmer_dataset \
  --steps 25
```

#### Option B: Automatic Download / Resolution
```bash
surflifegen \
  --auto-download \
  --output-dir ./my_swimmer_dataset \
  --steps 25
```

---

### 3. Programmatic Python API

```python
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator, generate_modular_prompt

pipe = SurfLifeGenPipeline(auto_download=True)
annotator = PrecisionSwimmerAnnotator(output_dir="./output")

# Generate modular prompt dynamically
config = generate_modular_prompt()
print("Prompt:", config["prompt"])

image = pipe.generate(prompt=config["prompt"], steps=25)
image.save("./sample.png")

# Auto-annotate matching the exact swimmer count
ann = annotator.annotate_image("./sample.png", target_count=config["swimmer_count"])
```

---

## 📄 License
MIT License
