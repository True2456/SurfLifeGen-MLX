# 🚁 SurfLifeGen-MLX

**High-Altitude Synthetic Aerial Maritime & Surf Life Saving Dataset Generator + Precision Automated YOLO Bounding Box Annotator** powered by native **Apple Silicon MLX 8-Bit Quantized Cosmos 3 Omni**.

---

## ✨ Features

- **⚡ Apple Silicon Native MLX 8-Bit Inference:** Runs directly on Apple Silicon (M1/M2/M3/M4/M5 Max/Ultra) using native `mlx` attention kernels, utilizing ~12GB memory while generating 1024×768 aerial photographs in ~25 seconds.
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

2. **Install in editable mode (or install requirements):**
   ```bash
   pip install -e .
   ```
   *(Requires Apple Silicon macOS with `mlx >= 0.22`, `torch`, `diffusers`, `transformers`, `opencv-python`)*

---

## 🚀 Quickstart & Usage

### 1. Command-Line Interface (`surflifegen`)

#### Option A: Point to an existing local Cosmos 3 MLX install location
If you already have `Cosmos3-Nano-MLX-8bit` downloaded on your machine, pass `--model-path`:

```bash
surflifegen \
  --model-path /Users/true/Documents/Mati_Train/models/Cosmos3-Nano-MLX-8bit \
  --output-dir ./my_swimmer_dataset \
  --steps 25
```

#### Option B: Automatic Download / Resolution
If you don't pass `--model-path` (or if the folder isn't found), `surflifegen` will automatically check your local cache or download the weights from Hugging Face Hub (`True2456/Cosmos3-Nano-MLX-8bit`):

```bash
surflifegen \
  --auto-download \
  --output-dir ./my_swimmer_dataset \
  --steps 25
```

#### Custom Single-Image Generation & Annotation
```bash
surflifegen \
  --model-path /path/to/Cosmos3-Nano-MLX-8bit \
  --prompt "Direct overhead nadir photograph looking straight down from 90 meters altitude over coastal ocean water at two swimmers wearing red lifeguard vests" \
  --swimmer-count 2 \
  --output-dir ./custom_output
```

---

### 2. Python API

You can use `SurfLifeGenPipeline` and `PrecisionSwimmerAnnotator` directly in your Python code or Jupyter notebooks:

```python
from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator

# 1. Initialize pipeline (points to local folder OR downloads automatically)
pipe = SurfLifeGenPipeline(
    model_path="/path/to/Cosmos3-Nano-MLX-8bit",
    auto_download=True
)

# 2. Generate aerial nadir image
prompt = (
    "Direct overhead nadir photograph looking straight down from 90 meters altitude "
    "over open coastal ocean water at a single human swimmer treading water. "
    "Swimmer wears a high-visibility red lifeguard vest."
)
image = pipe.generate(prompt=prompt, width=1024, height=768, steps=25)
image.save("./swimmer_90m.png")

# 3. Automatically detect swimmer & export YOLO bounding box (.txt)
annotator = PrecisionSwimmerAnnotator(output_dir="./output_dataset")
ann = annotator.annotate_image("./swimmer_90m.png", target_count=1)

print("YOLO Label Saved:", ann["yolo_label_file"])
```

---

## 🎯 Dataset Output Format

Every generated dataset includes:
```
my_swimmer_dataset/
├── *.png                     # High-resolution 1024x768 aerial nadir images
├── labels/                   # YOLO format coordinates (.txt)
│   └── swimmer_01.txt        # ClassID x_center y_center width height
├── annotated_previews/       # Visual PNGs with green bounding boxes drawn
└── annotated_gallery.html    # Standalone interactive inspection browser gallery
```

---

## 📄 License
MIT License
