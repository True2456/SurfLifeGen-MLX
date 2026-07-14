"""
Model resolution and automatic downloading utilities for Cosmos 3 MLX 8-Bit.
Allows users to specify a local install location OR automatically download/fetch weights.
"""

import os
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "True2456/Cosmos3-Nano-MLX-8bit"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/surflifegen/models/Cosmos3-Nano-MLX-8bit")

def resolve_or_download_cosmos3_mlx(model_path: str = None, auto_download: bool = True, repo_id: str = DEFAULT_REPO_ID) -> str:
    """
    Resolves the local folder containing the Cosmos 3 MLX 8-Bit pipeline weights.
    
    1. If `model_path` is provided and exists, returns it directly.
    2. If `model_path` is None or doesn't exist, checks common local paths.
    3. If not found and `auto_download` is True, downloads the snapshot from Hugging Face Hub.
    """
    # 1. Direct explicit user path
    if model_path and os.path.exists(model_path):
        print(f"[SurfLifeGen-MLX] Using existing local Cosmos 3 model at: {os.path.abspath(model_path)}")
        return os.path.abspath(model_path)

    # 2. Check common search locations
    search_paths = [
        model_path if model_path else "",
        "/Users/true/Documents/Mati_Train/models/Cosmos3-Nano-MLX-8bit",
        os.path.abspath("./models/Cosmos3-Nano-MLX-8bit"),
        DEFAULT_CACHE_DIR
    ]
    for sp in search_paths:
        if sp and os.path.exists(sp) and os.path.exists(os.path.join(sp, "transformer")):
            print(f"[SurfLifeGen-MLX] Found local Cosmos 3 model at: {sp}")
            return sp

    # 3. Automatic download if enabled
    if not auto_download:
        raise FileNotFoundError(
            f"Cosmos 3 MLX 8-Bit model not found at '{model_path}' and auto_download=False. "
            f"Please pass --model-path /path/to/Cosmos3-Nano-MLX-8bit or enable --auto-download."
        )

    print(f"[SurfLifeGen-MLX] Local Cosmos 3 model not found. Automatically downloading repo '{repo_id}'...")
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=DEFAULT_CACHE_DIR,
        ignore_patterns=["*.msgpack", "*.bin"]
    )
    print(f"[SurfLifeGen-MLX] Download complete! Saved to: {downloaded_path}")
    return downloaded_path
