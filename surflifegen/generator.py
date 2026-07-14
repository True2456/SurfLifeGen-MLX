"""
Apple Silicon native MLX 8-Bit Generator for Cosmos 3 Omni pipeline.
Generates high-altitude aerial synthetic imagery for maritime search & rescue.
"""

import os
import sys
import time
import torch
from typing import Optional
from PIL import Image

from .model_utils import resolve_or_download_cosmos3_mlx

class SurfLifeGenPipeline:
    def __init__(self, model_path: Optional[str] = None, auto_download: bool = True):
        resolved_path = resolve_or_download_cosmos3_mlx(model_path, auto_download=auto_download)
        if resolved_path not in sys.path:
            sys.path.insert(0, resolved_path)

        from mlx_pipeline import MLXCosmos3Transformer
        from diffusers import Cosmos3OmniPipeline, AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
        from transformers import AutoTokenizer

        print("[SurfLifeGen-MLX] Loading tokenizers, VAE, and scheduler...")
        vae = AutoencoderKLWan.from_pretrained(resolved_path, subfolder="vae", torch_dtype=torch.float32).eval()
        sched = UniPCMultistepScheduler.from_pretrained(resolved_path, subfolder="scheduler")
        tok = AutoTokenizer.from_pretrained(resolved_path, subfolder="text_tokenizer")
        st = Cosmos3AVAEAudioTokenizer.from_pretrained(resolved_path, subfolder="sound_tokenizer", torch_dtype=torch.float32).eval()

        print("[SurfLifeGen-MLX] Loading MLX 8-Bit Quantized Cosmos 3 Transformer...")
        mlx_transformer = MLXCosmos3Transformer(os.path.join(resolved_path, "transformer"))

        self.pipe = Cosmos3OmniPipeline(
            transformer=mlx_transformer,
            text_tokenizer=tok,
            vae=vae,
            scheduler=sched,
            sound_tokenizer=st,
            enable_safety_checker=False
        )
        print("[SurfLifeGen-MLX] Pipeline successfully initialized on Apple Silicon!")

    def generate(self, prompt: str, width: int = 1024, height: int = 768, steps: int = 25) -> Image.Image:
        """
        Generates a single high-resolution aerial synthetic maritime photograph.
        """
        out = self.pipe(
            prompt=prompt,
            num_frames=1,
            height=height,
            width=width,
            num_inference_steps=steps
        )
        img = out.video[0][0] if isinstance(out.video[0], list) else out.video[0]
        return img
