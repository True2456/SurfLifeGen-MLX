"""
SurfLifeGen-MLX: Native Apple Silicon MLX 8-Bit Synthetic Dataset Generator & Precision Annotator
with Qwen2.5-VL Native Vision-Language Tag Verification.
"""

__version__ = "1.2.0"

from .model_utils import resolve_or_download_cosmos3_mlx
from .generator import SurfLifeGenPipeline
from .annotator import PrecisionSwimmerAnnotator
from .prompt_builder import generate_modular_prompt
from .vlm_verifier import VLMTagVerifier
