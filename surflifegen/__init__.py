"""
SurfLifeGen-MLX: Native Apple Silicon MLX 8-Bit Synthetic Dataset Generator & Precision Annotator
with Qwen3-VL Native Vision-Language Tag Verification for Swimmers and Submerged Sharks.
"""

__version__ = "1.3.0"

from .model_utils import resolve_or_download_cosmos3_mlx
from .generator import SurfLifeGenPipeline
from .annotator import PrecisionSwimmerAnnotator
from .dino_annotator import GroundingDinoAnnotator
from .prompt_builder import generate_modular_prompt
from .shark_prompt_builder import generate_shark_prompt
from .highway_prompt_builder import generate_highway_prompt
from .highway_pipeline import HighwayWearPipeline
from .vlm_verifier import VLMTagVerifier
