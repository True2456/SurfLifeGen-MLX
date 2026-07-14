"""
SurfLifeGen-MLX: Native Apple Silicon MLX 8-Bit Synthetic Dataset Generator & Precision Annotator
for Aerial Surf Life Saving & Maritime Search and Rescue AI Models.
"""

__version__ = "1.0.0"

from .model_utils import resolve_or_download_cosmos3_mlx
from .generator import SurfLifeGenPipeline
from .annotator import PrecisionSwimmerAnnotator
