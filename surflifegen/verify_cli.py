"""
Command Line Interface for VLM Bounding Box Verification & Correction
using native Apple Silicon MLX Qwen VL models.
"""

import argparse
from .vlm_verifier import VLMTagVerifier, DEFAULT_VLM_MODEL

def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Native VLM Bounding Box Verifier & Tag Corrector (Qwen MLX)"
    )
    parser.add_argument("--dataset-dir", "-d", required=True, help="Path to generated dataset directory containing PNG images")
    parser.add_argument("--model-path", "-m", default=DEFAULT_VLM_MODEL, help="HuggingFace model ID or local path for MLX quantized VLM")
    parser.add_argument("--target", "-t", choices=["swimmer", "shark"], default="swimmer", help="Target class to verify: 'swimmer' or 'shark'")

    args = parser.parse_args()

    verifier = VLMTagVerifier(model_path=args.model_path)
    verifier.verify_and_correct_dataset(args.dataset_dir, target_type=args.target)

if __name__ == "__main__":
    main()
