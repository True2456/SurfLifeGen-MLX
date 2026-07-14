"""
Command Line Interface for VLM Bounding Box Verification & Correction
using native Apple Silicon MLX Qwen VL models or local LM Studio (e.g., Qwen 3.6 35B).
"""

import argparse
from .vlm_verifier import VLMTagVerifier, DEFAULT_VLM_MODEL

def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: Native VLM Bounding Box Verifier & Tag Corrector (MLX & LM Studio Qwen 3.6)"
    )
    parser.add_argument("--dataset-dir", "-d", required=True, help="Path to generated dataset directory containing PNG images")
    parser.add_argument("--model-path", "-m", default=DEFAULT_VLM_MODEL, help="HuggingFace model ID or local path for MLX quantized VLM (or 'lmstudio')")
    parser.add_argument("--target", "-t", choices=["swimmer", "shark"], default="swimmer", help="Target class to verify: 'swimmer' or 'shark'")
    parser.add_argument("--backend", "-b", choices=["mlx", "lmstudio"], default="mlx", help="Backend for Stage 2 patch verification ('mlx' or 'lmstudio')")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1/chat/completions", help="Endpoint URL for LM Studio API server")
    parser.add_argument("--lmstudio-model", default="qwen/qwen3.6-35b-a3b", help="Model ID loaded in LM Studio (e.g., 'qwen/qwen3.6-35b-a3b')")
    parser.add_argument("--audit-backend", choices=["mlx", "lmstudio"], default=None, help="Backend specifically for Stage 3 Global Audit (defaults to --backend, or can use 'lmstudio' while Stage 2 uses 'mlx')")
    parser.add_argument("--audit-model", default=None, help="Model specifically for Stage 3 Global Audit (e.g., 'qwen/qwen3.6-35b-a3b')")

    args = parser.parse_args()

    # If model-path is set to lmstudio or http://..., default backend to lmstudio
    backend = args.backend
    if args.model_path.lower() == "lmstudio" or args.model_path.startswith("http"):
        backend = "lmstudio"

    verifier = VLMTagVerifier(
        model_path=args.model_path,
        backend=backend,
        lmstudio_url=args.lmstudio_url,
        lmstudio_model=args.lmstudio_model,
        audit_backend=args.audit_backend,
        audit_model=args.audit_model
    )
    verifier.verify_and_correct_dataset(args.dataset_dir, target_type=args.target)

if __name__ == "__main__":
    main()
