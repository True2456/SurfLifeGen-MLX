"""
Command Line Interface for High-Altitude (`120m-400m`) Active Swimmer Synthetic Generation & Grounding Engine.
"""

import os
import argparse
from .highalt_pipeline import HighAltitudeSwimmerPipeline


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: High-Altitude (`120m-400m`) Active Swimmer Synthetic Generator & Grounding DINO Annotator"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./highalt_swimmer_dataset",
        help="Path to save generated high-altitude active swimmer images and DINO annotations"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of high-altitude synthetic scenes to generate (default: 5)"
    )
    parser.add_argument(
        "--altitude", "-a",
        type=int,
        default=100,
        help="Simulated altitude in meters (e.g. 80, 100, 120, 135). Default: 100"
    )
    parser.add_argument(
        "--swimmers", "-s",
        type=int,
        default=3,
        help="Number of active swimmers per scene (default: 3)"
    )

    args = parser.parse_args()

    pipeline = HighAltitudeSwimmerPipeline(output_dir=args.output_dir)
    print(f"[HighAlt Generator] Starting generation of {args.count} high-altitude (`alt{args.altitude}m`) active swimmer scenes...")

    for i in range(1, args.count + 1):
        clean_p, ann_p, meta = pipeline.generate_highalt_scene(
            altitude_m=args.altitude,
            swimmer_count=args.swimmers,
            filename_prefix=f"highalt_{i:04d}"
        )
        print(f"[{i}/{args.count}] Generated {os.path.basename(clean_p)} -> {meta.get('dino_summary')}")

    print(f"\n[HighAlt Generator] Completed! Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
