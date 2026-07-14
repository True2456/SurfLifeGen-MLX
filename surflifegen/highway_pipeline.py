# surflifegen/highway_pipeline.py
"""
Highway Defect & Surface Wear Synthetic Generation & Zero-Shot Annotation/Segmentation Pipeline.
Orchestrates Apple Silicon MLX quantized Cosmos 3 Omni generation and Grounded-SAM
for exact polygon masks and colored segmentation overlays of cracks, potholes, and road edges.
"""

import os
import time
import json
from typing import Dict, Any, Tuple, Optional
from PIL import Image

from .generator import SurfLifeGenPipeline
from .dino_annotator import GroundingDinoAnnotator
from .segmenter import GroundedSamSegmenter
from .highway_prompt_builder import generate_highway_prompt


class HighwayWearPipeline:
    """
    End-to-end pipeline for generating and segmenting highway pavement defects.
    """
    def __init__(
        self,
        output_dir: str = "./highway_dataset",
        model_path: Optional[str] = None,
        auto_download: bool = True,
        box_threshold: float = 0.22,
        text_threshold: float = 0.22,
        nms_iou_threshold: float = 0.30,
        no_annotate: bool = False,
        use_sam: bool = True
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[HighwayWear Generator] Initializing MLX Cosmos 3 Omni on Apple Silicon...")
        self.generator = SurfLifeGenPipeline(model_path=model_path, auto_download=auto_download)

        self.annotator = None
        self.segmenter = None
        self.use_sam = use_sam

        if not no_annotate:
            if self.use_sam:
                try:
                    print(f"[HighwayWear Generator] Initializing Grounded-SAM (Grounding DINO + Segment Anything) on Apple Silicon MPS...")
                    self.segmenter = GroundedSamSegmenter(
                        box_threshold=box_threshold,
                        text_threshold=text_threshold
                    )
                except Exception as e:
                    print(f"[HighwayWear Generator] Warning: Grounded-SAM initialization failed ({e}). Falling back to box annotator...")
                    self.use_sam = False

            if not self.use_sam:
                try:
                    print(f"[HighwayWear Generator] Initializing Grounding DINO box detector...")
                    self.annotator = GroundingDinoAnnotator(
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        nms_iou_threshold=nms_iou_threshold
                    )
                except Exception as e:
                    print(f"[HighwayWear Generator] Warning: Grounding DINO initialization failed ({e}). Running without auto-annotation.")

    def generate_scene(
        self,
        defect_type: str = "random",
        asphalt_type: str = "random",
        perspective: str = "random",
        custom_prompt: Optional[str] = None,
        steps: int = 25,
        width: int = 1024,
        height: int = 768,
        filename_prefix: str = "highway_defect",
        detection_prompt: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Synthesizes a single highway defect image, saves it with strict anti-overwrite guarantees,
        and runs Grounded-SAM segmentation or box annotation.
        Returns (clean_image_path, annotated_image_path, metadata).
        """
        t0 = time.time()
        config = generate_highway_prompt(
            defect_type=defect_type,
            asphalt_type=asphalt_type,
            perspective=perspective,
            custom_prompt=custom_prompt
        )

        prompt_text = config["prompt"]
        dino_query = detection_prompt if detection_prompt else config["dino_query"]

        print(f"\n[HighwayWear Generator] Synthesizing image ({width}x{height}, {steps} steps)...")
        print(f"  -> Prompt: {prompt_text[:120]}...")

        # Generate using MLX Cosmos 3 Omni
        image = self.generator.generate(prompt=prompt_text, width=width, height=height, steps=steps)

        # Strict anti-overwrite guarantee
        counter = 1
        clean_filename = f"{filename_prefix}_{config['defect_type']}_{counter:04d}.png"
        clean_path = os.path.join(self.output_dir, clean_filename)
        annotated_path = clean_path.replace(".png", "_segmented.png" if self.use_sam else "_dino.png")

        while os.path.exists(clean_path) or os.path.exists(annotated_path):
            counter += 1
            clean_filename = f"{filename_prefix}_{config['defect_type']}_{counter:04d}.png"
            clean_path = os.path.join(self.output_dir, clean_filename)
            annotated_path = clean_path.replace(".png", "_segmented.png" if self.use_sam else "_dino.png")

        image.save(clean_path)
        elapsed_gen = round(time.time() - t0, 2)
        print(f"  -> Saved clean image: {clean_filename} ({elapsed_gen}s)")

        detections = []
        summary = "Annotation skipped."
        if self.segmenter:
            print(f"  -> Running Grounded-SAM polygon mask segmentation (Query: '{dino_query}')...")
            detections, summary = self.segmenter.segment_image(
                clean_path,
                output_dir=self.output_dir,
                detection_prompt=dino_query
            )
            print(f"  -> {summary}")
        elif self.annotator:
            print(f"  -> Running Grounding DINO box localization (Query: '{dino_query}')...")
            detections, summary = self.annotator.annotate_image(
                clean_path,
                output_path=annotated_path,
                target_type="defect",
                detection_prompt=dino_query
            )
            print(f"  -> {summary}")

        metadata = {
            "clean_file": os.path.basename(clean_path),
            "annotated_file": os.path.basename(annotated_path) if (self.segmenter or self.annotator) else None,
            "defect_type": config["defect_type"],
            "asphalt": config["asphalt"],
            "perspective": config["perspective"],
            "prompt": prompt_text,
            "dino_query": dino_query,
            "generation_time_sec": elapsed_gen,
            "detections": detections,
            "summary": summary
        }

        # Append to bounding_boxes.json / segmentations.json
        json_path = os.path.join(self.output_dir, "segmentations.json" if self.use_sam else "bounding_boxes.json")
        existing_data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []

        existing_data.append(metadata)
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=2)

        return clean_path, annotated_path, metadata
