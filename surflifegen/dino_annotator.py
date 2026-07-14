"""
Grounding DINO + NMS Zero-Shot Auto-Annotator for SurfLifeGen-MLX.
Replaces brittle OpenCV morphological heuristics with open-vocabulary foundation detection
running natively on Apple Silicon PyTorch MPS.
"""

import os
import glob
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image

try:
    import torch
    import torchvision
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False


class GroundingDinoAnnotator:
    """
    Zero-shot open-vocabulary bounding box annotator using IDEA-Research/grounding-dino-base.
    Runs natively on Apple Silicon MPS with Non-Maximum Suppression (NMS).
    """
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = None,
        box_threshold: float = 0.22,
        text_threshold: float = 0.22,
        nms_iou_threshold: float = 0.30
    ):
        if not DINO_AVAILABLE:
            raise RuntimeError(
                "Grounding DINO dependencies missing. Install via: pip install transformers torch torchvision"
            )

        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold

        print(f"[Grounding DINO] Loading {model_id} onto device='{self.device}'...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print(f"[Grounding DINO] Loaded model successfully in {time.time() - t0:.2f}s")

    def detect(self, image_path: str, target_type: str = "swimmer", detection_prompt: str = None) -> List[Dict[str, Any]]:
        """
        Runs Grounding DINO zero-shot detection on the image.
        Returns a list of dicts: [{'box': [x1, y1, x2, y2], 'score': float, 'label': str}, ...]
        """
        img_pil = Image.open(image_path).convert("RGB")
        w, h = img_pil.size

        if detection_prompt:
            text = detection_prompt
        elif target_type == "shark":
            text = "submerged shark . shark in water . marine predator silhouette ."
        elif target_type == "defect":
            text = "crack in asphalt . road crack . pothole . pavement defect . road edge ."
        else:
            text = "swimmer . person floating in water . person in ocean ."

        raw_inputs = self.processor(images=img_pil, text=text, return_tensors="pt")
        inputs = {
            k: (v.to(dtype=torch.float32, device=self.device) if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else (v.to(self.device) if isinstance(v, torch.Tensor) else v))
            for k, v in raw_inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"] if "input_ids" in inputs else raw_inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img_pil.size[::-1]]
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["text_labels"] if "text_labels" in results else results["labels"]

        valid_boxes = []
        valid_scores = []
        valid_labels = []

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh

            # Filter out whole-image water detections or microscopic noise (< 40px)
            if area > (w * h * 0.40) or area < 40:
                continue

            # Ensure coordinates are bounded to image
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            valid_boxes.append([x1, y1, x2, y2])
            valid_scores.append(score.item())
            valid_labels.append(label)

        if not valid_boxes:
            return []

        # Run NMS (Non-Maximum Suppression) to remove duplicate overlapping boxes
        boxes_t = torch.tensor(valid_boxes, dtype=torch.float32)
        scores_t = torch.tensor(valid_scores, dtype=torch.float32)
        keep_indices = torchvision.ops.nms(boxes_t, scores_t, self.nms_iou_threshold)

        detections = []
        for idx in keep_indices.tolist():
            detections.append({
                "box": valid_boxes[idx],
                "score": valid_scores[idx],
                "label": valid_labels[idx]
            })

        return detections

    def annotate_image(self, image_path: str, output_path: str = None, target_type: str = "swimmer", detection_prompt: str = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        Detects targets and writes an annotated copy of the image with clean bounding boxes.
        """
        detections = self.detect(image_path, target_type=target_type, detection_prompt=detection_prompt)
        img_bgr = cv2.imread(image_path)

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if target_type == "swimmer":
                label_text = f"#{i} Swimmer ({score:.2f})"
            elif target_type == "shark":
                label_text = f"#{i} Shark ({score:.2f})"
            else:
                raw_label = det.get("label", target_type.title()) if isinstance(det.get("label"), str) else target_type.title()
                label_text = f"#{i} {raw_label.strip().title()} ({score:.2f})"
            cv2.putText(
                img_bgr,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if not output_path:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_dino{ext}"

        cv2.imwrite(output_path, img_bgr)
        summary = f"Detected {len(detections)} {target_type}(s) via Grounding DINO."
        return detections, summary

    def annotate_dataset(self, dataset_dir: str, target_type: str = "swimmer", detection_prompt: str = None):
        """
        Batch annotates all PNG/JPG/JPEG/WEBP files in a dataset directory.
        """
        all_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"):
            all_files.extend(glob.glob(os.path.join(dataset_dir, ext)))
        all_files = sorted(list(set(all_files)))

        # Filter out already annotated images or temp files
        valid_files = [
            f for f in all_files
            if not os.path.basename(f).startswith("_")
            and "_annotated" not in os.path.basename(f)
            and "_dino" not in os.path.basename(f)
            and not os.path.basename(f).startswith("dino_")
        ]

        print(f"\n[Grounding DINO] Batch processing {len(valid_files)} images from '{dataset_dir}'...")
        t_start = time.time()
        total_targets = 0

        for i, img_path in enumerate(valid_files, 1):
            base, ext = os.path.splitext(img_path)
            out_path = f"{base}_dino_annotated{ext}"
            detections, summary = self.annotate_image(img_path, out_path, target_type=target_type, detection_prompt=detection_prompt)
            total_targets += len(detections)
            print(f"[{i}/{len(valid_files)}] {os.path.basename(img_path)} -> {summary}")

        elapsed = time.time() - t_start
        print(f"\n[Grounding DINO] Completed batch processing in {elapsed:.1f}s!")
        print(f"[Grounding DINO] Total {target_type}s detected across dataset: {total_targets}")
