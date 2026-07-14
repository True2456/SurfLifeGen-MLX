# surflifegen/segmenter.py
"""
Grounded-SAM Zero-Shot Pavement Defect & Surface Segmentation Engine.
Combines Grounding DINO (open-vocabulary localization) with Segment Anything (SAM)
running natively on Apple Silicon PyTorch MPS to output binary masks and colored segmentation overlays
for cracks, potholes, and highway edges.
"""

import os
import time
import json
import glob
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor


class GroundedSamSegmenter:
    """
    Zero-Shot Grounded-SAM Segmenter for High-Altitude & Surface Highway Inspection.
    """
    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam-vit-base",
        box_threshold: float = 0.22,
        text_threshold: float = 0.22,
        device: Optional[str] = None
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"[Grounded-SAM] Initializing Grounding DINO ({dino_model_id}) on device='{self.device}'...")
        t0 = time.time()
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
        self.dino_model.eval()

        print(f"[Grounded-SAM] Initializing Segment Anything ({sam_model_id}) on device='{self.device}'...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        self.sam_model.eval()
        print(f"[Grounded-SAM] Loaded both models successfully in {time.time() - t0:.2f}s")

    def segment(
        self,
        image_path: str,
        detection_prompt: str = "crack in asphalt . pothole . road edge .",
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, Image.Image]:
        """
        Runs Grounded-SAM on an input image.
        Returns:
            - list of segmented instances with mask bounding boxes and labels
            - combined multi-class mask array (H, W) where 1=crack, 2=pothole, 3=edge, etc.
            - overlay Image (PIL) with colored translucent masks and contours
        """
        thresh = box_threshold if box_threshold is not None else self.box_threshold
        t_thresh = text_threshold if text_threshold is not None else self.text_threshold

        img_pil = Image.open(image_path).convert("RGB")
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # Step 1: Grounding DINO Box Proposals
        raw_dino_inputs = self.dino_processor(images=img_pil, text=detection_prompt, return_tensors="pt")
        inputs = {
            k: (v.to(dtype=torch.float32, device=self.device) if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else (v.to(self.device) if isinstance(v, torch.Tensor) else v))
            for k, v in raw_dino_inputs.items()
        }
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"] if "input_ids" in inputs else raw_dino_inputs.input_ids,
            threshold=thresh,
            text_threshold=t_thresh,
            target_sizes=[(h, w)]
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

            # Filter whole-image noise (>70%) or microscopic box noise (<40px)
            if area > (w * h * 0.70) or area < 40:
                continue

            # Ensure bounding inside image
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            valid_boxes.append([x1, y1, x2, y2])
            valid_scores.append(score.item())
            valid_labels.append(label)

        # If no boxes found, return clean image and empty mask
        if not valid_boxes:
            return [], np.zeros((h, w), dtype=np.uint8), img_pil

        # Step 2: Segment Anything (SAM) Precision Mask Generation
        input_boxes = [[[float(coord) for coord in b] for b in valid_boxes]]
        raw_sam_inputs = self.sam_processor(
            images=img_pil,
            input_boxes=input_boxes,
            return_tensors="pt"
        )
        sam_inputs = {
            k: (v.to(dtype=torch.float32, device=self.device) if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else (v.to(self.device) if isinstance(v, torch.Tensor) else v))
            for k, v in raw_sam_inputs.items()
        }

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        masks_t = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0]
        scores_t = sam_outputs.iou_scores.cpu()[0]

        instances = []
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        overlay_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Color mapping for translucent overlay and class indices
        color_map = {
            "crack": ((0, 0, 255), 1),       # Red in BGR
            "pothole": ((0, 140, 255), 2),   # Orange in BGR
            "edge": ((255, 255, 0), 3),      # Cyan in BGR
            "marking": ((255, 0, 255), 4),   # Magenta in BGR
            "default": ((0, 255, 0), 5)      # Green in BGR
        }

        for i, (box, score, label) in enumerate(zip(valid_boxes, valid_scores, valid_labels)):
            # SAM outputs 3 candidate masks per box; pick the highest IoU score mask
            best_idx = torch.argmax(scores_t[i]).item()
            mask_np = masks_t[i, best_idx].numpy().astype(bool)

            # Determine class color and mask index
            lbl_lower = str(label).lower()
            if "crack" in lbl_lower or "fracture" in lbl_lower:
                color, class_id = color_map["crack"]
            elif "pothole" in lbl_lower or "hole" in lbl_lower or "crater" in lbl_lower:
                color, class_id = color_map["pothole"]
            elif "edge" in lbl_lower or "boundary" in lbl_lower or "shoulder" in lbl_lower:
                color, class_id = color_map["edge"]
            elif "marking" in lbl_lower or "line" in lbl_lower or "paint" in lbl_lower:
                color, class_id = color_map["marking"]
            else:
                color, class_id = color_map["default"]

            # Update combined binary/class mask
            combined_mask[mask_np] = class_id

            # Apply translucent color overlay
            colored_mask = np.zeros_like(overlay_bgr)
            colored_mask[mask_np] = color
            overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored_mask, 0.45, 0)

            # Draw crisp contour around mask
            contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, color, 2)

            # Label near the centroid or top of mask
            if contours and len(contours[0]) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = box[0], box[1]
            else:
                cx, cy = box[0], box[1]

            label_str = f"{label.strip().title()} ({score:.2f})"
            cv2.putText(overlay_bgr, label_str, (cx, max(20, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(overlay_bgr, label_str, (cx, max(20, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            instances.append({
                "label": label,
                "score": score,
                "box": box,
                "class_id": class_id,
                "mask_area_pixels": int(np.sum(mask_np))
            })

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)

        return instances, combined_mask, overlay_pil

    def segment_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        detection_prompt: str = "crack in asphalt . pothole . road edge .",
        box_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Segments a single image and saves both `_segmented.png` and `_mask.png`.
        """
        instances, mask_np, overlay_pil = self.segment(
            image_path,
            detection_prompt=detection_prompt,
            box_threshold=box_threshold
        )

        base_dir = output_dir if output_dir else os.path.dirname(image_path)
        os.makedirs(base_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Remove any existing annotation suffixes
        for suffix in ("_dino", "_annotated", "_segmented", "_mask"):
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]

        seg_path = os.path.join(base_dir, f"{base_name}_segmented.png")
        mask_path = os.path.join(base_dir, f"{base_name}_mask.png")

        overlay_pil.save(seg_path)

        # Save binary/colored mask for training (0=background, 255/128/64 classes)
        mask_vis = (mask_np * 60).astype(np.uint8)
        Image.fromarray(mask_vis).save(mask_path)

        summary = f"Segmented {len(instances)} defects -> Saved {os.path.basename(seg_path)} & {os.path.basename(mask_path)}"
        return instances, summary

    def segment_dataset(
        self,
        dataset_dir: str,
        detection_prompt: str = "crack in asphalt . pothole . road edge .",
        box_threshold: Optional[float] = None
    ):
        """
        Batch segments all images in a dataset directory.
        """
        all_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"):
            all_files.extend(glob.glob(os.path.join(dataset_dir, ext)))
        all_files = sorted(list(set(all_files)))

        valid_files = [
            f for f in all_files
            if not os.path.basename(f).startswith("_")
            and "_segmented" not in os.path.basename(f)
            and "_mask" not in os.path.basename(f)
            and "_dino" not in os.path.basename(f)
            and "_annotated" not in os.path.basename(f)
        ]

        print(f"\n[Grounded-SAM] Batch segmenting {len(valid_files)} images from '{dataset_dir}'...")
        t_start = time.time()
        total_instances = 0

        for i, img_path in enumerate(valid_files, 1):
            instances, summary = self.segment_image(img_path, detection_prompt=detection_prompt, box_threshold=box_threshold)
            total_instances += len(instances)
            print(f"[{i}/{len(valid_files)}] {os.path.basename(img_path)} -> {summary}")

        elapsed = time.time() - t_start
        print(f"\n[Grounded-SAM] Completed dataset segmentation in {elapsed:.1f}s! Total instances: {total_instances}")
