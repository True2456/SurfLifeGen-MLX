# surflifegen/urban_segmenter.py
"""
UrbanSceneSegmenter: Multi-Class Aerial & Urban Scene Segmentation Engine for Apple Silicon MPS.
Combines Grounding DINO open-vocabulary localization with Segment Anything (SAM) to segment
buildings, roads, grass, vehicles, pedestrians, trees, road edges, and sidewalks into structured
multi-class polygon masks (`_urban_mask.png`), color-coded overlays (`_urban_segmented.png` with legend),
and JSON annotations.
"""

import os
import glob
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor


class UrbanSceneSegmenter:
    """
    Multi-class zero-shot segmentation pipeline for aerial, highway, and urban photography.
    """

    DEFAULT_CLASSES = [
        "building",
        "road",
        "grass",
        "car",
        "person",
        "tree",
        "edge of road",
        "sidewalk"
    ]

    # Pre-curated BGR color palette (high contrast, visually distinct)
    CLASS_COLORS_BGR = {
        "building": (0, 100, 255),       # Vibrant Orange-Red
        "road": (180, 105, 255),         # Deep Lavender / Pink
        "grass": (0, 220, 0),            # Bright Green
        "car": (255, 255, 0),            # Bright Cyan
        "person": (0, 255, 255),         # Bright Yellow
        "tree": (34, 139, 34),           # Forest Green
        "edge of road": (255, 0, 255),   # Bright Magenta
        "sidewalk": (200, 200, 200),     # Silver / Soft White
    }

    # Hierarchy priority tiers: lower tier number is drawn FIRST (as background), higher tier is drawn ON TOP
    HIERARCHY_TIERS = {
        "road": 0,
        "grass": 0,
        "building": 0,
        "sidewalk": 1,
        "edge of road": 1,
        "tree": 2,
        "car": 3,
        "person": 3,
    }

    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam-vit-base",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        classes: Optional[List[str]] = None,
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

        # Parse and normalize classes
        if classes is None:
            self.classes = list(self.DEFAULT_CLASSES)
        else:
            self.classes = [c.strip().lower() for c in classes if c.strip()]

        # Generate colors for any custom classes
        self.class_colors = {}
        for idx, cls_name in enumerate(self.classes):
            if cls_name in self.CLASS_COLORS_BGR:
                self.class_colors[cls_name] = self.CLASS_COLORS_BGR[cls_name]
            else:
                # Generate well-distributed BGR colors based on hue
                hue = int((idx * 47) % 180)
                hsv_pixel = np.array([[[hue, 220, 240]]], dtype=np.uint8)
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
                self.class_colors[cls_name] = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

        print(f"[UrbanSceneSegmenter] Initializing Grounding DINO ({dino_model_id}) on device='{self.device}'...")
        t0 = time.time()
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
        self.dino_model.eval()

        print(f"[UrbanSceneSegmenter] Initializing Segment Anything ({sam_model_id}) on device='{self.device}'...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        self.sam_model.eval()

        print(f"[UrbanSceneSegmenter] Multi-Class Pipeline Ready in {time.time() - t0:.2f}s | Classes: {len(self.classes)}")

    def _build_query_prompt(self) -> str:
        return " . ".join(self.classes) + " ."

    def _resolve_class(self, label: str) -> Tuple[int, str, Tuple[int, int, int]]:
        """Matches a detected label string to our target taxonomy index (0-indexed), normalized name, and BGR color."""
        lbl_clean = label.lower().strip()
        
        # Direct exact match or substring check against defined classes
        for idx, cls_name in enumerate(self.classes):
            if cls_name == lbl_clean or cls_name in lbl_clean or lbl_clean in cls_name:
                return idx, cls_name, self.class_colors[cls_name]

        # Synonyms and aliases checking if standard classes used
        synonyms = {
            "building": ["building", "buildings", "roof", "apartment", "house", "structure", "rooftop"],
            "road": ["road", "roads", "street", "highway", "asphalt", "pavement", "intersection", "lane"],
            "grass": ["grass", "lawn", "vegetation", "shrub", "bush", "greenery", "landscape", "field", "plant"],
            "car": ["car", "cars", "vehicle", "vehicles", "truck", "van", "bus", "suv", "automobile"],
            "person": ["person", "people", "pedestrian", "pedestrians", "human"],
            "tree": ["tree", "trees", "canopy", "palm tree", "foliage", "forest"],
            "edge of road": ["edge of road", "edge of roads", "road edge", "curb", "curbs", "shoulder", "road boundary"],
            "sidewalk": ["sidewalk", "sidewalks", "pavement walkway", "footpath", "pedestrian path", "crosswalk"]
        }

        for idx, cls_name in enumerate(self.classes):
            if cls_name in synonyms:
                for syn in synonyms[cls_name]:
                    if syn in lbl_clean or lbl_clean in syn:
                        return idx, cls_name, self.class_colors[cls_name]

        # Fallback to class 0 if unmatched
        return 0, self.classes[0], self.class_colors[self.classes[0]]

    def segment_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Segments a single image into categorical multi-class layers (`_urban_mask.png`), color overlay (`_urban_segmented.png`),
        and returns structured instance dictionaries.
        """
        thresh = box_threshold if box_threshold is not None else self.box_threshold
        t_thresh = text_threshold if text_threshold is not None else self.text_threshold

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img_pil = Image.open(image_path).convert("RGB")
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # Construct Grounding DINO dot-separated query prompt
        detection_prompt = " . ".join(self.classes) + " ."

        # Step 1: Grounding DINO Multi-Class Localization
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

        raw_instances = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh

            # Skip microscopic noise (<25px) or extreme full canvas noise (>95% of screen unless road/grass)
            if area < 25 or area > (w * h * 0.96):
                continue

            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            cls_idx, cls_name, cls_color = self._resolve_class(label)
            tier = self.HIERARCHY_TIERS.get(cls_name, 1)

            raw_instances.append({
                "box": [x1, y1, x2, y2],
                "score": float(score.item()),
                "raw_label": label,
                "class_id": cls_idx,
                "class_name": cls_name,
                "color": cls_color,
                "area": area,
                "tier": tier
            })

        # If no boxes detected, output base image and zero masks
        if not raw_instances:
            base_dir = output_dir if output_dir else os.path.dirname(image_path)
            os.makedirs(base_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            for suffix in ("_urban_segmented", "_urban_mask", "_urban_mask_vis"):
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]

            cv2.imwrite(os.path.join(base_dir, f"{base_name}_urban_segmented.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(base_dir, f"{base_name}_urban_mask.png"), np.zeros((h, w), dtype=np.uint8))
            return [], f"Detected 0 urban objects."

        # Sort instances by tier ascending (background to foreground), then by area descending
        raw_instances.sort(key=lambda item: (item["tier"], -item["area"]))

        # Step 2: Segment Anything (SAM) Precision Polygon Masks
        input_boxes = [[[float(coord) for coord in inst["box"]] for inst in raw_instances]]
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

        # Step 3: Composite categorical mask & visual overlay
        categorical_mask = np.zeros((h, w), dtype=np.uint8)
        overlay_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        instances = []
        class_counts = {cls_name: 0 for cls_name in self.classes}

        for i, inst in enumerate(raw_instances):
            best_idx = torch.argmax(scores_t[i]).item()
            mask_np = masks_t[i, best_idx].numpy().astype(bool)

            # Check if mask has non-zero pixels
            if not np.any(mask_np):
                continue

            cls_idx = inst["class_id"]
            cls_name = inst["class_name"]
            color = inst["color"]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # Categorical mask uses 1-indexed values (0 = background, 1 = class 0, 2 = class 1, etc.)
            categorical_mask[mask_np] = cls_idx + 1

            # Apply translucent color overlay
            colored_mask = np.zeros_like(overlay_bgr)
            colored_mask[mask_np] = color
            overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored_mask, 0.50, 0)

            # Draw crisp outline
            contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, color, 2)

            # Extract normalized polygon coordinates for JSON/YOLO
            polygon_coords = []
            if contours:
                # Find largest contour for clean polygon representation
                largest_cnt = max(contours, key=cv2.contourArea)
                epsilon = 0.002 * cv2.arcLength(largest_cnt, True)
                approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
                for pt in approx:
                    px, py = pt[0][0], pt[0][1]
                    polygon_coords.append([round(float(px) / w, 6), round(float(py) / h, 6)])

            instances.append({
                "id": len(instances) + 1,
                "class_id": cls_idx,
                "class_name": cls_name,
                "score": round(inst["score"], 4),
                "box": inst["box"],
                "polygon": polygon_coords
            })

        # Step 4: Draw Professional Legend Banner on Overlay
        self._draw_legend(overlay_bgr, class_counts)

        # Step 5: Save outputs
        base_dir = output_dir if output_dir else os.path.dirname(image_path)
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for suffix in ("_urban_segmented", "_urban_mask", "_urban_mask_vis", "_segmented", "_mask", "_dino", "_annotated"):
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]

        seg_path = os.path.join(base_dir, f"{base_name}_urban_segmented.png")
        mask_path = os.path.join(base_dir, f"{base_name}_urban_mask.png")
        vis_path = os.path.join(base_dir, f"{base_name}_urban_mask_vis.png")

        cv2.imwrite(seg_path, overlay_bgr)
        cv2.imwrite(mask_path, categorical_mask)

        # Generate visual palette render of categorical mask for easy inspection
        mask_vis_bgr = np.zeros_like(overlay_bgr)
        for idx, cls_name in enumerate(self.classes):
            cls_mask = (categorical_mask == (idx + 1))
            if np.any(cls_mask):
                mask_vis_bgr[cls_mask] = self.class_colors[cls_name]
        cv2.imwrite(vis_path, mask_vis_bgr)

        summary = f"Segmented {len(instances)} objects ({', '.join([f'{k}: {v}' for k, v in class_counts.items() if v > 0])}) -> Saved {os.path.basename(seg_path)} & {os.path.basename(mask_path)}"
        return instances, summary

    def _draw_legend(self, img_bgr: np.ndarray, class_counts: Dict[str, int]):
        """Draws a semi-transparent legend panel showing active classes and counts."""
        h, w, _ = img_bgr.shape
        active_classes = [(cls_name, count) for cls_name, count in class_counts.items() if count > 0]
        if not active_classes:
            return

        box_width = 240
        box_height = 36 + len(active_classes) * 26
        x_pad, y_pad = 20, 20

        # Create semi-transparent dark banner
        sub_img = img_bgr[y_pad:y_pad + box_height, x_pad:x_pad + box_width]
        rect = np.full_like(sub_img, (30, 30, 30))
        cv2.addWeighted(sub_img, 0.25, rect, 0.75, 0, sub_img)

        # Draw border
        cv2.rectangle(img_bgr, (x_pad, y_pad), (x_pad + box_width, y_pad + box_height), (255, 255, 255), 1)
        cv2.putText(img_bgr, "URBAN SEGMENTATION LEGEND", (x_pad + 12, y_pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw swatches
        for idx, (cls_name, count) in enumerate(active_classes):
            sy = y_pad + 44 + idx * 26
            color = self.class_colors.get(cls_name, (0, 255, 0))
            cv2.rectangle(img_bgr, (x_pad + 14, sy - 12), (x_pad + 32, sy + 6), color, -1)
            cv2.rectangle(img_bgr, (x_pad + 14, sy - 12), (x_pad + 32, sy + 6), (255, 255, 255), 1)
            cv2.putText(img_bgr, f"{cls_name.upper()} ({count})", (x_pad + 42, sy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    def segment_dataset(
        self,
        dataset_dir: str,
        output_dir: Optional[str] = None,
        box_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch segments all images inside a directory, producing clean categorical masks, overlays, and metadata JSON.
        """
        dataset_dir = os.path.abspath(dataset_dir)
        out_dir = os.path.abspath(output_dir) if output_dir else dataset_dir
        os.makedirs(out_dir, exist_ok=True)

        all_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"):
            all_files.extend(glob.glob(os.path.join(dataset_dir, ext)))
        all_files = sorted(list(set(all_files)))

        valid_files = [
            f for f in all_files
            if not os.path.basename(f).startswith("_")
            and "_urban_segmented" not in os.path.basename(f)
            and "_urban_mask" not in os.path.basename(f)
            and "_segmented" not in os.path.basename(f)
            and "_mask" not in os.path.basename(f)
            and "_dino" not in os.path.basename(f)
            and "_annotated" not in os.path.basename(f)
        ]

        print(f"\n[UrbanSceneSegmenter] Batch segmenting {len(valid_files)} images from '{dataset_dir}'...")
        t_start = time.time()
        dataset_records = []

        for i, img_path in enumerate(valid_files, 1):
            fname = os.path.basename(img_path)
            print(f"[{i}/{len(valid_files)}] Processing {fname}...")
            instances, summary = self.segment_image(img_path, output_dir=out_dir, box_threshold=box_threshold)
            print(f"  -> {summary}")

            dataset_records.append({
                "clean_file": fname,
                "mask_file": f"{os.path.splitext(fname)[0]}_urban_mask.png",
                "segmented_file": f"{os.path.splitext(fname)[0]}_urban_segmented.png",
                "instances": instances
            })

        meta_path = os.path.join(out_dir, "urban_segmentations.json")
        with open(meta_path, "w") as f:
            json.dump({
                "classes": self.classes,
                "class_colors_bgr": {k: list(v) for k, v in self.class_colors.items()},
                "records": dataset_records
            }, f, indent=2)

        print(f"\n[UrbanSceneSegmenter] Completed multi-class segmentation in {time.time() - t_start:.1f}s!")
        print(f"  -> Structured metadata saved to: {meta_path}")
        return dataset_records
