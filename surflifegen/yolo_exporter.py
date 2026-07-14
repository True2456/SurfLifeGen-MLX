# surflifegen/yolo_exporter.py
"""
YOLO Dataset Exporter for SurfLifeGen-MLX (`surflifegen-yolo-export`).
Converts synthetic or real annotated datasets (`bounding_boxes.json` or `segmentations.json` or `_mask.png`)
into Ultralytics YOLOv8/YOLOv11 ready formats (`images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`)
for both real-time Object Detection (`[class_id, cx, cy, w, h]`) and real-time Instance Segmentation (`[class_id, x1, y1, ... xN, yN]`).
"""

import os
import json
import glob
import shutil
import random
import yaml
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import cv2


class YoloDatasetExporter:
    """
    Exports clean images + Grounding DINO / Grounded-SAM annotations to Ultralytics YOLO dataset format.
    """
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str = "./yolo_dataset",
        split_ratio: float = 0.8,
        mode: str = "auto",  # 'detect', 'segment', or 'auto'
        class_mapping: Optional[Dict[str, int]] = None,
        copy_images: bool = True
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.split_ratio = split_ratio
        self.mode = mode
        self.copy_images = copy_images

        # Default standard classes across surflifegen pipelines
        if class_mapping is not None:
            self.class_mapping = class_mapping
        else:
            self.class_mapping = {
                "swimmer": 0,
                "person": 0,
                "shark": 1,
                "submerged shark": 1,
                "crack": 0,
                "alligator crack": 0,
                "longitudinal crack": 0,
                "transverse crack": 0,
                "pothole": 1,
                "edge": 2,
                "road edge": 2,
                "marking": 3,
                "faded lane marking": 3,
                "defect": 0
            }

    def resolve_class_id(self, label_str: str) -> int:
        """Resolves text label to integer YOLO class index."""
        lbl = label_str.strip().lower()
        if lbl in self.class_mapping:
            return self.class_mapping[lbl]
        for key, idx in self.class_mapping.items():
            if key in lbl:
                return idx
        return 0  # Default class 0 if unmapped

    def _convert_box_to_yolo(self, box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Converts [x1, y1, x2, y2] to normalized [cx, cy, w, h]."""
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)), max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))

    def _extract_polygon_from_mask(self, mask_path: str, class_id: int) -> List[str]:
        """Extracts normalized polygon contour strings from a binary/multi-class mask PNG for YOLOv8-seg."""
        if not os.path.exists(mask_path):
            return []
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return []
        h, w = mask_img.shape

        # Find all non-zero unique values or threshold
        unique_vals = np.unique(mask_img)
        unique_vals = [v for v in unique_vals if v > 0]
        if not unique_vals:
            return []

        lines = []
        for val in unique_vals:
            binary_mask = (mask_img == val).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                # Simplify contour slightly for smaller TXT files
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) < 3:
                    continue

                poly_coords = []
                for point in approx:
                    px, py = point[0][0], point[0][1]
                    norm_x = round(max(0.0, min(1.0, float(px) / w)), 6)
                    norm_y = round(max(0.0, min(1.0, float(py) / h)), 6)
                    poly_coords.extend([str(norm_x), str(norm_y)])

                if len(poly_coords) >= 6:
                    lines.append(f"{class_id} " + " ".join(poly_coords))
        return lines

    def export(self) -> Dict[str, Any]:
        """Runs the YOLO dataset export process."""
        print(f"\n[YOLO Exporter] Scanning dataset directory: '{self.dataset_dir}'...")

        # Find metadata JSON files
        json_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
        records = []
        is_seg_mode = (self.mode == "segment")

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
            except Exception as e:
                print(f"  [Warning] Failed to read JSON {jf}: {e}")

        # Auto-detect segmentation mode if segmentations.json is present or mask files exist
        if self.mode == "auto":
            if any("segmentations.json" in jf for jf in json_files) or glob.glob(os.path.join(self.dataset_dir, "*_mask.png")):
                is_seg_mode = True
                print("[YOLO Exporter] Auto-detected YOLOv8-seg Instance Segmentation format requirements.")
            else:
                is_seg_mode = False
                print("[YOLO Exporter] Auto-detected YOLOv8 Object Detection format requirements.")

        # Gather clean images and map their annotations
        all_images = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG"):
            all_images.extend(glob.glob(os.path.join(self.dataset_dir, ext)))

        clean_images = [
            f for f in sorted(list(set(all_images)))
            if not os.path.basename(f).startswith("_")
            and "_dino" not in f
            and "_annotated" not in f
            and "_segmented" not in f
            and "_mask" not in f
        ]

        # If no clean images found directly in dataset_dir, recover them from JSON records
        if not clean_images and records:
            for r in records:
                if not isinstance(r, dict):
                    continue
                orig = r.get("image_path", r.get("clean_file", ""))
                if orig and os.path.exists(orig):
                    # Copy clean image right into dataset_dir if not already there
                    dest_path = os.path.join(self.dataset_dir, os.path.basename(orig))
                    if not os.path.exists(dest_path):
                        shutil.copy2(orig, dest_path)
                    clean_images.append(dest_path)
            clean_images = sorted(list(set(clean_images)))

        if not clean_images:
            raise FileNotFoundError(f"No clean image files found inside {self.dataset_dir}!")

        print(f"[YOLO Exporter] Found {len(clean_images)} clean images. Mapping annotations...")

        # Create YOLO folder structure
        for split in ("train", "val"):
            os.makedirs(os.path.join(self.output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", split), exist_ok=True)

        random.seed(42)
        shuffled = list(clean_images)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * self.split_ratio)
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]

        # Map clean filenames and IDs to their JSON record
        record_map = {}
        for r in records:
            if not isinstance(r, dict):
                continue
            if "clean_file" in r and r["clean_file"]:
                record_map[os.path.basename(r["clean_file"])] = r
            if "image_path" in r and r["image_path"]:
                record_map[os.path.basename(r["image_path"])] = r
            if "image_id" in r and r["image_id"]:
                record_map[str(r["image_id"])] = r

        converted_count = 0
        total_labels = 0

        for split, file_list in [("train", train_files), ("val", val_files)]:
            for img_path in file_list:
                fname = os.path.basename(img_path)
                base_no_ext = os.path.splitext(fname)[0]

                # Destination paths
                dest_img = os.path.join(self.output_dir, "images", split, fname)
                dest_txt = os.path.join(self.output_dir, "labels", split, f"{base_no_ext}.txt")

                # Copy or symlink image
                if not os.path.exists(dest_img):
                    if self.copy_images:
                        shutil.copy2(img_path, dest_img)
                    else:
                        os.symlink(img_path, dest_img)

                # Get image dimensions
                img_pil = Image.open(img_path)
                w, h = img_pil.size

                label_lines = []
                rec = record_map.get(fname, record_map.get(base_no_ext, {}))

                # Case 1: urban_segmentations.json structure (`instances` with polygon or box)
                if rec and "instances" in rec:
                    for inst in rec["instances"]:
                        cls_id = inst.get("class_id", 0)
                        poly = inst.get("polygon", [])
                        if is_seg_mode and poly and len(poly) >= 3:
                            poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                            label_lines.append(f"{cls_id} {poly_str}")
                        elif "box" in inst and inst["box"] and len(inst["box"]) == 4:
                            cx, cy, bw, bh = self._convert_box_to_yolo(inst["box"], w, h)
                            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                # Case 2: old binary _mask.png files from single-class segmenter.py
                elif is_seg_mode and os.path.exists(os.path.join(self.dataset_dir, f"{base_no_ext}_mask.png")):
                    mask_path = os.path.join(self.dataset_dir, f"{base_no_ext}_mask.png")
                    class_id = 0
                    if rec and "detections" in rec and rec["detections"]:
                        class_id = self.resolve_class_id(str(rec["detections"][0].get("label", "crack")))
                    label_lines = self._extract_polygon_from_mask(mask_path, class_id=class_id)
                # Case 3: old segmentations.json detections array
                elif rec and "detections" in rec:
                    for det in rec["detections"]:
                        box = det.get("box")
                        if not box or len(box) != 4:
                            continue
                        label_str = str(det.get("label", "target"))
                        class_id = self.resolve_class_id(label_str)
                        cx, cy, bw, bh = self._convert_box_to_yolo(box, w, h)
                        label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                # Case 4: look for existing TXT file
                else:
                    existing_txt = os.path.join(self.dataset_dir, f"{base_no_ext}.txt")
                    if os.path.exists(existing_txt):
                        with open(existing_txt, "r") as et:
                            label_lines = [l.strip() for l in et.readlines() if l.strip()]

                with open(dest_txt, "w") as tf:
                    tf.write("\n".join(label_lines) + "\n" if label_lines else "")

                converted_count += 1
                total_labels += len(label_lines)

        # Build data.yaml
        # Invert class mapping to get unique classes in order
        sorted_classes = {}
        for name, cid in self.class_mapping.items():
            if cid not in sorted_classes:
                sorted_classes[cid] = name.title()

        class_names_list = [sorted_classes[i] for i in range(max(sorted_classes.keys()) + 1 if sorted_classes else 1)]

        data_yaml_content = {
            "path": self.output_dir,
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(class_names_list)}
        }

        yaml_path = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_path, "w") as yf:
            yaml.dump(data_yaml_content, yf, default_flow_style=False)

        summary = {
            "output_dir": self.output_dir,
            "train_images": len(train_files),
            "val_images": len(val_files),
            "total_labels": total_labels,
            "mode": "YOLOv8-Seg (Instance Segmentation)" if is_seg_mode else "YOLOv8 (Object Detection)",
            "data_yaml": yaml_path
        }

        print(f"\n[YOLO Exporter] Successfully exported dataset to: {self.output_dir}")
        print(f"  -> Mode: {summary['mode']}")
        print(f"  -> Train Images: {len(train_files)} | Val Images: {len(val_files)}")
        print(f"  -> Total Labels: {total_labels}")
        print(f"  -> Configuration saved to: {yaml_path}")
        return summary
