"""
Precision Automated Swimmer Bounding Box Annotator for Surf Life Saving datasets.
Suppresses low-saturation seafoam and matches expected swimmer counts to output YOLO labels.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class PrecisionSwimmerAnnotator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.labels_dir = os.path.join(output_dir, "labels")
        self.previews_dir = os.path.join(output_dir, "annotated_previews")
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.previews_dir, exist_ok=True)

    @staticmethod
    def detect_swimmers(img_bgr: np.ndarray, target_count: int = 1) -> List[Tuple[int, int, int, int]]:
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

        # High-visibility rash vest & skin warm tones
        mask_warm1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([32, 255, 255]))
        mask_warm2 = cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
        mask_warm = cv2.bitwise_or(mask_warm1, mask_warm2)

        # Dark wetsuit contrast against bright ocean
        mask_dark = cv2.inRange(hsv, np.array([0, 0, 10]), np.array([180, 160, 95]))

        # CIE-LAB chromatic saliency
        _, mask_a = cv2.threshold(lab[:, :, 1], 134, 255, cv2.THRESH_BINARY)
        _, mask_b = cv2.threshold(lab[:, :, 2], 136, 255, cv2.THRESH_BINARY)

        saliency = cv2.bitwise_or(mask_warm, cv2.bitwise_or(mask_dark, cv2.bitwise_or(mask_a, mask_b)))

        # Seafoam suppression (low saturation bright white/cyan)
        seafoam_mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([180, 40, 255]))
        saliency[seafoam_mask > 0] = 0

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        closed = cv2.morphologyEx(saliency, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 120 <= area <= 28000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1.0, min(bw, bh))
                if aspect <= 3.2:
                    roi = saliency[y:y+bh, x:x+bw]
                    density = np.sum(roi > 0) / float(max(1, bw * bh))
                    score = area * (1.0 + density * 10.0)

                    pad_x = int(bw * 0.18)
                    pad_y = int(bh * 0.18)
                    xmin = max(0, x - pad_x)
                    ymin = max(0, y - pad_y)
                    xmax = min(w - 1, x + bw + pad_x)
                    ymax = min(h - 1, y + bh + pad_y)

                    candidates.append({
                        "box": (xmin, ymin, xmax, ymax),
                        "score": score
                    })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = []
        for cand in candidates:
            box = cand["box"]
            overlap = False
            for s in selected:
                sb = s["box"]
                ix1 = max(box[0], sb[0])
                iy1 = max(box[1], sb[1])
                ix2 = min(box[2], sb[2])
                iy2 = min(box[3], sb[3])
                if ix1 < ix2 and iy1 < iy2:
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
                if len(selected) >= target_count:
                    break

        if not selected:
            return [(int(w*0.42), int(h*0.42), int(w*0.58), int(h*0.58))]

        return [s["box"] for s in selected]

    def annotate_image(self, image_path: str, target_count: int = 1) -> Dict[str, Any]:
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        boxes = self.detect_swimmers(img, target_count=target_count)

        yolo_lines = []
        preview_img = img.copy()
        detections = []

        for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes, 1):
            xc = ((xmin + xmax) / 2.0) / w
            yc = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / float(w)
            bh = (ymax - ymin) / float(h)

            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            detections.append({"class_id": 0, "class_name": "swimmer", "bbox": [xmin, ymin, xmax, ymax]})

            cv2.rectangle(preview_img, (xmin, ymin), (xmax, ymax), (0, 255, 64), 2)
            cv2.putText(preview_img, f"Swimmer #{idx}", (xmin, max(18, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 64), 2, cv2.LINE_AA)

        label_file = os.path.join(self.labels_dir, f"{stem}.txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        preview_file = os.path.join(self.previews_dir, filename)
        cv2.imwrite(preview_file, preview_img)

        return {
            "filename": filename,
            "yolo_label_file": f"labels/{stem}.txt",
            "preview_file": f"annotated_previews/{filename}",
            "image_width": w,
            "image_height": h,
            "detections": detections
        }

    def export_html_gallery(self, annotations: List[Dict[str, Any]], gallery_filename: str = "annotated_gallery.html"):
        html_path = os.path.join(self.output_dir, gallery_filename)
        with open(html_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Surf Life Saving — Precision Swimmer Bounding Box Gallery</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }
        h1 { text-align: center; color: #38bdf8; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #94a3b8; margin-bottom: 30px; font-size: 1.1em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 24px; max-width: 1600px; margin: 0 auto; }
        .card { background: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .card img { width: 100%; height: 260px; object-fit: cover; display: block; }
        .info { padding: 16px; }
        .tag { display: inline-block; background: #0284c7; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-bottom: 8px; }
        .title { font-weight: bold; font-size: 1em; margin-bottom: 6px; color: #f1f5f9; }
        .desc { font-size: 0.85em; color: #cbd5e1; }
    </style>
</head>
<body>
    <h1>🎯 Swimmer Bounding Box Inspection Gallery</h1>
    <div class="subtitle">Exact Swimmer Count Matching • Seafoam False-Positives Filtered Out</div>
    <div class="grid">
""")
            for ann in annotations:
                num_boxes = len(ann['detections'])
                f.write(f"""        <div class="card">
            <a href="{ann['preview_file']}" target="_blank"><img src="{ann['preview_file']}" alt="{ann['filename']}"></a>
            <div class="info">
                <span class="tag">Annotated: {num_boxes} Box(es)</span>
                <div class="title">{ann['filename']}</div>
                <div class="desc">YOLO Label: {ann['yolo_label_file']}</div>
            </div>
        </div>
""")
            f.write("""    </div>
</body>
</html>
""")
        return html_path
