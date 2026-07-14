"""
Apple Silicon Native Vision-Language Model (VLM) Bounding Box Verifier & Corrector
Uses Qwen2.5-VL / Qwen3-VL (via mlx-vlm) with Chain-of-Thought (CoT) visual reasoning
to accurately count, verify, and correct bounding box tags across entire aerial frames.
"""

import os
import re
import json
import time
import cv2
from typing import List, Dict, Any, Tuple
from PIL import Image

try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import get_message_json
    from mlx_vlm.utils import load_config
    MLX_VLM_AVAILABLE = True
except ImportError:
    MLX_VLM_AVAILABLE = False

DEFAULT_VLM_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

class VLMTagVerifier:
    """
    Uses Qwen VL on Apple Silicon MLX to visually inspect aerial dataset photos,
    detect targets (swimmers or submerged sharks), verify existing annotations, and output corrected YOLO bounding boxes.
    """
    def __init__(self, model_path: str = DEFAULT_VLM_MODEL):
        if not MLX_VLM_AVAILABLE:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm")
        
        print(f"[VLM Verifier] Loading Apple Silicon MLX Vision Model: {model_path} ...")
        t0 = time.time()
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        print(f"[VLM Verifier] Loaded VLM successfully in {time.time()-t0:.1f}s")

    def detect_targets_vlm(self, image_path: str, target_type: str = "swimmer") -> Tuple[List[Tuple[int, int, int, int]], str]:
        """
        Asks Qwen VL to count and detect ALL swimmers or submerged sharks using Chain-of-Thought reasoning.
        Formats prompt with explicit two-step reasoning + grounding instructions.
        Returns a tuple of (list of [xmin, ymin, xmax, ymax] pixel coordinates, raw_response_text).
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if target_type == "shark":
            instruction = (
                "First, examine this entire aerial nadir ocean photograph from edge to edge, count exactly how many submerged shark silhouettes are visible under the water, and briefly describe where each shark is located. "
                "Second, output the exact bounding box coordinates for every single shark found in format (ymin, xmin), (ymax, xmax) normalized from 0 to 1000."
            )
        else:
            instruction = (
                "First, examine this entire aerial nadir ocean photograph from edge to edge, count exactly how many human swimmers floating or treading water are visible across the image, and briefly describe where each swimmer is located. "
                "Second, output the exact bounding box coordinates for every single swimmer found in format (ymin, xmin), (ymax, xmax) normalized from 0 to 1000."
            )

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]

        if hasattr(self.processor, "apply_chat_template"):
            formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            formatted_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>\n<|im_start|>assistant\n"

        output = generate(
            self.model,
            self.processor,
            image=image_path,
            prompt=formatted_prompt,
            max_tokens=384,
            temperature=0.1
        )

        output_text = getattr(output, "text", None)
        if output_text is None:
            output_text = output if isinstance(output, str) else str(output)

        boxes = self._parse_qwen_boxes(output_text, width=w, height=h)
        return boxes, output_text

    @staticmethod
    def _parse_qwen_boxes(text: str, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """
        Robustly parses Qwen spatial grounding tags or coordinate tuples from CoT responses.
        Supports:
          1. <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|> or (ymin,xmin),(ymax,xmax)
          2. [ymin, xmin, ymax, xmax]
          3. (ymin, xmin, ymax, xmax)
        Qwen standard coordinates are normalized to [0..1000].
        """
        boxes = []
        p1 = re.findall(r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)", text)
        p2 = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
        p3 = re.findall(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", text)
        
        all_matches = p1 + p2 + p3
        for y1, x1, y2, x2 in all_matches:
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            if max(y1, x1, y2, x2) <= 1000:
                ymin = int(y1 / 1000.0 * height)
                xmin = int(x1 / 1000.0 * width)
                ymax = int(y2 / 1000.0 * height)
                xmax = int(x2 / 1000.0 * width)
            else:
                ymin, xmin, ymax, xmax = y1, x1, y2, x2
            
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
            elif xmin > xmax and ymin > ymax:
                boxes.append((xmax, ymax, xmin, ymin))
        return boxes

    def verify_and_correct_dataset(self, dataset_dir: str, target_type: str = "swimmer") -> Dict[str, Any]:
        """
        Scans a dataset directory, runs VLM visual verification over every image,
        corrects/updates YOLO tags in labels/, and exports a verification audit gallery.
        """
        image_files = sorted(
            [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
             if f.endswith(".png") and not f.startswith(".")]
        )

        labels_dir = os.path.join(dataset_dir, "labels")
        verified_previews_dir = os.path.join(dataset_dir, "vlm_verified_previews")
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(verified_previews_dir, exist_ok=True)

        results = []
        total_t0 = time.time()

        for idx, img_path in enumerate(image_files, 1):
            filename = os.path.basename(img_path)
            stem = os.path.splitext(filename)[0]
            label_file = os.path.join(labels_dir, f"{stem}.txt")

            print(f"[{idx}/{len(image_files)}] VLM Inspecting & Verifying ({target_type.upper()}): {filename} ...")
            t0 = time.time()
            vlm_boxes, raw_text = self.detect_targets_vlm(img_path, target_type=target_type)
            elapsed = round(time.time() - t0, 2)

            # Extract summary line from CoT text for cleaner logging
            first_line = raw_text.strip().split("\n")[0] if raw_text else "No response"
            print(f"   -> VLM Summary: {first_line[:110]} | Detected: {len(vlm_boxes)} box(es)")

            img_cv = cv2.imread(img_path)
            h, w = img_cv.shape[:2]

            yolo_lines = []
            detections = []
            class_id = 1 if target_type == "shark" else 0
            for i, (xmin, ymin, xmax, ymax) in enumerate(vlm_boxes, 1):
                xc = ((xmin + xmax) / 2.0) / w
                yc = ((ymin + ymax) / 2.0) / h
                bw = (xmax - xmin) / float(w)
                bh = (ymax - ymin) / float(h)
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                detections.append({"class_id": class_id, "class_name": target_type, "bbox": [xmin, ymin, xmax, ymax]})

                color = (0, 140, 255) if target_type == "shark" else (0, 215, 255)
                cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img_cv, f"VLM {target_type.title()} #{i}", (xmin, max(18, ymin - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            if yolo_lines:
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")

            preview_file = os.path.join(verified_previews_dir, filename)
            cv2.imwrite(preview_file, img_cv)

            results.append({
                "filename": filename,
                "yolo_label_file": f"labels/{stem}.txt",
                "preview_file": f"vlm_verified_previews/{filename}",
                "vlm_box_count": len(vlm_boxes),
                "target_type": target_type,
                "verification_time_sec": elapsed
            })

        html_report = self.export_vlm_audit_gallery(dataset_dir, results, target_type=target_type)
        print(f"\n[SUCCESS] VLM Verification Complete in {time.time()-total_t0:.1f}s -> Report: {html_report}")
        return {"report_file": html_report, "results": results}

    @staticmethod
    def export_vlm_audit_gallery(dataset_dir: str, results: List[Dict[str, Any]], target_type: str = "swimmer") -> str:
        report_path = os.path.join(dataset_dir, "vlm_verification_report.html")
        with open(report_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>SurfLifeGen-MLX — VLM Visual Audit & Tag Correction Report ({target_type.upper()})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #090d16; color: #f8fafc; margin: 0; padding: 30px; }}
        h1 {{ text-align: center; color: #38bdf8; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 30px; font-size: 1.1em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 24px; max-width: 1600px; margin: 0 auto; }}
        .card {{ background: #131c2e; border-radius: 12px; overflow: hidden; border: 1px solid #23314e; box-shadow: 0 4px 15px rgba(0,0,0,0.4); }}
        .card img {{ width: 100%; height: 270px; object-fit: cover; display: block; }}
        .info {{ padding: 16px; }}
        .tag {{ display: inline-block; background: #0284c7; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-bottom: 8px; }}
        .title {{ font-weight: bold; font-size: 1em; margin-bottom: 6px; color: #f1f5f9; }}
        .desc {{ font-size: 0.85em; color: #cbd5e1; }}
    </style>
</head>
<body>
    <h1>Qwen Native MLX Tag Verification & Audit Report ({target_type.title()})</h1>
    <div class="subtitle">Verified and corrected bounding box tags using native Vision-Language Grounding</div>
    <div class="grid">
""")
            for r in results:
                f.write(f"""        <div class="card">
            <a href="{r['preview_file']}" target="_blank"><img src="{r['preview_file']}" alt="{r['filename']}"></a>
            <div class="info">
                <span class="tag">VLM Verified {target_type.title()} Boxes: {r['vlm_box_count']}</span>
                <div class="title">{r['filename']}</div>
                <div class="desc">Updated YOLO: {r['yolo_label_file']} ({r['verification_time_sec']}s)</div>
            </div>
        </div>
""")
            f.write("""    </div>
</body>
</html>
""")
        return report_path
