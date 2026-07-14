"""
Apple Silicon Native Vision-Language Model (VLM) + Computer Vision Hybrid Verifier
Uses high-precision programmatic Computer Vision candidate detection combined with
Qwen2.5-VL / Qwen3-VL patch semantic verification (`YES`/`NO`) to eliminate
wave-ripple false positives and numerical coordinate hallucinations.
"""

import os
import re
import json
import time
import cv2
from typing import List, Dict, Any, Tuple
from PIL import Image
from .annotator import PrecisionSwimmerAnnotator

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
    Hybrid Computer Vision + Qwen VL Verifier for Surf Life Saving datasets.
    Extracts high-precision physical candidate bounding boxes using OpenCV morphology,
    then uses Qwen VL binary cropped-patch classification (`YES`/`NO`) to verify each target.
    """
    def __init__(self, model_path: str = DEFAULT_VLM_MODEL):
        if not MLX_VLM_AVAILABLE:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm")
        
        print(f"[VLM Verifier] Loading Apple Silicon MLX Vision Model: {model_path} ...")
        t0 = time.time()
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        print(f"[VLM Verifier] Loaded VLM successfully in {time.time()-t0:.1f}s")

    def verify_patch_vlm(self, patch_path: str, target_type: str = "swimmer") -> bool:
        """
        Asks Qwen VL to verify whether a cropped candidate image patch contains the target.
        Returns True if Qwen confirms YES, False if NO.
        """
        if target_type == "shark":
            prompt = "Look at this cropped ocean photograph. Is there a submerged shark silhouette under the water clearly visible in this crop? Answer only YES or NO."
        else:
            prompt = "Look at this cropped ocean photograph. Is there a human person/swimmer floating or treading water clearly visible in this crop? Answer only YES or NO."

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        if hasattr(self.processor, "apply_chat_template"):
            formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            formatted_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"

        output = generate(
            self.model,
            self.processor,
            image=patch_path,
            prompt=formatted_prompt,
            max_tokens=15,
            temperature=0.0
        )

        output_text = getattr(output, "text", None)
        if output_text is None:
            output_text = output if isinstance(output, str) else str(output)

        ans = output_text.strip().upper()
        return "YES" in ans and "NO" not in ans

    def detect_targets_vlm(self, image_path: str, target_type: str = "swimmer") -> Tuple[List[Tuple[int, int, int, int]], str]:
        """
        Hybrid Two-Stage Detection:
        1. Runs PrecisionSwimmerAnnotator.detect_targets to extract physical candidate boxes.
        2. Crops each candidate patch with context padding and asks Qwen VL binary classification.
        Returns verified bounding boxes and a verification summary.
        """
        img_cv = cv2.imread(image_path)
        h, w = img_cv.shape[:2]

        # Stage 1: Programmatic Computer Vision candidate detection (low threshold to catch all potentials)
        candidates = PrecisionSwimmerAnnotator.detect_targets(img_cv, target_type=target_type, min_score=30.0)

        verified_boxes = []
        answers = []
        temp_crop = os.path.join(os.path.dirname(image_path), "_temp_vlm_patch.png")

        for i, (xmin, ymin, xmax, ymax) in enumerate(candidates, 1):
            bw = xmax - xmin
            bh = ymax - ymin
            # Generous context padding so VLM can clearly see surrounding ocean
            pad_x = int(bw * 0.45)
            pad_y = int(bh * 0.45)
            crop = img_cv[max(0, ymin - pad_y):min(h, ymax + pad_y), max(0, xmin - pad_x):min(w, xmax + pad_x)]
            
            if crop.size == 0:
                continue
                
            cv2.imwrite(temp_crop, crop)
            is_valid = self.verify_patch_vlm(temp_crop, target_type=target_type)
            ans_str = "YES" if is_valid else "NO"
            answers.append(f"Box #{i} -> {ans_str}")

            if is_valid:
                verified_boxes.append((xmin, ymin, xmax, ymax))

        if os.path.exists(temp_crop):
            try:
                os.remove(temp_crop)
            except OSError:
                pass

        summary_text = f"Checked {len(candidates)} CV candidates | Verified {len(verified_boxes)} true {target_type}(s) | " + ", ".join(answers[:6])
        return verified_boxes, summary_text

    def verify_and_correct_dataset(self, dataset_dir: str, target_type: str = "swimmer") -> Dict[str, Any]:
        """
        Scans a dataset directory, runs Hybrid CV + VLM visual verification over every image,
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

            print(f"[{idx}/{len(image_files)}] Hybrid CV+VLM Verifying ({target_type.upper()}): {filename} ...")
            t0 = time.time()
            vlm_boxes, summary = self.detect_targets_vlm(img_path, target_type=target_type)
            elapsed = round(time.time() - t0, 2)

            print(f"   -> {summary} ({elapsed}s)")

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
                cv2.putText(img_cv, f"Verified {target_type.title()} #{i}", (xmin, max(18, ymin - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            if yolo_lines:
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
            elif os.path.exists(label_file):
                # Clear label file if zero verified targets found to remove previous false positives
                with open(label_file, "w") as f:
                    f.write("")

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
        print(f"\n[SUCCESS] Hybrid CV+VLM Verification Complete in {time.time()-total_t0:.1f}s -> Report: {html_report}")
        return {"report_file": html_report, "results": results}

    @staticmethod
    def export_vlm_audit_gallery(dataset_dir: str, results: List[Dict[str, Any]], target_type: str = "swimmer") -> str:
        report_path = os.path.join(dataset_dir, "vlm_verification_report.html")
        with open(report_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>SurfLifeGen-MLX — Hybrid CV+VLM Audit Report ({target_type.upper()})</title>
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
    <h1>Hybrid Programmatic CV + Qwen VL Audit Report ({target_type.title()})</h1>
    <div class="subtitle">Exact physical coordinate bounds verified by semantic patch classification</div>
    <div class="grid">
""")
            for r in results:
                f.write(f"""        <div class="card">
            <a href="{r['preview_file']}" target="_blank"><img src="{r['preview_file']}" alt="{r['filename']}"></a>
            <div class="info">
                <span class="tag">Verified {target_type.title()} Boxes: {r['vlm_box_count']}</span>
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
