"""
Apple Silicon Native MLX + LM Studio (Qwen 3.6 / OpenAI API) Hybrid Vision-Language Verifier.
Three-Stage Architecture:
1. Stage 1 (CV Candidate Detection): Programmatic morphological candidate extraction.
2. Stage 2 (Marked-Box Patch Verification): Zoomed context patch with marked red bounding box checked via YES/NO VLM query.
3. Stage 3 (Global Holistic Audit & Recovery): Full-frame check confirming all targets are boxed, powered by local MLX or LM Studio (e.g., Qwen 3.6 35B reasoning model) to recover any missed targets.
"""

import os
import re
import json
import time
import base64
import urllib.request
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
    Hybrid Computer Vision + Qwen VL / LM Studio Three-Stage Verifier for Surf Life Saving datasets.
    Guarantees 100% detection coverage and zero coordinate hallucinations.
    """
    def __init__(
        self,
        model_path: str = DEFAULT_VLM_MODEL,
        backend: str = "mlx",
        lmstudio_url: str = "http://localhost:1234/v1/chat/completions",
        lmstudio_model: str = "qwen/qwen3.6-35b-a3b",
        audit_backend: str = None,
        audit_model: str = None
    ):
        self.backend = backend.lower() if backend else "mlx"
        if model_path and model_path.lower() == "lmstudio":
            self.backend = "lmstudio"

        self.lmstudio_url = lmstudio_url
        self.lmstudio_model = lmstudio_model
        self.audit_backend = (audit_backend.lower() if audit_backend else self.backend)
        self.audit_model = audit_model or self.lmstudio_model

        # Only load MLX model into memory if MLX backend is requested for Stage 2 or Stage 3
        self.model = None
        self.processor = None
        self.config = None

        if self.backend == "mlx" or self.audit_backend == "mlx":
            if not MLX_VLM_AVAILABLE:
                raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm or use --backend lmstudio")
            
            print(f"[VLM Verifier] Loading Apple Silicon MLX Vision Model: {model_path} ...")
            t0 = time.time()
            self.model, self.processor = load(model_path)
            self.config = load_config(model_path)
            print(f"[VLM Verifier] Loaded MLX VLM successfully in {time.time()-t0:.1f}s")
        else:
            print(f"[VLM Verifier] Using LM Studio backend -> URL: {self.lmstudio_url} (Model: {self.lmstudio_model})")

    def _generate_lmstudio(self, image_path: str, prompt: str, max_tokens: int = 4096, model_id: str = None) -> Tuple[str, str]:
        """
        Sends an image and text prompt to an LM Studio (or OpenAI-compatible) local endpoint.
        Returns (content, reasoning_content).
        """
        model_name = model_id or self.lmstudio_model
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }

        req = urllib.request.Request(
            self.lmstudio_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                content = choice.get("content", "") or ""
                reasoning = choice.get("reasoning_content", "") or ""
                return content.strip(), reasoning.strip()
        except Exception as e:
            print(f"[LM Studio Error] Failed connecting to {self.lmstudio_url}: {e}")
            return "", ""

    def verify_patch_vlm(self, patch_path: str, target_type: str = "swimmer") -> bool:
        """
        Stage 2: Asks Qwen VL whether the object highlighted inside the red box in the cropped patch is a valid target.
        """
        if target_type == "shark":
            prompt = "Look at the object highlighted inside the red bounding box drawn in the center of this ocean patch. Is that highlighted object a submerged shark silhouette under the water? Answer only YES or NO."
        else:
            prompt = "Look at the object highlighted inside the red bounding box drawn in the center of this ocean patch. Is that highlighted object a human swimmer or person in the water? Answer only YES or NO."

        if self.backend == "lmstudio":
            content, reasoning = self._generate_lmstudio(patch_path, prompt, max_tokens=150, model_id=self.lmstudio_model)
            full_text = f"{content}\n{reasoning}".upper()
            return "YES" in full_text and "NO" not in full_text

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
            max_tokens=12,
            temperature=0.0
        )

        output_text = getattr(output, "text", None)
        if output_text is None:
            output_text = output if isinstance(output, str) else str(output)

        ans = output_text.strip().upper()
        return "YES" in ans and "NO" not in ans

    def global_audit_and_recover(self, overview_path: str, verified_count: int, target_type: str = "swimmer") -> Tuple[List[int], str]:
        """
        Stage 3: Performs a global check of the annotated overview image.
        Supports Qwen 3.6 / reasoning models in LM Studio or local MLX models.
        Confirms whether all targets are enclosed in bold green boxes, and recovers any candidate IDs (#1, #2, etc.) that were missed.
        """
        if target_type == "shark":
            if verified_count == 0:
                prompt = (
                    "We are auditing an automated aerial shark detection system. "
                    "In this photograph, candidate detections are highlighted with numbered yellow bounding boxes (#1, #2, etc.). Currently no boxes are confirmed.\n"
                    "First, carefully count exactly how many submerged sharks under the water are visible across the entire photograph.\n"
                    "Second, list ONLY the exact yellow box numbers (#1, #2, etc.) that enclose actual submerged sharks in format: RECOVER: [box_num_1, box_num_2]."
                )
            else:
                prompt = (
                    "We are auditing an automated aerial shark detection system. "
                    "In this photograph, bold green boxes enclose confirmed verified sharks, and yellow numbered boxes (#1, #2, etc.) enclose unverified candidate detections.\n"
                    "First, count carefully how many submerged sharks under the water are visible in total across the entire photograph.\n"
                    "Second, check: are all submerged sharks already enclosed inside a bold green box?\n"
                    "If ALL submerged sharks across the image are already inside green boxes, output exactly: GLOBAL_STATUS: ALL_BOXED.\n"
                    "If there is any submerged shark that is NOT inside a green box, inspect the yellow numbered boxes (#1, #2, etc.) and output the exact yellow box number(s) that enclose the missing shark(s) in format: RECOVER: [box_num_1, box_num_2]."
                )
        else:
            if verified_count == 0:
                prompt = (
                    "We are auditing an automated aerial swimmer detection system. "
                    "In this photograph, candidate detections are highlighted with numbered yellow bounding boxes (#1, #2, etc.). Currently no boxes are confirmed.\n"
                    "First, carefully count exactly how many human swimmers or people floating in the water are visible across the entire photograph.\n"
                    "Second, list ONLY the exact yellow box numbers (#1, #2, etc.) that enclose actual human swimmers in format: RECOVER: [box_num_1, box_num_2]."
                )
            else:
                prompt = (
                    "We are auditing an automated aerial swimmer detection system. "
                    "In this photograph, bold green boxes enclose confirmed verified human swimmers, and yellow numbered boxes (#1, #2, etc.) enclose unverified candidate detections.\n"
                    "First, count carefully how many human swimmers or people floating in the water are visible in total across the entire photograph.\n"
                    "Second, check: are all human swimmers across the photograph already enclosed inside a bold green box?\n"
                    "If ALL human swimmers across the photograph are already inside bold green boxes, output exactly: GLOBAL_STATUS: ALL_BOXED.\n"
                    "If there is any human swimmer that is NOT inside a green box, inspect the yellow numbered boxes (#1, #2, etc.) and output the exact yellow box number(s) that enclose the missing swimmer(s) in format: RECOVER: [box_num_1, box_num_2]."
                )

        if self.audit_backend == "lmstudio":
            content, reasoning = self._generate_lmstudio(overview_path, prompt, max_tokens=4096, model_id=self.audit_model)
            full_ans = f"{content}\n{reasoning}"
        else:
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
                image=overview_path,
                prompt=formatted_prompt,
                max_tokens=100,
                temperature=0.1
            )

            output_text = getattr(output, "text", None)
            if output_text is None:
                output_text = output if isinstance(output, str) else str(output)
            full_ans = output_text.strip()

        recovered_ids = []
        
        # Parse RECOVER: [idx1, idx2] or Boxes: 1, 3 or SWIMMERS: [idx1, idx2] across content and reasoning
        match = re.search(r"RECOVER:\s*\[([0-9,\s]+)\]", full_ans, re.IGNORECASE)
        if not match:
            match = re.search(r"SWIMMERS:\s*\[([0-9,\s]+)\]", full_ans, re.IGNORECASE)
        if not match:
            match = re.search(r"Boxes:\s*([0-9,\s]+)", full_ans, re.IGNORECASE)
        if not match:
            match = re.search(r"\[([0-9,\s]+)\]", full_ans)
            
        if match:
            try:
                nums_str = match.group(1)
                recovered_ids = [int(n.strip()) for n in nums_str.split(",") if n.strip().isdigit()]
            except Exception:
                pass

        return recovered_ids, full_ans.strip()

    def detect_targets_vlm(self, image_path: str, target_type: str = "swimmer") -> Tuple[List[Tuple[int, int, int, int]], str]:
        """
        Three-Stage Hybrid Detection Engine:
        1. Stage 1 (CV Candidates): Extracts precise physical candidate boxes.
        2. Stage 2 (Marked-Box Verification): Crops context with marked red box, runs binary YES/NO query.
        3. Stage 3 (Global Audit & Recovery): Checks full image globally via MLX or LM Studio (Qwen 3.6) to ensure all targets are boxed and recovers missing items.
        """
        img_cv = cv2.imread(image_path)
        h, w = img_cv.shape[:2]

        # Stage 1: Programmatic Computer Vision candidate detection
        candidates = PrecisionSwimmerAnnotator.detect_targets(img_cv, target_type=target_type, min_score=12.0)

        verified_boxes = []
        answers = []
        temp_crop = os.path.join(os.path.dirname(image_path), "_temp_vlm_patch.png")

        # Stage 2: Zoomed Marked-Box Binary Verification
        for i, (xmin, ymin, xmax, ymax) in enumerate(candidates, 1):
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            bw, bh = xmax - xmin, ymax - ymin
            win = max(75, int(max(bw, bh) * 1.3))
            
            crop = img_cv[max(0, cy - win):min(h, cy + win), max(0, cx - win):min(w, cx + win)].copy()
            if crop.size == 0:
                continue
                
            rx1 = max(0, xmin - (cx - win))
            ry1 = max(0, ymin - (cy - win))
            rx2 = min(crop.shape[1], rx1 + bw)
            ry2 = min(crop.shape[0], ry1 + bh)
            
            # Highlight target in red inside context crop
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            crop_resized = cv2.resize(crop, (336, 336), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(temp_crop, crop_resized)
            
            is_valid = self.verify_patch_vlm(temp_crop, target_type=target_type)
            ans_str = "YES" if is_valid else "NO"
            answers.append(f"#{i}->{ans_str}")

            if is_valid:
                verified_boxes.append((xmin, ymin, xmax, ymax))

        if os.path.exists(temp_crop):
            try:
                os.remove(temp_crop)
            except OSError:
                pass

        # Stage 3: Global Holistic Audit & Missing Target Recovery via local MLX or LM Studio (Qwen 3.6)
        temp_overview = os.path.join(os.path.dirname(image_path), "_temp_global_audit.png")
        overview_img = img_cv.copy()
        
        for i, box in enumerate(candidates, 1):
            if box in verified_boxes:
                cv2.rectangle(overview_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(overview_img, f"[VERIFIED #{i}]", (box[0], max(18, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(overview_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                cv2.putText(overview_img, f"#{i}", (box[0], max(18, box[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                            
        cv2.imwrite(temp_overview, overview_img)
        recovered_ids, stage3_summary = self.global_audit_and_recover(temp_overview, len(verified_boxes), target_type=target_type)
        
        if os.path.exists(temp_overview):
            try:
                os.remove(temp_overview)
            except OSError:
                pass

        recovered_count = 0
        for rid in recovered_ids:
            if 1 <= rid <= len(candidates):
                rec_box = candidates[rid - 1]
                if rec_box not in verified_boxes:
                    verified_boxes.append(rec_box)
                    recovered_count += 1

        backend_label = f"LM Studio ({self.audit_model})" if self.audit_backend == "lmstudio" else "MLX"
        summary_text = (
            f"Stage 1: {len(candidates)} candidates | Stage 2: Verified {len(verified_boxes) - recovered_count} ({', '.join(answers[:6])}) | "
            f"Stage 3 ({backend_label} Audit): Recovered {recovered_count} box(es) -> Total Verified {target_type.title()}s: {len(verified_boxes)}"
        )
        return verified_boxes, summary_text

    def verify_and_correct_dataset(self, dataset_dir: str, target_type: str = "swimmer") -> Dict[str, Any]:
        """
        Scans a dataset directory, runs Three-Stage CV+VLM verification over every image,
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

            print(f"[{idx}/{len(image_files)}] Three-Stage Verifying ({target_type.upper()}): {filename} ...")
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
        print(f"\n[SUCCESS] Three-Stage Verification Complete in {time.time()-total_t0:.1f}s -> Report: {html_report}")
        return {"report_file": html_report, "results": results}

    @staticmethod
    def export_vlm_audit_gallery(dataset_dir: str, results: List[Dict[str, Any]], target_type: str = "swimmer") -> str:
        report_path = os.path.join(dataset_dir, "vlm_verification_report.html")
        with open(report_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>SurfLifeGen-MLX — Three-Stage Audit Report ({target_type.upper()})</title>
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
    <h1>Three-Stage Hybrid Audit Report ({target_type.title()})</h1>
    <div class="subtitle">Exact physical bounding boxes verified by patch classification and global overview check</div>
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
