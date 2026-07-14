# surflifegen/urban_live.py
"""
UrbanLiveVideoPlayer: Interactive Video & Live Stream Segmenter with Real-Time Number Key Class Toggling.
Supports both Zero-Shot Grounded-SAM (`UrbanSceneSegmenter`) and real-time YOLO (`YOLOv8-Seg` / `YOLOv11-Seg`).
Press 1-9 to toggle individual classes ON/OFF in real time during video playback!
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from .urban_segmenter import UrbanSceneSegmenter


class UrbanLiveVideoPlayer:
    """
    Interactive video processing engine with real-time class layer toggling via keyboard shortcuts.
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        engine: str = "sam",
        model_path: Optional[str] = None,
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        max_size: int = 800,
        stride: int = 1,
        device: Optional[str] = None
    ):
        self.engine = engine.lower()
        self.model_path = model_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_size = max_size
        self.stride = max(1, stride)

        if classes is None:
            self.classes = list(UrbanSceneSegmenter.DEFAULT_CLASSES)
        else:
            self.classes = [c.strip().lower() for c in classes if c.strip()]

        # Active toggles for each class (True = shown, False = hidden)
        self.active_classes: Dict[str, bool] = {cls_name: True for cls_name in self.classes}

        self.segmenter: Optional[UrbanSceneSegmenter] = None
        self.yolo_model = None

        if self.engine == "yolo" or self.model_path is not None:
            try:
                from ultralytics import YOLO
                print(f"[UrbanLive] Loading YOLO segmentation model from '{self.model_path}'...")
                self.yolo_model = YOLO(self.model_path)
                # If YOLO model has names, we map them
                if hasattr(self.yolo_model, "names") and self.yolo_model.names:
                    self.classes = [str(v).lower() for v in self.yolo_model.names.values()]
                    self.active_classes = {cls_name: True for cls_name in self.classes}
                print(f"[UrbanLive] Loaded YOLO model successfully with {len(self.classes)} classes.")
            except ImportError:
                print("[UrbanLive Warning] `ultralytics` not installed or model load failed. Falling back to Grounded-SAM engine.")
                self.engine = "sam"
            except Exception as e:
                print(f"[UrbanLive Warning] Could not load YOLO model ({e}). Falling back to Grounded-SAM engine.")
                self.engine = "sam"

        if self.engine == "sam" and self.yolo_model is None:
            print(f"[UrbanLive] Initializing Zero-Shot Grounded-SAM engine for video processing...")
            self.segmenter = UrbanSceneSegmenter(
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                classes=self.classes,
                device=device
            )
            # Synchronize colors and classes from segmenter
            self.classes = self.segmenter.classes
            self.active_classes = {cls_name: True for cls_name in self.classes}

        print("\n" + "=" * 65)
        print(" 🎮 SURFLIFEGEN LIVE VIDEO SEGMENTER KEYBOARD CONTROLS 🎮")
        print("=" * 65)
        for idx, cls_name in enumerate(self.classes[:9], 1):
            print(f"   [{idx}] Toggle Class : {cls_name.upper()}")
        print("   [0 / A] Toggle     : ALL Classes ON / OFF")
        print("   [SPACE / P] Pause  : Pause playback & inspect / toggle layers")
        print("   [S] Snapshot       : Save current frame, overlay & mask to disk")
        print("   [Q / ESC] Quit     : Exit live video player")
        print("=" * 65 + "\n")

    def _process_frame_sam(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Processes a single BGR video frame using Grounded-SAM."""
        if self.segmenter is None:
            return []

        h, w, _ = frame_bgr.shape
        # Optional resize along long edge for faster video processing
        scale = 1.0
        if max(w, h) > self.max_size and self.max_size > 0:
            scale = float(self.max_size) / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_bgr = frame_bgr

        img_pil = Image.fromarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
        rw, rh = img_pil.size

        # DINO localization
        detection_prompt = " . ".join(self.segmenter.classes) + " ."
        raw_dino = self.segmenter.dino_processor(images=img_pil, text=detection_prompt, return_tensors="pt")
        inputs = {
            k: (v.to(dtype=torch.float32, device=self.segmenter.device) if hasattr(v, "dtype") and v.dtype == torch.float64 else (v.to(self.segmenter.device) if hasattr(v, "to") else v))
            for k, v in raw_dino.items()
        }
        import torch
        with torch.no_grad():
            outputs = self.segmenter.dino_model(**inputs)

        results = self.segmenter.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids", raw_dino.input_ids),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(rh, rw)]
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("text_labels", results["labels"])

        raw_instances = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            if (x2 - x1) * (y2 - y1) < 20:
                continue
            cls_idx, cls_name, cls_color = self.segmenter._resolve_class(label)
            tier = self.segmenter.HIERARCHY_TIERS.get(cls_name, 1)
            raw_instances.append({
                "box": [x1, y1, x2, y2],
                "score": float(score.item()),
                "class_id": cls_idx,
                "class_name": cls_name,
                "color": cls_color,
                "area": (x2 - x1) * (y2 - y1),
                "tier": tier
            })

        if not raw_instances:
            return []

        raw_instances.sort(key=lambda item: (item["tier"], -item["area"]))
        input_boxes = [[[float(coord) for coord in inst["box"]] for inst in raw_instances]]
        raw_sam = self.segmenter.sam_processor(images=img_pil, input_boxes=input_boxes, return_tensors="pt")
        sam_inputs = {
            k: (v.to(dtype=torch.float32, device=self.segmenter.device) if hasattr(v, "dtype") and v.dtype == torch.float64 else (v.to(self.segmenter.device) if hasattr(v, "to") else v))
            for k, v in raw_sam.items()
        }

        with torch.no_grad():
            sam_outputs = self.segmenter.sam_model(**sam_inputs)

        masks_t = self.segmenter.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0]
        scores_t = sam_outputs.iou_scores.cpu()[0]

        instances = []
        for i, inst in enumerate(raw_instances):
            best_idx = torch.argmax(scores_t[i]).item()
            mask_np = masks_t[i, best_idx].numpy().astype(bool)
            if not np.any(mask_np):
                continue

            # If frame was scaled down, resize mask back to original w, h
            if scale != 1.0:
                mask_full = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                bx = [int(coord / scale) for coord in inst["box"]]
            else:
                mask_full = mask_np
                bx = inst["box"]

            instances.append({
                "id": len(instances) + 1,
                "class_id": inst["class_id"],
                "class_name": inst["class_name"],
                "score": inst["score"],
                "box": bx,
                "color": inst["color"],
                "mask": mask_full
            })

        return instances

    def _process_frame_yolo(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Processes a single BGR video frame using Ultralytics YOLO segmentation."""
        if self.yolo_model is None:
            return []

        results = self.yolo_model.predict(frame_bgr, conf=self.box_threshold, verbose=False)[0]
        h, w, _ = frame_bgr.shape
        instances = []

        if results.boxes is None:
            return []

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        masks = results.masks.data.cpu().numpy() if results.masks is not None else None

        for i, (bx, score, cid) in enumerate(zip(boxes, scores, cls_ids)):
            cls_name = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
            # Color lookup or default
            if self.segmenter and cls_name in self.segmenter.class_colors:
                color = self.segmenter.class_colors[cls_name]
            else:
                hue = int((cid * 47) % 180)
                hsv = np.array([[[hue, 220, 240]]], dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

            mask_full = None
            if masks is not None and i < len(masks):
                m = masks[i]
                if m.shape[0] != h or m.shape[1] != w:
                    mask_full = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask_full = m.astype(bool)

            instances.append({
                "id": len(instances) + 1,
                "class_id": int(cid),
                "class_name": cls_name,
                "score": float(score),
                "box": bx.tolist(),
                "color": color,
                "mask": mask_full
            })

        return instances

    def render_overlay(self, frame_bgr: np.ndarray, instances: List[Dict[str, Any]]) -> np.ndarray:
        """Renders only the currently ACTIVE toggled class layers onto the BGR frame."""
        overlay_bgr = frame_bgr.copy()
        h, w, _ = frame_bgr.shape
        class_counts = {cls_name: 0 for cls_name in self.classes}

        for inst in instances:
            cls_name = inst["class_name"]
            # Check if this class is toggled ON
            if not self.active_classes.get(cls_name, True):
                continue

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            color = inst["color"]
            mask_np = inst.get("mask")

            if mask_np is not None and np.any(mask_np):
                colored_mask = np.zeros_like(overlay_bgr)
                colored_mask[mask_np] = color
                overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored_mask, 0.48, 0)

                contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_bgr, contours, -1, color, 2)
            else:
                # Draw box if mask unavailable
                bx = [int(v) for v in inst["box"]]
                cv2.rectangle(overlay_bgr, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)

        # Draw Legend Box with active toggles highlighted
        self._draw_interactive_legend(overlay_bgr, class_counts)
        return overlay_bgr

    def _draw_interactive_legend(self, img_bgr: np.ndarray, class_counts: Dict[str, int]):
        """Draws the legend banner showing hotkeys (1-9) and ON/OFF status."""
        box_width = 280
        box_height = 36 + len(self.classes) * 24
        x_pad, y_pad = 16, 16
        h, w, _ = img_bgr.shape

        sub_img = img_bgr[y_pad:y_pad + box_height, x_pad:x_pad + box_width]
        rect = np.full_like(sub_img, (25, 25, 25))
        cv2.addWeighted(sub_img, 0.20, rect, 0.80, 0, sub_img)

        cv2.rectangle(img_bgr, (x_pad, y_pad), (x_pad + box_width, y_pad + box_height), (240, 240, 240), 1)
        cv2.putText(img_bgr, "LIVE KEYBOARD TOGGLES (1-9)", (x_pad + 12, y_pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        for idx, cls_name in enumerate(self.classes):
            sy = y_pad + 44 + idx * 24
            is_active = self.active_classes.get(cls_name, True)
            color = self.segmenter.class_colors.get(cls_name, (0, 255, 0)) if self.segmenter else (0, 255, 0)
            if not is_active:
                color = (80, 80, 80) # Gray out disabled classes

            key_num = str(idx + 1) if idx < 9 else "-"
            status_str = "ON " if is_active else "OFF"
            count = class_counts.get(cls_name, 0)

            # Draw key box
            cv2.rectangle(img_bgr, (x_pad + 10, sy - 13), (x_pad + 32, sy + 5), color, -1)
            cv2.rectangle(img_bgr, (x_pad + 10, sy - 13), (x_pad + 32, sy + 5), (255, 255, 255), 1)
            cv2.putText(img_bgr, key_num, (x_pad + 17, sy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0) if is_active else (200, 200, 200), 2)

            label_text = f"[{status_str}] {cls_name.upper()} ({count})"
            cv2.putText(img_bgr, label_text, (x_pad + 40, sy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255) if is_active else (130, 130, 130), 1)

    def play_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        fps_delay_ms: int = 1
    ):
        """Opens video file, runs segmentation (every `stride` frames), and handles real-time numkey toggles."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[UrbanLive] Playing '{os.path.basename(video_path)}' ({w}x{h} @ {orig_fps:.1f} FPS | {total_frames} frames)")
        print(f"[UrbanLive] Engine: {self.engine.upper()} | Stride: {self.stride} | Press 1-9 to toggle classes!")

        video_writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, orig_fps / self.stride, (w, h))

        frame_idx = 0
        last_instances: List[Dict[str, Any]] = []
        paused = False
        window_name = f"SurfLifeGen Live Urban Segmenter - {os.path.basename(video_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(w, 1400), min(h, 900))

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[UrbanLive] Reached end of video or stream closed.")
                    break
                frame_idx += 1

                # Only run detection every N frames
                if (frame_idx - 1) % self.stride == 0:
                    t0 = time.time()
                    if self.engine == "yolo" and self.yolo_model is not None:
                        last_instances = self._process_frame_yolo(frame)
                    else:
                        last_instances = self._process_frame_sam(frame)
                    dt = time.time() - t0
                    print(f"\r[Frame {frame_idx}/{total_frames}] Segmented {len(last_instances)} objects in {dt:.2f}s", end="")

            # Render overlay with currently active toggles
            display_frame = self.render_overlay(frame, last_instances)

            if paused:
                cv2.putText(display_frame, "PAUSED (Press SPACE/P to Resume | 1-9 to Toggle)", (w // 2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, display_frame)
            if video_writer and not paused and (frame_idx - 1) % self.stride == 0:
                video_writer.write(display_frame)

            # Wait for key press and check interactive shortcuts
            key = cv2.waitKey(0 if paused else fps_delay_ms) & 0xFF

            # Number keys 1-9 toggle classes
            if ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.classes):
                    cls_name = self.classes[idx]
                    self.active_classes[cls_name] = not self.active_classes[cls_name]
                    status = "ON" if self.active_classes[cls_name] else "OFF"
                    print(f"\n  👉 [Live Toggle] Class '{cls_name.upper()}' (Key {idx+1}) -> {status}")
            elif key == ord('0') or key in (ord('a'), ord('A')):
                # If any is off, turn all ON; if all are ON, turn all OFF
                any_off = any(not v for v in self.active_classes.values())
                for c in self.classes:
                    self.active_classes[c] = any_off
                print(f"\n  👉 [Live Toggle] ALL Classes -> {'ON' if any_off else 'OFF'}")
            elif key in (ord(' '), ord('p'), ord('P'), 27): # Space or P or ESC pause/resume
                paused = not paused
                print(f"\n  👉 [Playback] {'PAUSED' if paused else 'RESUMED'}")
            elif key in (ord('s'), ord('S')):
                snap_dir = os.path.join(os.path.dirname(video_path), "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                snap_img = os.path.join(snap_dir, f"frame_{frame_idx:06d}_segmented.png")
                cv2.imwrite(snap_img, display_frame)
                print(f"\n  📸 [Snapshot] Saved frame {frame_idx} -> {snap_img}")
            elif key in (ord('q'), ord('Q')):
                print("\n[UrbanLive] Exiting video player upon user request.")
                break

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
