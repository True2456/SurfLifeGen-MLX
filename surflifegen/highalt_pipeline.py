"""
High-Altitude (`120m - 400m`) Active Swimmer Synthetic Generation & Grounding Engine.
Specialized pipeline for synthesizing extreme high-altitude surveillance views with
active swimming poses (`freestyle, breaststroke, vigorous splashing`) scaled to exact physical dimensions,
and automatically zero-shot annotated by Grounding DINO + NMS.
"""

import os
import time
import math
import json
import random
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

from .dino_annotator import GroundingDinoAnnotator, DINO_AVAILABLE


class HighAltitudeSwimmerPipeline:
    """
    Orchestrates high-altitude (`120m` to `400m`) maritime scene generation with active swimmers
    and exact Grounding DINO zero-shot bounding box annotation.
    """
    def __init__(
        self,
        output_dir: str = "./highalt_swimmer_dataset",
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = None
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.annotator = None
        if DINO_AVAILABLE:
            try:
                self.annotator = GroundingDinoAnnotator(
                    model_id=dino_model_id,
                    device=device,
                    box_threshold=0.18,  # Lower threshold for tiny high-altitude dots
                    text_threshold=0.18,
                    nms_iou_threshold=0.30
                )
            except Exception as e:
                print(f"[HighAlt Pipeline] Warning: Could not initialize Grounding DINO: {e}")

        # Active swimming attire and action descriptors
        self.active_actions = [
            ("freestyle", (255, 100, 50)),      # Orange lifeguard vest
            ("freestyle", (50, 50, 220)),       # Red swimwear
            ("breaststroke", (40, 240, 255)),   # Yellow rescue shirt
            ("breaststroke", (30, 30, 30)),     # Dark black neoprene wetsuit
            ("vigorous_splash", (255, 120, 0)), # Neon safety rash vest
            ("treading_splash", (40, 40, 200)), # Crimson swim cap & vest
        ]

        self.water_palettes = [
            ("turquoise_clear", (180, 140, 30), (140, 110, 20)),   # BGR base & ripple color
            ("deep_azure_swell", (150, 80, 20), (120, 60, 15)),
            ("emerald_coastal", (130, 150, 30), (100, 120, 20)),
            ("choppy_dark_blue", (160, 90, 30), (130, 70, 22)),
        ]

    def _calculate_swimmer_pixel_size(self, altitude_m: int, fov_deg: float = 60.0, img_width_px: int = 1024) -> int:
        """
        Calculates exact physical pixel length of a 1.8m human swimming at nadir view from given altitude.
        """
        ground_width_m = 2.0 * altitude_m * math.tan(math.radians(fov_deg / 2.0))
        px_per_meter = img_width_px / ground_width_m
        human_length_m = 1.8
        pixel_length = max(12, int(human_length_m * px_per_meter))
        return pixel_length

    def _generate_synthetic_highalt_background(self, width: int = 1024, height: int = 768, water_type: str = "turquoise_clear") -> np.ndarray:
        """
        Generates realistic high-altitude maritime background with deep swell and subtle foam texture.
        """
        base_bgr = (180, 140, 30)
        ripple_bgr = (140, 110, 20)
        for name, base, rpl in self.water_palettes:
            if name == water_type:
                base_bgr, ripple_bgr = base, rpl
                break

        # Base water gradient
        bg = np.full((height, width, 3), base_bgr, dtype=np.uint8)
        
        # Add smooth Perlin-like swell texture via multi-scale Gaussian noise
        noise_sm = np.random.normal(0, 4, (height // 8, width // 8, 3)).astype(np.float32)
        noise_sm = cv2.resize(noise_sm, (width, height), interpolation=cv2.INTER_CUBIC)
        
        noise_lg = np.random.normal(0, 10, (height // 32, width // 32, 3)).astype(np.float32)
        noise_lg = cv2.resize(noise_lg, (width, height), interpolation=cv2.INTER_CUBIC)

        combined = np.clip(bg.astype(np.float32) + noise_sm + noise_lg, 0, 255).astype(np.uint8)

        # Add subtle wave ripples
        num_ripples = random.randint(15, 35)
        for _ in range(num_ripples):
            rx = random.randint(0, width - 1)
            ry = random.randint(0, height - 1)
            rw = random.randint(60, 200)
            rh = random.randint(8, 25)
            angle = random.uniform(0, 180)
            cv2.ellipse(combined, (rx, ry), (rw, rh), angle, 0, 360, ripple_bgr, -1)

        combined = cv2.GaussianBlur(combined, (15, 15), 0)
        return combined

    def _render_active_swimmer_sprite(self, length_px: int, action: str, attire_bgr: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders a physically scaled active swimmer silhouette (freestyle arms, breaststroke legs, splash ring).
        Returns (sprite_bgr, alpha_mask).
        """
        width_px = max(8, int(length_px * 0.5))
        sprite = np.zeros((length_px + 20, length_px + 20, 3), dtype=np.uint8)
        alpha = np.zeros((length_px + 20, length_px + 20), dtype=np.float32)

        cx, cy = (length_px + 20) // 2, (length_px + 20) // 2

        # Draw water displacement splash ring around active swimmer
        cv2.circle(sprite, (cx, cy), int(length_px * 0.65), (245, 245, 250), 2)
        cv2.circle(alpha, (cx, cy), int(length_px * 0.65), 0.45, 2)
        cv2.circle(sprite, (cx, cy), int(length_px * 0.8), (230, 230, 240), 1)
        cv2.circle(alpha, (cx, cy), int(length_px * 0.8), 0.25, 1)

        # Draw torso (attire color)
        torso_len = max(6, int(length_px * 0.55))
        torso_w = max(4, int(width_px * 0.7))
        cv2.ellipse(sprite, (cx, cy), (torso_w, torso_len), 0, 0, 360, attire_bgr, -1)
        cv2.ellipse(alpha, (cx, cy), (torso_w, torso_len), 0, 0, 360, 1.0, -1)

        # Draw head (pinkish skin tone / swim cap)
        head_r = max(3, int(width_px * 0.35))
        head_y = cy - int(torso_len * 0.8)
        cv2.circle(sprite, (cx, head_y), head_r, (180, 200, 240), -1)
        cv2.circle(alpha, (cx, head_y), head_r, 1.0, -1)

        # Draw active swimming limbs (arms splashing out during freestyle/breaststroke)
        if action in ["freestyle", "vigorous_splash"]:
            # Right arm extended forward splashing
            arm_x = cx + int(torso_w * 1.1)
            arm_y = cy - int(torso_len * 0.6)
            cv2.line(sprite, (cx, cy - int(torso_len * 0.3)), (arm_x, arm_y), attire_bgr, max(2, int(width_px * 0.25)))
            cv2.line(alpha, (cx, cy - int(torso_len * 0.3)), (arm_x, arm_y), 1.0, max(2, int(width_px * 0.25)))
            # Splash foam at fingertip
            cv2.circle(sprite, (arm_x, arm_y - 2), max(2, int(head_r * 0.6)), (255, 255, 255), -1)
            cv2.circle(alpha, (arm_x, arm_y - 2), max(2, int(head_r * 0.6)), 0.85, -1)
        elif action == "breaststroke":
            # Both arms extended outward symmetrically
            arm_dx = int(torso_w * 1.3)
            arm_dy = int(torso_len * 0.4)
            cv2.line(sprite, (cx, cy - int(torso_len * 0.2)), (cx - arm_dx, cy - arm_dy), attire_bgr, max(2, int(width_px * 0.25)))
            cv2.line(sprite, (cx, cy - int(torso_len * 0.2)), (cx + arm_dx, cy - arm_dy), attire_bgr, max(2, int(width_px * 0.25)))
            cv2.line(alpha, (cx, cy - int(torso_len * 0.2)), (cx - arm_dx, cy - arm_dy), 1.0, max(2, int(width_px * 0.25)))
            cv2.line(alpha, (cx, cy - int(torso_len * 0.2)), (cx + arm_dx, cy - arm_dy), 1.0, max(2, int(width_px * 0.25)))

        # Legs kicking out behind
        leg_y = cy + int(torso_len * 0.8)
        cv2.line(sprite, (cx - 2, cy + int(torso_len * 0.4)), (cx - 4, leg_y + 4), (160, 180, 220), max(2, int(width_px * 0.22)))
        cv2.line(sprite, (cx + 2, cy + int(torso_len * 0.4)), (cx + 4, leg_y + 4), (160, 180, 220), max(2, int(width_px * 0.22)))
        cv2.line(alpha, (cx - 2, cy + int(torso_len * 0.4)), (cx - 4, leg_y + 4), 0.9, max(2, int(width_px * 0.22)))
        cv2.line(alpha, (cx + 2, cy + int(torso_len * 0.4)), (cx + 4, leg_y + 4), 0.9, max(2, int(width_px * 0.22)))

        return sprite, alpha

    def generate_highalt_scene(
        self,
        altitude_m: int = 200,
        swimmer_count: int = 3,
        water_type: str = "turquoise_clear",
        img_width: int = 1024,
        img_height: int = 768,
        filename_prefix: str = "highalt_scene"
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Synthesizes a high-altitude active swimmer scene and runs Grounding DINO zero-shot annotation.
        Returns (clean_image_path, annotated_image_path, metadata).
        """
        bg = self._generate_synthetic_highalt_background(img_width, img_height, water_type)
        swimmer_px = self._calculate_swimmer_pixel_size(altitude_m, img_width_px=img_width)

        ground_truth_boxes = []
        placed_centers = []

        for _ in range(swimmer_count):
            action, attire_bgr = random.choice(self.active_actions)
            sprite, alpha = self._render_active_swimmer_sprite(swimmer_px, action, attire_bgr)

            # Random rotation for active swimmer orientation
            angle = random.uniform(0, 360)
            sh, sw = sprite.shape[:2]
            M = cv2.getRotationMatrix2D((sw // 2, sh // 2), angle, 1.0)
            rot_sprite = cv2.warpAffine(sprite, M, (sw, sh), flags=cv2.INTER_LINEAR)
            rot_alpha = cv2.warpAffine(alpha, M, (sw, sh), flags=cv2.INTER_LINEAR)

            # Find non-overlapping position
            for _attempt in range(50):
                px = random.randint(sw, img_width - sw - 1)
                py = random.randint(sh, img_height - sh - 1)
                
                if all(math.hypot(px - cx, py - cy) > (swimmer_px * 2.5) for cx, cy in placed_centers):
                    placed_centers.append((px, py))
                    break
            else:
                continue

            # Composite sprite onto background using alpha blend
            x1, y1 = px - sw // 2, py - sh // 2
            x2, y2 = x1 + sw, y1 + sh

            roi = bg[y1:y2, x1:x2].astype(np.float32)
            rot_sprite_f = rot_sprite.astype(np.float32)
            mask_3c = np.dstack([rot_alpha, rot_alpha, rot_alpha])

            blended = roi * (1.0 - mask_3c) + rot_sprite_f * mask_3c
            bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

            # Record exact ground truth bounding box around active swimmer silhouette
            y_indices, x_indices = np.where(rot_alpha > 0.15)
            if len(y_indices) > 0 and len(x_indices) > 0:
                gx1 = x1 + int(np.min(x_indices))
                gy1 = y1 + int(np.min(y_indices))
                gx2 = x1 + int(np.max(x_indices))
                gy2 = y1 + int(np.max(y_indices))
                ground_truth_boxes.append([gx1, gy1, gx2, gy2])

        clean_path = os.path.join(self.output_dir, f"{filename_prefix}_alt{altitude_m}m.png")
        cv2.imwrite(clean_path, bg)

        # Run Grounding DINO zero-shot auto-annotation on the new high-altitude image
        dino_detections = []
        annotated_path = os.path.join(self.output_dir, f"{filename_prefix}_alt{altitude_m}m_dino.png")
        if self.annotator:
            dino_detections, summary = self.annotator.annotate_image(clean_path, output_path=annotated_path, target_type="swimmer")
        else:
            cv2.imwrite(annotated_path, bg)
            summary = "Grounding DINO not available."

        metadata = {
            "image_file": os.path.basename(clean_path),
            "annotated_file": os.path.basename(annotated_path),
            "altitude_m": altitude_m,
            "swimmer_count": len(ground_truth_boxes),
            "swimmer_pixel_length": swimmer_px,
            "ground_truth_boxes": ground_truth_boxes,
            "dino_detections": dino_detections,
            "dino_summary": summary
        }

        meta_path = os.path.join(self.output_dir, f"{filename_prefix}_alt{altitude_m}m.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return clean_path, annotated_path, metadata
