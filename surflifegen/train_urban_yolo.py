# surflifegen/train_urban_yolo.py
"""
Automated End-to-End YOLOv8 Segmentation Training & Video Inference Pipeline (`surflifegen-train-yolo`).
1. Extracts frames from video (`--video`) and/or images from a dataset (`--image-dir`).
2. Runs zero-shot Grounded-SAM (`UrbanSceneSegmenter`) to auto-annotate masks across target classes.
3. Exports high-quality YOLO segmentation polygons (`data.yaml`, `images/train`, `labels/train`).
4. Trains an Ultralytics YOLOv8-Seg model (`yolov8n-seg.pt`) natively on Apple Silicon MPS.
5. Runs the trained YOLO model at 60+ FPS across the target video (`--run-video`)!
"""

import os
import glob
import time
import argparse
import shutil
from typing import List, Optional
import cv2
import numpy as np

from .urban_segmenter import UrbanSceneSegmenter
from .yolo_exporter import YoloDatasetExporter
from .urban_live import UrbanLiveVideoPlayer


class UrbanYoloAutoTrainer:
    """Orchestrates auto-annotation, dataset export, YOLOv8-Seg training, and real-time video inference."""

    def __init__(
        self,
        classes: List[str],
        base_dir: str = "./yolo_urban_auto_dataset",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        max_size: int = 800,
        device: Optional[str] = None
    ):
        self.classes = [c.strip().lower() for c in classes if c.strip()]
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, "raw_annotated")
        self.export_dir = os.path.join(self.base_dir, "yolo_formatted")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_size = max_size
        self.device = device

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def auto_annotate_video_and_images(
        self,
        video_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        num_video_frames: int = 25,
        num_images: int = 25
    ) -> int:
        """Extracts frames/images and generates Grounded-SAM multi-class segmentation masks."""
        print(f"\n[AutoTrainer 🚀] Step 1/3: Auto-Annotating Dataset for {len(self.classes)} Classes...")
        print(f"  👉 Classes: {', '.join(self.classes)}")

        segmenter = UrbanSceneSegmenter(
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            classes=self.classes,
            device=self.device
        )

        extracted_img_paths = []

        # 1. Sample frames from video if provided
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0 and num_video_frames > 0:
                step = max(1, total_frames // num_video_frames)
                print(f"  🎬 Extracting ~{num_video_frames} representative frames from '{os.path.basename(video_path)}' (every {step} frames)...")
                frame_idx = 0
                extracted = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or extracted >= num_video_frames:
                        break
                    if frame_idx % step == 0:
                        v_stem = os.path.splitext(os.path.basename(video_path))[0]
                        f_path = os.path.join(self.raw_dir, f"{v_stem}_f{frame_idx:05d}.jpg")
                        cv2.imwrite(f_path, frame)
                        extracted_img_paths.append(f_path)
                        extracted += 1
                    frame_idx += 1
            cap.release()

        # 2. Sample from external image dir if provided
        if image_dir and os.path.exists(image_dir):
            img_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
            found = []
            for ext in img_exts:
                found.extend(glob.glob(os.path.join(image_dir, ext)))
            if found:
                found.sort()
                step = max(1, len(found) // max(1, num_images))
                selected = found[::step][:num_images]
                print(f"  🖼️ Sampling {len(selected)} images from '{image_dir}'...")
                for p in selected:
                    dest = os.path.join(self.raw_dir, os.path.basename(p))
                    if not os.path.exists(dest):
                        shutil.copy2(p, dest)
                    extracted_img_paths.append(dest)

        # De-duplicate
        extracted_img_paths = list(set(extracted_img_paths))
        extracted_img_paths.sort()

        if not extracted_img_paths:
            raise RuntimeError("No input images or video frames found for auto-annotation!")

        print(f"  🧠 Running zero-shot Grounded-SAM segmentation on {len(extracted_img_paths)} images...")
        t0 = time.time()
        for i, img_path in enumerate(extracted_img_paths, 1):
            ts = time.time()
            segmenter.segment_image(
                image_path=img_path,
                output_dir=self.raw_dir,
                save_overlay=True,
                save_mask=True,
                max_size=self.max_size
            )
            print(f"\r  [{i}/{len(extracted_img_paths)}] Annotated '{os.path.basename(img_path)}' ({time.time()-ts:.1f}s)", end="")

        print(f"\n[AutoTrainer] ✅ Completed auto-annotation of {len(extracted_img_paths)} images in {time.time()-t0:.1f}s!")
        return len(extracted_img_paths)

    def export_to_yolo(self) -> str:
        """Exports the raw segmented masks to Ultralytics YOLOv8-Seg format (`data.yaml`)."""
        print(f"\n[AutoTrainer 📦] Step 2/3: Exporting to Ultralytics YOLOv8-Seg Format...")
        exporter = YoloDatasetExporter(
            dataset_dir=self.raw_dir,
            output_dir=self.export_dir,
            mode="segment",
            split_ratio=0.8,
            use_symlinks=False
        )
        data_yaml_path = exporter.export()
        print(f"[AutoTrainer] ✅ YOLO segmentation dataset exported to: {self.export_dir}")
        print(f"  👉 Configuration file: {data_yaml_path}")
        return data_yaml_path

    def train_yolo_model(
        self,
        data_yaml_path: str,
        model_arch: str = "yolov8n-seg.pt",
        epochs: int = 15,
        imgsz: int = 640,
        batch: int = 8,
        run_name: str = "urban_seg_run"
    ) -> str:
        """Invokes Ultralytics YOLO to train a segmentation model natively on Apple Silicon MPS."""
        print(f"\n[AutoTrainer 🔥] Step 3/3: Training YOLOv8 Segmentation Model ({model_arch}) for {epochs} Epochs...")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install `ultralytics` (`pip install ultralytics`) before training!")

        model = YOLO(model_arch)
        # Train model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device if self.device else "mps",
            name=run_name,
            project=os.path.join(self.base_dir, "runs"),
            exist_ok=True,
            verbose=True
        )

        best_pt = os.path.join(self.base_dir, "runs", run_name, "weights", "best.pt")
        if not os.path.exists(best_pt):
            # Fallback check
            alt_pt = os.path.join("runs", "segment", run_name, "weights", "best.pt")
            if os.path.exists(alt_pt):
                best_pt = alt_pt

        print(f"\n[AutoTrainer] 🏁 Training Complete! Best model weights saved at: {best_pt}")
        return best_pt


def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX: End-to-End Automated YOLOv8 Segmentation Training & Video Inference (`surflifegen-train-yolo`)"
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="Path to video file (.mp4, .mov) to sample frames from and run real-time inference on."
    )
    parser.add_argument(
        "--image-dir", "-i",
        default=None,
        help="Optional directory containing additional images (e.g. VisDrone dataset) to include in training."
    )
    parser.add_argument(
        "--classes", "-c",
        type=str,
        default="building, road, grass, car, person, tree, edge of road, sidewalk",
        help="Comma-separated target urban classes (default: building, road, grass, car, person, tree, edge of road, sidewalk)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=25,
        help="Number of frames to extract from --video for auto-annotation (default: 25)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=25,
        help="Number of images to sample from --image-dir for auto-annotation (default: 25)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=12,
        help="Training epochs for YOLOv8-Seg (default: 12)"
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="yolov8n-seg.pt",
        help="Base YOLO model architecture to fine-tune (default: yolov8n-seg.pt)"
    )
    parser.add_argument(
        "--output-video", "-o",
        type=str,
        default=None,
        help="Path to save the real-time YOLO segmented video output"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./yolo_urban_auto_dataset",
        help="Directory to store raw annotations, YOLO dataset, and training runs (default: ./yolo_urban_auto_dataset)"
    )

    args = parser.parse_args()

    video_path = args.video
    if video_path is None:
        # Check ./videos for latest video
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_dir = os.path.join(repo_dir, "videos")
        if os.path.exists(video_dir):
            v_files = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.mov"))
            if v_files:
                video_path = max(v_files, key=os.path.getmtime)
                print(f"[surflifegen-train-yolo] Auto-selected latest video: '{os.path.basename(video_path)}'")

    classes_list = [c.strip() for c in args.classes.split(",") if c.strip()]

    trainer = UrbanYoloAutoTrainer(
        classes=classes_list,
        base_dir=args.dataset_dir
    )

    # 1. Auto-annotate
    trainer.auto_annotate_video_and_images(
        video_path=video_path,
        image_dir=args.image_dir,
        num_video_frames=args.num_frames,
        num_images=args.num_images
    )

    # 2. Export to YOLO
    data_yaml = trainer.export_to_yolo()

    # 3. Train YOLO
    best_weights = trainer.train_yolo_model(
        data_yaml_path=data_yaml,
        model_arch=args.model_arch,
        epochs=args.epochs,
        run_name="urban_diamond_seg"
    )

    # 4. Run trained model on video
    if video_path and os.path.exists(video_path) and os.path.exists(best_weights):
        print(f"\n[AutoTrainer 🎬] Launching Real-Time YOLO Video Inference with Trained Weights ({best_weights})...")
        player = UrbanLiveVideoPlayer(
            classes=classes_list,
            engine="yolo",
            model_path=best_weights,
            stride=1
        )
        out_vid = args.output_video
        if not out_vid:
            out_vid = os.path.join(os.path.dirname(video_path), f"yolo_realtime_{os.path.basename(video_path)}")
        player.play_video(video_path=video_path, output_video_path=out_vid)


if __name__ == "__main__":
    main()
