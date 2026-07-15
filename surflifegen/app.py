# surflifegen/app.py
import os
import sys
import glob
import json
import uuid
import time
import shutil
import logging
import asyncio
import subprocess
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import io
import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surflifegen-web")

app = FastAPI(title="SurfLifeGen-MLX Web Dashboard API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global constants and state
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(REPO_DIR, "web_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

CONFIG_PATH = os.path.join(REPO_DIR, "web_settings.json")
DEFAULT_SETTINGS = {
    "output_dir": "./surflife_dataset",
    "highway_output_dir": "./highway_defect_dataset",
    "yolo_dataset_dir": "./yolo_urban_auto_dataset",
    "epochs": 10,
    "batch_size": 8,
    "steps": 25,
    "box_threshold": 0.20,
    "text_threshold": 0.20,
    "model_arch": "yolov8n-seg.pt",
    "classes": "building, road, grass, car, person, tree, edge of road, sidewalk"
}

# Cached Grounded-SAM segmenter for sandbox
_cached_segmenter = None
_cached_segmenter_classes = None

def get_settings() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict[str, Any]):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

class Job:
    def __init__(self, job_id: str, job_type: str, command: List[str], cwd: str):
        self.job_id = job_id
        self.job_type = job_type
        self.command = command
        self.cwd = cwd
        self.status = "running"
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.exit_code = None
        self.process = None
        self.log_path = os.path.join(JOBS_DIR, f"{job_id}.log")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "command": " ".join(self.command),
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "log_path": self.log_path
        }

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.load_job_history()

    def load_job_history(self):
        history_path = os.path.join(JOBS_DIR, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        # Re-instantiate historical jobs as ended
                        j = Job(item["job_id"], item["job_type"], item["command"].split(), REPO_DIR)
                        j.status = item["status"]
                        j.start_time = item["start_time"]
                        j.end_time = item["end_time"]
                        j.exit_code = item["exit_code"]
                        self.jobs[j.job_id] = j
            except Exception as e:
                logger.error(f"Error loading job history: {e}")

    def save_job_history(self):
        history_path = os.path.join(JOBS_DIR, "history.json")
        try:
            data = [j.to_dict() for j in self.jobs.values()]
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving job history: {e}")

    def start_job(self, job_type: str, command: List[str]) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id, job_type, command, REPO_DIR)
        
        # Ensure log file starts empty
        with open(job.log_path, "w", encoding="utf-8") as f:
            f.write(f"=== JOB STARTED: {job_type.upper()} ===\n")
            f.write(f"Timestamp: {job.start_time}\n")
            f.write(f"Command: {' '.join(command)}\n\n")

        # Spawn background process
        try:
            job.process = subprocess.Popen(
                command,
                cwd=REPO_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.jobs[job_id] = job
            
            # Start background thread to capture logs
            threading.Thread(target=self._log_reader, args=(job,), daemon=True).start()
            self.save_job_history()
            return job_id
        except Exception as e:
            job.status = "failed"
            job.exit_code = -1
            job.end_time = datetime.now().isoformat()
            with open(job.log_path, "a", encoding="utf-8") as f:
                f.write(f"\nFailed to spawn process: {e}\n")
            self.jobs[job_id] = job
            self.save_job_history()
            return job_id

    def _log_reader(self, job: Job):
        # Read from process line by line
        with open(job.log_path, "a", encoding="utf-8", buffering=1) as f:
            for line in iter(job.process.stdout.readline, ""):
                f.write(line)
            
            job.process.stdout.close()
            exit_code = job.process.wait()
            job.exit_code = exit_code
            job.status = "completed" if exit_code == 0 else ("cancelled" if job.status == "cancelled" else "failed")
            job.end_time = datetime.now().isoformat()
            
            f.write(f"\n=== JOB ENDED: {job.status.upper()} (Exit Code: {exit_code}) ===\n")
            
        self.save_job_history()

    def cancel_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        job = self.jobs[job_id]
        if job.status == "running" and job.process:
            job.status = "cancelled"
            job.process.terminate()
            try:
                job.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                job.process.kill()
            self.save_job_history()
            return True
        return False

job_manager = JobManager()

# Settings API
class SettingsModel(BaseModel):
    output_dir: str
    highway_output_dir: str
    yolo_dataset_dir: str
    epochs: int
    batch_size: int
    steps: int
    box_threshold: float
    text_threshold: float
    model_arch: str
    classes: str

@app.get("/api/settings")
def get_settings_endpoint():
    return get_settings()

@app.post("/api/settings")
def save_settings_endpoint(settings: SettingsModel):
    save_settings(settings.dict())
    return {"status": "success", "settings": get_settings()}

# System Status API
@app.get("/api/status")
def get_status():
    import torch
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    # Check local Cosmos3 model path
    cosmos_found = os.path.exists(os.path.join(REPO_DIR, "models", "Cosmos3-Nano-MLX-8bit")) or \
                   os.path.exists(os.path.join(REPO_DIR, "Cosmos3-Nano-MLX-8bit"))

    return {
        "device": device,
        "cosmos_model_found": cosmos_found,
        "venv_python": sys.executable,
        "running_jobs": len([j for j in job_manager.jobs.values() if j.status == "running"])
    }

# File Explorer API
@app.get("/api/files/browse")
def browse_files(directory: Optional[str] = None):
    # Default to browsing REPO_DIR
    target_dir = os.path.abspath(os.path.join(REPO_DIR, directory) if directory else REPO_DIR)
    
    # Security: Restrict traversal outside the main workspace directory
    if not target_dir.startswith(os.path.abspath(REPO_DIR)):
        target_dir = os.path.abspath(REPO_DIR)

    if not os.path.exists(target_dir):
        return {"current_dir": target_dir, "files": [], "dirs": []}

    dirs = []
    files = []
    
    try:
        for entry in os.scandir(target_dir):
            # Ignore hidden folders
            if entry.name.startswith(".") and entry.name != "..":
                continue
            
            rel_path = os.path.relpath(entry.path, REPO_DIR)
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": rel_path})
            else:
                ext = os.path.splitext(entry.name)[1].lower()
                is_media = ext in [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".avi"]
                files.append({
                    "name": entry.name,
                    "path": rel_path,
                    "size_bytes": entry.stat().st_size,
                    "is_media": is_media
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Sort listings alphabetically
    dirs.sort(key=lambda x: x["name"])
    files.sort(key=lambda x: x["name"])

    return {
        "current_dir": os.path.relpath(target_dir, REPO_DIR),
        "dirs": dirs,
        "files": files
    }

# Serving Local Media Files
@app.get("/api/media/{path:path}")
def serve_media(path: str):
    file_path = os.path.abspath(os.path.join(REPO_DIR, path))
    if not file_path.startswith(os.path.abspath(REPO_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

# Jobs API
class JobStartModel(BaseModel):
    job_type: str  # 'generate_swimmers_sharks', 'generate_highway', 'train_yolo', 'segment_video'
    params: Dict[str, Any]

@app.get("/api/jobs")
def get_jobs():
    return [j.to_dict() for j in sorted(job_manager.jobs.values(), key=lambda x: x.start_time, reverse=True)]

@app.post("/api/jobs/start")
def start_job_endpoint(payload: JobStartModel):
    venv_bin = os.path.join(REPO_DIR, ".venv", "bin")
    
    job_type = payload.job_type
    p = payload.params

    if job_type == "generate_swimmers_sharks":
        # Launch bulk generation script
        cmd = [
            os.path.join(venv_bin, "surflifegen"),
            "--target", p.get("target", "swimmer"),
            "--output-dir", p.get("output_dir", "./surflife_dataset"),
            "--steps", str(p.get("steps", 25)),
            "--box-thresh", str(p.get("box_threshold", 0.22)),
            "--text-thresh", str(p.get("text_threshold", 0.22))
        ]
        if p.get("bulk_count"):
            cmd.extend(["--bulk-count", str(p["bulk_count"])])
        if p.get("prompt"):
            cmd.extend(["--prompt", p["prompt"]])
        if p.get("detection_prompt"):
            cmd.extend(["--detection-prompt", p["detection_prompt"]])
        if p.get("no_annotate"):
            cmd.append("--no-annotate")

    elif job_type == "generate_highway":
        cmd = [
            os.path.join(venv_bin, "surflifegen-highway"),
            "--defect-type", p.get("defect_type", "random"),
            "--output-dir", p.get("output_dir", "./highway_defect_dataset"),
            "--count", str(p.get("count", 5)),
            "--steps", str(p.get("steps", 25)),
            "--box-thresh", str(p.get("box_threshold", 0.18)),
            "--text-thresh", str(p.get("text_threshold", 0.18))
        ]
        if p.get("perspective") and p["perspective"] != "random":
            cmd.extend(["--perspective", p["perspective"]])
        if p.get("prompt"):
            cmd.extend(["--prompt", p["prompt"]])
        if p.get("detection_prompt"):
            cmd.extend(["--detection-prompt", p["detection_prompt"]])
        if p.get("no_annotate"):
            cmd.append("--no-annotate")
        if p.get("boxes_only"):
            cmd.append("--boxes-only")

    elif job_type == "train_yolo":
        cmd = [
            os.path.join(venv_bin, "surflifegen-train-yolo"),
            "--epochs", str(p.get("epochs", 10)),
            "--model-arch", p.get("model_arch", "yolov8n-seg.pt")
        ]
        if p.get("video"):
            cmd.extend(["--video", p["video"]])
        if p.get("image_dir"):
            cmd.extend(["--image-dir", p["image_dir"]])
        if p.get("num_frames"):
            cmd.extend(["--num-frames", str(p["num_frames"])])
        if p.get("num_images"):
            cmd.extend(["--num-images", str(p["num_images"])])
        if p.get("output_video"):
            cmd.extend(["--output-video", p["output_video"]])
        if p.get("dataset_dir"):
            cmd.extend(["--dataset-dir", p["dataset_dir"]])

    elif job_type == "segment_video":
        cmd = [
            os.path.join(venv_bin, "surflifegen-urban-live"),
            "--video", p.get("video"),
            "--engine", p.get("engine", "sam"),
            "--thresh", str(p.get("box_threshold", 0.20)),
            "--max-size", str(p.get("max_size", 800)),
            "--stride", str(p.get("stride", 1))
        ]
        if p.get("model"):
            cmd.extend(["--model", p["model"]])
        if p.get("output_video"):
            cmd.extend(["--output-video", p["output_video"]])
        if p.get("classes"):
            cmd.extend(["--classes", p["classes"]])

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported job type: {job_type}")

    job_id = job_manager.start_job(job_type, cmd)
    return {"status": "success", "job_id": job_id}

@app.post("/api/jobs/cancel/{job_id}")
def cancel_job_endpoint(job_id: str):
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not running or not found")
    return {"status": "success"}

# SSE Server-Sent Events Log Streaming
@app.get("/api/jobs/stream/{job_id}")
def stream_job_logs(job_id: str):
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_manager.jobs[job_id]
    
    async def log_generator():
        # Open and tail log file
        try:
            with open(job.log_path, "r", encoding="utf-8") as f:
                # Read all existing logs first
                content = f.read()
                if content:
                    yield f"data: {json.dumps({'type': 'log', 'text': content})}\n\n"
                
                # Keep polling for updates while job is running or until file ends
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"
                    else:
                        # Check status
                        if job.status != "running":
                            # Read any leftover content written at the very end
                            leftover = f.read()
                            if leftover:
                                yield f"data: {json.dumps({'type': 'log', 'text': leftover})}\n\n"
                            yield f"data: {json.dumps({'type': 'end', 'status': job.status, 'exit_code': job.exit_code})}\n\n"
                            break
                        await asyncio.sleep(0.2)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")

# Grounded-SAM Sandbox API
@app.post("/api/sandbox/test")
async def run_sandbox_test(
    image_path: Optional[str] = Form(None),
    classes: str = Form("building, road, grass, car, person, tree, edge of road, sidewalk"),
    box_threshold: float = Form(0.20),
    text_threshold: float = Form(0.20),
    file: Optional[UploadFile] = File(None)
):
    global _cached_segmenter, _cached_segmenter_classes

    # Load or retrieve image bytes
    img_bytes = None
    if file:
        img_bytes = await file.read()
    elif image_path:
        full_path = os.path.abspath(os.path.join(REPO_DIR, image_path))
        if os.path.exists(full_path) and full_path.startswith(os.path.abspath(REPO_DIR)):
            with open(full_path, "rb") as f:
                img_bytes = f.read()
    
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or a valid local image_path")

    # Load image using OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding")

    # Limit image size for quick sandbox response
    h, w, _ = img_bgr.shape
    max_size = 640
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_bgr_resized = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_bgr_resized = img_bgr

    classes_list = [c.strip() for c in classes.split(",") if c.strip()]

    # Lazy-load and cache the UrbanSceneSegmenter
    try:
        from .urban_segmenter import UrbanSceneSegmenter
        # If segmenter is not loaded or classes list has changed, re-initialize
        if _cached_segmenter is None or _cached_segmenter_classes != classes_list:
            logger.info("Initializing/Updating UrbanSceneSegmenter for sandbox testing...")
            _cached_segmenter = UrbanSceneSegmenter(
                dino_model_id="IDEA-Research/grounding-dino-base",
                sam_model_id="facebook/sam-vit-base",
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                classes=classes_list
            )
            _cached_segmenter_classes = classes_list
        else:
            # Dynamically adjust thresholds without reloading models
            _cached_segmenter.box_threshold = box_threshold
            _cached_segmenter.text_threshold = text_threshold
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load zero-shot segmenter models: {e}")

    # Process resized frame
    try:
        # Write to temporary file in JOBS_DIR
        temp_id = str(uuid.uuid4())[:8]
        temp_in_path = os.path.join(JOBS_DIR, f"temp_{temp_id}.png")
        
        # Save resized image BGR to disk
        cv2.imwrite(temp_in_path, img_bgr_resized)
        
        # Run segment_image
        instances, summary = _cached_segmenter.segment_image(
            image_path=temp_in_path,
            output_dir=JOBS_DIR,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            save_overlay=True,
            save_mask=False,
            max_size=640
        )
        
        # Segmented overlay path
        overlay_path = os.path.join(JOBS_DIR, f"temp_{temp_id}_urban_segmented.png")
        
        # Read overlay image and convert to base64
        overlay_b64 = ""
        if os.path.exists(overlay_path):
            with open(overlay_path, "rb") as f:
                overlay_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        overlay_url = f"data:image/png;base64,{overlay_b64}" if overlay_b64 else ""

        # Clean up temporary files
        for p in (temp_in_path, overlay_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        # Format instances for JSON response
        clean_instances = []
        for inst in instances:
            clean_instances.append({
                "class_id": inst["class_id"],
                "class_name": inst["class_name"],
                "score": float(inst["score"]),
                "box": [int(coord) for coord in inst["box"]]
            })

        return {
            "status": "success",
            "overlay_url": overlay_url,
            "instances": clean_instances,
            "width": img_bgr_resized.shape[1],
            "height": img_bgr_resized.shape[0],
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error during sandbox run: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Command Runner helper
def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    # Check if frontend is built, if so serve it
    frontend_dist = os.path.join(REPO_DIR, "web_ui", "dist")
    if os.path.exists(frontend_dist):
        logger.info(f"Serving compiled frontend static files from: {frontend_dist}")
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
    else:
        logger.warning(f"Frontend dist directory not found at '{frontend_dist}'. FastAPI will run in API-only mode.")

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
