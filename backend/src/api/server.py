import os
import sys
import json
import uuid
import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# IMP-32: Configure loguru — remove default, add rotating file + stdout
logger.remove()
logger.add(
    sys.stdout,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
    colorize=True,
)
logger.add(
    "logs/hawkeye_{time:YYYY-MM-DD}.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
)

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.pipeline import RoadEvaluationPipeline
from src.api.db import init_db, SessionLocal, Project, User

import gradio as gr
from src.api.potholes_app import create_pothole_app
from src.api.potholes_app import process_video as process_pothole_video

# ── JWT Config — IMP-11 (env-based secret) ────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# BUG-02 FIX: Capture the running event loop at startup for background threads
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# IMP-13: Max upload size (default 5 GB, configurable via env)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024 * 1024))

# IMP-12: Allowed video MIME types and extensions
ALLOWED_MIME = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska",
                "video/webm", "video/x-msvideo", "application/octet-stream"}
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# IMP-15: Rate limiter
limiter = Limiter(key_func=get_remote_address)

# --- SCHEMAS ---

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[dict] = None

# --- AUTH UTILS ---

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- APP SETUP ---

# IMP-32: Create logs directory
Path("logs").mkdir(exist_ok=True)

# ── Lifespan: replaces deprecated @app.on_event("startup") ───────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup (before yield) and shutdown (after yield)."""
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    logger.info("Event loop captured for background thread WebSocket dispatch.")
    yield
    # Cleanup on shutdown (close any open WebSocket connections gracefully)
    for ws in list(active_connections.values()):
        try:
            await ws.close()
        except Exception:
            pass
    logger.info("Hawkeye API shutdown complete.")

app = FastAPI(title="Hawkeye AI API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
init_db()

# Mount the Pothole Detection Gradio App
app = gr.mount_gradio_app(app, create_pothole_app(), path="/potholes")

# BUG-16 FIX: Restrict CORS to known frontend origins
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()}

# IMP-13: File size limit middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and "upload" in str(request.url):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024**3)} GB."
            )
    return await call_next(request)

UPLOAD_DIR = Path("data/uploads")
REPORT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

jobs: Dict[str, Dict] = {}
active_connections: Dict[str, WebSocket] = {}

# IMP-10: Cleanup jobs older than 24 hours to prevent memory leak
def _cleanup_old_jobs():
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    to_delete = [
        jid for jid, j in jobs.items()
        if j.get("created_at") and j["created_at"] < cutoff
    ]
    for jid in to_delete:
        jobs.pop(jid, None)
    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} stale job(s) from memory.")

# --- BACKGROUND TASKS ---

def run_pipeline_task(job_id: str, video_path: str, gps_path: Optional[str] = None):
    try:
        pipeline = RoadEvaluationPipeline()
        jobs[job_id]["status"] = "processing"

        def progress_callback(current, total, vis_frame, score, metrics):
            jobs[job_id]["progress"] = round((current / total) * 100, 2)

            update = {
                "type": "progress",
                "current": current,
                "total": total,
                "score": score.score if score else None,
                "metrics": metrics.__dict__ if metrics else None,
                "risk_level": score.risk_level if score else None
            }

            if vis_frame is not None:
                import cv2
                import base64
                # Resize for UI preview to reduce encoding time and websocket payload size
                preview_frame = cv2.resize(vis_frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                update["image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

            # BUG-02 FIX: Use the captured event loop from background thread
            if job_id in active_connections and _event_loop is not None:
                asyncio.run_coroutine_threadsafe(
                    active_connections[job_id].send_json(update),
                    _event_loop
                )

        report = pipeline.run(
            video_path=video_path,
            gps_path=gps_path,
            output_dir=str(REPORT_DIR / job_id),
            progress_callback=progress_callback
        )

        # IMP-24: Inject job_id into report metadata so frontend can use it for downloads
        if report and "metadata" in report:
            report["metadata"]["job_id"] = job_id

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["result"] = report

        # Notify WebSocket of completion
        if job_id in active_connections and _event_loop is not None:
            asyncio.run_coroutine_threadsafe(
                active_connections[job_id].send_json({"type": "complete", "summary": report}),
                _event_loop
            )

        # Save to DB
        db = SessionLocal()
        try:
            new_project = Project(
                id=job_id,
                user_id=jobs[job_id].get("user_id"),
                video_name=report["metadata"]["video"],
                safety_score=report["overall"]["safety_score"],
                risk_level=report["overall"]["risk_level"],
                road_width_avg=report["overall"]["road_width_avg_m"],
                total_estimated_cost=report["maintenance_budget"]["total_estimated_cost_inr"],
                full_report_json=report
            )
            db.add(new_project)
            db.commit()
        except Exception as e:
            logger.error(f"DB Save Error: {e}")
        finally:
            db.close()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Pipeline Error for job {job_id}: {e}", exc_info=True)
        if job_id in active_connections and _event_loop is not None:
            asyncio.run_coroutine_threadsafe(
                active_connections[job_id].send_json({"type": "error", "message": str(e)}),
                _event_loop
            )

def run_pothole_task(job_id: str, video_path: str, conf_threshold: float = 0.50, save_csv: bool = True, save_screenshots: bool = True):
    try:
        jobs[job_id]["status"] = "processing"

        def progress_callback(fraction, desc, vis_frame=None):
            jobs[job_id]["progress"] = round(fraction * 100, 2)

            update = {
                "type": "progress",
                "current": int(fraction * 100),
                "total": 100,
                "message": desc,
            }

            if vis_frame is not None:
                import cv2
                import base64
                preview_frame = cv2.resize(vis_frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                update["image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

            if job_id in active_connections and _event_loop is not None:
                asyncio.run_coroutine_threadsafe(
                    active_connections[job_id].send_json(update),
                    _event_loop
                )

        final_video_path, csv_file_path, screenshots_zip, summary = process_pothole_video(
            video_path=video_path,
            conf_threshold=conf_threshold,
            save_csv=save_csv,
            save_screenshots=save_screenshots,
            progress=progress_callback
        )
        
        result = {
            "video_path": final_video_path,
            "csv_path": csv_file_path,
            "zip_path": screenshots_zip,
            "summary": summary
        }
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["result"] = result

        if job_id in active_connections and _event_loop is not None:
            asyncio.run_coroutine_threadsafe(
                active_connections[job_id].send_json({"type": "complete", "summary": result}),
                _event_loop
            )

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Pothole Pipeline Error for job {job_id}: {e}", exc_info=True)
        if job_id in active_connections and _event_loop is not None:
            asyncio.run_coroutine_threadsafe(
                active_connections[job_id].send_json({"type": "error", "message": str(e)}),
                _event_loop
            )

# --- API ROUTES ---

@app.post("/signup")
async def signup(user: UserCreate):
    db = SessionLocal()
    if db.query(User).filter(User.username == user.username).first():
        db.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    if db.query(User).filter(User.email == user.email).first():
        db.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        new_user = User(
            username=user.username,
            email=user.email,
            hashed_password=User.get_password_hash(user.password)
        )
        db.add(new_user)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=400, detail="Registration failed due to database constraint.")
    finally:
        db.close()
    return {"message": "User created successfully"}

# IMP-15: Rate limit login to 10 attempts per minute
@app.post("/token", response_model=Token)
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = db.query(User).filter(User.username == form_data.username).first()
    db.close()

    if not user or not user.verify_password(form_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    gps_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user)
):
    # IMP-12: File extension validation
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not allowed. Accepted: {', '.join(ALLOWED_EXT)}"
        )

    # IMP-12: MIME type check (content_type can be None on some clients)
    mime = (file.content_type or "").lower()
    if mime and mime not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"MIME type '{mime}' not accepted. Only video files are allowed."
        )

    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}{suffix}"

    # IMP-10: Cleanup stale jobs before accepting new ones
    _cleanup_old_jobs()

    try:
        with file_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                buffer.write(chunk)
    finally:
        await file.close()

    gps_file_path = None
    if gps_file:
        g_suffix = Path(gps_file.filename or "").suffix.lower()
        gps_file_path = UPLOAD_DIR / f"{job_id}_gps{g_suffix}"
        try:
            with gps_file_path.open("wb") as buffer:
                while chunk := await gps_file.read(1024 * 1024):
                    buffer.write(chunk)
        finally:
            await gps_file.close()

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "filename": file.filename,
        "user_id": current_user.id,
        "created_at": datetime.now(tz=timezone.utc),  # IMP-10: For cleanup tracking
    }

    background_tasks.add_task(run_pipeline_task, job_id, str(file_path), str(gps_file_path) if gps_file_path else None)
    return {"job_id": job_id, "message": "Processing started"}

@app.post("/upload_pothole")
async def upload_pothole_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.50),
    save_csv: bool = Form(True),
    save_screenshots: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not allowed. Accepted: {', '.join(ALLOWED_EXT)}"
        )

    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_pothole{suffix}"

    _cleanup_old_jobs()

    try:
        with file_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
    finally:
        await file.close()

    jobs[job_id] = {
        "job_id": job_id,
        "type": "pothole",
        "status": "queued",
        "progress": 0.0,
        "filename": file.filename,
        "user_id": current_user.id,
        "created_at": datetime.now(tz=timezone.utc),
    }

    background_tasks.add_task(run_pothole_task, job_id, str(file_path), conf_threshold, save_csv, save_screenshots)
    return {"job_id": job_id, "message": "Pothole processing started"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        projects = db.query(Project).filter(
            Project.user_id == current_user.id
        ).order_by(Project.processed_at.desc()).all()
        # BUG-12 FIX: Serialize ORM objects to dicts
        return [
            {
                "id": p.id,
                "video_name": p.video_name,
                "processed_at": p.processed_at.isoformat() if p.processed_at else None,
                "safety_score": p.safety_score,
                "risk_level": p.risk_level,
                "road_width_avg": p.road_width_avg,
                "total_estimated_cost": p.total_estimated_cost,
                "full_report_json": p.full_report_json,
            }
            for p in projects
        ]
    finally:
        db.close()

@app.get("/geojson/{job_id}")
async def get_geojson(job_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    project = db.query(Project).filter(Project.id == job_id, Project.user_id == current_user.id).first()
    db.close()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # IMP-28: GeoJSON path now matches the filename that GeoJSONWriter.save() outputs
    geojson_path = REPORT_DIR / job_id / "analysis_track.geojson"
    if not geojson_path.exists():
        # Fallback: try legacy video-name-based path
        geojson_path = REPORT_DIR / job_id / f"{project.video_name}.geojson"

    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON not yet generated for this job.")

    with open(geojson_path, "r") as f:
        return json.load(f)

@app.get("/download/{job_id}/{file_type}")
async def download_report(job_id: str, file_type: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    project = db.query(Project).filter(Project.id == job_id, Project.user_id == current_user.id).first()
    db.close()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if file_type not in ("pdf", "csv", "txt"):
        raise HTTPException(status_code=400, detail="Invalid file type. Use: pdf, csv, txt")

    safe_name = project.video_name.replace(" ", "_")
    candidates = [
        REPORT_DIR / job_id / f"{safe_name}_report.{file_type}",
        REPORT_DIR / job_id / "analysis_results.csv" if file_type == "csv" else None,
    ]

    file_path = next((p for p in candidates if p and p.exists()), None)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"{file_type.upper()} report not found. Analysis may still be running.")

    return FileResponse(
        path=file_path,
        filename=f"Hawkeye_{safe_name}_report.{file_type}",
        media_type="application/octet-stream"
    )

@app.get("/download_pothole/{job_id}/{file_type}")
async def download_pothole_report(job_id: str, file_type: str):
    if job_id not in jobs or jobs[job_id].get("type") != "pothole":
        raise HTTPException(status_code=404, detail="Pothole job not found")
        
    result = jobs[job_id].get("result")
    if not result:
        raise HTTPException(status_code=404, detail="Result not ready")
        
    if file_type == "csv":
        file_path = result["csv_path"]
        media_type = "text/csv"
        filename = f"pothole_detections_{job_id[:8]}.csv"
    elif file_type == "zip":
        file_path = result["zip_path"]
        media_type = "application/zip"
        filename = f"pothole_screenshots_{job_id[:8]}.zip"
    elif file_type == "video":
        file_path = result["video_path"]
        media_type = "video/mp4"
        filename = f"pothole_video_{job_id[:8]}.mp4"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
        
    return FileResponse(path=file_path, filename=filename, media_type=media_type)

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_connections[job_id] = websocket

    # If job already completed before WS connected, send immediately
    if job_id in jobs and jobs[job_id].get("status") == "completed":
        await websocket.send_json({
            "type": "complete",
            "summary": jobs[job_id].get("result", {})
        })

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(job_id, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
