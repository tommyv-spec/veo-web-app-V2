# -*- coding: utf-8 -*-
"""
Veo Web App - Main FastAPI Application

Features:
- REST API for job management
- Server-Sent Events for real-time progress
- File upload handling
- Static file serving
- Password protection for private access
"""

import os
import sys
import json
import uuid
import shutil
import secrets
import hashlib
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager


# =============================================================================
# Helper Functions
# =============================================================================
def safe_images_dir(images_dir: Union[str, None]) -> Union[Path, None]:
    """
    Safely convert images_dir to Path, returning None for empty/blank strings.
    
    CRITICAL: Never call Path() on empty strings!
    Path("") becomes Path(".") which searches the current directory,
    leading to errors for Flow jobs where frames are in R2, not local disk.
    """
    if not images_dir or not images_dir.strip():
        return None
    return Path(images_dir)


# =============================================================================
# FFmpeg Setup - Cross-platform (Windows + Linux)
# =============================================================================
def setup_ffmpeg():
    """Set up FFMPEG_BIN and FFPROBE_BIN environment variables."""
    # Check if already set and valid
    if os.environ.get("FFMPEG_BIN"):
        ffmpeg_path = os.environ["FFMPEG_BIN"]
        if Path(ffmpeg_path).exists() or shutil.which(ffmpeg_path):
            print(f"[FFmpeg] Using configured: {ffmpeg_path}")
            return
    
    # Check if ffmpeg is in PATH (Linux/Docker typically)
    ffmpeg_in_path = shutil.which("ffmpeg")
    ffprobe_in_path = shutil.which("ffprobe")
    
    if ffmpeg_in_path:
        os.environ["FFMPEG_BIN"] = ffmpeg_in_path
        if ffprobe_in_path:
            os.environ["FFPROBE_BIN"] = ffprobe_in_path
        print(f"[FFmpeg] Found in PATH: {ffmpeg_in_path}")
        return
    
    # Windows-specific search
    if sys.platform == "win32":
        possible_paths = []
        
        # Check ImageIO_FFMPEG_EXE first (might be set by user)
        if os.environ.get("ImageIO_FFMPEG_EXE"):
            possible_paths.append(os.environ["ImageIO_FFMPEG_EXE"])
        
        # Common Windows installation paths
        possible_paths.extend([
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ])
        
        # Search in C:\ffmpeg for any ffmpeg.exe
        ffmpeg_base = Path(r"C:\ffmpeg")
        if ffmpeg_base.exists():
            for found in ffmpeg_base.glob("**/ffmpeg.exe"):
                possible_paths.append(str(found))
        
        for ffmpeg_path in possible_paths:
            if ffmpeg_path and Path(ffmpeg_path).exists():
                ffmpeg_path = str(ffmpeg_path)
                ffprobe_path = str(Path(ffmpeg_path).parent / "ffprobe.exe")
                
                os.environ["FFMPEG_BIN"] = ffmpeg_path
                if Path(ffprobe_path).exists():
                    os.environ["FFPROBE_BIN"] = ffprobe_path
                
                # Also add to PATH
                bin_dir = str(Path(ffmpeg_path).parent)
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                
                print(f"[FFmpeg] Found: {ffmpeg_path}")
                return
    
    # Linux - ffmpeg should be installed via apt
    print("[FFmpeg] Warning: ffmpeg not found. Install with: apt-get install ffmpeg")

# Run ffmpeg setup
setup_ffmpeg()

# =============================================================================
# Authentication Configuration (Google OAuth)
# =============================================================================
# Set these environment variables for Google OAuth:
# GOOGLE_CLIENT_ID - Google OAuth client ID
# GOOGLE_CLIENT_SECRET - Google OAuth client secret
# SESSION_SECRET - Secret key for sessions (auto-generated if not set)
# APP_URL - Your app URL (e.g., https://your-app.onrender.com)

from auth import (
    GOOGLE_AUTH_ENABLED, oauth, SESSION_SECRET,
    get_current_user, get_optional_user, validate_session,
    handle_google_login, handle_google_callback, delete_session
)

# =============================================================================
# FastAPI Imports and Setup
# =============================================================================
from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, 
    BackgroundTasks, Depends, Query, Request, Response, Cookie, Header
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from config import (
    app_config, VideoConfig, APIKeysConfig, DialogueLine,
    JobStatus, ClipStatus, SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_SIZE_BYTES, AspectRatio, Resolution, Duration,
    ApprovalStatus, api_keys_config
)
from models import (
    init_db, get_db_session, Job, Clip, JobLog, BlacklistEntry,
    get_job_logs_since, add_job_log, User, UserAPIKey
)
from worker import worker, WORKER_VERSION
from error_handler import ErrorCode


# ============ Pydantic Models ============

class DialogueLineInput(BaseModel):
    id: int
    text: str
    start_image_idx: Optional[int] = None  # Storyboard image assignment
    scene_index: Optional[int] = None      # Which scene this clip belongs to
    clip_mode: Optional[str] = "blend"     # 'blend' | 'continue' | 'fresh'
    scene_transition: Optional[str] = None # 'blend' | 'cut' | null (for first scene)


class SceneInput(BaseModel):
    sceneIndex: int
    imageIndex: int
    clipMode: str = "blend"        # 'blend' | 'continue' | 'fresh'
    transition: Optional[str] = None  # 'blend' | 'cut' | null for first scene
    clips: List[int] = []          # List of clip indices in this scene


class VideoConfigInput(BaseModel):
    aspect_ratio: str = "9:16"
    resolution: str = "720p"
    duration: str = "8"
    language: str = "English"
    use_interpolation: bool = True
    use_openai_prompt_tuning: bool = True
    use_frame_vision: bool = True
    max_retries_per_clip: int = 5
    custom_prompt: str = ""  # User's custom prompt when AI is disabled
    user_context: str = ""  # User context for AI prompt generation
    single_image_mode: bool = False  # Use same image for start/end frames
    storyboard_mode: bool = False    # Whether in storyboard mode
    generation_mode: str = "parallel"  # "parallel" (fast) or "sequential" (guaranteed smooth transitions)


class APIKeysInput(BaseModel):
    gemini_keys: List[str] = []
    openai_key: Optional[str] = None


class CreateJobRequest(BaseModel):
    config: VideoConfigInput
    dialogue_lines: List[DialogueLineInput]
    api_keys: APIKeysInput
    job_id: Optional[str] = None  # Use existing upload job_id if provided
    scenes: Optional[List[SceneInput]] = None  # Scene definitions for storyboard mode
    last_frame_index: Optional[int] = None  # Index of image to use as end frame for the video


class JobResponse(BaseModel):
    id: str
    status: str
    progress_percent: float
    total_clips: int
    completed_clips: int
    failed_clips: int
    skipped_clips: int
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class ClipResponse(BaseModel):
    id: int
    clip_index: int
    dialogue_id: int
    dialogue_text: str
    status: str
    retry_count: int
    start_frame: Optional[str]
    end_frame: Optional[str]
    output_filename: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]
    # New approval fields
    approval_status: str = "pending_review"
    generation_attempt: int = 1
    attempts_remaining: int = 2
    redo_reason: Optional[str] = None
    versions: List[Dict] = []
    # Variant fields
    selected_variant: int = 1
    total_variants: int = 0


class RedoRequest(BaseModel):
    reason: Optional[str] = None  # Optional reason for redo
    new_dialogue: Optional[str] = None  # Optional new dialogue text for the clip


class ApprovalResponse(BaseModel):
    clip_id: int
    status: str
    message: str
    attempts_remaining: int


class LogResponse(BaseModel):
    id: int
    created_at: str
    level: str
    category: Optional[str]
    clip_index: Optional[int]
    message: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict] = None


# ============ Application Setup ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    init_db()
    
    # === VERSION INFO - Proves which code is deployed ===
    print(f"[Build] WORKER_VERSION={WORKER_VERSION}", flush=True)
    print(f"[Build] RENDER_GIT_COMMIT={os.environ.get('RENDER_GIT_COMMIT', 'not set')}", flush=True)
    print(f"[Build] IMAGE_TAG={os.environ.get('IMAGE_TAG', 'not set')}", flush=True)
    
    worker.start()
    print("[App] Started")
    
    yield
    
    # Shutdown
    worker.stop()
    print("[App] Shutdown complete")


app = FastAPI(
    title="Veo 3.1 Video Generator",
    description="Web interface for generating videos with Google Veo 3.1",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication Middleware (Google OAuth)
# =============================================================================
from starlette.middleware.base import BaseHTTPMiddleware

# Add session middleware for OAuth (required by authlib)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to protect routes with Google OAuth authentication."""
    
    # Routes that don't require authentication
    PUBLIC_ROUTES = {
        "/login", "/auth/login", "/auth/google/callback", 
        "/auth/me", "/api/health", "/favicon.ico"
    }
    PUBLIC_PREFIXES = {"/static/", "/auth/", "/api/local-worker/"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth if Google OAuth is not configured
        if not GOOGLE_AUTH_ENABLED:
            return await call_next(request)
        
        path = request.url.path
        
        # Allow public routes
        if path in self.PUBLIC_ROUTES:
            return await call_next(request)
        
        # Allow routes with public prefixes
        for prefix in self.PUBLIC_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)
        
        # Check session cookie
        session_token = request.cookies.get("session")
        
        # Debug: log cookie check
        all_cookies = dict(request.cookies)
        if path == "/":
            print(f"[AuthMiddleware] Path: {path}, Cookies: {list(all_cookies.keys())}, Session token present: {bool(session_token)}", flush=True)
        
        if session_token:
            # Need to validate against database
            from models import get_db, User, UserSession
            from auth import validate_session as db_validate_session
            with get_db() as db:
                user = db_validate_session(db, session_token)
                if user and user.is_active:
                    return await call_next(request)
                else:
                    print(f"[AuthMiddleware] Session invalid or user inactive for token: {session_token[:8]}...", flush=True)
        
        # Not authenticated - redirect to login or return 401
        if path.startswith("/api/"):
            return Response(
                content=json.dumps({"detail": "Not authenticated"}),
                status_code=401,
                media_type="application/json"
            )
        else:
            return RedirectResponse(url="/login", status_code=302)

# Add auth middleware (only if Google OAuth is configured)
if GOOGLE_AUTH_ENABLED:
    app.add_middleware(AuthMiddleware)


# ============ Static Files ============

# Create static directory if not exists
static_dir = app_config.base_dir / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============ Version Endpoint (for deployment verification) ============

@app.get("/api/version")
def get_version():
    """Return version info to verify which code is deployed"""
    return {
        "app": "veo-web-app",
        "worker_version": WORKER_VERSION,
        "render_commit": os.environ.get("RENDER_GIT_COMMIT", "not set"),
    }


# ============ Authentication Endpoints (Google OAuth) ============

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login page with Google Sign-In"""
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Veo Studio</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .login-container {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            backdrop-filter: blur(10px);
            text-align: center;
        }
        .logo {
            margin-bottom: 30px;
        }
        .logo h1 {
            font-size: 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .logo .subtitle {
            color: rgba(255,255,255,0.6);
            font-size: 14px;
        }
        .divider {
            display: flex;
            align-items: center;
            margin: 30px 0;
            color: rgba(255,255,255,0.4);
            font-size: 13px;
        }
        .divider::before, .divider::after {
            content: '';
            flex: 1;
            height: 1px;
            background: rgba(255,255,255,0.1);
        }
        .divider span { padding: 0 15px; }
        
        .google-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            width: 100%;
            padding: 14px 20px;
            background: #fff;
            border: none;
            border-radius: 8px;
            color: #333;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .google-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255,255,255,0.2);
        }
        .google-btn:active {
            transform: translateY(0);
        }
        .google-btn svg {
            width: 20px;
            height: 20px;
        }
        .info {
            margin-top: 24px;
            font-size: 13px;
            color: rgba(255,255,255,0.4);
        }
        .error {
            background: rgba(255,59,48,0.2);
            border: 1px solid rgba(255,59,48,0.3);
            color: #ff6b6b;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            display: none;
        }
        .error.show { display: block; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <h1>🎬 Veo Studio</h1>
            <p class="subtitle">AI Video Generation Platform</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <a href="/auth/login" class="google-btn">
            <svg viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Sign in with Google
        </a>
        
        <p class="info">Your jobs and data are private to your account</p>
    </div>
    
    <script>
        // Check for error in URL
        const urlParams = new URLSearchParams(window.location.search);
        const error = urlParams.get('error');
        if (error) {
            const errorEl = document.getElementById('error');
            errorEl.textContent = decodeURIComponent(error);
            errorEl.classList.add('show');
        }
    </script>
</body>
</html>
    """)


@app.get("/auth/login")
async def auth_login(request: Request):
    """Initiate Google OAuth flow"""
    if not GOOGLE_AUTH_ENABLED:
        # If auth disabled, just redirect to home
        return RedirectResponse(url="/", status_code=302)
    
    return await handle_google_login(request)


@app.get("/auth/google/callback")
async def auth_callback(request: Request, db: DBSession = Depends(get_db_session)):
    """Handle Google OAuth callback"""
    if not GOOGLE_AUTH_ENABLED:
        return RedirectResponse(url="/", status_code=302)
    
    try:
        user, session_token = await handle_google_callback(request, db)
        
        print(f"[Auth] Cookie set for user {user.email}, token: {session_token[:8]}...", flush=True)
        
        # Return HTML page that sets cookie via JavaScript (more reliable than Set-Cookie on redirects)
        return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Logging in...</title>
    <script>
        // Set cookie via JavaScript
        document.cookie = "session={session_token}; path=/; max-age={7 * 24 * 3600}; secure; samesite=lax";
        // Redirect to home
        window.location.href = "/";
    </script>
</head>
<body style="background: #1a1a2e; color: white; font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
    <div style="text-align: center;">
        <div style="font-size: 24px; margin-bottom: 10px;">🔐</div>
        <div>Logging in...</div>
    </div>
</body>
</html>
""")
        
    except HTTPException as e:
        # Redirect to login with error
        error_msg = str(e.detail)
        return RedirectResponse(url=f"/login?error={error_msg}", status_code=302)
    except Exception as e:
        print(f"[Auth] Callback error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return RedirectResponse(url="/login?error=Authentication failed", status_code=302)


@app.get("/auth/me")
async def auth_me(request: Request, db: DBSession = Depends(get_db_session)):
    """Get current authenticated user info"""
    if not GOOGLE_AUTH_ENABLED:
        # Return default user when auth is disabled
        return {
            "authenticated": True,
            "user": {
                "id": "default",
                "email": "default@local",
                "name": "Default User",
                "picture": None
            }
        }
    
    session_token = request.cookies.get("session")
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    from auth import validate_session as db_validate_session
    user = db_validate_session(db, session_token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    
    return {
        "authenticated": True,
        "user": user.to_dict()
    }


@app.post("/auth/logout")
async def auth_logout(request: Request, response: Response, db: DBSession = Depends(get_db_session)):
    """Log out and clear session"""
    session_token = request.cookies.get("session")
    
    if session_token:
        delete_session(db, session_token)
    
    response.delete_cookie("session")
    return {"success": True, "message": "Logged out"}


# ============ User API Keys Management ============

class AddAPIKeyRequest(BaseModel):
    key: str
    name: Optional[str] = None


class AddAPIKeysRequest(BaseModel):
    keys: List[str]  # List of API keys


def validate_single_api_key(api_key: str) -> dict:
    """
    Validate a single API key by testing Veo submission.
    Returns: {"status": "working"|"rate_limited"|"invalid", "message": str}
    """
    VEO_MODEL = "veo-3.1-fast-generate-preview"
    TEST_PROMPT = "A calm blue ocean wave gently rolling onto a sandy beach at sunset"
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        # Step 1: Quick check if key works at all
        try:
            models = list(client.models.list())
        except Exception as e:
            error_str = str(e).lower()
            if "suspended" in error_str:
                return {"status": "invalid", "message": "Key suspended"}
            elif "invalid" in error_str or "api_key_invalid" in error_str:
                return {"status": "invalid", "message": "Invalid API key"}
            elif "401" in str(e):
                return {"status": "invalid", "message": "Unauthorized"}
            elif "403" in str(e):
                return {"status": "invalid", "message": "Permission denied"}
            else:
                return {"status": "invalid", "message": f"API error: {str(e)[:50]}"}
        
        # Step 2: Try to submit a Veo generation
        config = types.GenerateVideosConfig(
            aspect_ratio="9:16",
            resolution="720p",
            duration_seconds="8",
        )
        
        operation = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=TEST_PROMPT,
            config=config,
        )
        
        # If we get here, the key can submit to Veo!
        return {"status": "working", "message": "Key working"}
        
    except Exception as e:
        error_str = str(e).lower()
        
        if "429" in str(e) or "resource_exhausted" in error_str:
            return {"status": "rate_limited", "message": "Rate limited (quota exhausted)"}
        elif "suspended" in error_str:
            return {"status": "invalid", "message": "Key suspended"}
        elif "permission" in error_str or "403" in str(e):
            return {"status": "invalid", "message": "No Veo access"}
        elif "404" in str(e) or "not found" in error_str:
            return {"status": "invalid", "message": "Veo model not available"}
        else:
            # Unknown error - treat as rate limited to be safe
            return {"status": "rate_limited", "message": f"Error: {str(e)[:40]}"}


@app.get("/api/user/keys")
async def list_user_api_keys(
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """List all API keys for the current user with status summary"""
    keys = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == current_user.id
    ).order_by(UserAPIKey.created_at.desc()).all()
    
    # Calculate summary
    working = sum(1 for k in keys if k.key_status == "working" and k.is_active)
    rate_limited = sum(1 for k in keys if k.key_status == "rate_limited" and k.is_active)
    invalid = sum(1 for k in keys if k.key_status == "invalid" or not k.is_valid)
    inactive = sum(1 for k in keys if not k.is_active)
    
    return {
        "keys": [k.to_dict() for k in keys],
        "count": len(keys),
        "has_keys": len(keys) > 0,
        "summary": {
            "working": working,
            "rate_limited": rate_limited,
            "invalid": invalid,
            "inactive": inactive,
            "total": len(keys),
        }
    }


@app.post("/api/user/keys")
async def add_user_api_key(
    request: AddAPIKeyRequest,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Add a single API key for the current user - validates immediately"""
    key_value = request.key.strip()
    
    # Basic validation
    if not key_value.startswith("AIza"):
        raise HTTPException(status_code=400, detail="Invalid API key format. Gemini keys start with 'AIza'")
    
    if len(key_value) < 30:
        raise HTTPException(status_code=400, detail="API key is too short")
    
    # Check for duplicate
    existing = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == current_user.id,
        UserAPIKey.key_value == key_value
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="This API key is already added")
    
    # Validate the key with Veo API (with error handling)
    print(f"[API Keys] Validating new key ...{key_value[-6:]}", flush=True)
    try:
        validation = validate_single_api_key(key_value)
    except Exception as e:
        print(f"[API Keys] Validation error: {e}", flush=True)
        # If validation fails, still add the key with unknown status
        validation = {"status": "unknown", "message": f"Validation failed: {str(e)[:50]}"}
    
    # Create new key with validation status
    try:
        new_key = UserAPIKey(
            user_id=current_user.id,
            key_value=key_value,
            key_name=request.name,
            key_suffix=key_value[-6:],
            is_valid=(validation["status"] != "invalid"),
            is_active=True,
            key_status=validation["status"],
            last_error=validation["message"] if validation["status"] != "working" else None,
            last_checked=datetime.utcnow(),
        )
        
        db.add(new_key)
        db.commit()
        db.refresh(new_key)
    except Exception as e:
        print(f"[API Keys] Database error: {e}", flush=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)[:100]}")
    
    status_emoji = {"working": "✅", "rate_limited": "⚠️", "invalid": "❌", "unknown": "❓"}.get(validation["status"], "❓")
    
    return {
        "success": True,
        "key": new_key.to_dict(),
        "validation": validation,
        "message": f"{status_emoji} Key added - {validation['message']}"
    }


@app.post("/api/user/keys/bulk")
async def add_user_api_keys_bulk(
    request: AddAPIKeysRequest,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Add multiple API keys at once"""
    added = []
    skipped = []
    errors = []
    
    for i, key_value in enumerate(request.keys):
        key_value = key_value.strip()
        
        # Skip empty lines
        if not key_value:
            continue
        
        # Basic validation
        if not key_value.startswith("AIza"):
            errors.append(f"Key {i+1}: Invalid format (must start with 'AIza')")
            continue
        
        if len(key_value) < 30:
            errors.append(f"Key {i+1}: Too short")
            continue
        
        # Check for duplicate
        existing = db.query(UserAPIKey).filter(
            UserAPIKey.user_id == current_user.id,
            UserAPIKey.key_value == key_value
        ).first()
        
        if existing:
            skipped.append(f"...{key_value[-6:]}")
            continue
        
        # Create new key
        new_key = UserAPIKey(
            user_id=current_user.id,
            key_value=key_value,
            key_suffix=key_value[-6:],
            is_valid=True,
            is_active=True,
        )
        db.add(new_key)
        added.append(f"...{key_value[-6:]}")
    
    db.commit()
    
    return {
        "success": True,
        "added": len(added),
        "skipped": len(skipped),
        "errors": len(errors),
        "details": {
            "added": added,
            "skipped": skipped,
            "errors": errors,
        },
        "message": f"Added {len(added)} keys, skipped {len(skipped)} duplicates"
    }


@app.delete("/api/user/keys/{key_id}")
async def delete_user_api_key(
    key_id: int,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Delete a user's API key"""
    key = db.query(UserAPIKey).filter(
        UserAPIKey.id == key_id,
        UserAPIKey.user_id == current_user.id
    ).first()
    
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    db.delete(key)
    db.commit()
    
    return {"success": True, "message": "API key deleted"}


@app.put("/api/user/keys/{key_id}/toggle")
async def toggle_user_api_key(
    key_id: int,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Toggle a user's API key active/inactive"""
    key = db.query(UserAPIKey).filter(
        UserAPIKey.id == key_id,
        UserAPIKey.user_id == current_user.id
    ).first()
    
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    key.is_active = not key.is_active
    db.commit()
    
    return {
        "success": True,
        "is_active": key.is_active,
        "message": f"API key {'activated' if key.is_active else 'deactivated'}"
    }


@app.post("/api/user/keys/{key_id}/revalidate")
async def revalidate_user_api_key(
    key_id: int,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Re-validate a user's API key"""
    key = db.query(UserAPIKey).filter(
        UserAPIKey.id == key_id,
        UserAPIKey.user_id == current_user.id
    ).first()
    
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Validate the key
    print(f"[API Keys] Re-validating key ...{key.key_suffix}", flush=True)
    validation = validate_single_api_key(key.key_value)
    
    # Update status
    key.is_valid = (validation["status"] != "invalid")
    key.key_status = validation["status"]
    key.last_error = validation["message"] if validation["status"] != "working" else None
    key.last_checked = datetime.utcnow()
    db.commit()
    
    status_emoji = {"working": "✅", "rate_limited": "⚠️", "invalid": "❌"}.get(validation["status"], "❓")
    
    return {
        "success": True,
        "key": key.to_dict(),
        "validation": validation,
        "message": f"{status_emoji} {validation['message']}"
    }


@app.post("/api/user/keys/revalidate-all")
async def revalidate_all_user_api_keys(
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Re-validate all of user's API keys"""
    keys = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == current_user.id
    ).all()
    
    if not keys:
        return {"success": True, "message": "No keys to validate", "results": []}
    
    results = []
    for key in keys:
        print(f"[API Keys] Re-validating key ...{key.key_suffix}", flush=True)
        validation = validate_single_api_key(key.key_value)
        
        key.is_valid = (validation["status"] != "invalid")
        key.key_status = validation["status"]
        key.last_error = validation["message"] if validation["status"] != "working" else None
        key.last_checked = datetime.utcnow()
        
        results.append({
            "key_suffix": key.key_suffix,
            "status": validation["status"],
            "message": validation["message"]
        })
    
    db.commit()
    
    working = sum(1 for r in results if r["status"] == "working")
    rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
    invalid = sum(1 for r in results if r["status"] == "invalid")
    
    return {
        "success": True,
        "message": f"Validated {len(keys)} keys: {working} working, {rate_limited} rate-limited, {invalid} invalid",
        "summary": {"working": working, "rate_limited": rate_limited, "invalid": invalid},
        "results": results
    }


@app.delete("/api/user/keys")
async def delete_all_user_api_keys(
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Delete all API keys for the current user"""
    count = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == current_user.id
    ).delete()
    
    db.commit()
    
    return {"success": True, "deleted": count, "message": f"Deleted {count} API keys"}


# ============ Root / UI ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Veo Web App</h1><p>UI not found. Place index.html in static/</p>")


# ============ Image Upload ============

@app.post("/api/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    job_id: Optional[str] = Form(None),
):
    """
    Upload images for video generation.
    Creates a new job directory if job_id not provided.
    Images are renamed sequentially to ensure correct ordering.
    """
    # Create or get job directory
    if job_id is None:
        job_id = str(uuid.uuid4())
    
    job_dir = app_config.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Count existing images to continue numbering
    existing_images = [f for f in job_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS]
    next_index = len(existing_images)
    
    uploaded = []
    errors = []
    
    for file in files:
        # Validate file type
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            errors.append({
                "filename": file.filename,
                "error": f"Unsupported format: {ext}",
                "code": ErrorCode.IMAGE_INVALID_FORMAT.value,
            })
            continue
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_IMAGE_SIZE_BYTES:
            errors.append({
                "filename": file.filename,
                "error": f"File too large: {len(content) / 1024 / 1024:.1f}MB",
                "code": ErrorCode.IMAGE_TOO_LARGE.value,
            })
            continue
        
        # Save file with sequential name to ensure correct ordering
        try:
            # Use sequential naming: image_00.png, image_01.png, etc.
            new_filename = f"image_{next_index:02d}{ext}"
            filepath = job_dir / new_filename
            with open(filepath, "wb") as f:
                f.write(content)
            uploaded.append({
                "filename": new_filename,
                "original_filename": file.filename,
                "size": len(content),
                "path": str(filepath),
                "index": next_index,
            })
            next_index += 1
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "code": ErrorCode.FILE_WRITE_ERROR.value,
            })
    
    return {
        "job_id": job_id,
        "uploaded": uploaded,
        "errors": errors,
        "total_uploaded": len(uploaded),
        "total_errors": len(errors),
    }


@app.get("/api/upload/{job_id}/images")
async def list_uploaded_images(job_id: str):
    """List images uploaded for a job"""
    job_dir = app_config.uploads_dir / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    images = []
    for f in job_dir.iterdir():
        if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            images.append({
                "filename": f.name,
                "size": f.stat().st_size,
            })
    
    images.sort(key=lambda x: x["filename"])
    
    return {"job_id": job_id, "images": images, "count": len(images)}


@app.delete("/api/upload/{job_id}")
async def delete_uploaded_images(job_id: str):
    """Delete all uploaded images for a job"""
    job_dir = app_config.uploads_dir / job_id
    
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    return {"status": "deleted", "job_id": job_id}


@app.delete("/api/upload/{job_id}/image/{filename}")
async def delete_single_image(job_id: str, filename: str):
    """Delete a single uploaded image"""
    job_dir = app_config.uploads_dir / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Upload folder not found")
    
    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    file_path = job_dir / safe_filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {safe_filename} not found")
    
    # Delete the file
    file_path.unlink()
    
    # Return remaining images
    remaining = [f.name for f in job_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS]
    remaining.sort()
    
    return {
        "status": "deleted",
        "deleted": safe_filename,
        "remaining": remaining,
        "count": len(remaining)
    }


# ============ Job Management ============

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(
    request: CreateJobRequest,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Create a new video generation job"""
    # Use provided job_id (from upload) or generate new one
    job_id = request.job_id if request.job_id else str(uuid.uuid4())
    
    # Validate images exist
    images_dir = app_config.uploads_dir / job_id
    
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise HTTPException(
            status_code=400,
            detail={"errors": ["No images uploaded. Please upload images first."], "code": ErrorCode.NO_IMAGES.value}
        )
    
    # Create output directory
    output_dir = app_config.outputs_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate config
    config = request.config
    print(f"[main.py] Received config from UI: language={config.language}, user_context='{config.user_context[:50] if config.user_context else 'empty'}'")
    errors = []
    
    if config.resolution == "1080p" and config.duration != "8":
        errors.append("1080p requires 8 second duration")
    
    if config.use_interpolation and config.duration != "8":
        errors.append("Interpolation requires 8 second duration")
    
    if not request.dialogue_lines:
        errors.append("At least one dialogue line is required")
    
    # Get user's API keys (if any)
    user_keys = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == current_user.id,
        UserAPIKey.is_active == True,
        UserAPIKey.is_valid == True
    ).all()
    
    # --- Get user keys (UI first, then DB) ---
    ui_gemini_keys = [k.strip() for k in (request.api_keys.gemini_keys if request.api_keys else []) if k and k.strip()]
    db_gemini_keys = [k.key_value for k in user_keys] if user_keys else []
    user_effective_keys = ui_gemini_keys or db_gemini_keys
    
    # --- Decide backend BEFORE any fallback ---
    from backends.selector import choose_backend_for_job, BackendType, is_flow_enabled
    
    backend = choose_backend_for_job(db, current_user.id, user_effective_keys)
    print(f"[main.py] Backend selected: {backend.value} (user keys: {len(user_effective_keys)})", flush=True)
    
    # --- Validate requirements based on backend ---
    if backend == BackendType.API:
        if not user_effective_keys:
            errors.append("No user API keys provided. Add your Gemini keys in Settings.")
        
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"errors": errors, "code": ErrorCode.INVALID_CONFIG.value}
            )
        
        api_keys_data = {
            "gemini_keys": user_effective_keys,             # ONLY user keys
            "openai_key": api_keys_config.openai_api_key    # server-side ok
        }
        print(f"[main.py] API backend: using {len(user_effective_keys)} user keys", flush=True)
    
    elif backend == BackendType.FLOW:
        if not is_flow_enabled():
            errors.append("Flow backend is not configured/enabled on server.")
            raise HTTPException(
                status_code=500,
                detail={"errors": errors, "code": ErrorCode.INVALID_CONFIG.value}
            )
        
        # Clear any non-key-related errors for Flow
        errors = [e for e in errors if "API key" not in e and "Gemini" not in e]
        
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"errors": errors, "code": ErrorCode.INVALID_CONFIG.value}
            )
        
        api_keys_data = {
            "gemini_keys": [],                              # Flow doesn't need Gemini keys
            "openai_key": api_keys_config.openai_api_key
        }
        print(f"[main.py] FLOW backend: no Gemini keys needed", flush=True)
    
    # Create job record
    config_dict = config.model_dump()
    print(f"[main.py] Creating job with config: language={config_dict.get('language')}, user_context='{config_dict.get('user_context', '')[:50] if config_dict.get('user_context') else 'empty'}'")
    
    # Convert dialogue lines to dict, preserving all clip settings
    dialogue_list = [d.model_dump() for d in request.dialogue_lines]
    print(f"[main.py] Dialogue lines with clip settings: {json.dumps(dialogue_list, indent=2)}")
    
    # Convert scenes if provided (storyboard mode)
    scenes_list = None
    if request.scenes:
        scenes_list = [s.model_dump() for s in request.scenes]
        print(f"[main.py] Scenes structure: {json.dumps(scenes_list, indent=2)}")
    
    # Log last frame index if set
    if request.last_frame_index is not None:
        print(f"[main.py] Last frame index: {request.last_frame_index}")
    
    # For Flow jobs, use a special status that API worker will never pick up
    initial_status = JobStatus.PENDING.value
    if backend == BackendType.FLOW:
        initial_status = "queued_for_flow"  # Special status only Flow worker handles
    
    job = Job(
        id=job_id,
        user_id=current_user.id,  # Associate job with current user
        status=initial_status,
        config_json=json.dumps(config_dict),
        dialogue_json=json.dumps({
            "lines": dialogue_list, 
            "scenes": scenes_list,
            "last_frame_index": request.last_frame_index
        }),
        api_keys_json=json.dumps(api_keys_data),
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        total_clips=len(request.dialogue_lines),
        backend=backend.value,  # Set backend type
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    add_job_log(db, job_id, f"Job created (backend: {backend.value})", "INFO", "system")
    
    # === Upload frames to R2 for ALL jobs (enables reliable redos) ===
    # This ensures frames are available even if local ephemeral storage is cleared
    from backends.storage import is_storage_configured, get_storage
    
    frames_storage_keys = {}  # Will store {filename: r2_key}
    first_frame_local_path = None
    
    if is_storage_configured():
        try:
            storage = get_storage()
            
            # Get all images from the job's images directory
            if images_dir.exists():
                image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
                for img_file in sorted(images_dir.iterdir()):
                    if img_file.suffix.lower() in image_extensions:
                        try:
                            # Keep track of first frame for analysis
                            if not first_frame_local_path:
                                first_frame_local_path = img_file
                            
                            # Upload to R2
                            remote_key = storage.upload_job_frame(
                                job_id, 
                                img_file.name, 
                                img_file
                            )
                            frames_storage_keys[img_file.name] = remote_key
                            print(f"[main.py] Uploaded frame to R2: {remote_key}")
                        except Exception as e:
                            print(f"[main.py] Failed to upload frame {img_file.name}: {e}")
            
            if frames_storage_keys:
                # Store R2 keys in database for redo recovery
                job.frames_storage_keys = json.dumps(frames_storage_keys)
                db.commit()
                add_job_log(db, job_id, f"Uploaded {len(frames_storage_keys)} frames to storage", "INFO", "system")
            else:
                print(f"[main.py] WARNING: No frames uploaded to R2")
        except Exception as e:
            print(f"[main.py] WARNING: Failed to upload frames to R2: {e}")
            # Continue anyway - local files still exist for initial processing
    else:
        print(f"[main.py] INFO: Object storage not configured - redos may fail if server restarts")
        # Still get the first frame for prompt generation
        if images_dir.exists():
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
            for img_file in sorted(images_dir.iterdir()):
                if img_file.suffix.lower() in image_extensions:
                    first_frame_local_path = img_file
                    break
    
    # If Flow backend, generate prompts and additional Flow-specific setup
    if backend == BackendType.FLOW:
        try:
            # === Generate prompts using the same engine as API backend ===
            from pathlib import Path
            from config import VideoConfig
            from veo_generator import (
                build_prompt,
                analyze_frame,
                process_user_context,
                get_default_voice_profile,
            )
            
            # Parse config for prompt generation
            config_data = json.loads(job.config_json) if job.config_json else {}
            language = config_data.get("language", "English")
            
            video_config = VideoConfig(
                aspect_ratio=config_data.get("aspect_ratio", "9:16"),
                resolution=config_data.get("resolution", "720p"),
                duration=config_data.get("duration", "8"),
                language=language,
                use_interpolation=config_data.get("use_interpolation", True),
                use_openai_prompt_tuning=config_data.get("use_openai_prompt_tuning", True),
                use_frame_vision=config_data.get("use_frame_vision", True),
                custom_prompt=config_data.get("custom_prompt", ""),
                user_context=config_data.get("user_context", ""),
            )
            
            # Get OpenAI key from environment (web app already has it)
            openai_key = os.environ.get("OPENAI_API_KEY")
            
            # Process user context once for all clips
            user_context_enriched = {}
            if video_config.user_context and openai_key:
                try:
                    print(f"[main.py] Processing user context for Flow job...")
                    user_context_enriched = process_user_context(
                        video_config.user_context,
                        language,
                        openai_key
                    )
                except Exception as e:
                    print(f"[main.py] User context processing failed: {e}")
            
            # Analyze first frame for voice profile
            frame_analysis = {}
            if first_frame_local_path and openai_key and video_config.use_frame_vision:
                try:
                    print(f"[main.py] Analyzing frame for voice profile...")
                    frame_analysis = analyze_frame(str(first_frame_local_path), openai_key)
                except Exception as e:
                    print(f"[main.py] Frame analysis failed: {e}")
            
            # Get voice profile
            voice_profile = get_default_voice_profile(language, video_config.user_context)
            
            # === FRAME ASSIGNMENT LOGIC (same as API worker) ===
            # 
            # For each clip, we need to determine:
            # 1. START FRAME: Where the clip begins
            # 2. END FRAME: Where the clip ends (can be None for no interpolation)
            #
            # The logic depends on clip_mode:
            #
            # BLEND mode (standard):
            #   - Start: assigned image
            #   - End: depends on NEXT clip (allows smooth transitions)
            #
            # CONTINUE mode:
            #   - Start: extracted from previous clip's last frame (set at runtime)
            #   - End: depends on NEXT clip (allows smooth transitions)
            #
            # FRESH mode:
            #   - Start: always original image
            #   - End: NONE (completely standalone clips, no interpolation)
            
            # Convert frames_storage_keys dict to list (sorted by filename)
            uploaded_frames_list = [frames_storage_keys[k] for k in sorted(frames_storage_keys.keys())] if frames_storage_keys else []
            num_images = len(uploaded_frames_list)
            use_interpolation = config_data.get("use_interpolation", True)
            single_image_mode = num_images == 1
            
            # Parse scenes data if present
            dialogue_raw = json.loads(job.dialogue_json) if job.dialogue_json else {}
            scenes_data = dialogue_raw.get("scenes", []) if isinstance(dialogue_raw, dict) else []
            last_frame_index = dialogue_raw.get("last_frame_index") if isinstance(dialogue_raw, dict) else None
            
            auto_cycle_mode = not scenes_data or len(scenes_data) == 0
            
            print(f"[main.py] Frame assignment: {num_images} images, single_image={single_image_mode}, interpolation={use_interpolation}, auto_cycle={auto_cycle_mode}")
            
            # Build clip info with frame assignments
            clip_info_list = []
            for i, line in enumerate(dialogue_list):
                # Determine clip_mode from line or scene
                clip_mode = line.get("clip_mode")
                scene_idx = line.get("scene_index", 0)
                
                if not clip_mode and scenes_data:
                    for scene in scenes_data:
                        if scene.get("sceneIndex") == scene_idx or scene.get("scene_index") == scene_idx:
                            clip_mode = scene.get("clipMode") or scene.get("mode", "blend")
                            break
                
                if not clip_mode:
                    clip_mode = "blend"
                
                clip_info_list.append({
                    "index": i,
                    "text": line.get("text", ""),
                    "dialogue_id": line.get("id", i + 1),
                    "image_idx": line.get("start_image_idx", i % num_images) if not single_image_mode else 0,
                    "scene_index": scene_idx,
                    "clip_mode": clip_mode,
                    "scene_transition": line.get("scene_transition"),
                })
            
            # Create clips with proper frame assignments
            for i, info in enumerate(clip_info_list):
                start_idx = info["image_idx"]
                clip_mode = info["clip_mode"]
                scene_index = info["scene_index"]
                dialogue_text = info["text"]
                
                # Determine END FRAME based on what comes AFTER this clip
                use_end_frame = False
                actual_end_idx = None
                end_frame_reason = ""
                
                # SINGLE IMAGE MODE: Always use same image as end frame for interpolation
                if single_image_mode and use_interpolation:
                    use_end_frame = True
                    actual_end_idx = start_idx  # Same image for smoother motion
                    end_frame_reason = "single image mode, same frame for interpolation"
                elif use_interpolation:
                    is_last_clip = (i == len(clip_info_list) - 1)
                    scene_transition_handled = False
                    
                    if not is_last_clip:
                        next_info = clip_info_list[i + 1]
                        next_scene = next_info["scene_index"]
                        next_image_idx = next_info["image_idx"]
                        
                        if auto_cycle_mode:
                            # AUTO-CYCLE MODE: Check if next clip uses a different image
                            if next_image_idx != start_idx:
                                use_end_frame = True
                                actual_end_idx = next_image_idx
                                end_frame_reason = f"auto-cycle: blend to next image {next_image_idx + 1}"
                                scene_transition_handled = True
                        elif next_scene != scene_index:
                            # STORYBOARD MODE: Next clip is in DIFFERENT scene
                            next_transition = next_info.get("scene_transition")
                            
                            if next_transition != "cut":
                                use_end_frame = True
                                actual_end_idx = next_info["image_idx"]
                                end_frame_reason = f"scene transition to scene {next_scene} (blend)"
                                scene_transition_handled = True
                            else:
                                use_end_frame = False
                                end_frame_reason = f"scene transition to scene {next_scene} (CUT)"
                                scene_transition_handled = True
                    
                    if not scene_transition_handled:
                        if is_last_clip and last_frame_index is not None and last_frame_index < num_images:
                            use_end_frame = True
                            actual_end_idx = last_frame_index
                            end_frame_reason = f"last clip with explicit end frame"
                        elif is_last_clip and auto_cycle_mode:
                            # Cycle back to different image
                            for offset in range(1, num_images):
                                next_idx = (start_idx + offset) % num_images
                                if next_idx != start_idx:
                                    use_end_frame = True
                                    actual_end_idx = next_idx
                                    end_frame_reason = f"last clip: cycle to image {next_idx + 1}"
                                    break
                            else:
                                use_end_frame = False
                                end_frame_reason = "last clip: single image"
                        elif is_last_clip:
                            use_end_frame = False
                            end_frame_reason = "last clip (storyboard)"
                        elif clip_mode == "blend":
                            if auto_cycle_mode:
                                for offset in range(1, num_images):
                                    next_idx = (start_idx + offset) % num_images
                                    if next_idx != start_idx:
                                        use_end_frame = True
                                        actual_end_idx = next_idx
                                        end_frame_reason = f"blend: cycle to image {next_idx + 1}"
                                        break
                                else:
                                    use_end_frame = False
                                    end_frame_reason = "blend: single image"
                            else:
                                # Storyboard blend - same image for smooth motion
                                use_end_frame = True
                                actual_end_idx = start_idx
                                end_frame_reason = "blend: same image for interpolation"
                        else:
                            # FRESH or CONTINUE mode
                            use_end_frame = False
                            end_frame_reason = f"{clip_mode} mode"
                
                # Get frame keys
                start_frame_key = uploaded_frames_list[start_idx] if uploaded_frames_list else None
                end_frame_key = uploaded_frames_list[actual_end_idx] if use_end_frame and actual_end_idx is not None and uploaded_frames_list else None
                
                print(f"[main.py] Clip {i}: start={start_idx}, end={actual_end_idx if use_end_frame else 'NONE'} ({end_frame_reason})")
                
                # Generate prompt using same function as API backend
                try:
                    prompt_text = build_prompt(
                        dialogue_line=dialogue_text,
                        start_frame_path=Path(first_frame_local_path) if first_frame_local_path else None,
                        end_frame_path=None,
                        clip_index=i,
                        language=language,
                        voice_profile=voice_profile,
                        config=video_config,
                        openai_key=openai_key,
                        frame_analysis=frame_analysis,
                        user_context_override=user_context_enriched,
                    )
                    print(f"[main.py] Generated prompt for clip {i}: {len(prompt_text)} chars")
                except Exception as e:
                    print(f"[main.py] Prompt generation failed for clip {i}, using fallback: {e}")
                    prompt_text = f"""Medium shot, static locked-off camera, sharp focus on subject.
The subject in the frame speaks directly to camera with natural expression.
The character says in {language}, "{dialogue_text}"
Voice: natural voice, clear and authentic.
Style: Raw realistic footage, natural lighting, photorealistic.
No subtitles, no text overlays. No background music. Only the speaker's voice.
(no subtitles)"""
                
                clip = Clip(
                    job_id=job_id,
                    clip_index=i,
                    dialogue_id=info["dialogue_id"],
                    dialogue_text=dialogue_text,
                    status=ClipStatus.PENDING.value,
                    start_frame=start_frame_key,
                    end_frame=end_frame_key,
                    prompt_text=prompt_text,
                )
                db.add(clip)
            
            db.commit()
            print(f"[main.py] Created {len(dialogue_list)} clips with prompts for Flow job")
            add_job_log(db, job_id, f"Generated prompts for {len(dialogue_list)} clips", "INFO", "flow")
            
            # Clear images_dir for Flow jobs - frames are in R2, not local disk
            # This prevents any code from mistakenly trying to access local filesystem
            job.images_dir = ""  # Empty string instead of None (DB has NOT NULL constraint)
            db.commit()
            print(f"[main.py] Cleared images_dir for Flow job (frames stored in R2)")
            
            # NOTE: No Redis queue needed - local Flow worker polls /api/local-worker/jobs/pending directly
            # The job is already in DB with status "queued_for_flow", which the local worker will pick up
            add_job_log(db, job_id, "Job ready for Flow processing", "INFO", "flow")
            print(f"[main.py] Job {job_id} ready for Flow backend (local worker will poll)")
        except Exception as e:
            print(f"[main.py] Failed to setup Flow job: {e}")
            import traceback
            traceback.print_exc()
            add_job_log(db, job_id, f"Failed to setup Flow job: {e}", "ERROR", "flow")
    
    return JobResponse(
        id=job.id,
        status=job.status,
        progress_percent=job.progress_percent,
        total_clips=job.total_clips,
        completed_clips=job.completed_clips,
        failed_clips=job.failed_clips,
        skipped_clips=job.skipped_clips,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@app.get("/api/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """List all jobs for the current user only"""
    query = db.query(Job).filter(
        Job.user_id == current_user.id
    )
    
    if status:
        query = query.filter(Job.status == status)
    
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        JobResponse(
            id=j.id,
            status=j.status,
            progress_percent=j.progress_percent,
            total_clips=j.total_clips,
            completed_clips=j.completed_clips,
            failed_clips=j.failed_clips,
            skipped_clips=j.skipped_clips,
            created_at=j.created_at.isoformat() if j.created_at else None,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
        )
        for j in jobs
    ]


def get_user_job(db: DBSession, job_id: str, user: User) -> Job:
    """Helper to get a job and verify ownership"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Strict ownership check - user must own the job
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return job


def get_user_clip(db: DBSession, clip_id: int, user: User) -> Clip:
    """Helper to get a clip and verify ownership via job"""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    # Verify job ownership - strict check
    job = db.query(Job).filter(Job.id == clip.job_id).first()
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return clip


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get job details"""
    job = get_user_job(db, job_id, current_user)
    
    return JobResponse(
        id=job.id,
        status=job.status,
        progress_percent=job.progress_percent,
        total_clips=job.total_clips,
        completed_clips=job.completed_clips,
        failed_clips=job.failed_clips,
        skipped_clips=job.skipped_clips,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@app.get("/api/jobs/{job_id}/config")
async def get_job_config(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get job configuration for cloning - returns config and dialogue data"""
    job = get_user_job(db, job_id, current_user)
    
    # Parse config and dialogue
    config_data = json.loads(job.config_json) if job.config_json else {}
    dialogue_raw = json.loads(job.dialogue_json) if job.dialogue_json else []
    
    # Handle both old format (list) and new format (dict with lines/scenes)
    if isinstance(dialogue_raw, list):
        dialogue_lines = dialogue_raw
        scenes = None
    else:
        dialogue_lines = dialogue_raw.get("lines", [])
        scenes = dialogue_raw.get("scenes", None)
    
    # Get list of images - check local filesystem first, then R2
    images = []
    
    # Method 1: Check local filesystem (legacy) - use safe_images_dir helper
    images_path = safe_images_dir(job.images_dir)
    if images_path and images_path.exists():
        # Support all common image formats
        for ext in ["png", "jpg", "jpeg", "webp"]:
            for img_file in sorted(images_path.glob(f"image_*.{ext}")):
                images.append({
                    "filename": img_file.name,
                    "url": f"/api/jobs/{job_id}/images/{img_file.name}"
                })
    
    # Method 2: Check R2 storage if no local images found
    if not images:
        try:
            from backends.storage import is_storage_configured, get_storage
            
            if is_storage_configured():
                storage = get_storage()
                # List images in job's R2 folder
                r2_prefix = f"jobs/{job_id}/frames/"
                r2_keys = storage.list_objects(prefix=r2_prefix, max_keys=100)
                
                for key in sorted(r2_keys):
                    filename = key.split("/")[-1]
                    if filename and any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                        images.append({
                            "filename": filename,
                            "url": f"/api/jobs/{job_id}/images/{filename}"
                        })
                
                if images:
                    print(f"[Config] Found {len(images)} images in R2 for job {job_id}", flush=True)
        except Exception as e:
            print(f"[Config] Error checking R2 for images: {e}", flush=True)
    
    return {
        "job_id": job_id,
        "config": config_data,
        "dialogue_lines": dialogue_lines,
        "scenes": scenes,
        "images": images,
        "images_dir": job.images_dir
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Delete a job and its data"""
    job = get_user_job(db, job_id, current_user)
    
    # Cancel if running
    if job.status == JobStatus.RUNNING.value:
        worker.cancel_job(job_id)
    
    # Delete files - use safe_images_dir helper to avoid Path("") issues
    images_dir = safe_images_dir(job.images_dir)
    output_dir = safe_images_dir(job.output_dir)  # Also protect output_dir
    
    if images_dir and images_dir.exists():
        shutil.rmtree(images_dir)
    if output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Delete database records
    db.delete(job)
    db.commit()
    
    return {"status": "deleted", "job_id": job_id}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Cancel a running job"""
    job = get_user_job(db, job_id, current_user)
    
    # Allow cancel even if status already changed (handle race conditions)
    if job.status not in [JobStatus.RUNNING.value, JobStatus.PENDING.value]:
        # Job already completed/failed/cancelled - just return success
        return {"status": job.status, "job_id": job_id, "message": "Job already finished"}
    
    success = worker.cancel_job(job_id)
    
    if success:
        add_job_log(db, job_id, "Job cancelled by user", "INFO", "system")
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Pause a running job"""
    job = get_user_job(db, job_id, current_user)
    
    if job.status != JobStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Job is not running")
    
    success = worker.pause_job(job_id)
    
    if success:
        add_job_log(db, job_id, "Job paused by user", "INFO", "system")
        return {"status": "paused", "job_id": job_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to pause job")


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Resume a paused job - reloads user's current API keys"""
    job = get_user_job(db, job_id, current_user)
    
    if job.status != JobStatus.PAUSED.value:
        raise HTTPException(status_code=400, detail="Job is not paused")
    
    # Reload user's current API keys (they may have added new ones)
    user_keys = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == current_user.id,
        UserAPIKey.is_active == True,
        UserAPIKey.is_valid == True
    ).all()
    
    user_gemini_keys = [k.key_value for k in user_keys] if user_keys else []
    
    # Update job with new keys (user's or fallback to server)
    if user_gemini_keys:
        print(f"[Resume] Reloading {len(user_gemini_keys)} user API keys for job {job_id[:8]}", flush=True)
        api_keys_data = {
            "gemini_keys": user_gemini_keys,
            "openai_key": api_keys_config.openai_api_key
        }
    else:
        print(f"[Resume] Using server API keys for job {job_id[:8]}", flush=True)
        api_keys_data = {
            "gemini_keys": api_keys_config.gemini_api_keys,
            "openai_key": api_keys_config.openai_api_key
        }
    
    # Update job with fresh keys
    job.api_keys_json = json.dumps(api_keys_data)
    db.commit()
    
    add_job_log(db, job_id, f"Job resumed with {len(api_keys_data['gemini_keys'])} API keys", "INFO", "system")
    
    success = worker.resume_job(job_id)
    
    if success:
        return {"status": "resumed", "job_id": job_id, "keys_loaded": len(api_keys_data['gemini_keys'])}
    else:
        raise HTTPException(status_code=500, detail="Failed to resume job")


# ============ Clips ============

def deduplicate_versions(versions_json: str) -> list:
    """Deduplicate versions by version_key (attempt.variant), keeping all unique variants"""
    if not versions_json:
        return []
    versions = json.loads(versions_json)
    seen = {}
    for v in versions:
        # Use version_key if available (new format: "1.1", "1.2")
        # Otherwise, fall back to attempt.variant or just attempt
        version_key = v.get("version_key")
        if not version_key:
            attempt = v.get("attempt", 1)
            variant = v.get("variant", 1)
            version_key = f"{attempt}.{variant}"
        
        # Keep the latest entry for each version_key
        seen[version_key] = v
    
    # Sort by attempt, then variant
    return sorted(seen.values(), key=lambda x: (x.get("attempt", 1), x.get("variant", 1)))

def get_actual_versions_count(clip) -> int:
    """Calculate actual number of successful versions for a clip (including all variants)."""
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    
    # Deduplicate by version_key (attempt.variant)
    seen = {}
    for v in versions:
        version_key = v.get("version_key")
        if not version_key:
            attempt = v.get("attempt", 1)
            variant = v.get("variant", 1)
            version_key = f"{attempt}.{variant}"
        seen[version_key] = v
    
    # Add current if completed and not in list
    current_attempt = clip.generation_attempt or 1
    current_key = f"{current_attempt}.1"  # Main output is always variant 1
    if clip.status == ClipStatus.COMPLETED.value and clip.output_filename and current_key not in seen:
        seen[current_key] = {"attempt": current_attempt, "variant": 1, "filename": clip.output_filename}
    
    return len(seen)

@app.get("/api/jobs/{job_id}/clips", response_model=List[ClipResponse])
async def get_job_clips(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get all clips for a job"""
    job = get_user_job(db, job_id, current_user)
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).order_by(Clip.clip_index).all()
    
    return [
        ClipResponse(
            id=c.id,
            clip_index=c.clip_index,
            dialogue_id=c.dialogue_id,
            dialogue_text=c.dialogue_text,
            status=c.status,
            retry_count=c.retry_count,
            start_frame=c.start_frame,
            end_frame=c.end_frame,
            output_filename=c.output_filename,
            error_code=c.error_code,
            error_message=c.error_message,
            approval_status=c.approval_status or "pending_review",
            generation_attempt=c.generation_attempt or 1,
            attempts_remaining=3 - (c.generation_attempt or 1),
            redo_reason=c.redo_reason,
            versions=deduplicate_versions(c.versions_json),
            # selected_variant: use stored value, or default to latest (total count)
            selected_variant=c.selected_variant if c.selected_variant else get_actual_versions_count(c),
            # total_variants: actual count of successful versions only
            total_variants=get_actual_versions_count(c),
        )
        for c in clips
    ]


# ============ Clip Review & Approval ============

@app.post("/api/clips/{clip_id}/approve", response_model=ApprovalResponse)
async def approve_clip(
    clip_id: int, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Approve a clip - marks it as accepted by the user.
    For 'continue' mode scenes, this allows the next clip to start generating.
    """
    clip = get_user_clip(db, clip_id, current_user)
    
    if clip.status != ClipStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Can only approve completed clips")
    
    if clip.approval_status == "max_attempts":
        raise HTTPException(status_code=400, detail="Clip has reached max attempts - contact support")
    
    # Update approval status
    clip.approval_status = "approved"
    
    # Update versions history
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    for v in versions:
        if v.get("attempt") == clip.generation_attempt:
            v["approved"] = True
    clip.versions_json = json.dumps(versions)
    
    db.commit()
    
    add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 1} approved by user", "INFO", "approval")
    
    # Check if there's a next clip waiting for this approval (continue mode)
    next_clip = db.query(Clip).filter(
        Clip.job_id == clip.job_id,
        Clip.clip_index == clip.clip_index + 1
    ).first()
    
    next_clip_triggered = False
    if next_clip and next_clip.status == ClipStatus.WAITING_APPROVAL.value:
        # Update next clip to PENDING so worker will pick it up
        next_clip.status = ClipStatus.PENDING.value
        db.commit()
        add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 2} now pending (was waiting for clip {clip.clip_index + 1} approval)", "INFO", "approval")
        next_clip_triggered = True
    
    return ApprovalResponse(
        clip_id=clip.id,
        status="approved",
        message="Clip approved" + (" - next clip will start generating" if next_clip_triggered else ""),
        attempts_remaining=3 - clip.generation_attempt
    )


@app.post("/api/clips/{clip_id}/reject", response_model=ApprovalResponse)
async def reject_clip(
    clip_id: int, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Reject a clip without triggering redo.
    User can later choose to redo or leave as rejected.
    """
    clip = get_user_clip(db, clip_id, current_user)
    
    if clip.status != ClipStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Can only reject completed clips")
    
    clip.approval_status = "rejected"
    db.commit()
    
    add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 1} rejected by user", "INFO", "approval")
    
    return ApprovalResponse(
        clip_id=clip.id,
        status="rejected",
        message="Clip has been rejected. You can redo it or leave as is.",
        attempts_remaining=3 - clip.generation_attempt
    )


@app.delete("/api/clips/{clip_id}")
async def delete_clip(
    clip_id: int, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a clip and its video file.
    """
    clip = get_user_clip(db, clip_id, current_user)
    
    job_id = clip.job_id
    clip_index = clip.clip_index
    
    # Delete video file if exists
    if clip.output_path:
        try:
            video_path = Path(clip.output_path)
            if video_path.exists():
                video_path.unlink()
        except Exception as e:
            print(f"Error deleting video file: {e}", flush=True)
    
    # Delete from database
    db.delete(clip)
    db.commit()
    
    # Update job stats
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        remaining_clips = db.query(Clip).filter(Clip.job_id == job_id).count()
        job.total_clips = remaining_clips
        completed = db.query(Clip).filter(Clip.job_id == job_id, Clip.status == ClipStatus.COMPLETED.value).count()
        job.completed_clips = completed
        if remaining_clips > 0:
            job.progress_percent = int((completed / remaining_clips) * 100)
        db.commit()
    
    add_job_log(db, job_id, f"Clip {clip_index + 1} deleted by user", "INFO", "deletion")
    
    return {"success": True, "message": f"Clip {clip_index + 1} deleted"}


@app.post("/api/clips/{clip_id}/select-variant/{variant_num}")
async def select_clip_variant(
    clip_id: int, 
    variant_num: int, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Select a specific variant for a clip.
    variant_num is 1-indexed position in the versions list.
    Handles both old format (attempt only) and new format (attempt.variant).
    Updates output_filename to point to the selected variant's video.
    """
    clip = get_user_clip(db, clip_id, current_user)
    
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    
    # Add current version if it's completed and not already in the list
    current_attempt = clip.generation_attempt or 1
    
    # Check if current output is already in versions BY FILENAME (not by attempt/variant combo)
    already_exists = False
    for v in versions:
        if v.get("filename") == clip.output_filename:
            already_exists = True
            break
    
    if clip.status == ClipStatus.COMPLETED.value and clip.output_filename and not already_exists:
        # Add as new format entry
        versions.append({
            "attempt": current_attempt,
            "variant": 1,
            "version_key": f"{current_attempt}.1",
            "filename": clip.output_filename,
            "url": clip.output_url,
            "generated_at": clip.completed_at.isoformat() if clip.completed_at else None,
            "approved": clip.approval_status == "approved",
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
        })
    
    # Sort versions by attempt, then variant
    versions.sort(key=lambda x: (x.get("attempt", 1), x.get("variant", 1)))
    
    # Save cleaned versions back
    clip.versions_json = json.dumps(versions)
    
    if not versions:
        raise HTTPException(status_code=400, detail="No variants available")
    
    # Check variant is in valid range (1-indexed position)
    if variant_num < 1 or variant_num > len(versions):
        raise HTTPException(status_code=400, detail=f"Variant must be between 1 and {len(versions)}")
    
    # Get variant by position (1-indexed)
    variant = versions[variant_num - 1]
    
    if not variant or not variant.get("filename"):
        raise HTTPException(status_code=404, detail=f"Variant {variant_num} has no video file")
    
    # Update selected variant and output filename
    clip.selected_variant = variant_num  # Store position
    clip.output_filename = variant.get("filename")
    if variant.get("url"):
        clip.output_url = variant.get("url")
    clip.approval_status = "pending_review"  # Reset approval when switching
    db.commit()
    
    add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 1} switched to variant {variant_num}", "INFO", "variant")
    
    # Return full clip data for UI update
    return {
        "success": True,
        "selected_variant": variant_num,
        "filename": variant.get("filename"),
        "total_variants": len(versions),
        "clip": ClipResponse(
            id=clip.id,
            clip_index=clip.clip_index,
            dialogue_id=clip.dialogue_id or 0,
            dialogue_text=clip.dialogue_text or "",
            status=clip.status,
            retry_count=clip.retry_count or 0,
            start_frame=clip.start_frame,
            end_frame=clip.end_frame,
            output_filename=clip.output_filename,
            error_code=clip.error_code,
            error_message=clip.error_message,
            approval_status=clip.approval_status or "pending_review",
            generation_attempt=clip.generation_attempt or 1,
            attempts_remaining=3 - (clip.generation_attempt or 1),
            redo_reason=clip.redo_reason,
            selected_variant=variant_num,
            total_variants=len(versions),
            versions=versions if versions else []
        )
    }


@app.post("/api/clips/{clip_id}/redo", response_model=ApprovalResponse)
async def request_clip_redo(
    clip_id: int, 
    request: RedoRequest = None,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Request a redo for a clip.
    
    - Attempt 1 → 2: Uses same logged parameters
    - Attempt 2 → 3: Uses fresh parameters (no log)
    - Attempt 3: No more redos allowed, must contact support
    
    For Flow backend jobs: sets status to 'flow_redo_queued' (handled by Flow worker)
    For API backend jobs: sets status to 'redo_queued' (handled by API worker)
    """
    clip = get_user_clip(db, clip_id, current_user)
    
    # Get the job early for backend detection
    job = db.query(Job).filter(Job.id == clip.job_id).first()
    is_flow = job and job.backend == 'flow'
    
    # Check if already queued or generating - prevent duplicate requests
    # Accept both redo_queued and flow_redo_queued as "already queued"
    if clip.status in [ClipStatus.REDO_QUEUED.value, ClipStatus.FLOW_REDO_QUEUED.value]:
        return ApprovalResponse(
            clip_id=clip.id,
            status="redo_queued",
            message="Redo already queued - please wait",
            attempts_remaining=3 - clip.generation_attempt
        )
    
    if clip.status == ClipStatus.GENERATING.value:
        raise HTTPException(status_code=400, detail="Clip is currently generating - please wait")
    
    if clip.status == ClipStatus.PENDING.value:
        raise HTTPException(status_code=400, detail="Clip is pending initial generation")
    
    # Allow redo for completed or failed clips
    if clip.status not in [ClipStatus.COMPLETED.value, ClipStatus.FAILED.value]:
        raise HTTPException(status_code=400, detail=f"Can only redo completed or failed clips (current status: {clip.status})")
    
    # Check attempt limit
    if clip.generation_attempt >= 3:
        clip.approval_status = "max_attempts"
        db.commit()
        raise HTTPException(
            status_code=400, 
            detail={
                "code": "MAX_ATTEMPTS_REACHED",
                "message": "Maximum 3 attempts reached. Please contact support for assistance.",
                "support_email": "support@yourdomain.com"
            }
        )
    
    # Save current version to history before redo (avoid duplicates)
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    existing_attempts = [v.get('attempt') for v in versions]
    
    # Only add if this attempt isn't already saved (avoid duplicates from worker)
    if clip.generation_attempt not in existing_attempts and clip.output_filename:
        versions.append({
            "attempt": clip.generation_attempt,
            "filename": clip.output_filename,
            "generated_at": clip.completed_at.isoformat() if clip.completed_at else None,
            "approved": False,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
        })
        clip.versions_json = json.dumps(versions)
    
    # Increment attempt
    new_attempt = clip.generation_attempt + 1
    clip.generation_attempt = new_attempt
    
    # Determine if we use logged params
    # Attempt 2: use logged params (same settings)
    # Attempt 3: fresh generation (no logged params)
    clip.use_logged_params = (new_attempt == 2)
    
    # Set status for redo queue based on backend type
    # This is the KEY SEPARATION: Flow worker only sees flow_redo_queued, API worker only sees redo_queued
    if is_flow:
        clip.status = ClipStatus.FLOW_REDO_QUEUED.value
        status_for_response = "flow_redo_queued"
    else:
        clip.status = ClipStatus.REDO_QUEUED.value
        status_for_response = "redo_queued"
    
    clip.approval_status = "rejected"
    clip.redo_reason = request.reason if request else None
    
    # Clear worker claim fields so the appropriate worker can claim again
    clip.claimed_by_worker = None
    clip.claimed_at = None
    
    # Update dialogue if provided
    if request and request.new_dialogue is not None:
        clip.dialogue_text = request.new_dialogue.strip()
        add_job_log(
            db, clip.job_id,
            f"Clip {clip.clip_index + 1} dialogue updated for redo",
            "INFO", "approval",
            details={"new_dialogue": clip.dialogue_text}
        )
    
    # Clear previous output (keep in versions history)
    clip.output_filename = None
    clip.error_code = None
    clip.error_message = None
    
    # Part C: Add debug log to prove DB state at redo time
    add_job_log(
        db, job.id,
        f"[RedoDebug] clip={clip.id} backend={job.backend} flow_url={'yes' if job.flow_project_url else 'no'} -> status={clip.status}",
        "DEBUG", "redo"
    )
    
    db.commit()
    
    add_job_log(
        db, clip.job_id, 
        f"Clip {clip.clip_index + 1} redo requested (attempt {new_attempt}/3, {'with' if clip.use_logged_params else 'without'} logged params, backend={job.backend})",
        "INFO", "approval",
        details={"reason": request.reason if request else None, "use_logged_params": clip.use_logged_params, "backend": job.backend}
    )
    
    return ApprovalResponse(
        clip_id=clip.id,
        status="redo_queued",  # UI always sees "redo_queued" for display purposes
        message=f"Redo queued (attempt {new_attempt}/3). {'Using same parameters.' if clip.use_logged_params else 'Using fresh parameters.'}",
        attempts_remaining=3 - new_attempt
    )


@app.get("/api/clips/{clip_id}")
async def get_clip(
    clip_id: int, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get a single clip's data"""
    clip = get_user_clip(db, clip_id, current_user)
    
    return {
        "id": clip.id,
        "clip_index": clip.clip_index,
        "dialogue_id": clip.dialogue_id,
        "dialogue_text": clip.dialogue_text or "",
        "status": clip.status,
        "approval_status": clip.approval_status,
        "generation_attempt": clip.generation_attempt,
        "attempts_remaining": 3 - clip.generation_attempt,
    }


@app.get("/api/clips/{clip_id}/versions")
async def get_clip_versions(
    clip_id: int, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get all generated versions of a clip"""
    clip = get_user_clip(db, clip_id, current_user)
    
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    
    # Add current version if completed
    if clip.status == ClipStatus.COMPLETED.value and clip.output_filename:
        versions.append({
            "attempt": clip.generation_attempt,
            "filename": clip.output_filename,
            "generated_at": clip.completed_at.isoformat() if clip.completed_at else None,
            "approved": clip.approval_status == "approved",
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "current": True,
        })
    
    return {
        "clip_id": clip_id,
        "dialogue_id": clip.dialogue_id,
        "total_attempts": clip.generation_attempt,
        "attempts_remaining": 3 - clip.generation_attempt,
        "versions": versions,
    }


@app.post("/api/jobs/{job_id}/cleanup-versions")
async def cleanup_clip_versions(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Clean up duplicate versions in all clips of a job.
    Call this to fix clips that have duplicate entries in versions_json.
    """
    job = get_user_job(db, job_id, current_user)
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).all()
    cleaned_count = 0
    
    for clip in clips:
        if not clip.versions_json:
            continue
            
        versions = json.loads(clip.versions_json)
        original_count = len(versions)
        
        # Deduplicate by version_key (attempt.variant) - keeps 1.1 and 1.2 separate
        seen = {}
        for v in versions:
            # Build version key from attempt.variant
            attempt = v.get("attempt", 1)
            variant = v.get("variant", 1)
            version_key = v.get("version_key") or f"{attempt}.{variant}"
            
            # Also dedupe by filename as backup (in case version_key is missing)
            filename = v.get("filename", "")
            key = version_key if version_key else filename
            
            if key:
                seen[key] = v
        
        cleaned_versions = sorted(seen.values(), key=lambda x: (x.get("attempt", 1), x.get("variant", 1)))
        
        if len(cleaned_versions) < original_count:
            clip.versions_json = json.dumps(cleaned_versions)
            cleaned_count += 1
            print(f"[Cleanup] Clip {clip.clip_index}: {original_count} -> {len(cleaned_versions)} versions", flush=True)
    
    db.commit()
    
    add_job_log(db, job_id, f"Cleaned up versions for {cleaned_count} clips", "INFO", "cleanup")
    
    return {
        "success": True,
        "clips_cleaned": cleaned_count,
        "total_clips": len(clips)
    }


@app.get("/api/jobs/{job_id}/review-status")
async def get_job_review_status(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get summary of clip approval statuses for a job"""
    job = get_user_job(db, job_id, current_user)
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).all()
    
    summary = {
        "total": len(clips),
        "pending_review": 0,
        "approved": 0,
        "redo_queued": 0,
        "max_attempts": 0,
        "generating": 0,
        "failed": 0,
    }
    
    for c in clips:
        if c.status == ClipStatus.COMPLETED.value:
            if c.approval_status == "approved":
                summary["approved"] += 1
            elif c.approval_status == "max_attempts":
                summary["max_attempts"] += 1
            else:
                summary["pending_review"] += 1
        elif c.status in [ClipStatus.REDO_QUEUED.value, ClipStatus.FLOW_REDO_QUEUED.value]:
            summary["redo_queued"] += 1
        elif c.status in [ClipStatus.GENERATING.value, ClipStatus.PENDING.value]:
            summary["generating"] += 1
        elif c.status == ClipStatus.FAILED.value:
            summary["failed"] += 1
    
    summary["all_approved"] = summary["approved"] == summary["total"]
    # Can export if we have approved clips (even if some failed)
    summary["can_export"] = summary["approved"] > 0 and summary["generating"] == 0 and summary["redo_queued"] == 0 and summary["pending_review"] == 0
    summary["needs_attention"] = summary["max_attempts"] > 0 or summary["failed"] > 0
    
    return summary


# ============ Logs ============

@app.get("/api/jobs/{job_id}/logs", response_model=List[LogResponse])
async def get_job_logs(
    job_id: str,
    since_id: int = 0,
    limit: int = Query(default=100, le=500),
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get logs for a job (supports polling with since_id)"""
    job = get_user_job(db, job_id, current_user)
    
    logs = get_job_logs_since(db, job_id, since_id)[:limit]
    
    return [
        LogResponse(
            id=log.id,
            created_at=log.created_at.isoformat() if log.created_at else "",
            level=log.level,
            category=log.category,
            clip_index=log.clip_index,
            message=log.message,
        )
        for log in logs
    ]


# ============ Server-Sent Events ============

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_events(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Stream job events via Server-Sent Events.
    
    Events:
    - progress: Clip progress update
    - clip_started: Clip generation started
    - clip_completed: Clip generation completed
    - error: Error occurred
    - job_completed: Job finished
    """
    job = get_user_job(db, job_id, current_user)
    
    async def event_generator():
        event_queue = worker.subscribe(job_id)
        
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'progress': job.progress_percent})}\n\n"
            
            while True:
                try:
                    # Non-blocking check
                    event = event_queue.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # Stop streaming if job completed
                    if event.get("type") == "job_completed":
                        break
                        
                except Exception:
                    # Send keepalive
                    yield f": keepalive\n\n"
                    
                    # Check if job is still active
                    from models import get_db
                    with get_db() as check_db:
                        check_job = check_db.query(Job).filter(Job.id == job_id).first()
                        if check_job and check_job.status in [
                            JobStatus.COMPLETED.value,
                            JobStatus.FAILED.value,
                            JobStatus.CANCELLED.value,
                        ]:
                            break
        finally:
            worker.unsubscribe(job_id, event_queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============ Downloads ============

@app.get("/api/jobs/{job_id}/outputs")
async def list_outputs(
    job_id: str, 
    approved_only: bool = False,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    List generated videos for a job.
    
    If approved_only=True, only returns videos from approved clips (selected variants).
    Falls back to filesystem listing if job not in database (e.g., after server restart).
    Supports both local filesystem and R2 storage.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    
    # Try to find output directory even without database entry
    if job:
        output_dir = Path(job.output_dir)
    else:
        # Fallback: check if directory exists directly
        output_dir = app_config.outputs_dir / job_id
    
    videos = []
    
    if approved_only and job:
        # Only return approved clips' selected variants (requires DB)
        clips = db.query(Clip).filter(
            Clip.job_id == job_id,
            Clip.approval_status == "approved"
        ).order_by(Clip.clip_index).all()
        
        for clip in clips:
            if clip.output_filename:
                # Check local filesystem first
                filepath = output_dir / clip.output_filename if output_dir.exists() else None
                if filepath and filepath.exists():
                    videos.append({
                        "filename": clip.output_filename,
                        "size": filepath.stat().st_size,
                        "url": f"/api/jobs/{job_id}/outputs/{clip.output_filename}",
                        "clip_index": clip.clip_index,
                        "variant": clip.selected_variant,
                    })
                elif clip.output_url:
                    # R2 storage - use the API endpoint which proxies from R2
                    videos.append({
                        "filename": clip.output_filename,
                        "size": 0,  # Unknown size for R2 files
                        "url": f"/api/jobs/{job_id}/outputs/{clip.output_filename}",
                        "clip_index": clip.clip_index,
                        "variant": clip.selected_variant,
                        "storage": "r2"
                    })
    else:
        # Return all videos from filesystem
        if output_dir.exists():
            for f in output_dir.glob("*.mp4"):
                # Try to extract clip index from filename (e.g., "1_image_00_..." -> clip 1)
                clip_idx = None
                try:
                    parts = f.stem.split("_")
                    if parts[0].isdigit():
                        clip_idx = int(parts[0])
                except:
                    pass
                
                videos.append({
                    "filename": f.name,
                    "size": f.stat().st_size,
                    "url": f"/api/jobs/{job_id}/outputs/{f.name}",
                    "clip_index": clip_idx,
                })
        
        # Also check R2 for Flow job outputs
        if job and job.backend == "flow":
            try:
                from backends.storage import is_storage_configured, get_storage
                
                if is_storage_configured():
                    storage = get_storage()
                    r2_prefix = f"jobs/{job_id}/outputs/"
                    r2_keys = storage.list_objects(prefix=r2_prefix, max_keys=100)
                    
                    existing_filenames = {v["filename"] for v in videos}
                    
                    for key in r2_keys:
                        filename = key.split("/")[-1]
                        if filename and filename.endswith(".mp4") and filename not in existing_filenames:
                            # Extract clip index from filename (clip_0.mp4 -> 0)
                            clip_idx = None
                            try:
                                if filename.startswith("clip_"):
                                    clip_idx = int(filename.replace("clip_", "").replace(".mp4", ""))
                            except:
                                pass
                            
                            videos.append({
                                "filename": filename,
                                "size": 0,
                                "url": f"/api/jobs/{job_id}/outputs/{filename}",
                                "clip_index": clip_idx,
                                "storage": "r2"
                            })
            except Exception as e:
                print(f"[Outputs] R2 list error: {e}", flush=True)
    
    videos.sort(key=lambda x: x.get("clip_index") or 0 if approved_only else x["filename"])
    
    return {"job_id": job_id, "videos": videos, "count": len(videos)}


@app.get("/api/jobs/{job_id}/outputs/{filename}")
async def download_output(
    job_id: str, 
    filename: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Download a generated video. Works with local filesystem or R2 storage."""
    from fastapi.responses import Response
    
    job = get_user_job(db, job_id, current_user)
    
    # Method 1: Check local filesystem first
    output_dir = Path(job.output_dir)
    filepath = output_dir / filename
    
    if filepath.exists():
        return FileResponse(
            filepath,
            media_type="video/mp4",
            filename=filename,
        )
    
    # Method 2: Check R2 storage (for Flow jobs)
    try:
        from backends.storage import is_storage_configured, get_storage
        
        if is_storage_configured():
            storage = get_storage()
            r2_key = f"jobs/{job_id}/outputs/{filename}"
            
            if storage.exists(r2_key):
                # Download from R2 and serve
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp_path = tmp.name
                
                storage.download_file(r2_key, tmp_path)
                
                with open(tmp_path, 'rb') as f:
                    content = f.read()
                
                os.remove(tmp_path)
                
                return Response(
                    content=content,
                    media_type="video/mp4",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
    except Exception as e:
        print(f"[Download] R2 error: {e}", flush=True)
    
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/jobs/{job_id}/missing-clips")
async def download_missing_clips(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Download the missing clips Excel file for celebrity-filtered clips."""
    job = get_user_job(db, job_id, current_user)
    
    output_dir = Path(job.output_dir)
    
    # Try xlsx first, then csv, then json (fallback)
    for ext, media_type in [
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("csv", "text/csv"),
        ("json", "application/json")
    ]:
        filepath = output_dir / f"missing_clips.{ext}"
        if filepath.exists():
            return FileResponse(
                filepath,
                media_type=media_type,
                filename=f"missing_clips.{ext}",
            )
    
    raise HTTPException(status_code=404, detail="Missing clips file not found")


# ============ Final Video Export ============

class ExportSettings(BaseModel):
    frames_to_cut_start: int = Field(default=7, ge=0, le=30)
    frames_to_cut_end: int = Field(default=7, ge=0, le=30)
    smart_trim: bool = True  # Don't trim first clip / cut-to scenes
    remove_silence: bool = False
    vad_threshold: float = Field(default=0.5, ge=0.1, le=0.9)
    vad_min_gap: float = Field(default=1.0, ge=0.1, le=5.0)
    vad_pad_before: float = Field(default=0.1, ge=0.0, le=1.0)
    vad_pad_after: float = Field(default=0.2, ge=0.0, le=1.0)
    # Individual audio enhancement toggles
    remove_laughter: bool = False  # noisereduce (treats laughter as noise)
    denoise_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    apply_deepfilter: bool = False  # DeepFilterNet (removes hiss/static)
    apply_voice_filter: bool = False  # Compressor, gate, limiter
    apply_loudnorm: bool = False  # EBU R128 -16 LUFS
    # Legacy (backwards compatibility)
    enhance_audio: bool = False


@app.post("/api/jobs/{job_id}/export-final")
async def export_final_video(
    job_id: str,
    settings: ExportSettings,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Export all approved clips as a single final video.
    Optionally applies trimming and Voice Activity Detection (VAD).
    
    Works even after server restart by falling back to filesystem.
    
    Rules for start frame trimming:
    - Never trim start frames from the FIRST clip (clip_index 0)
    - Never trim start frames from clips that start a "cut" transition scene
    """
    from video_processor import export_final_video as process_export, check_vad_available
    
    job = get_user_job(db, job_id, current_user)
    
    # Determine output directory
    output_dir = Path(job.output_dir)
    dialogue_json = job.dialogue_json
    
    # Get approved clips from database
    clip_info = []
    cut_scene_first_clips = set()
    
    clips = db.query(Clip).filter(
        Clip.job_id == job_id,
        Clip.approval_status == "approved"
    ).order_by(Clip.clip_index).all()
    
    if not clips:
        raise HTTPException(status_code=400, detail="No approved clips to export")
    
    # Parse scenes for smart trim
    try:
        dialogue_data = json.loads(dialogue_json) if dialogue_json else {}
        scenes = dialogue_data.get("scenes", [])
        
        if scenes and settings.smart_trim:
            for scene in scenes:
                transition = scene.get("transition", None)
                scene_clips = scene.get("clips", [])
                if transition == "cut" and scene_clips:
                    first_clip_of_scene = min(scene_clips)
                    cut_scene_first_clips.add(first_clip_of_scene)
                    print(f"[Export] Scene with 'cut' transition starts at clip {first_clip_of_scene}")
    except Exception as e:
        print(f"[Export] Warning: Could not parse scenes: {e}")
    
    # Import storage helper
    from backends.storage import is_storage_configured, get_storage
    storage = None
    if is_storage_configured():
        try:
            storage = get_storage()
        except Exception as e:
            print(f"[Export] Storage init warning: {e}")
    
    # Collect clip file paths - download from R2 if needed
    for clip in clips:
        if clip.output_filename:
            clip_path = output_dir / clip.output_filename
            
            # Check if file exists locally, if not try R2
            if not clip_path.exists() and storage:
                try:
                    r2_key = f"jobs/{job_id}/outputs/{clip.output_filename}"
                    if storage.exists(r2_key):
                        print(f"[Export] Downloading clip {clip.clip_index} from R2: {clip.output_filename}")
                        # Ensure output directory exists
                        output_dir.mkdir(parents=True, exist_ok=True)
                        storage.download_file(r2_key, str(clip_path))
                except Exception as e:
                    print(f"[Export] R2 download error for clip {clip.clip_index}: {e}")
            
            if clip_path.exists():
                skip_start_trim = False
                if settings.smart_trim:
                    skip_start_trim = (clip.clip_index == 0 or clip.clip_index in cut_scene_first_clips)
                
                clip_info.append({
                    "path": clip_path,
                    "clip_index": clip.clip_index,
                    "skip_start_trim": skip_start_trim
                })
                
                if skip_start_trim:
                    print(f"[Export] Clip {clip.clip_index}: SKIP start frame trim")
    
    # Check VAD availability if requested
    if settings.remove_silence and not check_vad_available():
        raise HTTPException(
            status_code=400,
            detail="VAD requires torch and numpy. Install with: pip install torch numpy"
        )
    
    if not clip_info:
        # Debug: Log what we tried
        for clip in clips:
            print(f"[Export] DEBUG: Clip {clip.clip_index} output_filename={clip.output_filename}, approval_status={clip.approval_status}")
        print(f"[Export] DEBUG: Storage configured={is_storage_configured() if 'is_storage_configured' in dir() else 'N/A'}, storage={storage is not None}")
        raise HTTPException(status_code=400, detail="No valid clip files found")
    
    print(f"[Export] Smart trim: {settings.smart_trim}, Start frames: {settings.frames_to_cut_start}, End frames: {settings.frames_to_cut_end}")
    
    # Create output filename with unique suffix to prevent collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]  # 6 char random suffix
    output_filename = f"final_export_{timestamp}_{unique_suffix}.mp4"
    output_path = output_dir / output_filename
    
    try:
        print(f"[Export] Starting export for job {job_id}")
        print(f"[Export] Clips to process: {len(clip_info)}")
        print(f"[Export] Output path: {output_path}")
        
        # Process the export with per-clip trim settings (non-blocking)
        stats = await asyncio.to_thread(
            process_export,
            clip_info=clip_info,
            output_path=output_path,
            frames_to_cut_start=settings.frames_to_cut_start,
            frames_to_cut_end=settings.frames_to_cut_end,
            remove_silence=settings.remove_silence,
            vad_threshold=settings.vad_threshold,
            vad_min_gap=settings.vad_min_gap,
            vad_pad_before=settings.vad_pad_before,
            vad_pad_after=settings.vad_pad_after
        )
        
        print(f"[Export] Success! Stats: {stats}")
        
        # Apply audio enhancement if any audio toggle is enabled
        any_audio_enabled = settings.remove_laughter or settings.apply_deepfilter or settings.apply_voice_filter or settings.apply_loudnorm
        
        if any_audio_enabled:
            try:
                enabled_steps = []
                if settings.remove_laughter: enabled_steps.append(f"laughter({settings.denoise_strength})")
                if settings.apply_deepfilter: enabled_steps.append("deepfilter")
                if settings.apply_voice_filter: enabled_steps.append("voicefilter")
                if settings.apply_loudnorm: enabled_steps.append("loudnorm")
                print(f"[Export] Applying audio enhancement: {', '.join(enabled_steps)}")
                
                # Enhance the exported video
                enhanced_path = output_dir / f"enhanced_{output_filename}"
                
                from audio_processor import enhance_audio
                audio_stats = await asyncio.to_thread(
                    enhance_audio,
                    output_path,
                    enhanced_path,
                    remove_laughter=settings.remove_laughter,
                    denoise_strength=settings.denoise_strength,
                    apply_deepfilter=settings.apply_deepfilter,
                    apply_voice_filter=settings.apply_voice_filter,
                    apply_loudnorm=settings.apply_loudnorm
                )
                
                if audio_stats.get("enhanced"):
                    # Replace original with enhanced
                    import os
                    os.replace(enhanced_path, output_path)
                    stats["audio_enhanced"] = True
                    stats["audio_stats"] = audio_stats
                    print(f"[Export] Audio enhancement applied: {audio_stats}")
                else:
                    print(f"[Export] Audio enhancement skipped: {audio_stats.get('reason')}")
                    stats["audio_enhanced"] = False
            except Exception as e:
                print(f"[Export] Audio enhancement failed (non-fatal): {e}")
                import traceback
                traceback.print_exc()
                stats["audio_enhanced"] = False
                stats["audio_error"] = str(e)
        
        return {
            "success": True,
            "filename": output_filename,
            "download_url": f"/api/jobs/{job_id}/outputs/{output_filename}",
            "stats": stats
        }
        
    except Exception as e:
        import traceback
        print(f"[Export] ERROR: {str(e)}")
        print(f"[Export] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/vad-available")
async def check_vad_availability():
    """Check if VAD dependencies are installed."""
    from video_processor import check_vad_available
    return {"available": check_vad_available()}


@app.get("/api/audio-enhance-available")
async def check_audio_enhance_availability():
    """Check if audio enhancement dependencies are installed."""
    try:
        import numpy
        import soundfile
        import noisereduce
        return {"available": True}
    except ImportError:
        return {"available": False}


@app.get("/api/jobs/{job_id}/export-audio/{filename}")
async def export_audio_from_video(
    job_id: str,
    filename: str,
    enhance: bool = True,
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Export audio from a video file as WAV.
    Use this to download audio for external processing (OpenVoice, ElevenLabs).
    
    Args:
        job_id: Job ID
        filename: Video filename (e.g., "final_export_xxx.mp4")
        enhance: Apply basic noise reduction before export
    """
    from audio_processor import export_audio_only
    
    job = get_user_job(db, job_id, current_user)
    output_dir = Path(job.output_dir)
    
    video_path = output_dir / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Create audio output path
    audio_filename = f"{video_path.stem}_audio.wav"
    audio_path = output_dir / audio_filename
    
    try:
        success = export_audio_only(video_path, audio_path, enhance=enhance)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to export audio")
        
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=audio_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio export failed: {str(e)}")


@app.post("/api/jobs/{job_id}/import-audio/{video_filename}")
async def import_audio_to_video(
    job_id: str,
    video_filename: str,
    audio_file: UploadFile = File(...),
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Import external audio into a video (replace existing audio).
    Use this after processing audio with OpenVoice, ElevenLabs, etc.
    
    Args:
        job_id: Job ID
        video_filename: Original video filename to replace audio in
        audio_file: New audio file (WAV or MP3)
    
    Returns:
        New video file with replaced audio
    """
    from audio_processor import import_audio
    
    job = get_user_job(db, job_id, current_user)
    output_dir = Path(job.output_dir)
    
    video_path = output_dir / video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Save uploaded audio
    audio_ext = Path(audio_file.filename).suffix or ".wav"
    temp_audio = output_dir / f"temp_imported_audio{audio_ext}"
    
    try:
        # Save uploaded file
        content = await audio_file.read()
        with open(temp_audio, "wb") as f:
            f.write(content)
        
        # Create output with new audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        output_filename = f"voice_swapped_{timestamp}_{unique_suffix}.mp4"
        output_path = output_dir / output_filename
        
        success = import_audio(video_path, temp_audio, output_path)
        
        # Cleanup temp file
        if temp_audio.exists():
            temp_audio.unlink()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to import audio")
        
        return {
            "success": True,
            "filename": output_filename,
            "download_url": f"/api/jobs/{job_id}/outputs/{output_filename}"
        }
        
    except Exception as e:
        if temp_audio.exists():
            temp_audio.unlink()
        raise HTTPException(status_code=500, detail=f"Audio import failed: {str(e)}")


@app.get("/api/voice-clone-available")
async def check_voice_clone_availability():
    """Check if voice cloning (Replicate) is configured"""
    from voice_cloner import check_replicate_available
    return check_replicate_available()


@app.post("/api/voice-clone-warmup")
async def warmup_voice_clone():
    """
    Trigger warmup of voice clone server (Modal).
    Call this early (e.g., when Export Final is clicked) so the server is warm
    by the time the user wants to voice clone.
    """
    import asyncio
    
    async def warmup_openvoice():
        try:
            from voice_cloner import check_openvoice_available
            result = await asyncio.to_thread(check_openvoice_available)
            print(f"[Warmup] OpenVoice: {result.get('message', 'unknown')}", flush=True)
        except Exception as e:
            print(f"[Warmup] OpenVoice warmup failed: {e}", flush=True)
    
    # Fire and forget - don't wait for warmup to complete
    asyncio.create_task(warmup_openvoice())
    
    return {"status": "warmup_initiated"}


@app.get("/api/jobs/{job_id}/list-outputs")
async def list_job_outputs(
    job_id: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """List all output files for a job"""
    job = get_user_job(db, job_id, current_user)
    output_dir = Path(job.output_dir)
    
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")
    
    files = []
    for f in output_dir.iterdir():
        if f.is_file():
            files.append(f.name)
    
    return {"files": sorted(files)}


class VoiceSwapRequest(BaseModel):
    video_filename: str


@app.post("/api/jobs/{job_id}/voice-swap")
async def voice_swap_video_endpoint(
    job_id: str,
    video_filename: str = Form(...),
    voice_sample: UploadFile = File(None),
    reference_clips: str = Form(None),  # JSON array of clip filenames
    tau: str = Form("0.3"),  # Voice similarity (0.1-0.5, lower = more similar)
    pitch_normalize: str = Form("0.0"),  # Pitch normalization (0.0-1.0, 0 = off)
    provider: str = Form("openvoice"),  # "openvoice" or "elevenlabs"
    elevenlabs_api_key: str = Form(None),
    elevenlabs_voice_id: str = Form(None),
    elevenlabs_stability: str = Form("0.5"),
    elevenlabs_similarity: str = Form("0.75"),
    elevenlabs_style: str = Form("0"),
    elevenlabs_remove_noise: str = Form("true"),
    elevenlabs_speaker_boost: str = Form("true"),
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Swap voice in video using AI voice cloning.
    
    Supports two providers:
    - OpenVoice v2: Self-hosted (~$0.01/run)
    - ElevenLabs: Premium quality (uses your API credits)
    
    Args:
        job_id: Job ID
        video_filename: Video file to process
        voice_sample: Reference voice audio file (OpenVoice only)
        reference_clips: OR use clips' audio as reference (OpenVoice only)
        tau: Voice similarity for OpenVoice
        pitch_normalize: Pitch compression for OpenVoice
        provider: "openvoice" or "elevenlabs"
        elevenlabs_api_key: Your ElevenLabs API key (ElevenLabs only)
        elevenlabs_voice_id: Target voice ID (ElevenLabs only)
    
    Returns:
        New video file with cloned voice
    """
    from audio_processor import extract_audio, concatenate_audio_files, replace_audio
    
    job = get_user_job(db, job_id, current_user)
    output_dir = Path(job.output_dir)
    
    video_path = output_dir / video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Route to appropriate provider
    if provider == "elevenlabs":
        el_settings = {
            "stability": float(elevenlabs_stability),
            "similarity_boost": float(elevenlabs_similarity),
            "style": float(elevenlabs_style),
            "use_speaker_boost": elevenlabs_speaker_boost.lower() == "true",
            "remove_background_noise": elevenlabs_remove_noise.lower() == "true"
        }
        return await voice_swap_elevenlabs(
            job_id, video_path, output_dir, 
            elevenlabs_api_key, elevenlabs_voice_id, el_settings
        )
    else:
        return await voice_swap_openvoice(
            job_id, video_path, output_dir,
            voice_sample, reference_clips, 
            float(tau), float(pitch_normalize)
        )


async def voice_swap_elevenlabs(
    job_id: str, 
    video_path: Path, 
    output_dir: Path,
    api_key: str, 
    voice_id: str,
    settings: dict = None
):
    """Handle ElevenLabs speech-to-speech voice swap"""
    import httpx
    from audio_processor import extract_audio, replace_audio
    
    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key required")
    if not voice_id:
        raise HTTPException(status_code=400, detail="ElevenLabs Voice ID required")
    
    # Default settings
    if settings is None:
        settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "remove_background_noise": True
        }
    
    temp_audio = output_dir / "temp_source_audio.mp3"
    converted_audio = output_dir / "temp_converted_audio.mp3"
    
    try:
        # Step 1: Extract audio from video as mp3
        print(f"[ElevenLabs] Extracting audio from video...")
        if not extract_audio(video_path, temp_audio, format="mp3"):
            raise HTTPException(status_code=500, detail="Failed to extract audio from video")
        
        print(f"[ElevenLabs] Audio extracted: {temp_audio.stat().st_size} bytes")
        
        # Step 2: Call ElevenLabs Speech-to-Speech API
        print(f"[ElevenLabs] Calling speech-to-speech API for voice: {voice_id}")
        print(f"[ElevenLabs] Settings: stability={settings['stability']}, similarity={settings['similarity_boost']}, style={settings['style']}")
        
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}?output_format=mp3_44100_128"
        
        headers = {
            "xi-api-key": api_key
        }
        
        # Read file content first for async compatibility
        with open(temp_audio, "rb") as f:
            audio_content = f.read()
        
        # Build voice_settings JSON
        voice_settings = {
            "stability": settings["stability"],
            "similarity_boost": settings["similarity_boost"],
            "style": settings["style"],
            "use_speaker_boost": settings["use_speaker_boost"]
        }
        
        files = {
            "audio": ("audio.mp3", audio_content, "audio/mpeg"),
        }
        data = {
            "model_id": "eleven_multilingual_sts_v2",
            "voice_settings": json.dumps(voice_settings),
            "remove_background_noise": str(settings["remove_background_noise"]).lower()
        }
        
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
        
        print(f"[ElevenLabs] Response status: {response.status_code}")
        
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid ElevenLabs API key")
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Voice ID not found: {voice_id}")
        if response.status_code == 422:
            error_detail = response.text[:300] if response.text else "Validation error"
            raise HTTPException(status_code=422, detail=f"ElevenLabs validation error: {error_detail}")
        if response.status_code != 200:
            error_detail = response.text[:200] if response.text else "Unknown error"
            raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs API error: {error_detail}")
        
        # Save converted audio
        with open(converted_audio, "wb") as f:
            f.write(response.content)
        
        print(f"[ElevenLabs] Received {len(response.content)} bytes of converted audio")
        
        # Step 3: Replace audio in video (non-blocking)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        output_filename = f"voice_cloned_el_{timestamp}_{unique_suffix}.mp4"
        output_path = output_dir / output_filename
        
        success = await asyncio.to_thread(replace_audio, video_path, converted_audio, output_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create output video")
        
        print(f"[ElevenLabs] Success! Output: {output_filename}")
        
        return {
            "success": True,
            "filename": output_filename,
            "download_url": f"/api/jobs/{job_id}/outputs/{output_filename}",
            "provider": "elevenlabs",
            "voice_id": voice_id
        }
        
    finally:
        # Cleanup temp files
        if temp_audio.exists():
            temp_audio.unlink()
        if converted_audio.exists():
            converted_audio.unlink()


async def voice_swap_openvoice(
    job_id: str,
    video_path: Path,
    output_dir: Path,
    voice_sample: UploadFile,
    reference_clips: str,
    tau: float,
    pitch_normalize: float
):
    """Handle OpenVoice voice swap (original logic)"""
    from voice_cloner import check_replicate_available, voice_swap_video_sync
    from audio_processor import extract_audio, concatenate_audio_files, enhance_audio_for_voice_clone
    
    # Check if configured
    status = check_replicate_available()
    if not status["available"]:
        raise HTTPException(
            status_code=503, 
            detail=status.get("message", "OpenVoice endpoint not available")
        )
    
    # Parse reference clips if provided
    clip_filenames = []
    if reference_clips:
        try:
            clip_filenames = json.loads(reference_clips)
            if len(clip_filenames) > 4:
                raise HTTPException(status_code=400, detail="Maximum 4 reference clips allowed")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid reference_clips format")
    
    # Must have either voice_sample or reference_clips
    if not voice_sample and not clip_filenames:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either voice_sample file or reference_clips array"
        )
    
    temp_voice = None
    temp_audio_files = []
    
    try:
        # Get voice reference - either from upload or from clip audio(s)
        if voice_sample and voice_sample.filename:
            # Use uploaded voice sample
            voice_ext = Path(voice_sample.filename).suffix or ".wav"
            temp_voice = output_dir / f"temp_voice_sample{voice_ext}"
            content = await voice_sample.read()
            with open(temp_voice, "wb") as f:
                f.write(content)
            print(f"[VoiceSwap] Using uploaded voice sample: {voice_sample.filename}")
        elif clip_filenames:
            # Extract audio from each reference clip, concatenate, then enhance once
            print(f"[VoiceSwap] Extracting voice from {len(clip_filenames)} clips")
            
            for i, clip_name in enumerate(clip_filenames):
                clip_path = output_dir / clip_name
                if not clip_path.exists():
                    print(f"[VoiceSwap] Warning: Clip not found: {clip_name}")
                    continue
                
                temp_audio = output_dir / f"temp_clip_voice_{i}.wav"
                # Basic extraction only (we'll enhance after combining)
                await asyncio.to_thread(extract_audio, clip_path, temp_audio)
                print(f"[VoiceSwap] Extracted audio from: {clip_name}")
                temp_audio_files.append(temp_audio)
            
            if not temp_audio_files:
                raise HTTPException(status_code=404, detail="No valid reference clips found")
            
            # Concatenate all audio files
            if len(temp_audio_files) == 1:
                combined_audio = temp_audio_files[0]
            else:
                combined_audio = output_dir / "temp_combined_voice_raw.wav"
                await asyncio.to_thread(concatenate_audio_files, temp_audio_files, combined_audio, False)
                print(f"[VoiceSwap] Combined {len(temp_audio_files)} clips into single reference")
            
            # Enhance the combined audio once with DeepFilterNet
            temp_voice = output_dir / "temp_voice_enhanced.wav"
            print(f"[VoiceSwap] Applying DeepFilterNet enhancement to combined voice reference...")
            result = await asyncio.to_thread(
                enhance_audio_for_voice_clone, combined_audio, temp_voice,
                denoise=True, denoise_strength=0.8  # Strong denoise for clean voice reference
            )
            if result.get("enhanced"):
                print(f"[VoiceSwap] Voice reference enhanced successfully (denoise: {result.get('denoise_applied')})")
            else:
                # Fallback to unenhanced if enhancement fails
                temp_voice = combined_audio
                print(f"[VoiceSwap] Enhancement skipped, using raw combined audio")
        
        if not temp_voice or not temp_voice.exists():
            raise HTTPException(status_code=400, detail="Failed to prepare voice reference")
        
        # Create output path with unique suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        output_filename = f"voice_cloned_{timestamp}_{unique_suffix}.mp4"
        output_path = output_dir / output_filename
        
        # Run voice swap (non-blocking)
        print(f"[VoiceSwap] Starting voice clone using OpenVoice (tau={tau}, pitch_norm={pitch_normalize})")
        result = await asyncio.to_thread(
            voice_swap_video_sync,
            video_path=video_path,
            reference_voice_path=temp_voice,
            output_path=output_path,
            tau=tau,
            pitch_normalize=pitch_normalize
        )
        
        # Cleanup temp files
        for f in temp_audio_files:
            if f and f.exists():
                f.unlink()
        # Clean combined raw audio if it exists
        combined_raw = output_dir / "temp_combined_voice_raw.wav"
        if combined_raw.exists():
            combined_raw.unlink()
        # Clean enhanced voice file
        if temp_voice and temp_voice.exists() and temp_voice not in temp_audio_files:
            temp_voice.unlink()
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500, 
                detail=f"Voice cloning failed: {result.get('error', 'Unknown error')}"
            )
        
        print(f"[VoiceSwap] Success! Output: {output_filename}")
        
        return {
            "success": True,
            "filename": output_filename,
            "download_url": f"/api/jobs/{job_id}/outputs/{output_filename}",
            "cost_estimate": result.get("cost_estimate", "$0.06"),
            "model_used": result.get("model", "HierSpeech++")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp files
        for f in temp_audio_files:
            if f and f.exists():
                try:
                    f.unlink()
                except:
                    pass
        # Clean combined raw audio
        combined_raw = output_dir / "temp_combined_voice_raw.wav"
        if combined_raw.exists():
            try:
                combined_raw.unlink()
            except:
                pass
        if temp_voice and temp_voice.exists():
            try:
                temp_voice.unlink()
            except:
                pass
        import traceback
        print(f"[VoiceSwap] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Voice swap failed: {str(e)}")


@app.get("/api/jobs/{job_id}/images/{filename}")
async def get_job_image(
    job_id: str, 
    filename: str, 
    db: DBSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Get an image from a job's images directory (local or R2)"""
    from fastapi.responses import Response
    
    job = get_user_job(db, job_id, current_user)
    
    # Determine media type
    suffix = Path(filename).suffix.lower()
    media_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
    media_type = media_types.get(suffix, 'image/png')
    
    # Method 1: Check local filesystem first - use safe_images_dir helper
    images_path = safe_images_dir(job.images_dir)
    if images_path:
        filepath = images_path / filename
        if filepath.exists():
            return FileResponse(filepath, media_type=media_type)
    
    # Method 2: Download from R2 and serve (avoid CORS issues with redirect)
    try:
        from backends.storage import is_storage_configured, get_storage
        
        if is_storage_configured():
            storage = get_storage()
            r2_key = f"jobs/{job_id}/frames/{filename}"
            
            # Check if file exists in R2
            if storage.exists(r2_key):
                # Download bytes and serve directly (avoids CORS redirect issues)
                import tempfile
                temp_path = tempfile.mktemp(suffix=suffix)
                storage.download_file(r2_key, temp_path)
                
                # Read and return
                with open(temp_path, 'rb') as f:
                    content = f.read()
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                return Response(content=content, media_type=media_type)
    except Exception as e:
        print(f"[Images] Error fetching from R2: {e}", flush=True)
    
    raise HTTPException(status_code=404, detail="Image not found")


# ============ Script Splitting ============

class ScriptSplitRequest(BaseModel):
    script: str
    language: str = "English"

# Speaking rates by language (words per second) for natural speech
LANGUAGE_SPEAKING_RATES = {
    "English": 2.5,      # ~150 wpm → 17-18 words per 7 sec
    "Italian": 2.8,      # ~168 wpm → 19-20 words per 7 sec  
    "Spanish": 2.8,      # ~168 wpm → 19-20 words per 7 sec
    "French": 2.5,       # ~150 wpm → 17-18 words per 7 sec
    "German": 2.2,       # ~132 wpm → 15-16 words per 7 sec
    "Portuguese": 2.6,   # ~156 wpm → 18-19 words per 7 sec
    "Dutch": 2.3,        # ~138 wpm → 16-17 words per 7 sec
    "Polish": 2.4,       # ~144 wpm → 17 words per 7 sec
    "Russian": 2.3,      # ~138 wpm → 16-17 words per 7 sec
    "Japanese": 4.0,     # ~240 morae/min → 28 chars per 7 sec
    "Chinese": 3.5,      # ~210 chars/min → 24-25 chars per 7 sec
    "Korean": 3.5,       # Similar to Chinese
    "Arabic": 2.5,       # ~150 wpm → 17-18 words per 7 sec
    "Hindi": 2.6,        # ~156 wpm → 18-19 words per 7 sec
}

TARGET_DURATION_SECONDS = 7

@app.post("/api/split-script")
async def split_script(request: ScriptSplitRequest):
    """
    Split a full script into ~7 second dialogue lines using OpenAI.
    Preserves the EXACT original text - only splits, never rewrites.
    Every line MUST be approximately 7 seconds (enforced via post-processing).
    """
    import os
    
    # Get OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    # Get language-specific rate
    words_per_sec = LANGUAGE_SPEAKING_RATES.get(request.language, 2.5)
    target_words = int(words_per_sec * TARGET_DURATION_SECONDS)
    min_words = max(10, target_words - 5)  # Minimum words per line
    
    # Count total words to estimate expected clips
    total_words = len(request.script.split())
    expected_clips = max(1, round(total_words / target_words))
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        prompt = f"""TASK: Split this script into chunks of EXACTLY ~{target_words} words each.

⚠️ ABSOLUTE REQUIREMENTS:
1. EVERY chunk MUST have AT LEAST {min_words} words (this is ~7 seconds of speech)
2. NEVER create a chunk with less than {min_words} words
3. If a sentence is short, COMBINE it with the next sentence(s) until you reach {min_words}+ words
4. The LAST chunk can be slightly shorter only if all remaining text is less than {min_words} words
5. Preserve EXACT original text - do NOT add, remove, or change any words

ORIGINAL SCRIPT ({total_words} total words):
"{request.script}"

MATH: {total_words} words ÷ {target_words} words = ~{expected_clips} chunks expected

EXAMPLES of what NOT to do:
❌ ["Short sentence.", "Another short one."] - BAD, each under {min_words} words
✅ ["Short sentence. Another short one. And more text here."] - GOOD, combined to reach {min_words}+ words

OUTPUT: JSON array only. Each string MUST have {min_words}+ words.
["chunk with {min_words}+ words here", "another chunk with {min_words}+ words"]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON - handle potential markdown code blocks
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        lines = json.loads(result)
        
        if not isinstance(lines, list) or len(lines) == 0:
            raise ValueError("Invalid response format")
        
        # POST-PROCESSING: Merge any lines that are too short
        merged_lines = []
        buffer = ""
        
        for line in lines:
            if buffer:
                buffer += " " + line.strip()
            else:
                buffer = line.strip()
            
            word_count = len(buffer.split())
            
            # If buffer has enough words, add it to merged_lines
            if word_count >= min_words:
                merged_lines.append(buffer)
                buffer = ""
        
        # Handle remaining buffer
        if buffer:
            if merged_lines:
                # Append to last line if buffer is too short
                buffer_words = len(buffer.split())
                if buffer_words < min_words:
                    merged_lines[-1] = merged_lines[-1] + " " + buffer
                else:
                    merged_lines.append(buffer)
            else:
                # Only one line in total
                merged_lines.append(buffer)
        
        # Clean up whitespace
        merged_lines = [" ".join(line.split()) for line in merged_lines]
        
        # Calculate average duration estimate using language-specific rate
        total_words_result = sum(len(line.split()) for line in merged_lines)
        avg_words = total_words_result / len(merged_lines) if merged_lines else 0
        avg_duration = round(avg_words / words_per_sec, 1)
        
        # Calculate per-line stats
        line_stats = []
        for line in merged_lines:
            wc = len(line.split())
            dur = round(wc / words_per_sec, 1)
            line_stats.append({"words": wc, "duration_sec": dur})
        
        return {
            "success": True,
            "lines": merged_lines,
            "count": len(merged_lines),
            "avg_duration": avg_duration,
            "total_words": total_words_result,
            "target_words_per_line": target_words,
            "min_words_per_line": min_words,
            "language": request.language,
            "line_stats": line_stats
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenAI library not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Script splitting failed: {str(e)}")


# ============ Error Codes Reference ============

@app.get("/api/error-codes")
async def get_error_codes():
    """Get list of all error codes and their meanings"""
    return {
        code.value: {
            "name": code.name,
            "value": code.value,
        }
        for code in ErrorCode
    }


# ============ Health Check ============

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    """Health check endpoint (supports GET and HEAD for monitoring services)"""
    # Check if genai SDK is available
    try:
        from veo_generator import GENAI_AVAILABLE
        sdk_status = "installed" if GENAI_AVAILABLE else "not_installed"
    except:
        sdk_status = "unknown"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "workers": {
            "running_jobs": len(worker.running_jobs),
            "max_workers": worker.max_workers,
        },
        "sdk": {
            "google_genai": sdk_status,
            "message": "Video generation available" if sdk_status == "installed" else "Install google-genai for video generation"
        }
    }


# ============ Admin - API Keys ============

@app.get("/api/admin/keys")
async def get_api_keys_status():
    """
    Check status of API keys configured on the server.
    
    Keys are loaded from .env file:
    - GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    - OPENAI_API_KEY (optional)
    """
    from config import key_pool
    
    status = api_keys_config.get_status()
    pool_status = key_pool.get_status()
    
    # Add masked preview of keys with block status
    masked_keys = []
    for i, key in enumerate(api_keys_config.gemini_api_keys):
        if len(key) > 12:
            masked = f"{key[:8]}...{key[-4:]}"
        else:
            masked = "***"
        
        is_blocked = api_keys_config.is_key_blocked(i)
        blocked_info = None
        if is_blocked and i in api_keys_config.blocked_keys:
            from datetime import datetime, timedelta
            block_time = api_keys_config.blocked_keys[i]
            unblock_time = block_time + timedelta(hours=api_keys_config.block_duration_hours)
            remaining = unblock_time - datetime.now()
            blocked_info = {
                "blocked_at": block_time.isoformat(),
                "unblocks_at": unblock_time.isoformat(),
                "remaining_hours": round(max(0, remaining.total_seconds() / 3600), 1)
            }
        
        # Check if reserved by a job
        reserved_by = pool_status["reservations"].get(i, None)
        
        masked_keys.append({
            "index": i + 1,
            "masked": masked,
            "is_current": i == api_keys_config.current_key_index,
            "is_blocked": is_blocked,
            "blocked_info": blocked_info,
            "reserved_by": reserved_by[:8] if reserved_by else None,
        })
    
    return {
        **status,
        "pool_status": pool_status,
        "gemini_keys": masked_keys,
        "openai_masked": f"{api_keys_config.openai_api_key[:8]}...{api_keys_config.openai_api_key[-4:]}" if api_keys_config.openai_api_key else None,
        "config_file": ".env",
        "block_duration_hours": api_keys_config.block_duration_hours,
        "instructions": "Add keys to .env file and restart server to update"
    }


@app.post("/api/admin/keys/unblock/{key_index}")
async def unblock_api_key(key_index: int):
    """
    Manually unblock a specific API key before the 12h timeout.
    key_index is 1-based (1, 2, 3, etc.)
    """
    actual_index = key_index - 1  # Convert to 0-based
    
    if actual_index < 0 or actual_index >= len(api_keys_config.gemini_api_keys):
        raise HTTPException(status_code=400, detail=f"Invalid key index. Must be 1-{len(api_keys_config.gemini_api_keys)}")
    
    if actual_index not in api_keys_config.blocked_keys:
        return {
            "success": True,
            "message": f"Key {key_index} was not blocked",
            "key_index": key_index
        }
    
    del api_keys_config.blocked_keys[actual_index]
    api_keys_config._save_blocked_keys()  # Persist to disk
    
    return {
        "success": True,
        "message": f"Key {key_index} has been unblocked",
        "key_index": key_index,
        "available_keys": api_keys_config.get_available_key_count(),
        "blocked_keys": len(api_keys_config.blocked_keys)
    }


@app.post("/api/admin/keys/unblock-all")
async def unblock_all_api_keys():
    """
    Unblock all API keys at once.
    """
    blocked_count = len(api_keys_config.blocked_keys)
    api_keys_config.blocked_keys.clear()
    api_keys_config._save_blocked_keys()  # Persist to disk
    
    return {
        "success": True,
        "message": f"Unblocked {blocked_count} keys",
        "unblocked_count": blocked_count,
        "available_keys": api_keys_config.get_available_key_count()
    }


@app.post("/api/admin/keys/rotate")
async def rotate_api_key(block_current: bool = False):
    """Manually rotate to the next Gemini API key"""
    if not api_keys_config.gemini_api_keys:
        raise HTTPException(status_code=400, detail="No Gemini keys configured")
    
    old_index = api_keys_config.current_key_index
    api_keys_config.rotate_key(block_current=block_current)
    new_index = api_keys_config.current_key_index
    
    return {
        "success": True,
        "previous_index": old_index,
        "current_index": new_index,
        "total_keys": len(api_keys_config.gemini_api_keys)
    }


@app.post("/api/admin/keys/reload")
async def reload_api_keys():
    """
    Reload API keys from .env file without restarting server.
    Useful after updating .env file.
    """
    from config import get_gemini_keys_from_env, get_openai_key_from_env
    from dotenv import load_dotenv
    
    # Reload .env file
    load_dotenv(override=True)
    
    # Update keys
    old_count = len(api_keys_config.gemini_api_keys)
    api_keys_config.gemini_api_keys = get_gemini_keys_from_env()
    api_keys_config.openai_api_key = get_openai_key_from_env()
    api_keys_config.current_key_index = 0  # Reset to first key
    
    new_count = len(api_keys_config.gemini_api_keys)
    
    return {
        "success": True,
        "previous_gemini_count": old_count,
        "current_gemini_count": new_count,
        "openai_configured": api_keys_config.openai_api_key is not None,
        "message": f"Loaded {new_count} Gemini key(s) from .env"
    }


class ValidateKeyRequest(BaseModel):
    api_key: str


@app.post("/api/admin/keys/validate")
async def validate_gemini_key(request: ValidateKeyRequest):
    """
    Validate a Gemini API key by making a test API call.
    Returns whether the key is valid, quota status, and any errors.
    """
    import httpx
    
    api_key = request.api_key.strip()
    
    if not api_key:
        return {
            "valid": False,
            "error": "No API key provided",
            "details": None
        }
    
    # Mask key for logging
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    
    try:
        # Test with a simple models list request (doesn't consume quota)
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, try listing models (free, no quota)
            list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            response = await client.get(list_url)
            
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])
                
                # Check for Veo model specifically
                veo_available = any("veo" in m.get("name", "").lower() for m in models)
                gemini_available = any("gemini" in m.get("name", "").lower() for m in models)
                
                # Try a minimal generateContent request to check quota
                # Using gemini-2.0-flash which is fast and cheap
                generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                test_payload = {
                    "contents": [{"parts": [{"text": "Say 'OK'"}]}],
                    "generationConfig": {"maxOutputTokens": 5}
                }
                
                gen_response = await client.post(generate_url, json=test_payload)
                
                quota_ok = gen_response.status_code == 200
                quota_error = None
                
                if not quota_ok:
                    error_data = gen_response.json() if gen_response.content else {}
                    quota_error = error_data.get("error", {}).get("message", f"Status {gen_response.status_code}")
                
                return {
                    "valid": True,
                    "key_preview": masked_key,
                    "models_accessible": len(models),
                    "veo_available": veo_available,
                    "gemini_available": gemini_available,
                    "quota_ok": quota_ok,
                    "quota_error": quota_error,
                    "message": "✅ Key is valid" + (" and has quota" if quota_ok else " but quota may be exhausted")
                }
            
            elif response.status_code == 400:
                return {
                    "valid": False,
                    "key_preview": masked_key,
                    "error": "Invalid API key format",
                    "details": response.json() if response.content else None
                }
            
            elif response.status_code == 403:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", "Access denied")
                return {
                    "valid": False,
                    "key_preview": masked_key,
                    "error": f"API key not authorized: {error_msg}",
                    "details": error_data
                }
            
            elif response.status_code == 429:
                return {
                    "valid": True,
                    "key_preview": masked_key,
                    "quota_ok": False,
                    "error": "Rate limited - key is valid but quota exhausted",
                    "message": "⚠️ Key is valid but currently rate limited"
                }
            
            else:
                return {
                    "valid": False,
                    "key_preview": masked_key,
                    "error": f"Unexpected response: {response.status_code}",
                    "details": response.text[:500] if response.text else None
                }
                
    except httpx.TimeoutException:
        return {
            "valid": None,
            "key_preview": masked_key,
            "error": "Request timed out - could not verify key",
            "message": "⚠️ Could not verify key (timeout)"
        }
    except Exception as e:
        return {
            "valid": None,
            "key_preview": masked_key,
            "error": f"Validation error: {str(e)}",
            "message": "⚠️ Could not verify key"
        }


# ============ Debug Clip Versions ============

@app.get("/api/debug/clip/{clip_id}/versions")
async def debug_clip_versions(
    clip_id: int,
    db: DBSession = Depends(get_db_session),
):
    """Debug endpoint to view raw versions_json for a clip"""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    return {
        "clip_id": clip.id,
        "clip_index": clip.clip_index,
        "status": clip.status,
        "output_filename": clip.output_filename,
        "generation_attempt": clip.generation_attempt,
        "versions_json_raw": clip.versions_json,
        "versions_count": len(versions),
        "versions": versions
    }


# ============ Debug Screenshots ============

@app.get("/api/debug/screenshots")
async def list_debug_screenshots(
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    List debug screenshots stored in R2.
    Returns list of screenshots with presigned URLs.
    """
    from backends.storage import is_storage_configured, get_storage
    
    if not is_storage_configured():
        raise HTTPException(status_code=503, detail="Storage not configured")
    
    try:
        storage = get_storage()
        
        # List screenshots in debug folder
        keys = storage.list_objects(prefix="debug/screenshots/", max_keys=limit)
        
        # Generate presigned URLs for each (valid for 1 hour)
        screenshots = []
        for key in reversed(keys):  # Newest first (by filename which has timestamp)
            filename = key.split("/")[-1]
            url = storage.get_presigned_url(key, expires_in=3600)
            
            # Parse timestamp from filename (format: YYYYMMDD_HHMMSS_name.png)
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 3:
                date_str = parts[0]
                time_str = parts[1]
                name = "_".join(parts[2:])
                try:
                    timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                except Exception:
                    timestamp = "unknown"
            else:
                name = filename
                timestamp = "unknown"
            
            screenshots.append({
                "key": key,
                "filename": filename,
                "name": name,
                "timestamp": timestamp,
                "url": url
            })
        
        return {
            "count": len(screenshots),
            "screenshots": screenshots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list screenshots: {str(e)}")


@app.get("/debug/screenshots", response_class=HTMLResponse)
async def debug_screenshots_gallery(
    request: Request,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """
    HTML gallery page for viewing debug screenshots.
    """
    from backends.storage import is_storage_configured, get_storage
    
    if not is_storage_configured():
        return HTMLResponse("<h1>Storage not configured</h1>")
    
    try:
        storage = get_storage()
        keys = storage.list_objects(prefix="debug/screenshots/", max_keys=limit)
        
        # Build screenshots list (newest first)
        screenshots = []
        for key in reversed(keys):
            filename = key.split("/")[-1]
            url = storage.get_presigned_url(key, expires_in=3600)
            
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 3:
                date_str = parts[0]
                time_str = parts[1]
                name = "_".join(parts[2:])
                try:
                    timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                except Exception:
                    timestamp = "unknown"
            else:
                name = filename
                timestamp = "unknown"
            
            screenshots.append({
                "name": name,
                "timestamp": timestamp,
                "url": url
            })
        
        # Generate HTML
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Flow Debug Screenshots</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 20px;
        }
        .info {
            color: #888;
            margin-bottom: 20px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .screenshot {
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .screenshot img {
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }
        .screenshot img:hover {
            opacity: 0.9;
        }
        .screenshot .info-bar {
            padding: 10px;
            background: #0f3460;
        }
        .screenshot .name {
            font-weight: bold;
            color: #00d4ff;
        }
        .screenshot .timestamp {
            font-size: 12px;
            color: #888;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            cursor: pointer;
        }
        .modal img {
            max-width: 95%;
            max-height: 95%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #00d4ff;
            text-decoration: none;
        }
        .back-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <a href="/" class="back-link">← Back to Studio</a>
    <h1>🔍 Flow Debug Screenshots</h1>
    <p class="info">Showing """ + str(len(screenshots)) + """ screenshots (newest first). Click to enlarge.</p>
    
    <div class="gallery">
"""
        
        for s in screenshots:
            html += f"""
        <div class="screenshot">
            <img src="{s['url']}" alt="{s['name']}" onclick="showModal(this.src)">
            <div class="info-bar">
                <div class="name">{s['name']}</div>
                <div class="timestamp">{s['timestamp']}</div>
            </div>
        </div>
"""
        
        html += """
    </div>
    
    <div class="modal" id="modal" onclick="hideModal()">
        <img id="modal-img" src="">
    </div>
    
    <script>
        function showModal(src) {
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'block';
        }
        function hideModal() {
            document.getElementById('modal').style.display = 'none';
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') hideModal();
        });
    </script>
</body>
</html>
"""
        return HTMLResponse(html)
        
    except Exception as e:
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>")


@app.delete("/api/debug/screenshots")
async def delete_old_screenshots(
    older_than_hours: int = 24,
    current_user: User = Depends(get_current_user)
):
    """
    Delete debug screenshots older than specified hours.
    Helps clean up storage.
    """
    from backends.storage import is_storage_configured, get_storage
    from datetime import datetime, timedelta
    
    if not is_storage_configured():
        raise HTTPException(status_code=503, detail="Storage not configured")
    
    try:
        storage = get_storage()
        
        # List all screenshots
        keys = storage.list_objects(prefix="debug/screenshots/", max_keys=1000)
        
        cutoff = datetime.now() - timedelta(hours=older_than_hours)
        deleted = 0
        
        for key in keys:
            filename = key.split("/")[-1]
            # Parse timestamp from filename
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 2:
                try:
                    date_str = parts[0]
                    time_str = parts[1]
                    file_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                    
                    if file_time < cutoff:
                        storage.client.delete_object(Bucket=storage.bucket_name, Key=key)
                        deleted += 1
                except Exception:
                    pass
        
        return {
            "deleted": deleted,
            "message": f"Deleted {deleted} screenshots older than {older_than_hours} hours"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete screenshots: {str(e)}")


# ============ Local Worker API ============
# These routes allow a local worker to fetch jobs and update status

LOCAL_WORKER_API_KEY = os.environ.get("LOCAL_WORKER_API_KEY", "local-worker-secret-key-12345")

def verify_local_worker_key(authorization: str = Header(None)):
    """Verify local worker API key"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization[7:]
    if token != LOCAL_WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@app.get("/api/local-worker/health")
async def local_worker_health():
    """Health check for local worker"""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/local-worker/jobs/pending")
async def local_worker_get_pending_job(
    request: Request,
    worker_id: Optional[str] = Query(None, description="Worker ID for claiming"),
    db: DBSession = Depends(get_db_session),
    authorized: bool = Depends(verify_local_worker_key)
):
    """Get next pending Flow job with all clips.
    
    If worker_id is provided, atomically claims the job for that worker.
    Jobs claimed more than 10 minutes ago without completion are released.
    """
    from sqlalchemy import or_, and_
    
    # Release stale claims (claimed > 10 minutes ago and not started)
    claim_timeout = datetime.utcnow() - timedelta(minutes=10)
    stale_jobs = db.query(Job).filter(
        Job.backend == 'flow',
        Job.status.in_(['pending', 'queued_for_flow']),
        Job.claimed_by_worker.isnot(None),
        Job.claimed_at < claim_timeout
    ).all()
    
    for stale_job in stale_jobs:
        print(f"[Worker] Releasing stale claim on job {stale_job.id[:8]} (was claimed by {stale_job.claimed_by_worker})", flush=True)
        stale_job.claimed_by_worker = None
        stale_job.claimed_at = None
    
    if stale_jobs:
        db.commit()
    
    # Build query for available jobs
    # Either: unclaimed, OR claimed by this same worker
    if worker_id:
        job = db.query(Job).filter(
            Job.backend == 'flow',
            Job.status.in_(['pending', 'queued_for_flow']),
            or_(
                Job.claimed_by_worker.is_(None),
                Job.claimed_by_worker == worker_id
            )
        ).order_by(Job.created_at.asc()).first()
        
        if job and job.claimed_by_worker != worker_id:
            # Claim it
            job.claimed_by_worker = worker_id
            job.claimed_at = datetime.utcnow()
            db.commit()
            print(f"[Worker] Job {job.id[:8]} claimed by {worker_id}", flush=True)
    else:
        # No worker_id - just get unclaimed (legacy behavior)
        job = db.query(Job).filter(
            Job.backend == 'flow',
            Job.status.in_(['pending', 'queued_for_flow']),
            Job.claimed_by_worker.is_(None)
        ).order_by(Job.created_at.asc()).first()
    
    if not job:
        return {"job": None}
    
    clips = db.query(Clip).filter(Clip.job_id == job.id).order_by(Clip.clip_index.asc()).all()
    
    # Parse config JSON
    config = json.loads(job.config_json) if job.config_json else {}
    use_interpolation = config.get("use_interpolation", True)
    
    # Build base URL for frame downloads (use proxy to avoid SSL issues on Windows)
    base_url = str(request.base_url).rstrip('/')
    
    # Determine if single image mode (all clips have same start_frame or only one unique frame)
    unique_frames = set(c.start_frame for c in clips if c.start_frame)
    single_image_mode = len(unique_frames) <= 1
    
    clips_data = []
    for clip in clips:
        # Get frame keys
        start_frame_key = clip.start_frame
        end_frame_key = clip.end_frame
        
        # FALLBACK: If end_frame is None but interpolation is on, use start_frame
        # This handles clips created before the fix
        if use_interpolation and not end_frame_key and start_frame_key:
            if single_image_mode:
                # Single image mode: use same frame for end
                end_frame_key = start_frame_key
            # For multi-image, leave as None (worker will handle transitions)
        
        # Extract filename from R2 key (format: jobs/{job_id}/frames/{filename})
        start_filename = start_frame_key.split('/')[-1] if start_frame_key else None
        end_filename = end_frame_key.split('/')[-1] if end_frame_key else None
        
        clip_data = {
            "id": clip.id,
            "clip_index": clip.clip_index,
            "dialogue_text": clip.dialogue_text,
            "prompt": clip.prompt_text,  # Generated prompt
            "start_frame_key": start_frame_key,  # R2 key for frame
            "end_frame_key": end_frame_key,
            "status": clip.status,
            # Use proxy URLs instead of direct R2 presigned URLs
            "start_frame_url": f"{base_url}/api/local-worker/frames/{job.id}/{start_filename}" if start_filename else None,
            "end_frame_url": f"{base_url}/api/local-worker/frames/{job.id}/{end_filename}" if end_filename else None,
        }
        
        clips_data.append(clip_data)
    
    return {
        "job": {
            "id": job.id,
            "aspect_ratio": config.get("aspect_ratio", "9:16"),
            "duration": config.get("duration", "8"),
            "resolution": config.get("resolution", "720p"),
            "use_interpolation": use_interpolation,
            "single_image_mode": single_image_mode,
            "flow_project_url": job.flow_project_url,
            "clips": clips_data,
            "claimed_by": job.claimed_by_worker
        }
    }


@app.get("/api/local-worker/clips/redo-pending")
async def local_worker_get_redo_clips(
    request: Request,
    worker_id: Optional[str] = Query(None, description="Worker ID for claiming"),
    db: DBSession = Depends(get_db_session),
    authorized: bool = Depends(verify_local_worker_key)
):
    """Get clips that need regeneration for Flow jobs.
    
    If worker_id is provided, atomically claims clips for that worker.
    Claims expire after 10 minutes if not completed.
    
    This ONLY handles Flow backend redos via 'flow_redo_queued' status.
    API backend redos use 'redo_queued' and are handled by the API worker.
    """
    from sqlalchemy import or_, and_
    
    # Release stale claims (claimed > 10 minutes ago)
    # NOTE: Now filtering for flow_redo_queued instead of redo_queued
    claim_timeout = datetime.utcnow() - timedelta(minutes=10)
    stale_clips = db.query(Clip).join(Job).filter(
        Job.backend == 'flow',
        Clip.status == ClipStatus.FLOW_REDO_QUEUED.value,  # Changed from 'redo_queued'
        Clip.claimed_by_worker.isnot(None),
        Clip.claimed_at < claim_timeout
    ).all()
    
    for stale_clip in stale_clips:
        print(f"[Worker] Releasing stale claim on clip {stale_clip.id} (was claimed by {stale_clip.claimed_by_worker})", flush=True)
        stale_clip.claimed_by_worker = None
        stale_clip.claimed_at = None
    
    if stale_clips:
        db.commit()
    
    # Build query for available redo clips
    # IMPORTANT: Now using flow_redo_queued status for proper separation
    if worker_id:
        # Either: unclaimed, OR claimed by this same worker
        redo_clips = db.query(Clip).join(Job).filter(
            Job.backend == 'flow',
            or_(
                # Normal Flow redo queue - unclaimed or claimed by this worker
                and_(
                    Clip.status == ClipStatus.FLOW_REDO_QUEUED.value,  # Changed from 'redo_queued'
                    or_(
                        Clip.claimed_by_worker.is_(None),
                        Clip.claimed_by_worker == worker_id
                    )
                ),
                # Failed Flow redos that API worker wrongly tried to process (legacy recovery)
                and_(
                    Clip.status == 'failed',
                    Clip.generation_attempt > 1,
                    Clip.error_message.ilike('%file not found%')
                )
            )
        ).order_by(Clip.id.asc()).all()
    else:
        # No worker_id - get unclaimed only (legacy behavior)
        redo_clips = db.query(Clip).join(Job).filter(
            Job.backend == 'flow',
            or_(
                and_(
                    Clip.status == ClipStatus.FLOW_REDO_QUEUED.value,  # Changed from 'redo_queued'
                    Clip.claimed_by_worker.is_(None)
                ),
                and_(
                    Clip.status == 'failed',
                    Clip.generation_attempt > 1,
                    Clip.error_message.ilike('%file not found%')
                )
            )
        ).order_by(Clip.id.asc()).all()
    
    if not redo_clips:
        return {"clips": []}
    
    base_url = str(request.base_url).rstrip('/')
    
    clips_data = []
    for clip in redo_clips:
        job = clip.job
        
        # If this is a failed clip being recovered, reset its status to flow_redo_queued
        if clip.status == 'failed':
            print(f"[LocalWorker] Recovering Flow redo: clip {clip.id} (job {job.id[:8]})", flush=True)
            clip.status = ClipStatus.FLOW_REDO_QUEUED.value  # Changed from 'redo_queued'
            clip.error_message = None
            db.commit()
            # No job log - this is expected behavior, not worth cluttering logs
        
        # Claim clip if worker_id provided and not already claimed by this worker
        if worker_id and clip.claimed_by_worker != worker_id:
            clip.claimed_by_worker = worker_id
            clip.claimed_at = datetime.utcnow()
            # CRITICAL: Change status to GENERATING to prevent infinite claim loop
            # Without this, the clip keeps getting returned on every poll
            clip.status = ClipStatus.GENERATING.value
            clip.started_at = datetime.utcnow()
            db.commit()
            print(f"[Worker] Clip {clip.id} (redo) claimed by {worker_id}, status → generating", flush=True)
            add_job_log(db, clip.job_id, f"Flow redo for clip {clip.clip_index + 1} claimed by local worker", "INFO", "redo")
        elif worker_id and clip.claimed_by_worker == worker_id:
            # Already claimed by this worker - skip (don't log again to avoid spam)
            pass
        
        # Get frame URLs
        start_filename = clip.start_frame.split('/')[-1] if clip.start_frame else None
        end_filename = clip.end_frame.split('/')[-1] if clip.end_frame else None
        
        clips_data.append({
            "id": clip.id,
            "job_id": job.id,
            "clip_index": clip.clip_index,
            "dialogue_text": clip.dialogue_text,
            "prompt": clip.prompt_text,
            "start_frame_url": f"{base_url}/api/local-worker/frames/{job.id}/{start_filename}" if start_filename else None,
            "end_frame_url": f"{base_url}/api/local-worker/frames/{job.id}/{end_filename}" if end_filename else None,
            "flow_project_url": job.flow_project_url,
            "generation_attempt": clip.generation_attempt,
            "redo_reason": clip.redo_reason,
            "claimed_by": clip.claimed_by_worker
        })
    
    return {"clips": clips_data}


class LocalWorkerJobUpdate(BaseModel):
    status: Optional[str] = None
    error_message: Optional[str] = None
    flow_project_url: Optional[str] = None


@app.post("/api/local-worker/jobs/{job_id}/status")
async def local_worker_update_job_status(
    job_id: str,
    update: LocalWorkerJobUpdate,
    db: DBSession = Depends(get_db_session),
    authorized: bool = Depends(verify_local_worker_key)
):
    """Update job status"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if update.status:
        job.status = update.status
        # Clear claim when job is completed or failed
        if update.status in ['completed', 'failed', 'cancelled']:
            job.claimed_by_worker = None
            job.claimed_at = None
    if update.error_message:
        job.error_message = update.error_message
    if update.flow_project_url:
        job.flow_project_url = update.flow_project_url
    job.updated_at = datetime.utcnow()
    
    db.commit()
    return {"success": True, "job_id": job_id, "status": job.status}


class LocalWorkerClipUpdate(BaseModel):
    status: Optional[str] = None
    output_url: Optional[str] = None
    output_key: Optional[str] = None
    error_message: Optional[str] = None


@app.post("/api/local-worker/clips/{clip_id}/status")
async def local_worker_update_clip_status(
    clip_id: str,
    update: LocalWorkerClipUpdate,
    db: DBSession = Depends(get_db_session),
    authorized: bool = Depends(verify_local_worker_key)
):
    """Update clip status"""
    # Use FOR UPDATE to prevent race condition with upload endpoint
    clip = db.query(Clip).filter(Clip.id == clip_id).with_for_update().first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    # Clean up images_dir for Flow jobs (frames are in R2, not local disk)
    # This fixes existing Flow jobs that still have local paths set
    job = db.query(Job).filter(Job.id == clip.job_id).first()
    if job and job.backend == 'flow' and job.images_dir:
        print(f"[LocalWorker] Cleaning up images_dir for Flow job {job.id[:8]}", flush=True)
        job.images_dir = ""  # Empty string instead of None (DB has NOT NULL constraint)
    
    old_status = clip.status
    
    if update.status:
        clip.status = update.status
        # Clear claim when clip is completed or failed
        if update.status in ['completed', 'failed']:
            clip.claimed_by_worker = None
            clip.claimed_at = None
            
            # Log Flow redo failures for debugging
            if update.status == 'failed' and old_status == ClipStatus.GENERATING.value:
                error_msg = update.error_message or "Unknown error"
                add_job_log(
                    db, clip.job_id,
                    f"⚠️ Flow redo for clip {clip.clip_index + 1} failed: {error_msg[:100]}",
                    "ERROR", "redo"
                )
        # Clear error state when status is NOT failed (e.g., generating, completed)
        if update.status != 'failed':
            clip.error_message = None
            clip.error_code = None
    if update.output_url:
        clip.output_url = update.output_url
    if update.output_key:
        clip.output_key = update.output_key
    if update.error_message:
        clip.error_message = update.error_message
    
    # When completing a clip (from redo or initial generation), update approval status
    # Include flow_redo_queued for Flow backend redos
    if update.status == 'completed' and old_status in ['generating', 'redo_queued', 'flow_redo_queued']:
        clip.approval_status = 'pending_review'
        clip.completed_at = datetime.utcnow()
        
        # Extract filename from output_url for video playback
        if update.output_url:
            # URL format: .../outputs/clip_X.mp4
            import re
            match = re.search(r'/outputs/([^/]+\.mp4)', update.output_url)
            if match:
                clip.output_filename = match.group(1)
        
        # Set selected_variant based on actual versions count (already populated by upload endpoint)
        versions = json.loads(clip.versions_json) if clip.versions_json else []
        if versions:
            # Default to last variant of current attempt
            current_attempt_versions = [v for v in versions if v.get("attempt") == clip.generation_attempt]
            clip.selected_variant = len(versions) if not current_attempt_versions else len([v for v in versions if v.get("attempt") < clip.generation_attempt]) + len(current_attempt_versions)
        
        # Log completion
        add_job_log(
            db, clip.job_id,
            f"Clip {clip.clip_index + 1} completed via Flow backend (all variants uploaded)",
            "INFO", "flow"
        )
        
        # Update job's completed_clips counter
        job = db.query(Job).filter(Job.id == clip.job_id).first()
        if job:
            completed = db.query(Clip).filter(
                Clip.job_id == clip.job_id,
                Clip.status == ClipStatus.COMPLETED.value
            ).count()
            # Add 1 for this clip since we haven't committed yet
            if old_status != 'completed':
                completed += 1
            job.completed_clips = completed
            if job.total_clips > 0:
                job.progress_percent = int((completed / job.total_clips) * 100)
            # Check if all clips are completed
            if completed >= job.total_clips:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
    
    clip.updated_at = datetime.utcnow()
    
    db.commit()
    return {"success": True, "clip_id": clip_id, "status": clip.status}


@app.get("/api/local-worker/frames/{job_id}/{filename}")
async def local_worker_download_frame(
    job_id: str,
    filename: str,
    authorized: bool = Depends(verify_local_worker_key)
):
    """
    Download a frame for local worker.
    Proxies from R2 to avoid SSL issues on Windows.
    """
    from fastapi.responses import Response
    from backends.storage import is_storage_configured, get_storage
    
    # Build R2 key
    r2_key = f"jobs/{job_id}/frames/{filename}"
    
    # Check local filesystem first
    local_path = app_config.uploads_dir / job_id / filename
    if local_path.exists():
        with open(local_path, 'rb') as f:
            content = f.read()
        media_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'
        return Response(content=content, media_type=media_type)
    
    # Download from R2
    if not is_storage_configured():
        raise HTTPException(status_code=503, detail="Storage not configured")
    
    try:
        storage = get_storage()
        
        # Download to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp_path = tmp.name
        
        storage.download_file(r2_key, tmp_path)
        
        with open(tmp_path, 'rb') as f:
            content = f.read()
        
        # Clean up
        os.remove(tmp_path)
        
        media_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'
        return Response(content=content, media_type=media_type)
        
    except Exception as e:
        print(f"[LocalWorker] Frame download error: {e}", flush=True)
        raise HTTPException(status_code=404, detail=f"Frame not found: {filename}")


@app.post("/api/local-worker/jobs/{job_id}/upload-video/{clip_index}")
async def local_worker_upload_video(
    job_id: str,
    clip_index: int,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db_session),
    authorized: bool = Depends(verify_local_worker_key)
):
    """
    Upload a generated video from local worker.
    Proxies to R2 to avoid SSL issues on Windows.
    
    Filename format: clip_{clip_index}_{attempt}.{variant}.mp4
    Example: clip_0_1.1.mp4 (clip 0, attempt 1, variant 1)
    """
    from backends.storage import is_storage_configured, get_storage
    import re
    
    if not is_storage_configured():
        raise HTTPException(status_code=503, detail="Storage not configured")
    
    try:
        storage = get_storage()
        
        # Parse filename to extract attempt and variant
        # Expected format: clip_{index}_{attempt}.{variant}.mp4
        filename = file.filename or f"clip_{clip_index}_1.1.mp4"
        match = re.match(r'clip_(\d+)_(\d+)\.(\d+)\.mp4', filename)
        
        if match:
            attempt = int(match.group(2))
            variant = int(match.group(3))
        else:
            # Fallback for old format: clip_{index}.mp4
            attempt = 1
            variant = 1
        
        print(f"[LocalWorker] Uploading clip {clip_index}, attempt {attempt}, variant {variant}", flush=True)
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Upload to R2 with unique key including attempt.variant
        r2_key = f"jobs/{job_id}/outputs/clip_{clip_index}_{attempt}.{variant}.mp4"
        storage.upload_file(tmp_path, r2_key, content_type='video/mp4')
        
        # Generate URL
        output_url = storage.get_presigned_url(r2_key, expires_in=86400 * 7)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Update clip in database - use FOR UPDATE to prevent race condition
        # between variant 1.1 and 1.2 uploads happening simultaneously
        clip = db.query(Clip).filter(
            Clip.job_id == job_id,
            Clip.clip_index == clip_index
        ).with_for_update().first()
        
        if clip:
            old_status = clip.status
            
            # DEBUG: Log current state before modification
            print(f"[DEBUG-UPLOAD] Clip {clip_index} variant {attempt}.{variant}: old_status={old_status}", flush=True)
            print(f"[DEBUG-UPLOAD] Clip {clip_index} BEFORE: versions_json={clip.versions_json}", flush=True)
            
            # Load existing versions
            versions = json.loads(clip.versions_json) if clip.versions_json else []
            print(f"[DEBUG-UPLOAD] Clip {clip_index} parsed versions count: {len(versions)}", flush=True)
            
            # Create version key for this attempt.variant
            version_key = f"{attempt}.{variant}"
            
            # Check if this version already exists (by attempt.variant combo)
            existing_idx = None
            for idx, v in enumerate(versions):
                v_attempt = v.get("attempt", 1)
                v_variant = v.get("variant", 1)
                if v_attempt == attempt and v_variant == variant:
                    existing_idx = idx
                    break
            
            # Create version entry
            version_entry = {
                "attempt": attempt,
                "variant": variant,
                "version_key": version_key,
                "filename": f"clip_{clip_index}_{attempt}.{variant}.mp4",
                "url": output_url,
                "generated_at": datetime.utcnow().isoformat(),
            }
            
            if existing_idx is not None:
                # Update existing
                versions[existing_idx] = version_entry
            else:
                # Add new
                versions.append(version_entry)
            
            # Sort versions by attempt, then variant
            versions.sort(key=lambda x: (x.get("attempt", 1), x.get("variant", 1)))
            
            clip.versions_json = json.dumps(versions)
            print(f"[DEBUG-UPLOAD] Clip {clip_index} AFTER: versions_json={clip.versions_json}", flush=True)
            print(f"[DEBUG-UPLOAD] Clip {clip_index} new versions count: {len(versions)}", flush=True)
            
            # Only update main output for the primary variant (X.1)
            if variant == 1:
                clip.output_url = output_url
                clip.output_filename = f"clip_{clip_index}_{attempt}.{variant}.mp4"
                clip.generation_attempt = attempt
                
                # Update selected_variant to point to this version
                clip.selected_variant = len([v for v in versions if v.get("attempt") == attempt and v.get("variant") <= variant])
            
            # NOTE: Do NOT set clip.status = "completed" or mark job as completed here!
            # The worker will call update_clip_status(clip_id, 'completed') AFTER all variants
            # are uploaded, which properly handles status and job completion.
            # This prevents the race condition where polling stops before variant 2 is uploaded.
            
            add_job_log(
                db, job_id,
                f"Clip {clip_index + 1} variant {attempt}.{variant} uploaded via Flow backend",
                "INFO", "flow"
            )
            
            db.commit()
            
            # DEBUG: Verify what was actually saved
            db.refresh(clip)
            print(f"[DEBUG-UPLOAD] Clip {clip_index} AFTER COMMIT: versions_json={clip.versions_json}", flush=True)
            
            print(f"[LocalWorker] Uploaded video for clip {clip_index} ({attempt}.{variant}): {r2_key}", flush=True)
        
        return {
            "success": True,
            "key": r2_key,
            "url": output_url,
            "attempt": attempt,
            "variant": variant
        }
        
    except Exception as e:
        print(f"[LocalWorker] Video upload error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ============ Main Entry Point ============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
    )