# -*- coding: utf-8 -*-
"""
Configuration module for Veo Web App
Centralizes all settings with validation and defaults
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_gemini_keys_from_env() -> List[str]:
    """Load all Gemini API keys from environment variables"""
    keys = []
    found_vars = []
    
    # Method 1: Scan ALL environment variables for any GEMINI/GOOGLE key patterns
    for var_name, var_value in os.environ.items():
        if var_name.startswith(("GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY")):
            if var_value and var_value.strip() and not var_value.startswith("your-"):
                key = var_value.strip()
                if key not in keys:
                    keys.append(key)
                    found_vars.append(var_name)
    
    # Method 2: Also check single key formats (in case they weren't caught)
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        key = os.environ.get(var)
        if key and key.strip() and not key.startswith("your-") and key.strip() not in keys:
            keys.append(key.strip())
            if var not in found_vars:
                found_vars.append(var)
    
    print(f"[Config] Loaded {len(keys)} Gemini API keys from environment", flush=True)
    if found_vars:
        # Sort and log found variable names (without revealing the keys)
        found_vars.sort()
        print(f"[Config] Found key variables: {found_vars}", flush=True)
    
    return keys


def get_openai_key_from_env() -> Optional[str]:
    """Load OpenAI API key from environment"""
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip() and not key.startswith("sk-your"):
        return key.strip()
    return None


class AspectRatio(str, Enum):
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"


class Resolution(str, Enum):
    HD = "720p"
    FULL_HD = "1080p"


class Duration(str, Enum):
    SHORT = "4"
    MEDIUM = "6"
    LONG = "8"


class PersonGeneration(str, Enum):
    ALLOW_ALL = "allow_all"
    ALLOW_ADULT = "allow_adult"
    DONT_ALLOW = "dont_allow"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    REDO_QUEUED = "redo_queued"  # Waiting for API redo generation
    FLOW_REDO_QUEUED = "flow_redo_queued"  # Waiting for Flow redo generation (handled by Flow worker only)
    WAITING_APPROVAL = "waiting_approval"  # Continue mode: waiting for previous clip approval


class ApprovalStatus(str, Enum):
    PENDING_REVIEW = "pending_review"  # Waiting for user to review
    APPROVED = "approved"              # User accepted the clip
    REJECTED = "rejected"              # User requested redo (in progress)
    MAX_ATTEMPTS = "max_attempts"      # Hit 3 attempts, needs support contact


class ErrorCode(str, Enum):
    # API Errors
    RATE_LIMIT = "RATE_LIMIT_429"
    API_KEY_INVALID = "API_KEY_INVALID"
    API_QUOTA_EXCEEDED = "API_QUOTA_EXCEEDED"
    API_NETWORK_ERROR = "API_NETWORK_ERROR"
    API_TIMEOUT = "API_TIMEOUT"
    
    # Content Filtering
    CELEBRITY_FILTER = "CELEBRITY_RAI_FILTER"
    CONTENT_POLICY = "CONTENT_POLICY_VIOLATION"
    SAFETY_FILTER = "SAFETY_FILTER"
    
    # Image Errors
    IMAGE_INVALID_FORMAT = "IMAGE_INVALID_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_CORRUPTED = "IMAGE_CORRUPTED"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    ALL_IMAGES_BLACKLISTED = "ALL_IMAGES_BLACKLISTED"
    
    # Generation Errors
    VIDEO_GENERATION_FAILED = "VIDEO_GENERATION_FAILED"
    PROMPT_TOO_LONG = "PROMPT_TOO_LONG"
    OPENAI_PROMPT_FAILED = "OPENAI_PROMPT_FAILED"
    
    # System Errors
    STORAGE_FULL = "STORAGE_FULL"
    FILE_WRITE_ERROR = "FILE_WRITE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    WORKER_CRASHED = "WORKER_CRASHED"
    
    # User Errors
    INVALID_CONFIG = "INVALID_CONFIG"
    NO_IMAGES = "NO_IMAGES"
    NO_DIALOGUE = "NO_DIALOGUE"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    
    # Unknown
    UNKNOWN = "UNKNOWN_ERROR"


@dataclass
class AppConfig:
    """Application-wide configuration"""
    
    # Paths - Can be overridden by environment variables
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    uploads_dir: Path = field(default=None)
    outputs_dir: Path = field(default=None)
    data_dir: Path = field(default=None)
    
    # Database
    database_url: str = field(default=None)
    
    # Server
    host: str = "0.0.0.0"
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.environ.get("DEBUG", "false").lower() == "true")
    
    # Environment
    is_production: bool = field(default_factory=lambda: os.environ.get("RENDER", "") == "true" or os.environ.get("ENV", "").lower() == "production")
    
    # Workers
    # MAX_JOB_WORKERS env var controls parallel video generation jobs
    # - Each job is mostly API calls to Veo (low local RAM usage)
    # - Reduced to 2 to avoid rate limit issues with Veo API
    max_workers: int = field(default_factory=lambda: int(os.environ.get("MAX_JOB_WORKERS", "2")))
    worker_poll_interval: float = 1.0
    
    # File limits
    max_upload_size_mb: int = 50
    max_images_per_job: int = 100
    max_dialogue_lines: int = 50
    
    # Cleanup
    keep_uploads_days: int = 7
    keep_outputs_days: int = 30
    
    def __post_init__(self):
        # Determine data root directory
        # Priority: DATA_DIR env var > /app/data (Docker) > ./data (local)
        data_root = os.environ.get("DATA_DIR")
        if data_root:
            data_root = Path(data_root)
        elif self.is_production:
            # In production (Render/Docker), use /app/data
            data_root = Path("/app/data")
            if not data_root.exists():
                # Fallback to relative path if /app/data doesn't exist
                data_root = self.base_dir / "data"
        else:
            data_root = self.base_dir / "data"
        
        # Set directory paths
        if self.uploads_dir is None:
            uploads_env = os.environ.get("UPLOADS_DIR")
            self.uploads_dir = Path(uploads_env) if uploads_env else data_root / "uploads"
        
        if self.outputs_dir is None:
            outputs_env = os.environ.get("OUTPUTS_DIR")
            self.outputs_dir = Path(outputs_env) if outputs_env else data_root / "outputs"
        
        if self.data_dir is None:
            self.data_dir = data_root
        
        # Database URL
        if self.database_url is None:
            db_env = os.environ.get("DATABASE_URL")
            if db_env:
                self.database_url = db_env
            else:
                self.database_url = f"sqlite:///{self.data_dir / 'jobs.db'}"
        
        # Create directories with error handling
        for dir_path in [self.uploads_dir, self.outputs_dir, self.data_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"[Config] Warning: Cannot create {dir_path} - using temp directory")
                # Fallback to temp directory
                import tempfile
                temp_base = Path(tempfile.gettempdir()) / "veo-app"
                temp_base.mkdir(parents=True, exist_ok=True)
                if dir_path == self.uploads_dir:
                    self.uploads_dir = temp_base / "uploads"
                elif dir_path == self.outputs_dir:
                    self.outputs_dir = temp_base / "outputs"
                else:
                    self.data_dir = temp_base / "data"
                    self.database_url = f"sqlite:///{self.data_dir / 'jobs.db'}"
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Log configuration in production
        if self.is_production:
            print(f"[Config] Production mode enabled")
            print(f"[Config] Data directory: {self.data_dir}")
            print(f"[Config] Uploads: {self.uploads_dir}")
            print(f"[Config] Outputs: {self.outputs_dir}")
            print(f"[Config] Database: {self.database_url}")


@dataclass
class VideoConfig:
    """Video generation configuration"""
    
    # Video settings
    aspect_ratio: AspectRatio = AspectRatio.PORTRAIT
    resolution: Resolution = Resolution.HD
    duration: Duration = Duration.LONG
    
    # Language
    language: str = "English"
    
    # Person generation
    person_generation: PersonGeneration = PersonGeneration.ALLOW_ADULT
    
    # Features
    use_interpolation: bool = True
    use_openai_prompt_tuning: bool = True
    use_frame_vision: bool = True
    timestamp_names: bool = True
    
    # Custom prompt (used when use_openai_prompt_tuning is False)
    custom_prompt: str = ""
    
    # User context (additional info for AI prompt generation)
    user_context: str = ""
    
    # Single image mode (use same image for start/end frames)
    single_image_mode: bool = False
    
    # Retry settings
    max_retries_per_clip: int = 5
    max_image_attempts: int = 15
    max_retries_submit: int = 15  # Try up to 15 times to cycle through all API keys
    poll_interval_sec: int = 10
    
    # Celebrity filter handling
    skip_on_celebrity_filter: bool = False  # If True, skip clip immediately on celebrity filter (storyboard mode)
    
    # Parallel clip generation
    parallel_clips: int = 6  # Number of clips to generate simultaneously (reduced to avoid rate limits)
    
    # Generation mode: "parallel" (fast, may have transition gaps) or "sequential" (slower, guaranteed smooth transitions)
    generation_mode: str = "staggered"
    
    # Regeneration
    reuse_logged_params: bool = True
    
    # Image selection
    images_sort_key: str = "name"  # "name" or "date"
    images_sort_reverse: bool = False
    skip_first_pairs: int = 0
    skip_last_pairs: int = 0
    max_clips: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Resolution/duration constraints
        if self.resolution == Resolution.FULL_HD and self.duration != Duration.LONG:
            errors.append("1080p resolution requires 8 second duration")
        
        if self.use_interpolation and self.duration != Duration.LONG:
            errors.append("Interpolation requires 8 second duration")
        
        # EU compliance
        if self.person_generation == PersonGeneration.ALLOW_ALL:
            errors.append("'allow_all' not permitted for EU compliance")
        
        return errors


@dataclass
class APIKeysConfig:
    """API keys configuration - loads from environment by default"""
    
    # Google/Gemini keys (multiple for rotation)
    gemini_api_keys: List[str] = field(default_factory=get_gemini_keys_from_env)
    
    # OpenAI key (optional)
    openai_api_key: Optional[str] = field(default_factory=get_openai_key_from_env)
    
    # Key rotation settings
    rotate_keys_on_429: bool = True
    current_key_index: int = 0
    
    # Key blocking - maps key index to block timestamp
    blocked_keys: dict = field(default_factory=dict)
    block_duration_hours: int = 1  # Rate limits typically reset within 1 hour
    
    # Failure tracking - only block after N failures
    key_failures: dict = field(default_factory=dict)
    max_failures_before_block: int = 2  # Block key after 2 rate limit errors
    
    # Track permanently invalid keys (suspended, invalid API keys)
    _invalid_keys: set = field(default_factory=set, repr=False)
    
    # Persistence file path
    _blocked_keys_file: Path = field(default=None, repr=False)
    
    def __post_init__(self):
        """Load persisted blocked keys on init"""
        if self._blocked_keys_file is None:
            self._blocked_keys_file = Path(__file__).parent / "data" / "blocked_keys.json"
        self._load_blocked_keys()
    
    def _load_blocked_keys(self):
        """
        Load blocked keys from disk.
        
        NOTE: We now use KeyPoolManager for 60s rate limiting.
        The old 1-hour blocking is deprecated. Clear any existing blocks.
        """
        # Clear any existing blocks from old system
        self.blocked_keys = {}
        
        # Also clear the file to prevent confusion
        try:
            if self._blocked_keys_file and self._blocked_keys_file.exists():
                self._blocked_keys_file.unlink()
                print(f"[APIKeys] Cleared old blocked_keys file (now using KeyPoolManager)", flush=True)
        except Exception as e:
            pass  # Ignore errors clearing the file
        
        print(f"[APIKeys] All keys available (rate limiting via KeyPoolManager)", flush=True)
    
    def _save_blocked_keys(self):
        """Save blocked keys to disk"""
        try:
            if self._blocked_keys_file:
                import json
                self._blocked_keys_file.parent.mkdir(parents=True, exist_ok=True)
                # Convert datetime to ISO strings
                data = {
                    str(k): v.isoformat() 
                    for k, v in self.blocked_keys.items()
                }
                with open(self._blocked_keys_file, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"[APIKeys] Could not save blocked keys: {e}", flush=True)
    
    def is_key_blocked(self, key_index: int) -> bool:
        """Check if a key is currently blocked or permanently invalid"""
        from datetime import datetime, timedelta
        
        # Check if permanently invalid (suspended, etc.)
        if key_index in self._invalid_keys:
            return True
        
        if key_index not in self.blocked_keys:
            return False
        
        block_time = self.blocked_keys[key_index]
        unblock_time = block_time + timedelta(hours=self.block_duration_hours)
        
        if datetime.now() >= unblock_time:
            # Block expired, remove it
            del self.blocked_keys[key_index]
            print(f"[APIKeys] ✅ Key {key_index + 1} unblocked (12h expired)", flush=True)
            self._save_blocked_keys()  # Persist to disk
            return False
        
        return True
    
    def block_key(self, key_index: int):
        """Record a failure for a key - only block after max_failures_before_block failures"""
        from datetime import datetime
        
        # Increment failure count
        if key_index not in self.key_failures:
            self.key_failures[key_index] = 0
        self.key_failures[key_index] += 1
        
        failure_count = self.key_failures[key_index]
        key = self.gemini_api_keys[key_index] if key_index < len(self.gemini_api_keys) else "?"
        key_suffix = key[-8:] if key else "?"
        
        if failure_count < self.max_failures_before_block:
            print(f"[APIKeys] ⚠️ Key {key_index + 1} (...{key_suffix}) failed ({failure_count}/{self.max_failures_before_block})", flush=True)
            return
        
        # Reached max failures - actually block the key
        self.blocked_keys[key_index] = datetime.now()
        self.key_failures[key_index] = 0  # Reset failure count
        print(f"[APIKeys] 🚫 Key {key_index + 1} (...{key_suffix}) BLOCKED for {self.block_duration_hours}h (failed {self.max_failures_before_block}x)", flush=True)
        self._save_blocked_keys()  # Persist to disk
    
    def reset_key_failures(self, key_index: int):
        """Reset failure count for a key (call on success)"""
        if key_index in self.key_failures and self.key_failures[key_index] > 0:
            self.key_failures[key_index] = 0
            key = self.gemini_api_keys[key_index] if key_index < len(self.gemini_api_keys) else "?"
            key_suffix = key[-8:] if key else "?"
            print(f"[APIKeys] ✅ Key {key_index + 1} (...{key_suffix}) success - failures reset", flush=True)
    
    def get_available_key_count(self) -> int:
        """Count how many keys are currently available (not blocked or invalid)"""
        available = 0
        for i in range(len(self.gemini_api_keys)):
            if not self.is_key_blocked(i) and i not in self._invalid_keys:
                available += 1
        return available
    
    def get_current_gemini_key(self) -> Optional[str]:
        """Get current Gemini API key (skips blocked keys)"""
        if not self.gemini_api_keys:
            return None
        
        if self.current_key_index >= len(self.gemini_api_keys):
            self.current_key_index = 0
        
        # If current key is blocked, find next available
        if self.is_key_blocked(self.current_key_index):
            self._find_next_available_key()
        
        # Check if we have any available keys
        if self.get_available_key_count() == 0:
            print(f"[APIKeys] ⚠️ ALL {len(self.gemini_api_keys)} keys are blocked!", flush=True)
            return None
        
        return self.gemini_api_keys[self.current_key_index]
    
    def _find_next_available_key(self):
        """Find the next non-blocked key"""
        start_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.gemini_api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
            if not self.is_key_blocked(self.current_key_index):
                return
            attempts += 1
        
        # All keys blocked, reset to original
        self.current_key_index = start_index
    
    def rotate_key(self, block_current: bool = False):
        """Rotate to next API key, optionally blocking the current one"""
        if not self.gemini_api_keys:
            return
        
        if block_current:
            self.block_key(self.current_key_index)
        
        self._find_next_available_key()
    
    def get_status(self) -> dict:
        """Get status of API keys for admin dashboard"""
        blocked_info = []
        from datetime import datetime, timedelta
        
        for idx, block_time in self.blocked_keys.items():
            unblock_time = block_time + timedelta(hours=self.block_duration_hours)
            remaining = unblock_time - datetime.now()
            remaining_hours = max(0, remaining.total_seconds() / 3600)
            key = self.gemini_api_keys[idx] if idx < len(self.gemini_api_keys) else "?"
            blocked_info.append({
                "index": idx + 1,
                "key_suffix": key[-8:] if key else "?",
                "blocked_at": block_time.isoformat(),
                "unblocks_at": unblock_time.isoformat(),
                "remaining_hours": round(remaining_hours, 1)
            })
        
        # Build key status with failure counts
        key_status = []
        for idx in range(len(self.gemini_api_keys)):
            key = self.gemini_api_keys[idx]
            failures = self.key_failures.get(idx, 0)
            is_blocked = self.is_key_blocked(idx)
            is_invalid = idx in self._invalid_keys
            key_status.append({
                "index": idx + 1,
                "key_suffix": key[-8:] if key else "?",
                "failures": failures,
                "max_failures": self.max_failures_before_block,
                "blocked": is_blocked,
                "invalid": is_invalid,
                "status": "invalid" if is_invalid else ("blocked" if is_blocked else "available")
            })
        
        return {
            "gemini_keys_count": len(self.gemini_api_keys),
            "gemini_keys_configured": len(self.gemini_api_keys) > 0,
            "gemini_current_index": self.current_key_index,
            "gemini_available_keys": self.get_available_key_count(),
            "gemini_blocked_keys": len(self.blocked_keys),
            "gemini_invalid_keys": len(self._invalid_keys),
            "blocked_details": blocked_info,
            "key_status": key_status,
            "max_failures_before_block": self.max_failures_before_block,
            "openai_configured": self.openai_api_key is not None,
        }
    
    def validate(self) -> List[str]:
        """Validate API keys configuration"""
        errors = []
        
        if not self.get_current_gemini_key():
            if self.get_available_key_count() == 0 and len(self.gemini_api_keys) > 0:
                errors.append(f"All {len(self.gemini_api_keys)} Gemini API keys are blocked (quota exhausted). Wait for unblock or add new keys.")
            else:
                errors.append("No Gemini API keys configured. Add keys to .env file.")
        
        return errors
    
    def validate_keys_with_api(self, log_callback=None) -> int:
        """
        Test all API keys by actually submitting a Veo video generation request.
        This is the ONLY reliable way to check if a key has Veo quota available.
        
        - Keys that can submit: marked as valid
        - Keys that get 429: marked as rate-limited (blocked 300s)
        - Keys that fail with other errors: marked as invalid
        
        Returns the number of valid keys (including rate-limited ones).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # Import key_pool for marking rate-limited keys
        from config import key_pool
        
        # Veo config for test
        VEO_MODEL = "veo-3.1-fast-generate-preview"
        TEST_PROMPT = "A calm blue ocean wave gently rolling onto a sandy beach at sunset"
        
        def log(msg):
            print(msg, flush=True)
            if log_callback:
                log_callback(msg)
        
        if not self.gemini_api_keys:
            log("[APIKeys] No API keys configured")
            return 0
        
        log(f"[APIKeys] Validating {len(self.gemini_api_keys)} API keys (testing Veo submission)...")
        
        def test_single_key(key_index: int, api_key: str) -> tuple:
            """
            Test a single key by actually submitting to Veo.
            Returns (key_index, is_valid, is_rate_limited, error_msg)
            """
            key_suffix = api_key[-8:] if len(api_key) > 8 else "***"
            
            # Skip already known invalid keys
            if key_index in self._invalid_keys:
                return (key_index, False, False, "Previously marked invalid")
            
            try:
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=api_key)
                
                # Step 1: Quick check if key works at all (models.list is fast)
                try:
                    models = list(client.models.list())
                except Exception as e:
                    error_str = str(e).lower()
                    if "suspended" in error_str:
                        return (key_index, False, False, "SUSPENDED")
                    elif "invalid" in error_str or "api_key_invalid" in error_str:
                        return (key_index, False, False, "INVALID KEY")
                    elif "401" in str(e):
                        return (key_index, False, False, "UNAUTHORIZED")
                    elif "403" in str(e):
                        return (key_index, False, False, "PERMISSION DENIED")
                    else:
                        return (key_index, False, False, f"API error: {str(e)[:50]}")
                
                # Step 2: Actually try to submit a Veo generation
                # This is the ONLY way to test Veo quota
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
                # The video will generate in background (we'll ignore it)
                return (key_index, True, False, f"✓ Veo OK (test submitted)")
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "429" in str(e) or "resource_exhausted" in error_str:
                    # Rate limited - key is valid but quota exhausted
                    return (key_index, True, True, "429 rate-limited")
                elif "suspended" in error_str:
                    return (key_index, False, False, "SUSPENDED")
                elif "permission" in error_str or "403" in str(e):
                    return (key_index, False, False, "NO VEO ACCESS")
                elif "404" in str(e) or "not found" in error_str:
                    return (key_index, False, False, "VEO MODEL NOT AVAILABLE")
                else:
                    # Unknown error during Veo submit - treat as rate-limited to be safe
                    return (key_index, True, True, f"Error: {str(e)[:40]}")
        
        valid_count = 0
        invalid_count = 0
        rate_limited_count = 0
        working_count = 0  # Keys that actually submitted successfully
        results = []
        
        # Test keys sequentially to avoid overwhelming the API
        # (parallel testing can cause false 429s)
        for i, key in enumerate(self.gemini_api_keys):
            key_index, is_valid, is_rate_limited, message = test_single_key(i, key)
            key_suffix = self.gemini_api_keys[key_index][-8:]
            
            if is_valid:
                valid_count += 1
                if is_rate_limited:
                    rate_limited_count += 1
                    # Mark as rate-limited in KeyPoolManager for 300 seconds (5 minutes)
                    # Google's rate limits typically last 1-5 minutes
                    key_pool.mark_key_rate_limited(key_index, duration_seconds=300)
                    log(f"[APIKeys] ⚠ Key {key_index + 1} (...{key_suffix}): {message} - blocked 300s")
                else:
                    working_count += 1
                    log(f"[APIKeys] ✓ Key {key_index + 1} (...{key_suffix}): {message}")
            else:
                invalid_count += 1
                # Mark as permanently invalid (won't retry)
                self._invalid_keys.add(key_index)
                log(f"[APIKeys] ✗ Key {key_index + 1} (...{key_suffix}): {message} - DISABLED")
            
            results.append((key_index, is_valid, is_rate_limited, message))
            
            # Small delay between tests to avoid triggering rate limits
            if i < len(self.gemini_api_keys) - 1:
                time.sleep(0.5)
        
        log(f"[APIKeys] Validation complete:")
        log(f"[APIKeys]   ✓ Working NOW: {working_count}")
        log(f"[APIKeys]   ⚠ Rate-limited (will recover): {rate_limited_count}")
        log(f"[APIKeys]   ✗ Invalid/disabled: {invalid_count}")
        
        # If NO keys are working right now, this is a problem
        if working_count == 0:
            if rate_limited_count > 0:
                log(f"[APIKeys] ⚠️ All keys are rate-limited! Will wait for recovery...")
            else:
                log(f"[APIKeys] ❌ No valid API keys available!")
        
        # Return tuple: (working_now, rate_limited, invalid)
        return working_count, rate_limited_count, invalid_count


# Global API keys instance (loaded from environment)
api_keys_config = APIKeysConfig()


@dataclass
class DialogueLine:
    """Single dialogue line"""
    id: int
    text: str
    
    def validate(self) -> List[str]:
        errors = []
        if self.id < 1:
            errors.append(f"Invalid ID {self.id}: must be positive")
        if not self.text or not self.text.strip():
            errors.append(f"Line {self.id}: text cannot be empty")
        if len(self.text) > 2000:
            errors.append(f"Line {self.id}: text too long (max 2000 chars)")
        return errors


# Veo model configuration
VEO_MODEL = "veo-3.1-fast-generate-preview"
OPENAI_MODEL = "gpt-4.1"

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

# Max image file size (in bytes)
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


class KeyPoolManager:
    """
    Manages API key allocation across jobs with smart rotation.
    
    Features:
    - Per-job key reservation: each job gets N dedicated keys
    - Smart rotation: keys have cooldown between uses (prevents burning all keys instantly)
    - Rate limit tracking: short-term blocking (60s) vs long-term blocking (1h+)
    - Fallback: if all assigned keys exhausted, can borrow free keys
    """
    
    _instance = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        import threading
        self._lock = threading.RLock()
        
        # Key state tracking
        self._key_last_used: Dict[int, datetime] = {}  # key_index -> last_used timestamp
        self._key_rate_limited_until: Dict[int, datetime] = {}  # key_index -> rate_limit_expires
        self._key_reserved_by: Dict[int, str] = {}  # key_index -> job_id
        
        # Configuration
        self._min_key_cooldown_seconds = 8  # Don't reuse a key within 8 seconds
        self._rate_limit_duration_seconds = 300  # Short-term rate limit block (5 minutes)
        
        self._initialized = True
        print("[KeyPoolManager] Initialized", flush=True)
    
    def get_reserved_keys_for_job(self, job_id: str) -> List[int]:
        """Get list of keys already reserved by this job."""
        with self._lock:
            return [idx for idx, jid in self._key_reserved_by.items() if jid == job_id]
    
    def reserve_keys_for_job(self, job_id: str, num_keys: int, api_keys_config: 'APIKeysConfig') -> List[int]:
        """
        Reserve N keys for a specific job.
        If job already has reserved keys, returns those instead of reserving more.
        Returns list of key indices that were reserved.
        
        Prioritizes keys that are NOT rate-limited (working keys first).
        """
        with self._lock:
            # Check if this job already has reserved keys
            existing = [idx for idx, jid in self._key_reserved_by.items() if jid == job_id]
            if existing:
                print(f"[KeyPoolManager] Job {job_id[:8]}: Using {len(existing)} already-reserved keys", flush=True)
                return existing
            
            total_keys = len(api_keys_config.gemini_api_keys)
            now = datetime.now()
            
            # Separate keys into working and rate-limited
            working_keys = []
            rate_limited_keys = []
            
            for idx in range(total_keys):
                if idx in api_keys_config._invalid_keys:
                    continue
                if idx in self._key_reserved_by:
                    continue
                
                # Check if rate-limited
                if idx in self._key_rate_limited_until and self._key_rate_limited_until[idx] > now:
                    rate_limited_keys.append(idx)
                else:
                    working_keys.append(idx)
            
            # Prioritize working keys, then fall back to rate-limited if needed
            available_indices = working_keys + rate_limited_keys
            
            # Reserve up to num_keys
            reserved = available_indices[:num_keys]
            for idx in reserved:
                self._key_reserved_by[idx] = job_id
            
            # Log what we reserved
            working_reserved = [i for i in reserved if i in working_keys]
            rate_limited_reserved = [i for i in reserved if i in rate_limited_keys]
            
            # Safe access with bounds checking
            total_available = len(api_keys_config.gemini_api_keys)
            key_suffixes = [api_keys_config.gemini_api_keys[i][-8:] if i < total_available else f"?{i}?" for i in reserved]
            print(f"[KeyPoolManager] Job {job_id[:8]}: Reserved {len(reserved)} keys ({len(working_reserved)} working, {len(rate_limited_reserved)} rate-limited): {key_suffixes}", flush=True)
            
            return reserved
    
    def release_keys_for_job(self, job_id: str):
        """Release all keys reserved by a job."""
        with self._lock:
            released = []
            for idx, reserved_job in list(self._key_reserved_by.items()):
                if reserved_job == job_id:
                    del self._key_reserved_by[idx]
                    released.append(idx)
            
            if released:
                print(f"[KeyPoolManager] Job {job_id[:8]}: Released {len(released)} keys back to pool", flush=True)
    
    def get_any_available_key(self, api_keys_config: 'APIKeysConfig') -> Optional[Tuple[int, str]]:
        """
        Get ANY available key from the pool - fully dynamic, no reservations.
        
        This is the NEW approach for parallel jobs:
        - No static key assignments
        - All jobs share all keys dynamically
        - Rate-limited keys are skipped
        - Cooldown keys are waited for (short wait)
        
        Returns (key_index, api_key) or None if ALL keys are rate-limited.
        """
        # First pass: try to get a ready key without waiting
        with self._lock:
            now = datetime.now()
            total_keys = len(api_keys_config.gemini_api_keys)
            
            # Clean up expired rate limits
            for idx in list(self._key_rate_limited_until.keys()):
                if self._key_rate_limited_until[idx] <= now:
                    del self._key_rate_limited_until[idx]
            
            # Categorize all keys
            ready_keys = []
            cooldown_keys = []
            rate_limited_keys = []
            
            for idx in range(total_keys):
                # Skip permanently invalid keys
                if idx in api_keys_config._invalid_keys:
                    continue
                
                # Check if rate-limited
                if idx in self._key_rate_limited_until:
                    remaining = (self._key_rate_limited_until[idx] - now).total_seconds()
                    if remaining > 0:
                        rate_limited_keys.append((idx, remaining))
                        continue
                
                # Check cooldown (8s between uses of same key)
                last_used = self._key_last_used.get(idx)
                if last_used:
                    elapsed = (now - last_used).total_seconds()
                    if elapsed < self._min_key_cooldown_seconds:
                        cooldown_keys.append((idx, elapsed))
                        continue
                
                ready_keys.append(idx)
            
            # Priority 1: Use a ready key (no waiting needed)
            if ready_keys:
                key_idx = ready_keys[0]
                self._key_last_used[key_idx] = now
                key_suffix = api_keys_config.gemini_api_keys[key_idx][-8:]
                print(f"[KeyPool] Using key {key_idx+1} (...{key_suffix}) - {len(ready_keys)} ready, {len(cooldown_keys)} cooldown, {len(rate_limited_keys)} rate-limited", flush=True)
                return (key_idx, api_keys_config.gemini_api_keys[key_idx])
            
            # Priority 2: If we have cooldown keys, pick one to wait for
            # RELEASE LOCK before sleeping to allow other threads to proceed
            if cooldown_keys:
                # Pick the key closest to being ready
                cooldown_keys.sort(key=lambda x: x[1], reverse=True)  # Sort by elapsed time descending
                key_idx, elapsed = cooldown_keys[0]
                wait_time = self._min_key_cooldown_seconds - elapsed
                key_to_wait = (key_idx, wait_time, api_keys_config.gemini_api_keys[key_idx])
            else:
                key_to_wait = None
            
            # Priority 3: Check rate-limited keys
            if not key_to_wait and rate_limited_keys:
                soonest_idx, soonest_time = min(rate_limited_keys, key=lambda x: x[1])
                
                # If soonest recovery is within 30 seconds, wait for it
                if soonest_time <= 30:
                    key_to_wait = (soonest_idx, soonest_time + 1, api_keys_config.gemini_api_keys[soonest_idx])
                    is_rate_limited_wait = True
                else:
                    # Long wait - return None and let caller handle it (pause job)
                    print(f"[KeyPool] All {total_keys} keys rate-limited, soonest recovery: key {soonest_idx+1} in {soonest_time:.0f}s - returning None", flush=True)
                    return None
            else:
                is_rate_limited_wait = False
            
            if not key_to_wait and not ready_keys:
                print(f"[KeyPool] No keys available (all invalid?)", flush=True)
                return None
        
        # Wait OUTSIDE the lock so other threads aren't blocked
        if key_to_wait:
            key_idx, wait_time, api_key = key_to_wait
            if wait_time > 0:
                wait_type = "rate-limit recovery" if is_rate_limited_wait else "cooldown"
                print(f"[KeyPool] Waiting {wait_time:.1f}s for key {key_idx+1} ({wait_type})", flush=True)
                import time
                time.sleep(wait_time)
            
            # Re-acquire lock to mark key as used
            with self._lock:
                # Check if key is still available (another thread might have taken it)
                now = datetime.now()
                if is_rate_limited_wait and key_idx in self._key_rate_limited_until:
                    del self._key_rate_limited_until[key_idx]
                self._key_last_used[key_idx] = now
                key_suffix = api_key[-8:]
                print(f"[KeyPool] Using key {key_idx+1} (...{key_suffix}) after wait", flush=True)
                return (key_idx, api_key)
        
        return None
    
    def get_pool_status_summary(self, api_keys_config: 'APIKeysConfig') -> dict:
        """Get a summary of pool status for logging/debugging."""
        with self._lock:
            now = datetime.now()
            total = len(api_keys_config.gemini_api_keys)
            invalid = len(api_keys_config._invalid_keys)
            rate_limited = sum(1 for idx in self._key_rate_limited_until 
                              if self._key_rate_limited_until.get(idx, now) > now)
            available = total - invalid - rate_limited
            
            return {
                "total": total,
                "available": available,
                "rate_limited": rate_limited,
                "invalid": invalid,
            }

    def get_best_key(self, job_id: str, reserved_keys: List[int], api_keys_config: 'APIKeysConfig') -> Optional[Tuple[int, str]]:
        """
        Get the best available key for a job.
        
        Priority:
        1. Reserved keys that aren't on cooldown or rate-limited (60s)
        2. Reserved keys that are on cooldown but not rate-limited (if desperate)
        3. Free keys from pool (borrowed)
        
        NOTE: We only check for permanently invalid keys, NOT the old 1-hour blocking.
        Rate limits use the 60s system in KeyPoolManager.
        
        Returns (key_index, api_key) or None if no keys available.
        """
        with self._lock:
            now = datetime.now()
            
            # Clean up expired rate limits
            for idx in list(self._key_rate_limited_until.keys()):
                if self._key_rate_limited_until[idx] <= now:
                    del self._key_rate_limited_until[idx]
            
            # Priority 1: Reserved keys that are ready (no cooldown, no rate limit)
            ready_keys = []
            cooldown_keys = []
            rate_limited_keys = []
            
            for idx in reserved_keys:
                # Skip permanently invalid keys (suspended, etc.)
                if idx in api_keys_config._invalid_keys:
                    continue
                
                # Check if rate-limited (60s block)
                if idx in self._key_rate_limited_until:
                    remaining = (self._key_rate_limited_until[idx] - now).total_seconds()
                    rate_limited_keys.append((idx, remaining))
                    continue
                
                # Check cooldown (8s between uses)
                last_used = self._key_last_used.get(idx)
                if last_used and (now - last_used).total_seconds() < self._min_key_cooldown_seconds:
                    cooldown_keys.append((idx, last_used))
                else:
                    ready_keys.append(idx)
            
            # Use a ready key if available
            if ready_keys:
                key_idx = ready_keys[0]
                if key_idx >= len(api_keys_config.gemini_api_keys):
                    print(f"[KeyPoolManager] ERROR: key_idx {key_idx} >= len(keys) {len(api_keys_config.gemini_api_keys)}", flush=True)
                    return None
                self._key_last_used[key_idx] = now
                key_suffix = api_keys_config.gemini_api_keys[key_idx][-8:]
                print(f"[KeyPoolManager] Using key {key_idx+1} (...{key_suffix}) - {len(ready_keys)} ready, {len(rate_limited_keys)} rate-limited", flush=True)
                return (key_idx, api_keys_config.gemini_api_keys[key_idx])
            
            # Priority 2: Use cooldown key if no ready keys (pick the oldest one)
            if cooldown_keys:
                cooldown_keys.sort(key=lambda x: x[1])  # Sort by last_used ascending
                key_idx = cooldown_keys[0][0]
                if key_idx >= len(api_keys_config.gemini_api_keys):
                    print(f"[KeyPoolManager] ERROR: cooldown key_idx {key_idx} >= len(keys) {len(api_keys_config.gemini_api_keys)}", flush=True)
                    return None
                wait_time = self._min_key_cooldown_seconds - (now - cooldown_keys[0][1]).total_seconds()
                if wait_time > 0:
                    print(f"[KeyPoolManager] All {len(reserved_keys)} reserved keys on cooldown, waiting {wait_time:.1f}s", flush=True)
                    import time
                    time.sleep(wait_time)
                self._key_last_used[key_idx] = datetime.now()
                key_suffix = api_keys_config.gemini_api_keys[key_idx][-8:]
                print(f"[KeyPoolManager] Using key {key_idx+1} (...{key_suffix}) after cooldown", flush=True)
                return (key_idx, api_keys_config.gemini_api_keys[key_idx])
            
            # Priority 3: If all keys are rate-limited, DON'T BLOCK - return None
            # Let the caller handle it (pause job, retry later, etc.)
            # Blocking here would freeze the thread for minutes!
            if rate_limited_keys:
                rate_limited_keys.sort(key=lambda x: x[1])  # Sort by remaining time
                key_idx, remaining = rate_limited_keys[0]
                if key_idx >= len(api_keys_config.gemini_api_keys):
                    print(f"[KeyPoolManager] ERROR: rate-limited key_idx {key_idx} >= len(keys) {len(api_keys_config.gemini_api_keys)}", flush=True)
                    return None
                
                # If remaining time is very short (< 5s), wait for it
                if remaining > 0 and remaining <= 5:
                    print(f"[KeyPoolManager] Waiting {remaining:.1f}s for key {key_idx+1} to recover (short wait)", flush=True)
                    import time
                    time.sleep(remaining + 1)
                    # Clear the rate limit and use this key
                    if key_idx in self._key_rate_limited_until:
                        del self._key_rate_limited_until[key_idx]
                    self._key_last_used[key_idx] = datetime.now()
                    key_suffix = api_keys_config.gemini_api_keys[key_idx][-8:]
                    print(f"[KeyPoolManager] Using key {key_idx+1} (...{key_suffix}) after short wait", flush=True)
                    return (key_idx, api_keys_config.gemini_api_keys[key_idx])
                else:
                    # Long wait - don't block, return None
                    print(f"[KeyPoolManager] All {len(reserved_keys)} keys rate-limited, soonest recovery in {remaining:.0f}s - returning None (caller should pause)", flush=True)
                    return None
            
            # Priority 4: Try to borrow a free key from the pool
            borrowed_key = self._try_borrow_free_key(job_id, api_keys_config)
            if borrowed_key:
                return borrowed_key
            
            return None
    
    def _try_borrow_free_key(self, job_id: str, api_keys_config: 'APIKeysConfig') -> Optional[Tuple[int, str]]:
        """Try to borrow a free (unreserved) key from the pool."""
        now = datetime.now()
        total_keys = len(api_keys_config.gemini_api_keys)
        
        for idx in range(total_keys):
            # Skip permanently invalid keys (suspended, etc.)
            if idx in api_keys_config._invalid_keys:
                continue
            
            # Skip reserved keys
            if idx in self._key_reserved_by:
                continue
            
            # Skip rate-limited keys (60s block)
            if idx in self._key_rate_limited_until:
                if self._key_rate_limited_until[idx] > now:
                    continue
                else:
                    # Expired, remove it
                    del self._key_rate_limited_until[idx]
            
            # Found a free key! Temporarily reserve it
            self._key_reserved_by[idx] = job_id
            self._key_last_used[idx] = now
            key_suffix = api_keys_config.gemini_api_keys[idx][-8:]
            print(f"[KeyPoolManager] Job {job_id[:8]}: Borrowed free key {idx+1} (...{key_suffix})", flush=True)
            return (idx, api_keys_config.gemini_api_keys[idx])
        
        return None
    
    def mark_key_rate_limited(self, key_index: int, duration_seconds: int = None):
        """Mark a key as temporarily rate-limited (short-term, ~60s)."""
        with self._lock:
            duration = duration_seconds or self._rate_limit_duration_seconds
            self._key_rate_limited_until[key_index] = datetime.now() + timedelta(seconds=duration)
            print(f"[KeyPoolManager] Key {key_index+1} rate-limited for {duration}s", flush=True)
    
    def mark_key_used(self, key_index: int):
        """Mark that a key was just used (update cooldown timer)."""
        with self._lock:
            self._key_last_used[key_index] = datetime.now()
    
    def get_all_reserved_keys_rate_limited(self, reserved_keys: List[int]) -> bool:
        """Check if ALL reserved keys are currently rate-limited."""
        with self._lock:
            now = datetime.now()
            for idx in reserved_keys:
                if idx not in self._key_rate_limited_until:
                    return False
                if self._key_rate_limited_until[idx] <= now:
                    return False
            return True
    
    def wait_for_any_key(self, reserved_keys: List[int], timeout_seconds: int = 120) -> bool:
        """
        Wait until at least one reserved key becomes available.
        Returns True if a key became available, False if timeout.
        """
        import time
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout_seconds:
            with self._lock:
                now = datetime.now()
                for idx in reserved_keys:
                    if idx not in self._key_rate_limited_until:
                        return True
                    if self._key_rate_limited_until[idx] <= now:
                        del self._key_rate_limited_until[idx]
                        return True
                
                # Find minimum wait time
                min_wait = timeout_seconds
                for idx in reserved_keys:
                    if idx in self._key_rate_limited_until:
                        wait = (self._key_rate_limited_until[idx] - now).total_seconds()
                        min_wait = min(min_wait, wait)
            
            # Wait a bit before checking again
            time.sleep(min(min_wait + 1, 10))
        
        return False
    
    def get_status(self) -> dict:
        """Get current pool status for debugging."""
        with self._lock:
            now = datetime.now()
            return {
                "reserved_count": len(self._key_reserved_by),
                "rate_limited_count": sum(1 for t in self._key_rate_limited_until.values() if t > now),
                "reservations": dict(self._key_reserved_by),
                "rate_limits": {k: v.isoformat() for k, v in self._key_rate_limited_until.items() if v > now}
            }


# Singleton key pool manager
key_pool = KeyPoolManager()


# Default prompts and instructions (from original script)
BASE_PROMPT = """
A realistic vertical video that maintains natural continuity.
The visual style, camera setup, and environment are based on the input frames.
The subject's facial expressions, gestures, and body language adapt naturally to communicate the spoken message.
"""

NO_TEXT_INSTRUCTION = (
    "CRITICAL: No text, subtitles, captions, words, letters, numbers, graphics, or overlays "
    "may appear on screen at any time. No burned-in text. No visual text elements whatsoever. "
    "Any text visible in the background (whiteboards, signs) must remain static environmental props."
)

AUDIO_TIMING_INSTRUCTION = (
    "The narrator must stop speaking exactly at 7.0 seconds. "
    "From 7.0 to 8.0 seconds: silence, no words, no breaths, no mouth sounds. "
    "Only natural room tone or ambient sound. "
    "Total duration: 8 seconds (7 seconds speech + 1 second quiet ambience)."
)

AUDIO_QUALITY_INSTRUCTION = (
    "Clean, natural audio recording with professional broadcast quality. "
    "Studio microphone sound, low noise floor, no clipping, no distortion, no metallic or robotic artifacts. "
    "Stable loudness over the whole line, gentle broadcast-style compression, minimal room reverb."
)

PRONUNCIATION_TEMPLATE = (
    "Pronunciation must be native-level {language}, with correct stress on every word. "
    "Avoid foreign accents and avoid moving the stress to the wrong syllable. "
    "Follow standard dictionary stress patterns for {language}. "
    "Use authentic {language} pronunciation with natural rhythm and intonation."
)


# Singleton app config
app_config = AppConfig()