# -*- coding: utf-8 -*-
"""
Flow Backend for Veo Web App

Browser automation backend using Playwright to automate Google Flow UI.
This is used when users don't have their own API keys.

Adapted from test_for_jobs2.py with:
- Database integration (replaces Excel/JSON cache)
- Object storage integration (replaces local file system)
- Headless operation with stored auth state
- Error recovery and retry logic
"""

import os
import re
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field

try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Try to import playwright-stealth for better anti-detection
try:
    from playwright_stealth import stealth_sync
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False


# Configuration
FLOW_HOME_URL = "https://labs.google/fx/tools/flow"
GENERATION_TIMEOUT = 180  # Seconds to wait for generation
DEFAULT_WAIT_AFTER_SUBMIT = 120  # Seconds to wait before attempting download


@dataclass
class FlowClip:
    """Represents a clip to be generated via Flow"""
    clip_index: int
    dialogue_text: str
    start_frame_path: Optional[str] = None
    end_frame_path: Optional[str] = None
    start_frame_key: Optional[str] = None  # S3 key if using object storage
    end_frame_key: Optional[str] = None    # S3 key if using object storage
    prompt: Optional[str] = None  # Pre-built prompt from API prompt engine
    voice_profile: Optional[str] = None  # Voice profile for prompt building
    duration: float = 8.0  # Clip duration in seconds
    
    # Output tracking
    flow_clip_id: Optional[str] = None
    output_url: Optional[str] = None
    output_key: Optional[str] = None
    status: str = "pending"  # pending, submitted, generating, completed, failed
    error_message: Optional[str] = None


@dataclass
class FlowJob:
    """Represents a job to be processed via Flow"""
    job_id: str
    clips: List[FlowClip]
    
    # Flow project tracking
    project_url: Optional[str] = None
    state_json: Optional[str] = None
    
    # Callbacks
    on_progress: Optional[Callable[[int, str, str], None]] = None
    on_error: Optional[Callable[[str], None]] = None


def get_prompt(
    dialogue: str, 
    language: str = "English",
    voice_profile: str = None,
    duration: float = 8.0,
    facial_expression: str = None,
    body_language: str = None,
    delivery_style: str = None,
    emotion: str = None,
    redo_feedback: str = None,
) -> str:
    """
    Generate the video generation prompt from dialogue text.
    
    This matches the structure from veo_generator.py build_prompt() to ensure
    consistency between API and Flow backends.
    
    KEY PRINCIPLES (from veo_generator.py):
    1. NO VISUAL REDESCRIPTION: The image locks appearance
    2. RAW/DOCUMENTARY STYLE: Not "cinematic" - prevents AI glossy look
    3. STATIC CAMERA: For talking heads, locked-off camera preserves lip-sync
    4. VOICE PROFILE: Extract and pass voice traits correctly
    5. "Character says:" syntax for Veo lip-sync engine
    
    Args:
        dialogue: The dialogue line to speak
        language: Language for pronunciation
        voice_profile: Full voice profile text
        duration: Clip duration in seconds
        facial_expression: Expression description
        body_language: Body language description
        delivery_style: How the line should be delivered
        emotion: Emotional tone
        redo_feedback: Priority feedback for redo attempts
        
    Returns:
        Formatted prompt string
    """
    dialogue = dialogue.strip().strip('"').strip("'")
    
    # Defaults
    if facial_expression is None:
        facial_expression = "natural engaged expression"
    if body_language is None:
        body_language = "natural posture"
    if delivery_style is None:
        delivery_style = "natural conversational"
    if emotion is None:
        emotion = "neutral"
    
    # Calculate timing
    speech_end_time = duration - 1.0
    
    # Extract voice traits from voice profile if provided
    voice_texture = ""
    voice_tone = ""
    voice_accent = ""
    voice_signature = ""
    
    if voice_profile:
        lines = voice_profile.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            line_clean = line.strip()
            
            # Extract texture (raspy, smooth, gravelly, etc.)
            if 'texture:' in line_lower:
                voice_texture = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            elif 'quality:' in line_lower and not voice_texture:
                voice_texture = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            
            # Extract tone
            if 'tone:' in line_lower:
                voice_tone = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            
            # Extract accent
            if 'accent:' in line_lower:
                accent_val = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
                if accent_val and 'none' not in accent_val.lower() and 'neutral' not in accent_val.lower():
                    voice_accent = accent_val
            
            # Extract signature traits
            if 'signature' in line_lower and 'trait' in line_lower:
                voice_signature = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
    
    # Build consolidated voice instruction
    voice_parts = []
    if voice_texture:
        voice_parts.append(voice_texture)
    if voice_tone:
        voice_parts.append(voice_tone)
    if voice_signature:
        voice_parts.append(voice_signature)
    if voice_accent:
        voice_parts.append(f"accent: {voice_accent}")
    
    short_voice = ", ".join(voice_parts) if voice_parts else "natural voice"
    
    # Build prompt following Veo 3.1 official structure (same as veo_generator.py)
    if voice_profile:
        final_prompt = f"""=== VOICE PROFILE ===
{voice_profile}
===

Medium shot, static locked-off camera, sharp focus on subject.

The subject in the frame speaks directly to camera with {facial_expression}, {body_language}.

The character says in {language}, "{dialogue}"

Voice: {short_voice}. {delivery_style}, {emotion} emotion.

Ambient noise: Complete silence, professional recording booth, no room ambiance.

Style: Raw realistic footage, natural lighting, photorealistic. Speech timing: 0s to {speech_end_time:.1f}s, then silence.

No subtitles, no text overlays, no captions, no watermarks. No background music, no laughter, no applause, no crowd sounds, no ambient noise. No morphing, no face distortion, no jerky movements. Only the speaker's isolated voice.

(no subtitles)"""
    else:
        # Simpler format without voice profile section
        final_prompt = f"""Medium shot, static locked-off camera, sharp focus on subject.

The subject in the frame speaks directly to camera with {facial_expression}, {body_language}.

The character says in {language}, "{dialogue}"

Voice: {short_voice}. {delivery_style}, {emotion} emotion.

Ambient noise: Complete silence, professional recording booth, no room ambiance.

Style: Raw realistic footage, natural lighting, photorealistic. Speech timing: 0s to {speech_end_time:.1f}s, then silence.

No subtitles, no text overlays, no captions, no watermarks. No background music, no laughter, no applause, no crowd sounds, no ambient noise. No morphing, no face distortion, no jerky movements. Only the speaker's isolated voice.

(no subtitles)"""

    # Add redo feedback at the top if present (same as veo_generator.py)
    if redo_feedback:
        final_prompt = f"""=== PRIORITY ===
{redo_feedback}
===

{final_prompt}"""
    
    return final_prompt


def clean_prompt_for_flow(
    prompt: str, 
    dialogue: str, 
    language: str = "English",
    voice_profile: str = None,
    duration: float = 8.0
) -> str:
    """
    Clean an API-generated prompt for use with Flow UI.
    
    The API prompt includes markers like === VOICE PROFILE === that confuse Flow.
    This function extracts the essential content and creates a Flow-compatible prompt.
    
    Args:
        prompt: The API-generated prompt (may have markers)
        dialogue: The dialogue text (fallback if prompt is too broken)
        language: Language for the dialogue
        voice_profile: Voice profile for prompt building
        duration: Clip duration in seconds
        
    Returns:
        Clean prompt suitable for Flow
    """
    # TESTING: Always use the simple fallback prompt
    # The API prompts seem to be causing "Failed Generation" errors
    # Uncomment below to restore API prompt cleaning
    print(f"[Flow] Using simple fallback prompt (API prompt bypassed for testing)", flush=True)
    return get_prompt(dialogue, language, voice_profile=voice_profile, duration=duration)
    
    # === Original cleaning logic (disabled for testing) ===
    # if not prompt:
    #     return get_prompt(dialogue, language, voice_profile=voice_profile, duration=duration)
    # 
    # # If prompt doesn't have our markers or voice profile hints, return as-is
    # has_markers = "===" in prompt or "---" in prompt
    # has_voice_hints = any(x in prompt.lower() for x in ['pacing:', 'accent:', 'gender:', 'pitch:'])
    # 
    # if not has_markers and not has_voice_hints:
    #     return prompt
    # ... rest of cleaning logic ...


def get_video_id(src: str) -> Optional[str]:
    """Extract video ID from video source URL"""
    if not src:
        return None
    match = re.search(r'/video/([a-f0-9-]+)\?', src)
    return match.group(1) if match else None


class FlowBackend:
    """
    Flow backend for browser automation video generation.
    
    Uses Playwright to automate Google Flow UI for users without API keys.
    """
    
    def __init__(
        self,
        storage_state_path: Optional[str] = None,
        storage_state_url: Optional[str] = None,
        headless: bool = True,
        download_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        proxy_server: Optional[str] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
        browser_type: str = "chromium"  # "chromium" or "firefox"
    ):
        """
        Initialize Flow backend.
        
        Args:
            storage_state_path: Local path to browser session folder (user_data_dir)
                              OR path to Playwright storage state JSON (legacy)
            storage_state_url: Not used - session is downloaded from R2 at auth/flow_session.zip
            headless: Whether to run browser headlessly
            download_dir: Directory for downloads
            temp_dir: Directory for temporary files
            proxy_server: Not used in persistent context mode
            proxy_username: Not used in persistent context mode
            proxy_password: Not used in persistent context mode
            browser_type: Not used - always uses Chromium for persistent context
        
        Note: This backend uses launch_persistent_context with a real Chrome profile
        for better compatibility. Upload your working flow_session folder to R2
        at auth/flow_session.zip using the upload_session.py script.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for Flow backend. "
                "Install with: pip install playwright && playwright install chromium"
            )
        
        self.storage_state_path = storage_state_path or os.environ.get("FLOW_STORAGE_STATE_PATH")
        self.storage_state_url = storage_state_url or os.environ.get("FLOW_STORAGE_STATE_URL")
        self.headless = headless
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="flow_downloads_")
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="flow_temp_")
        
        # Proxy configuration - from params or environment variables
        self.proxy_server = proxy_server or os.environ.get("FLOW_PROXY_SERVER")
        self.proxy_username = proxy_username or os.environ.get("FLOW_PROXY_USERNAME")
        self.proxy_password = proxy_password or os.environ.get("FLOW_PROXY_PASSWORD")
        
        # Log proxy configuration
        if self.proxy_server:
            masked_proxy = self.proxy_server.split('@')[-1] if '@' in self.proxy_server else self.proxy_server
            print(f"[Flow] Proxy configured: {masked_proxy}", flush=True)
        else:
            print("[Flow] WARNING: No proxy configured - session may not work from server IP!", flush=True)
        
        # Browser type - chromium or firefox
        self.browser_type = browser_type or os.environ.get("FLOW_BROWSER_TYPE", "chromium")
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
        self._needs_auth = False
        self._cancelled = False
        
        # Use persistent context for more realistic browser behavior
        self._use_persistent_context = True
        self._user_data_dir = tempfile.mkdtemp(prefix="flow_browser_")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def _download_session_from_r2(self) -> Optional[str]:
        """Download and extract browser session from R2."""
        try:
            from .storage import is_storage_configured, get_storage
            
            if not is_storage_configured():
                print("[Flow] Storage not configured, cannot download session", flush=True)
                return None
            
            storage = get_storage()
            
            # Check if session exists in R2
            session_key = "auth/flow_session.zip"
            if not storage.exists(session_key):
                print(f"[Flow] Session not found in R2: {session_key}", flush=True)
                return None
            
            # Download zip file
            zip_path = os.path.join(self.temp_dir, "flow_session.zip")
            print(f"[Flow] Downloading session from R2: {session_key}", flush=True)
            storage.download_file(session_key, zip_path)
            
            # Extract to session directory
            import zipfile
            session_dir = os.path.join(self.temp_dir, "flow_session")
            print(f"[Flow] Extracting session to: {session_dir}", flush=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(session_dir)
            
            # Clean up zip
            os.remove(zip_path)
            
            print(f"[Flow] ✓ Session extracted successfully", flush=True)
            return session_dir
            
        except Exception as e:
            print(f"[Flow] Failed to download session from R2: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def start(self):
        """Start the browser with persistent context (like working local script)"""
        print("[Flow] Starting browser with persistent context...", flush=True)
        
        # Get session directory
        session_dir = None
        
        # Option 1: Use local path if provided and exists
        if self.storage_state_path and os.path.isdir(self.storage_state_path):
            session_dir = self.storage_state_path
            print(f"[Flow] Using local session: {session_dir}", flush=True)
        else:
            # Option 2: Download from R2
            session_dir = self._download_session_from_r2()
        
        if not session_dir:
            # Option 3: Create empty session directory (will need manual login)
            session_dir = os.path.join(self.temp_dir, "flow_session")
            os.makedirs(session_dir, exist_ok=True)
            print(f"[Flow] Created empty session: {session_dir}", flush=True)
            print("[Flow] WARNING: No existing session - may need manual login", flush=True)
        
        self._user_data_dir = session_dir
        self._playwright = sync_playwright().start()
        
        # Use EXACTLY the same args as the working script
        browser_args = [
            "--disable-blink-features=AutomationControlled",
            "--window-size=1280,720",
            "--no-sandbox",
            "--disable-setuid-sandbox", 
            "--disable-dev-shm-usage",
        ]
        
        # Parse proxy configuration
        proxy_config = None
        if self.proxy_server:
            # Handle proxy URL with embedded credentials: http://user:pass@host:port
            if '@' in self.proxy_server:
                # Parse: http://user:pass@host:port
                proto_and_creds, host_port = self.proxy_server.rsplit('@', 1)
                if '://' in proto_and_creds:
                    proto, creds = proto_and_creds.split('://', 1)
                    if ':' in creds:
                        user, password = creds.split(':', 1)
                        proxy_config = {
                            "server": f"{proto}://{host_port}",
                            "username": user,
                            "password": password
                        }
                        print(f"[Flow] Proxy configured: {host_port} (with auth)", flush=True)
                    else:
                        proxy_config = {"server": self.proxy_server}
                        print(f"[Flow] Proxy configured: {self.proxy_server}", flush=True)
                else:
                    proxy_config = {"server": self.proxy_server}
            else:
                # Simple proxy without credentials in URL
                proxy_config = {"server": self.proxy_server}
                if self.proxy_username and self.proxy_password:
                    proxy_config["username"] = self.proxy_username
                    proxy_config["password"] = self.proxy_password
                    print(f"[Flow] Proxy configured: {self.proxy_server} (with auth)", flush=True)
                else:
                    print(f"[Flow] Proxy configured: {self.proxy_server}", flush=True)
            
            # Add SSL bypass args for proxy
            browser_args.extend([
                "--ignore-certificate-errors",
                "--ignore-ssl-errors",
            ])
        
        # Build launch options
        launch_options = {
            "headless": self.headless,
            "slow_mo": 200 if not self.headless else 50,
            "viewport": {"width": 1280, "height": 720},
            "args": browser_args,
            "ignore_default_args": ["--enable-automation"],
            "accept_downloads": True,
            "ignore_https_errors": True,  # Important for proxy
        }
        
        # Add proxy if configured
        if proxy_config:
            launch_options["proxy"] = proxy_config
            print(f"[Flow] Proxy server: {proxy_config['server']}", flush=True)
        
        # Launch with persistent context - THIS IS THE KEY DIFFERENCE
        print(f"[Flow] Launching persistent context from: {session_dir}", flush=True)
        
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=session_dir,
            **launch_options
        )
        
        self._page = self._context.pages[0]
        
        # Add anti-detection script - SAME as working script
        self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)
        
        print("[Flow] ✓ Browser started with persistent context", flush=True)
        print(f"[Flow] Session directory: {session_dir}", flush=True)
    
    def _human_delay(self, min_ms: int = 500, max_ms: int = 1500):
        """Add a random human-like delay"""
        import random
        delay = random.randint(min_ms, max_ms) / 1000
        time.sleep(delay)
    
    def _human_click(self, locator, description: str = "element"):
        """Click an element with human-like behavior"""
        import random
        
        try:
            locator.wait_for(state="visible", timeout=10000)
            box = locator.bounding_box()
            if box:
                x = box['x'] + box['width'] / 2 + random.randint(-5, 5)
                y = box['y'] + box['height'] / 2 + random.randint(-3, 3)
                self._page.mouse.move(x, y, steps=random.randint(5, 15))
                time.sleep(random.uniform(0.1, 0.3))
                self._page.mouse.click(x, y)
                print(f"[Flow] Human-clicked: {description}", flush=True)
            else:
                locator.click()
                print(f"[Flow] Clicked (fallback): {description}", flush=True)
        except Exception as e:
            print(f"[Flow] Human click failed for {description}: {e}", flush=True)
            locator.click(force=True)
    
    def _human_type(self, locator, text: str, description: str = "field"):
        """Type text with human-like behavior"""
        import random
        
        try:
            locator.click()
            self._human_delay(200, 400)
            locator.fill("")
            self._human_delay(100, 200)
            
            for i, char in enumerate(text):
                locator.type(char, delay=random.randint(10, 50))
                if random.random() < 0.02:
                    time.sleep(random.uniform(0.2, 0.5))
            
            print(f"[Flow] Human-typed into: {description} ({len(text)} chars)", flush=True)
        except Exception as e:
            print(f"[Flow] Human type failed for {description}: {e}, using fill()", flush=True)
            locator.fill(text)
    
    def _scroll_into_view(self, locator):
        """Scroll element into view with human-like behavior"""
        try:
            locator.scroll_into_view_if_needed()
            self._human_delay(200, 400)
        except Exception:
            pass
    
    def stop(self):
        """Stop the browser"""
        print("[Flow] Stopping browser...", flush=True)
        
        # With persistent context, we only close the context (no separate browser)
        if self._context:
            self._context.close()
            self._context = None
        
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        
        print("[Flow] Browser stopped", flush=True)
    
    def cancel(self):
        """Cancel current operation"""
        self._cancelled = True
    
    def _get_storage_state(self) -> Optional[dict]:
        """Get Playwright storage state for authentication"""
        if self.storage_state_path and os.path.exists(self.storage_state_path):
            print(f"[Flow] Loading storage state from: {self.storage_state_path}", flush=True)
            with open(self.storage_state_path, 'r') as f:
                return json.load(f)
        
        if self.storage_state_url:
            try:
                from .storage import get_storage
                storage = get_storage()
                state = storage.download_flow_auth_state()
                if state:
                    print("[Flow] Loaded storage state from object storage", flush=True)
                    return state
            except Exception as e:
                print(f"[Flow] Failed to load storage state from S3: {e}", flush=True)
        
        print("[Flow] No storage state found - will need manual login", flush=True)
        return None
    
    def _check_and_dismiss_popup(self) -> bool:
        """Check for and dismiss 'I agree' popups"""
        try:
            agree_btn = self._page.locator("text=I agree")
            if agree_btn.count() > 0 and agree_btn.is_visible():
                agree_btn.click(force=True)
                print("[Flow] Dismissed 'I agree' popup", flush=True)
                time.sleep(1)
                return True
        except Exception:
            pass
        return False
    
    def _check_for_errors(self, context: str = "") -> Optional[str]:
        """
        Check the page for error messages and popups.
        Takes a screenshot if any errors found.
        
        Args:
            context: Description of when this check is happening
            
        Returns:
            Error message if found, None otherwise
        """
        error_found = None
        
        # List of error indicators to check
        error_selectors = [
            ("text=Something went wrong", "something_went_wrong"),
            ("text=Failed Generation", "failed_generation"),
            ("text=Error", "error_message"),
            ("text=limit reached", "limit_reached"),
            ("text=quota exceeded", "quota_exceeded"),
            ("text=try again", "try_again"),
            ("text=unavailable", "unavailable"),
            ("[role='alert']", "alert_role"),
            (".error-message", "error_class"),
            ("text=blocked", "blocked"),
            ("text=violated", "policy_violation"),
        ]
        
        for selector, error_name in error_selectors:
            try:
                el = self._page.locator(selector).first
                if el.count() > 0 and el.is_visible():
                    error_text = el.text_content()[:100] if el.text_content() else error_name
                    print(f"[Flow] ⚠ Error detected ({context}): {error_text}", flush=True)
                    
                    # Take screenshot of the error
                    self._screenshot(f"error_{error_name}_{context.replace(' ', '_')}")
                    
                    if not error_found:
                        error_found = error_text
            except Exception:
                pass
        
        # Also check for popup dialogs
        try:
            dialogs = self._page.locator("[role='dialog']")
            if dialogs.count() > 0:
                for i in range(dialogs.count()):
                    dialog = dialogs.nth(i)
                    if dialog.is_visible():
                        dialog_text = dialog.text_content()[:200] if dialog.text_content() else "unknown dialog"
                        print(f"[Flow] 📋 Dialog detected ({context}): {dialog_text}", flush=True)
                        self._screenshot(f"dialog_{context.replace(' ', '_')}")
        except Exception:
            pass
        
        return error_found
    
    def _screenshot(self, name: str, upload: bool = True) -> Optional[str]:
        """Take a screenshot and optionally upload to R2."""
        local_path = f"/tmp/flow_{name}.png"
        
        try:
            self._page.screenshot(path=local_path)
            print(f"[Flow] Screenshot saved: {local_path}", flush=True)
        except Exception as e:
            print(f"[Flow] Failed to take screenshot: {e}", flush=True)
            return None
        
        if not upload:
            return local_path
        
        try:
            from .storage import is_storage_configured, get_storage
            
            if not is_storage_configured():
                return local_path
            
            storage = get_storage()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            remote_key = f"debug/screenshots/{timestamp}_{name}.png"
            storage.upload_file(local_path, remote_key, content_type="image/png")
            url = storage.get_presigned_url(remote_key, expires_in=86400)
            print(f"[Flow] 📸 Screenshot uploaded: {url}", flush=True)
            return url
        except Exception as e:
            print(f"[Flow] Failed to upload screenshot: {e}", flush=True)
            return local_path
    
    def _check_login_required(self) -> bool:
        """Check if Google login is required"""
        current_url = self._page.url.lower()
        login_indicators = ["accounts.google.com", "identifier", "signin"]
        is_login_page = any(indicator in current_url for indicator in login_indicators)
        
        if not is_login_page:
            try:
                sign_in_text = self._page.locator("text=Sign in").count() > 0
                email_input = self._page.locator("input[type='email']").count() > 0
                is_login_page = sign_in_text and email_input
            except Exception:
                pass
        
        return is_login_page
    
    def _wait_for_login(self, timeout: int = 300) -> bool:
        """Wait for login to complete (for interactive mode)."""
        if not self._check_login_required():
            return True
        
        print("[Flow] Login required - waiting for manual login...", flush=True)
        self._needs_auth = True
        
        if self.headless:
            print("[Flow] ERROR: Login required but running headless. Export auth state first.", flush=True)
            return False
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(2)
            if not self._check_login_required():
                print("[Flow] Login completed!", flush=True)
                self._needs_auth = False
                return True
        
        print("[Flow] Login timeout", flush=True)
        return False
    
    def export_auth_state(self, output_path: str = None, upload_to_s3: bool = False) -> Optional[str]:
        """Export current browser authentication state."""
        if not self._context:
            raise RuntimeError("Browser not started. Call start() first.")
        
        self._page.goto(FLOW_HOME_URL, timeout=60000, wait_until="load")
        time.sleep(5)
        
        if not self._wait_for_login(timeout=300):
            print("[Flow] Failed to complete login for auth export", flush=True)
            return None
        
        storage_state = self._context.storage_state()
        output_path = output_path or "flow_storage_state.json"
        with open(output_path, 'w') as f:
            json.dump(storage_state, f, indent=2)
        
        print(f"[Flow] Auth state exported to: {output_path}", flush=True)
        
        if upload_to_s3:
            try:
                from .storage import get_storage
                storage = get_storage()
                storage.upload_flow_auth_state(storage_state)
                print("[Flow] Auth state uploaded to object storage", flush=True)
            except Exception as e:
                print(f"[Flow] Failed to upload auth state: {e}", flush=True)
        
        return output_path
    
    def create_new_project(self) -> str:
        """Create a new Flow project."""
        print("[Flow] Creating new project...", flush=True)
        
        # Navigate with longer timeout for proxy
        try:
            print("[Flow] Navigating to Flow homepage...", flush=True)
            self._page.goto(FLOW_HOME_URL, timeout=90000, wait_until="load")
            print("[Flow] ✓ Page loaded successfully", flush=True)
        except Exception as e:
            print(f"[Flow] ✗ Navigation failed: {e}", flush=True)
            self._screenshot("navigation_failed")
            raise
        
        print("[Flow] Waiting for JS initialization...", flush=True)
        time.sleep(5)
        
        # Take screenshot after navigation
        self._screenshot("after_navigation")
        
        # Check for errors
        self._check_for_errors("after_navigation")
        
        # Dismiss any popups
        self._check_and_dismiss_popup()
        
        # Check if we're on the PUBLIC landing page (not logged in)
        # This has "Create with Flow" button instead of "New project"
        create_with_flow_btn = self._page.locator("button:has-text('Create with Flow'), a:has-text('Create with Flow')")
        if create_with_flow_btn.count() > 0:
            print("[Flow] Detected public landing page - clicking 'Create with Flow'...", flush=True)
            try:
                create_with_flow_btn.first.click(force=True)
                time.sleep(5)
                self._screenshot("after_create_with_flow")
            except Exception as e:
                print(f"[Flow] Failed to click 'Create with Flow': {e}", flush=True)
        
        # Now check if login is required
        if self._check_login_required():
            print("[Flow] Login required - session may have expired", flush=True)
            self._screenshot("login_required")
            if not self._wait_for_login():
                raise RuntimeError("Login required but could not complete. Please update the flow_session on R2.")
        
        self._check_and_dismiss_popup()
        
        # Now we should be on the dashboard - look for "New project" button
        # It might also be visible as just "New project" text or icon
        new_project_selectors = [
            "button:has-text('New project')",
            "text=New project",
            "[aria-label='New project']",
            "button:has-text('New')",
        ]
        
        clicked = False
        for selector in new_project_selectors:
            try:
                btn = self._page.locator(selector)
                if btn.count() > 0 and btn.first.is_visible():
                    print(f"[Flow] Found New project button with selector: {selector}", flush=True)
                    with self._page.expect_navigation(wait_until="load", timeout=60000):
                        btn.first.click(force=True)
                    print("[Flow] Clicked New project button", flush=True)
                    clicked = True
                    break
            except Exception as e:
                print(f"[Flow] Selector {selector} failed: {e}", flush=True)
                continue
        
        if not clicked:
            print("[Flow] WARNING: Could not find 'New project' button!", flush=True)
            self._screenshot("no_new_project_button")
            # Try to continue anyway - maybe we're already in a project
        
        time.sleep(5)
        
        # Wait for the prompt input area
        print("[Flow] Waiting for project UI to be ready...", flush=True)
        try:
            self._page.wait_for_selector("#PINHOLE_TEXT_AREA_ELEMENT_ID", timeout=30000)
            print("[Flow] Prompt textarea found - page is ready", flush=True)
        except Exception as e:
            print(f"[Flow] Warning: Prompt textarea not found: {e}", flush=True)
            self._screenshot("project_not_ready")
        
        time.sleep(2)
        
        project_url = self._page.url
        print(f"[Flow] Created project: {project_url}", flush=True)
        
        if "/project/" not in project_url:
            print("[Flow] Warning: Project URL may be invalid, waiting more...", flush=True)
            time.sleep(5)
            project_url = self._page.url
        
        self._screenshot("project_created")
        
        return project_url
    
    def _upload_frame_with_button(
        self, 
        image_path: str, 
        button_selector: str, 
        is_first: bool = True,
        frame_name: str = "frame"
    ):
        """Click a button and upload a frame."""
        print(f"[Flow] Uploading {frame_name}: {image_path}", flush=True)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"[Flow] Image file exists, size: {os.path.getsize(image_path)} bytes", flush=True)
        
        self._check_and_dismiss_popup()
        
        if is_first:
            btn = self._page.locator(button_selector).first
        else:
            btn = self._page.locator(button_selector).last
        
        if btn.count() == 0:
            print(f"[Flow] WARNING: Button not found with selector: {button_selector}", flush=True)
            self._screenshot(f"button_not_found_{frame_name.replace(' ', '_')}")
            raise RuntimeError(f"Add frame button not found")
        
        file_chooser = None
        
        try:
            print(f"[Flow] Clicking {frame_name} button (expecting file chooser)...", flush=True)
            
            with self._page.expect_file_chooser(timeout=5000) as fc_info:
                btn.click(force=True)
                print(f"[Flow] Clicked {frame_name} button", flush=True)
            
            file_chooser = fc_info.value
            print(f"[Flow] File chooser opened directly from button click", flush=True)
            
        except Exception as e:
            print(f"[Flow] No direct file chooser (this is normal): {e}", flush=True)
            print(f"[Flow] Looking for Upload button in modal...", flush=True)
            
            time.sleep(2)
            self._check_and_dismiss_popup()
            
            upload_btn = None
            for selector in ["text=Upload", "button:has-text('Upload')", "[aria-label='Upload']", "text=Choose file"]:
                try:
                    candidate = self._page.locator(selector).first
                    if candidate.count() > 0 and candidate.is_visible():
                        upload_btn = candidate
                        print(f"[Flow] Found upload button with selector: {selector}", flush=True)
                        break
                except Exception:
                    pass
            
            if upload_btn:
                try:
                    with self._page.expect_file_chooser(timeout=10000) as fc_info:
                        upload_btn.click(force=True)
                        print(f"[Flow] Clicked Upload button", flush=True)
                    file_chooser = fc_info.value
                except Exception as e2:
                    print(f"[Flow] Failed to get file chooser from Upload button: {e2}", flush=True)
                    self._screenshot(f"upload_failed_{frame_name.replace(' ', '_')}")
                    raise
            else:
                print(f"[Flow] No Upload button found", flush=True)
                self._screenshot(f"no_upload_btn_{frame_name.replace(' ', '_')}")
                raise RuntimeError("Could not find way to upload file")
        
        if file_chooser:
            file_chooser.set_files(image_path)
            print(f"[Flow] File selected: {os.path.basename(image_path)}", flush=True)
        else:
            raise RuntimeError("No file chooser available")
        
        time.sleep(3)
        self._check_and_dismiss_popup()
        
        print(f"[Flow] Waiting for crop dialog...", flush=True)
        try:
            self._page.wait_for_selector("text=Crop and Save", timeout=15000)
            print(f"[Flow] Crop dialog opened for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Crop dialog not found (may not be needed): {e}", flush=True)
            time.sleep(2)
            return
        
        time.sleep(1)
        self._check_and_dismiss_popup()
        
        print(f"[Flow] Selecting orientation...", flush=True)
        try:
            orientation_selectors = [
                "div.sc-19de2353-4.boKhUT button.sc-a84519cc-0.fsaXDA",
                "button:has-text('Landscape')",
                "button:has-text('Portrait')",
                "[aria-label='Aspect ratio']"
            ]
            
            orientation_btn = None
            for selector in orientation_selectors:
                try:
                    candidate = self._page.locator(selector).first
                    if candidate.count() > 0 and candidate.is_visible():
                        orientation_btn = candidate
                        break
                except Exception:
                    pass
            
            if orientation_btn:
                for attempt in range(3):
                    try:
                        orientation_btn.click()
                        time.sleep(0.5)
                        
                        portrait_opt = self._page.locator("text=Portrait").first
                        if portrait_opt.count() > 0 and portrait_opt.is_visible():
                            portrait_opt.click(force=True)
                            print(f"[Flow] Selected Portrait for {frame_name}", flush=True)
                            break
                    except Exception:
                        pass
                    time.sleep(0.5)
        except Exception as e:
            print(f"[Flow] Could not set orientation (continuing anyway): {e}", flush=True)
        
        time.sleep(1)
        
        try:
            self._page.click("text=Crop and Save")
            print(f"[Flow] Clicked Crop and Save for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Could not click Crop and Save: {e}", flush=True)
            try:
                self._page.click("text=Save")
                print(f"[Flow] Clicked Save instead", flush=True)
            except Exception:
                pass
        
        time.sleep(2)
        print(f"[Flow] {frame_name} upload complete", flush=True)

    def _upload_frame(self, image_path: str, frame_name: str = "frame"):
        """Upload a frame image"""
        print(f"[Flow] Starting upload for {frame_name}: {image_path}", flush=True)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"[Flow] Image file exists, size: {os.path.getsize(image_path)} bytes", flush=True)
        
        self._check_and_dismiss_popup()
        
        try:
            print("[Flow] Waiting for file chooser...", flush=True)
            with self._page.expect_file_chooser(timeout=10000) as fc_info:
                upload_clicked = False
                for selector in ["text=Upload", "button:has-text('Upload')", "[aria-label='Upload']"]:
                    try:
                        btn = self._page.locator(selector).first
                        if btn.count() > 0 and btn.is_visible():
                            btn.click(force=True)
                            upload_clicked = True
                            print(f"[Flow] Clicked upload button with selector: {selector}", flush=True)
                            break
                    except Exception as e:
                        print(f"[Flow] Selector {selector} failed: {e}", flush=True)
                
                if not upload_clicked:
                    raise RuntimeError("Could not find upload button")
            
            file_chooser = fc_info.value
            file_chooser.set_files(image_path)
            print(f"[Flow] Uploaded image for {frame_name}", flush=True)
            
        except Exception as e:
            print(f"[Flow] Error during file upload: {e}", flush=True)
            self._screenshot(f"upload_error_{frame_name.replace(' ', '_')}")
            raise
        
        time.sleep(3)
        self._check_and_dismiss_popup()
        
        print("[Flow] Waiting for crop dialog...", flush=True)
        try:
            self._page.wait_for_selector("text=Crop and Save", timeout=15000)
            print(f"[Flow] Crop dialog opened for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Crop dialog not found: {e}", flush=True)
            self._screenshot(f"crop_error_{frame_name.replace(' ', '_')}")
            raise
        
        time.sleep(1)
        self._check_and_dismiss_popup()
        
        print("[Flow] Selecting Portrait orientation...", flush=True)
        landscape_btn = self._page.locator("div.sc-19de2353-4.boKhUT button.sc-a84519cc-0.fsaXDA")
        
        for attempt in range(5):
            try:
                landscape_btn.focus()
                time.sleep(0.3)
                self._page.keyboard.press("Space")
                time.sleep(0.5)
                
                if self._page.locator("[role='option']:has-text('Portrait')").is_visible():
                    print("[Flow] Dropdown opened", flush=True)
                    break
            except Exception as e:
                print(f"[Flow] Attempt {attempt + 1} to open dropdown failed: {e}", flush=True)
        
        time.sleep(1)
        
        try:
            self._page.locator("text=Portrait").first.click(force=True)
            print(f"[Flow] Selected Portrait for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Could not select Portrait: {e}", flush=True)
        
        time.sleep(1)
        
        try:
            self._page.click("text=Crop and Save")
            print(f"[Flow] Clicked Crop and Save for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Could not click Crop and Save: {e}", flush=True)
            raise
        
        time.sleep(2)
    
    def _submit_clip(
        self,
        clip: FlowClip,
        is_first_clip: bool,
        has_new_frames: bool,
        language: str = "English"
    ) -> bool:
        """Submit a single clip for generation."""
        if self._cancelled:
            return False
        
        # Use pre-built prompt from API engine if available
        if clip.prompt:
            prompt = clean_prompt_for_flow(
                clip.prompt, 
                clip.dialogue_text, 
                language,
                voice_profile=clip.voice_profile,
                duration=clip.duration
            )
            print(f"[Flow] Cleaned API prompt for Flow ({len(prompt)} chars)", flush=True)
            print(f"[Flow] Prompt preview: {prompt[:150]}...", flush=True)
        else:
            prompt = get_prompt(
                clip.dialogue_text, 
                language,
                voice_profile=clip.voice_profile,
                duration=clip.duration
            )
            print(f"[Flow] Using fallback prompt ({len(prompt)} chars)", flush=True)
        
        try:
            if is_first_clip:
                print(f"[Flow] Setting up first clip with frames...", flush=True)
                self._screenshot("before_mode_select")
                
                # Select Frames to Video mode
                print("[Flow] Selecting 'Frames to Video' mode...", flush=True)
                
                mode_changed = False
                for attempt in range(3):
                    print(f"[Flow] Mode selection attempt {attempt + 1}/3", flush=True)
                    
                    mode_button = None
                    for selector in [
                        "text=Text to Video",
                        "button:has-text('Text to Video')",
                        "[aria-haspopup='listbox']",
                        "div:has-text('Text to Video') >> button",
                    ]:
                        try:
                            btn = self._page.locator(selector).first
                            if btn.count() > 0 and btn.is_visible():
                                mode_button = btn
                                print(f"[Flow] Found mode dropdown with selector: {selector}", flush=True)
                                break
                        except Exception:
                            pass
                    
                    if not mode_button:
                        print("[Flow] Mode dropdown not found, trying to locate any dropdown...", flush=True)
                        self._screenshot(f"no_dropdown_attempt_{attempt}")
                        time.sleep(2)
                        continue
                    
                    mode_button.click()
                    print("[Flow] Clicked mode dropdown", flush=True)
                    time.sleep(1.5)
                    
                    try:
                        self._page.wait_for_selector("text=Frames to Video", timeout=5000)
                        print("[Flow] Dropdown options visible", flush=True)
                    except Exception:
                        print("[Flow] Dropdown options not visible, retrying...", flush=True)
                        self._page.keyboard.press("Escape")
                        time.sleep(1)
                        continue
                    
                    frames_option = self._page.locator("text=Frames to Video").first
                    if frames_option.count() > 0:
                        try:
                            frames_option.click()
                            print("[Flow] Clicked 'Frames to Video' option", flush=True)
                            time.sleep(2)
                        except Exception as e:
                            print(f"[Flow] Click failed: {e}, trying JavaScript click...", flush=True)
                            try:
                                frames_option.evaluate("el => el.click()")
                            except Exception:
                                pass
                            time.sleep(2)
                    
                    time.sleep(1)
                    frame_buttons = self._page.locator("button.sc-d02e9a37-1.hvUQuN")
                    if frame_buttons.count() > 0:
                        print(f"[Flow] SUCCESS: Mode changed! Found {frame_buttons.count()} frame button(s)", flush=True)
                        mode_changed = True
                        break
                    else:
                        print("[Flow] Frame buttons not found, mode may not have changed", flush=True)
                        self._screenshot(f"mode_attempt_{attempt}")
                
                if not mode_changed:
                    print("[Flow] ERROR: Failed to switch to 'Frames to Video' mode after 3 attempts!", flush=True)
                    self._screenshot("mode_failed")
                    raise RuntimeError("Could not switch to Frames to Video mode")
                
                self._check_and_dismiss_popup()
                
                # Upload START frame
                if clip.start_frame_path:
                    self._upload_frame_with_button(
                        clip.start_frame_path, 
                        "button.sc-d02e9a37-1.hvUQuN", 
                        is_first=True,
                        frame_name="START frame"
                    )
                
                # Upload END frame
                if clip.end_frame_path:
                    self._check_and_dismiss_popup()
                    self._upload_frame_with_button(
                        clip.end_frame_path,
                        "button.sc-d02e9a37-1.hvUQuN",
                        is_first=False,
                        frame_name="END frame"
                    )
                
                # Enter prompt
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                textarea.click()
                time.sleep(0.5)
                textarea.fill(prompt)
                print(f"[Flow] Entered prompt: {clip.dialogue_text[:50]}...", flush=True)
                time.sleep(10)
                
            elif has_new_frames:
                print(f"[Flow] Clip {clip.clip_index + 1}: Uploading new frames...", flush=True)
                
                if clip.start_frame_path:
                    self._check_and_dismiss_popup()
                    self._upload_frame_with_button(
                        clip.start_frame_path,
                        "button.sc-d02e9a37-1.hvUQuN",
                        is_first=True,
                        frame_name="START frame"
                    )
                
                if clip.end_frame_path:
                    self._check_and_dismiss_popup()
                    self._upload_frame_with_button(
                        clip.end_frame_path,
                        "button.sc-d02e9a37-1.hvUQuN",
                        is_first=False,
                        frame_name="END frame"
                    )
                
                print(f"[Flow] Entering prompt ({len(prompt)} chars)...", flush=True)
                
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                
                try:
                    textarea.wait_for(state="visible", timeout=5000)
                except Exception as e:
                    print(f"[Flow] Textarea not immediately visible: {e}", flush=True)
                    self._screenshot("no_textarea")
                
                textarea.click()
                time.sleep(0.5)
                textarea.fill("")
                time.sleep(0.3)
                textarea.fill(prompt)
                print(f"[Flow] Filled prompt into textarea", flush=True)
                time.sleep(1)
                
                try:
                    textarea.evaluate("el => el.dispatchEvent(new Event('input', { bubbles: true }))")
                    textarea.evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
                except Exception:
                    pass
                
                time.sleep(1)
                entered_text = textarea.input_value()
                if entered_text and len(entered_text) > 50:
                    print(f"[Flow] ✓ Prompt verified ({len(entered_text)} chars in textarea)", flush=True)
                else:
                    print(f"[Flow] ⚠ Prompt may not have been entered. Trying press_sequentially...", flush=True)
                    textarea.fill("")
                    textarea.press_sequentially(prompt[:200], delay=10)
                    time.sleep(0.5)
                    textarea.fill(prompt)
                    time.sleep(1)
                
                self._screenshot("prompt_entered")
                time.sleep(2)
                
            else:
                print(f"[Flow] Clip {clip.clip_index + 1}: Reusing frames...", flush=True)
                
                self._page.click("i:text('wrap_text')", force=True)
                print("[Flow] Clicked Reuse prompt", flush=True)
                time.sleep(2)
                
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                textarea.click()
                time.sleep(0.5)
                textarea.fill("")
                time.sleep(0.3)
                textarea.fill(prompt)
                print(f"[Flow] Entered prompt: {clip.dialogue_text[:50]}...", flush=True)
                time.sleep(1)
                
                self._page.click("i:text('arrow_forward')", force=True)
                print(f"[Flow] Clip {clip.clip_index + 1}: Generation started (reuse)", flush=True)
                time.sleep(3)
                
                clip.status = "generating"
                return True
            
            # Verify prompt was entered
            print("[Flow] Verifying prompt was entered...", flush=True)
            time.sleep(1)
            
            try:
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                current_text = textarea.input_value()
                if current_text and len(current_text) > 10:
                    print(f"[Flow] ✓ Prompt verified in textarea ({len(current_text)} chars)", flush=True)
                else:
                    print(f"[Flow] ⚠ Textarea seems empty or short", flush=True)
                    textarea.click()
                    time.sleep(0.5)
                    textarea.fill(prompt)
                    print("[Flow] Re-entered prompt", flush=True)
                    time.sleep(2)
            except Exception as e:
                print(f"[Flow] Could not verify prompt: {e}", flush=True)
            
            self._screenshot("before_generate")
            self._human_delay(1000, 2000)
            
            # Click Generate button
            print("[Flow] Looking for Generate button (arrow icon)...", flush=True)
            
            generate_clicked = False
            
            # Method 1: Click arrow_forward icon
            try:
                arrow_icon = self._page.locator("i:text('arrow_forward')").first
                if arrow_icon.count() > 0 and arrow_icon.is_visible():
                    print("[Flow] Found arrow_forward icon, human-clicking...", flush=True)
                    self._scroll_into_view(arrow_icon)
                    self._human_click(arrow_icon, "arrow_forward icon")
                    generate_clicked = True
                    print("[Flow] ✓ Human-clicked arrow_forward icon", flush=True)
            except Exception as e:
                print(f"[Flow] arrow_forward icon click failed: {e}", flush=True)
            
            # Method 2: Button containing arrow icon
            if not generate_clicked:
                try:
                    arrow_btn = self._page.locator("button:has(i:text('arrow_forward'))").first
                    if arrow_btn.count() > 0:
                        print("[Flow] Found button with arrow icon, human-clicking...", flush=True)
                        self._human_click(arrow_btn, "button with arrow icon")
                        generate_clicked = True
                except Exception as e:
                    print(f"[Flow] Button with arrow icon failed: {e}", flush=True)
            
            # Method 3: CSS selector
            if not generate_clicked:
                try:
                    btn = self._page.locator("div.sc-408537d4-1.eiHkev > button").first
                    if btn.count() > 0 and btn.is_visible():
                        self._human_click(btn, "CSS selector button")
                        generate_clicked = True
                except Exception as e:
                    print(f"[Flow] CSS selector failed: {e}", flush=True)
            
            # Method 4: Keyboard shortcut
            if not generate_clicked:
                try:
                    print("[Flow] Trying keyboard shortcut...", flush=True)
                    textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                    textarea.focus()
                    time.sleep(0.3)
                    self._page.keyboard.press("Control+Enter")
                    time.sleep(1)
                    self._page.keyboard.press("Tab")
                    time.sleep(0.3)
                    self._page.keyboard.press("Enter")
                    generate_clicked = True
                except Exception as e:
                    print(f"[Flow] Keyboard shortcut failed: {e}", flush=True)
            
            if not generate_clicked:
                print("[Flow] ERROR: All Generate button methods failed!", flush=True)
                self._screenshot("generate_failed")
                raise RuntimeError("Could not click Generate button")
            
            # Wait and check for generation
            print("[Flow] Waiting for generation to start...", flush=True)
            time.sleep(5)
            
            self._screenshot("after_generate")
            
            # Check for errors
            error = self._check_for_errors("after_generate_click")
            if error:
                print(f"[Flow] ⚠ Error after generate click: {error}", flush=True)
            
            # Look for generation indicators
            generation_started = False
            for indicator in ["text=Generating", "text=Queued", "text=Processing", "text=in progress"]:
                try:
                    if self._page.locator(indicator).count() > 0:
                        print(f"[Flow] ✓ Found generation indicator: {indicator}", flush=True)
                        generation_started = True
                        break
                except Exception:
                    pass
            
            try:
                video_count = self._page.locator("video").count()
                if video_count > 0:
                    print(f"[Flow] ✓ Found {video_count} video element(s)", flush=True)
                    generation_started = True
            except Exception:
                pass
            
            if not generation_started:
                print("[Flow] ⚠ Could not verify generation started - check screenshots", flush=True)
            
            print(f"[Flow] Clip {clip.clip_index + 1}: Generation started", flush=True)
            time.sleep(5)
            
            clip.status = "generating"
            return True
            
        except Exception as e:
            print(f"[Flow] Error submitting clip {clip.clip_index + 1}: {e}", flush=True)
            self._screenshot(f"clip_{clip.clip_index}_error")
            clip.status = "failed"
            clip.error_message = str(e)
            return False
    
    def _monitor_generation(self, timeout_seconds: int = 120, check_interval: int = 10) -> bool:
        """
        Monitor generation progress with periodic screenshots and error checks.
        
        Args:
            timeout_seconds: Total time to monitor
            check_interval: Seconds between checks
            
        Returns:
            True if generation appears successful, False if errors detected
        """
        print(f"[Flow] Monitoring generation for {timeout_seconds}s...", flush=True)
        
        start_time = time.time()
        check_count = 0
        last_progress = ""
        
        while time.time() - start_time < timeout_seconds:
            check_count += 1
            elapsed = int(time.time() - start_time)
            
            # Check for errors
            error = self._check_for_errors(f"monitor_{elapsed}s")
            if error:
                print(f"[Flow] ❌ Error detected during generation at {elapsed}s: {error}", flush=True)
                return False
            
            # Check progress indicators
            progress_found = False
            progress_text = ""
            
            # Look for percentage indicators
            try:
                progress_el = self._page.locator("text=/\\d+%/").first
                if progress_el.count() > 0:
                    progress_text = progress_el.text_content()
                    progress_found = True
            except Exception:
                pass
            
            # Look for "Generating" text
            try:
                if self._page.locator("text=Generating").count() > 0:
                    progress_found = True
                    progress_text = progress_text or "Generating..."
            except Exception:
                pass
            
            # Look for completed videos
            try:
                video_count = self._page.locator("video").count()
                if video_count > 0:
                    progress_text = f"{video_count} video(s)"
                    progress_found = True
            except Exception:
                pass
            
            # Log progress changes
            if progress_text and progress_text != last_progress:
                print(f"[Flow] 📊 Progress at {elapsed}s: {progress_text}", flush=True)
                last_progress = progress_text
            
            # Periodic screenshot (every 30 seconds)
            if check_count % 3 == 0:
                self._screenshot(f"progress_{elapsed}s")
            
            time.sleep(check_interval)
        
        print(f"[Flow] ✓ Generation monitoring complete ({timeout_seconds}s)", flush=True)
        self._screenshot("generation_complete")
        return True
    
    def _download_clip(
        self,
        clip: FlowClip,
        project_url: str,
        line_mapping: Dict[str, int]
    ) -> bool:
        """Download a generated clip."""
        try:
            print(f"[Flow] Navigating to project for download: {project_url}", flush=True)
            self._page.goto(project_url, timeout=60000, wait_until="load")
            time.sleep(5)
            
            if self._check_login_required():
                if not self._wait_for_login():
                    raise RuntimeError("Login required for download")
            
            self._check_and_dismiss_popup()
            time.sleep(2)
            
            # Check for errors before looking for videos
            self._check_for_errors("before_download")
            
            print("[Flow] Waiting for clips to load...", flush=True)
            video_found = False
            
            for attempt in range(60):
                video_count = self._page.locator("video").count()
                generating = self._page.locator("text=Generating").count()
                queued = self._page.locator("text=Queued").count()
                
                if video_count > 0:
                    print(f"[Flow] Found {video_count} video element(s)", flush=True)
                    video_found = True
                    break
                
                if attempt % 10 == 0:
                    print(f"[Flow] Still waiting... videos={video_count}, generating={generating}, queued={queued}", flush=True)
                    
                    if attempt == 30:
                        self._screenshot(f"debug_clip_{clip.clip_index}")
                
                time.sleep(1)
            
            if not video_found:
                print(f"[Flow] No video elements found after 60s wait", flush=True)
                self._screenshot("no_videos_found")
                clip.status = "generating"
                clip.error_message = "Video still generating - check project URL manually"
                return False
            
            downloaded = False
            
            container = self._page.locator("div[data-index='0']")
            if container.count() > 0:
                print("[Flow] Found data-index container", flush=True)
                
                video = container.locator("video").first
                if video.count() > 0:
                    try:
                        video.scroll_into_view_if_needed()
                        time.sleep(1)
                        video.hover(force=True)
                        time.sleep(1)
                        
                        src = video.get_attribute("src")
                        video_id = get_video_id(src) or f"clip_{clip.clip_index}"
                        clip.flow_clip_id = video_id
                        print(f"[Flow] Video ID: {video_id}", flush=True)
                        
                        download_btn = None
                        for selector in [
                            "i:text('download')",
                            "[aria-label='Download']",
                            "button:has-text('download')",
                            ".download-button",
                        ]:
                            btn = container.locator(selector).first
                            if btn.count() > 0:
                                download_btn = btn
                                break
                        
                        if download_btn:
                            download_btn.click(force=True)
                            time.sleep(2)
                            
                            for option_text in ["Original size (720p)", "720p", "Download", "Original"]:
                                option = self._page.locator(f"text={option_text}").first
                                if option.count() > 0 and option.is_visible():
                                    try:
                                        with self._page.expect_download(timeout=30000) as download_info:
                                            option.click()
                                        
                                        download = download_info.value
                                        save_path = os.path.join(
                                            self.download_dir,
                                            f"clip_{clip.clip_index + 1}_{video_id}.mp4"
                                        )
                                        download.save_as(save_path)
                                        
                                        print(f"[Flow] Downloaded: {save_path}", flush=True)
                                        clip.status = "completed"
                                        clip.output_url = save_path
                                        downloaded = True
                                        break
                                    except Exception as e:
                                        print(f"[Flow] Download attempt failed: {e}", flush=True)
                    except Exception as e:
                        print(f"[Flow] Error interacting with video: {e}", flush=True)
            
            if not downloaded:
                print("[Flow] Trying alternative video detection...", flush=True)
                all_videos = self._page.locator("video")
                if all_videos.count() > 0:
                    print(f"[Flow] Found {all_videos.count()} videos on page", flush=True)
                    clip.status = "generating"
                    clip.error_message = "Video generated but download failed - check project URL"
                    return False
            
            return downloaded
            
        except Exception as e:
            print(f"[Flow] Error downloading clip {clip.clip_index + 1}: {e}", flush=True)
            self._screenshot(f"download_error_clip_{clip.clip_index}")
            clip.error_message = str(e)
            clip.status = "generating"
            return False
    
    def process_job(
        self,
        job: FlowJob,
        language: str = "English",
        wait_for_generation: bool = True
    ) -> bool:
        """Process a complete job (submit all clips)."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        print(f"[Flow] Processing job {job.job_id} with {len(job.clips)} clips", flush=True)
        
        try:
            if job.project_url and "/project/" in job.project_url:
                print(f"[Flow] Resuming project: {job.project_url}", flush=True)
                self._page.goto(job.project_url, timeout=60000, wait_until="load")
                time.sleep(5)
                
                if self._check_login_required():
                    if not self._wait_for_login():
                        raise RuntimeError("Login required")
            else:
                job.project_url = self.create_new_project()
            
            # Process each clip
            for i, clip in enumerate(job.clips):
                if self._cancelled:
                    print("[Flow] Job cancelled", flush=True)
                    return False
                
                if clip.status in ("completed", "generating"):
                    print(f"[Flow] Skipping clip {i + 1} (status: {clip.status})", flush=True)
                    continue
                
                is_first = (i == 0)
                has_frames = bool(clip.start_frame_path or clip.end_frame_path)
                
                success = self._submit_clip(
                    clip,
                    is_first_clip=is_first,
                    has_new_frames=has_frames,
                    language=language
                )
                
                if success and job.on_progress:
                    job.on_progress(i, "generating", f"Submitted clip {i + 1}")
            
            print(f"[Flow] All clips submitted for job {job.job_id}", flush=True)
            print(f"[Flow] Project URL: {job.project_url}", flush=True)
            
            if wait_for_generation:
                # Monitor generation with periodic screenshots
                print(f"[Flow] Waiting {DEFAULT_WAIT_AFTER_SUBMIT}s for generation...", flush=True)
                self._monitor_generation(timeout_seconds=DEFAULT_WAIT_AFTER_SUBMIT, check_interval=10)
                
                line_mapping = {}
                for clip in job.clips:
                    dialogue = clip.dialogue_text.strip().strip('"').strip("'")
                    line_mapping[dialogue] = clip.clip_index + 1
                
                for clip in job.clips:
                    if clip.status == "generating":
                        self._download_clip(clip, job.project_url, line_mapping)
            
            return True
            
        except Exception as e:
            print(f"[Flow] Error processing job: {e}", flush=True)
            self._screenshot("job_error")
            if job.on_error:
                job.on_error(str(e))
            return False
    
    @property
    def needs_auth(self) -> bool:
        """Whether authentication is needed"""
        return self._needs_auth


# === Helper functions for integration ===

def create_flow_job_from_db(
    job_id: str,
    clips_data: List[dict],
    project_url: Optional[str] = None
) -> FlowJob:
    """Create a FlowJob from database data."""
    clips = []
    for i, clip_data in enumerate(clips_data):
        clips.append(FlowClip(
            clip_index=i,
            dialogue_text=clip_data.get("dialogue_text", ""),
            start_frame_path=clip_data.get("start_frame"),
            end_frame_path=clip_data.get("end_frame"),
            prompt=clip_data.get("prompt"),
        ))
    
    return FlowJob(
        job_id=job_id,
        clips=clips,
        project_url=project_url
    )


def export_auth_state_command():
    """CLI command to export auth state interactively."""
    print("=" * 50)
    print("FLOW AUTH STATE EXPORT")
    print("=" * 50)
    print("\nThis will open a browser window.")
    print("Please log in to your Google account when prompted.")
    print("The auth state will be saved for headless operation.\n")
    
    with FlowBackend(headless=False) as flow:
        output_path = flow.export_auth_state(
            output_path="flow_storage_state.json",
            upload_to_s3=True
        )
        
        if output_path:
            print(f"\n✓ Auth state exported successfully!")
            print(f"  Local: {output_path}")
            print("\nYou can now run the Flow worker headlessly.")
        else:
            print("\n✗ Failed to export auth state")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--export-auth":
        export_auth_state_command()
    else:
        print("Usage: python -m backends.flow_backend --export-auth")