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


def get_prompt(dialogue: str, language: str = "English") -> str:
    """
    Generate the video generation prompt from dialogue text.
    
    Args:
        dialogue: The dialogue line to speak
        language: Language for pronunciation
        
    Returns:
        Formatted prompt string
    """
    dialogue = dialogue.strip().strip('"').strip("'")
    
    return f"""Medium shot, static locked-off camera, sharp focus on subject. The subject in the frame speaks directly to camera with steady gaze with slight smile, upright posture, shoulders back. The character says in {language}, "{dialogue}" Voice: smooth - slight crispness on emphasis. clear and authoritative, confident emotion. Ambient noise: Complete silence, professional recording booth, no room ambiance. Style: Raw realistic footage, natural lighting, photorealistic. Speech timing: 0s to 7.0s, then silence. No subtitles, no text overlays, no captions, no watermarks. No background music, no laughter, no applause, no crowd sounds, no ambient noise. No morphing, no face distortion, no jerky movements. Only the speaker's isolated voice. (no subtitles)"""


def clean_prompt_for_flow(prompt: str, dialogue: str, language: str = "English") -> str:
    """
    Clean an API-generated prompt for use with Flow UI.
    
    The API prompt includes markers like === VOICE PROFILE === that confuse Flow.
    This function extracts the essential content and creates a Flow-compatible prompt.
    
    Args:
        prompt: The API-generated prompt (may have markers)
        dialogue: The dialogue text (fallback if prompt is too broken)
        language: Language for the dialogue
        
    Returns:
        Clean prompt suitable for Flow
    """
    if not prompt:
        return get_prompt(dialogue, language)
    
    # If prompt doesn't have our markers or voice profile hints, return as-is
    has_markers = "===" in prompt or "---" in prompt
    has_voice_hints = any(x in prompt.lower() for x in ['pacing:', 'accent:', 'gender:', 'pitch:'])
    
    if not has_markers and not has_voice_hints:
        return prompt
    
    # Extract the useful part - everything after the voice profile section
    # The format is:
    # === VOICE PROFILE ===
    # ...voice profile data...
    # ===
    # 
    # Medium shot, static locked-off camera...
    
    lines = prompt.split('\n')
    clean_lines = []
    skip_until_content = False
    found_content = False
    
    # Lines to skip (voice/style metadata that shouldn't be in Flow prompt)
    skip_patterns = [
        'gender:', 'pitch:', 'timbre:', 'texture:', 'tone:', 'age:',
        'pacing:', 'accent:', 'speaking style:', 'delivery:'
    ]
    
    for line in lines:
        stripped = line.strip()
        
        # Skip voice profile markers and content
        if stripped.startswith('===') or stripped.startswith('---'):
            skip_until_content = True
            continue
        
        # Skip empty lines at the start
        if not found_content and not stripped:
            continue
            
        # Once we hit real content, start collecting
        if stripped and not stripped.startswith('===') and not stripped.startswith('---'):
            # Skip lines that look like voice profile data
            if any(x in stripped.lower() for x in skip_patterns):
                continue
            
            found_content = True
            clean_lines.append(line)
    
    if clean_lines:
        clean_prompt = '\n'.join(clean_lines).strip()
        # Make sure we have something useful
        if len(clean_prompt) > 50:
            return clean_prompt
    
    # Fallback: extract just the essential elements
    # Try to find the dialogue line and build a simple prompt
    import re
    
    # Try to extract the dialogue from the prompt
    dialogue_match = re.search(r'says?\s+in\s+\w+,\s*"([^"]+)"', prompt)
    if dialogue_match:
        extracted_dialogue = dialogue_match.group(1)
    else:
        extracted_dialogue = dialogue
    
    # Build a clean, simple prompt
    return f"""Medium shot, static locked-off camera, sharp focus on subject.

The subject in the frame speaks directly to camera with natural expression, relaxed posture.

The character says in {language}, "{extracted_dialogue}"

Voice: natural voice, clear and authentic.

Style: Raw realistic footage, natural lighting, photorealistic.

No subtitles, no text overlays. No background music. Only the speaker's voice.

(no subtitles)"""


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
        proxy_password: Optional[str] = None
    ):
        """
        Initialize Flow backend.
        
        Args:
            storage_state_path: Local path to Playwright storage state JSON
            storage_state_url: S3 URL/key to download storage state from
            headless: Whether to run browser headlessly
            download_dir: Directory for downloads
            temp_dir: Directory for temporary files
            proxy_server: Proxy server URL (e.g., "http://proxy.example.com:8080")
            proxy_username: Proxy username (optional)
            proxy_password: Proxy password (optional)
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
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
        self._needs_auth = False
        self._cancelled = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self):
        """Start the browser with stealth measures and optional proxy"""
        print("[Flow] Starting browser with stealth mode...", flush=True)
        
        # Log proxy status
        if self.proxy_server:
            masked_proxy = self.proxy_server
            if '@' in masked_proxy:
                # Mask password in URL
                masked_proxy = masked_proxy.split('@')[-1]
            print(f"[Flow] Using proxy: {masked_proxy}", flush=True)
        else:
            print("[Flow] No proxy configured (using direct connection)", flush=True)
        
        self._playwright = sync_playwright().start()
        
        # Comprehensive stealth browser args
        stealth_args = [
            # Basic stealth
            "--disable-blink-features=AutomationControlled",
            
            # Sandbox (required for containers)
            "--no-sandbox",
            "--disable-setuid-sandbox",
            
            # Performance
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--no-first-run",
            "--no-zygote",
            
            # Make it look like a real browser
            "--disable-infobars",
            "--disable-extensions",
            "--disable-plugins-discovery",
            "--disable-default-apps",
            
            # Prevent detection
            "--disable-component-update",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-translate",
            "--hide-scrollbars",
            "--mute-audio",
            
            # Window settings
            "--window-size=1920,1080",
            "--start-maximized",
            
            # GPU - enable for more realistic fingerprint
            "--enable-webgl",
            "--use-gl=swiftshader",
        ]
        
        # Add SSL bypass flags if using proxy
        if self.proxy_server:
            stealth_args.extend([
                "--ignore-certificate-errors",
                "--ignore-ssl-errors",
                "--ignore-certificate-errors-spki-list",
                "--allow-insecure-localhost",
            ])
            print("[Flow] Added SSL bypass flags for proxy", flush=True)
        
        # Build launch options
        launch_options = {
            "headless": self.headless,
            "args": stealth_args
        }
        
        # Add proxy if configured
        if self.proxy_server:
            # Try embedding credentials in URL (some proxies require this format)
            if self.proxy_username and self.proxy_password:
                # Format: http://user:pass@host:port
                server_parts = self.proxy_server.replace("http://", "").replace("https://", "")
                proxy_url = f"http://{self.proxy_username}:{self.proxy_password}@{server_parts}"
                proxy_config = {"server": proxy_url}
                print(f"[Flow] Proxy configured with embedded auth: {server_parts}", flush=True)
            else:
                proxy_config = {"server": self.proxy_server}
                print(f"[Flow] Proxy configured (no auth): {self.proxy_server}", flush=True)
            launch_options["proxy"] = proxy_config
        
        self._browser = self._playwright.chromium.launch(**launch_options)
        
        # Get storage state
        storage_state = self._get_storage_state()
        
        # Create context with realistic settings
        context_options = {
            "accept_downloads": True,
            "viewport": {"width": 1920, "height": 1080},
            "screen": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "color_scheme": "dark",
            "has_touch": False,
            "is_mobile": False,
            "device_scale_factor": 1,
            "java_script_enabled": True,
            "bypass_csp": False,
            "extra_http_headers": {
                "Accept-Language": "en-US,en;q=0.9",
            },
            "permissions": ["geolocation"],
        }
        
        # Ignore HTTPS errors when using proxy (required for Bright Data and similar proxies)
        if self.proxy_server:
            context_options["ignore_https_errors"] = True
            print("[Flow] SSL certificate validation disabled (proxy mode)", flush=True)
        
        if storage_state:
            context_options["storage_state"] = storage_state
        
        self._context = self._browser.new_context(**context_options)
        self._page = self._context.new_page()
        
        # Comprehensive anti-detection scripts
        self._page.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Add chrome runtime
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
            
            // Fix permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Fix plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                    { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                    { name: 'Native Client', filename: 'internal-nacl-plugin' }
                ]
            });
            
            // Fix languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            // Fix platform
            Object.defineProperty(navigator, 'platform', {
                get: () => 'Win32'
            });
            
            // Fix hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8
            });
            
            // Fix device memory
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8
            });
            
            // Override the toString method
            const oldCall = Function.prototype.call;
            function call() {
                return oldCall.apply(this, arguments);
            }
            Function.prototype.call = call;
            
            // Fix iframe detection
            const originalAttachShadow = Element.prototype.attachShadow;
            Element.prototype.attachShadow = function() {
                return originalAttachShadow.apply(this, arguments);
            };
            
            // Hide automation
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        """)
        
        print("[Flow] Browser started with stealth mode", flush=True)
        
        # Test proxy connection if configured
        if self.proxy_server:
            self._test_proxy_connection()
    
    def _test_proxy_connection(self):
        """Test if the proxy is working by loading a simple page"""
        print("[Flow] Testing proxy connection...", flush=True)
        try:
            # Try to load a simple, fast page to test proxy
            self._page.goto("https://httpbin.org/ip", timeout=30000, wait_until="load")
            
            # Get the response to see what IP we're using
            content = self._page.content()
            print(f"[Flow] Proxy test response: {content[:200]}", flush=True)
            
            # Check if we got a valid response
            if "origin" in content:
                print("[Flow] ✓ Proxy connection working!", flush=True)
            else:
                print("[Flow] ⚠ Proxy test: unexpected response", flush=True)
                
        except Exception as e:
            print(f"[Flow] ✗ Proxy test failed: {e}", flush=True)
            print("[Flow] Will try to continue anyway...", flush=True)
    
    def _human_delay(self, min_ms: int = 500, max_ms: int = 1500):
        """Add a random human-like delay"""
        import random
        delay = random.randint(min_ms, max_ms) / 1000
        time.sleep(delay)
    
    def _human_click(self, locator, description: str = "element"):
        """
        Click an element with human-like behavior:
        - Move mouse to element
        - Small random offset
        - Random delay before click
        """
        import random
        
        try:
            # Wait for element to be visible
            locator.wait_for(state="visible", timeout=10000)
            
            # Get element bounding box
            box = locator.bounding_box()
            if box:
                # Calculate click position with slight randomness
                x = box['x'] + box['width'] / 2 + random.randint(-5, 5)
                y = box['y'] + box['height'] / 2 + random.randint(-3, 3)
                
                # Move mouse to element (human-like)
                self._page.mouse.move(x, y, steps=random.randint(5, 15))
                
                # Small delay before clicking
                time.sleep(random.uniform(0.1, 0.3))
                
                # Click
                self._page.mouse.click(x, y)
                print(f"[Flow] Human-clicked: {description}", flush=True)
            else:
                # Fallback to regular click
                locator.click()
                print(f"[Flow] Clicked (fallback): {description}", flush=True)
                
        except Exception as e:
            print(f"[Flow] Human click failed for {description}: {e}", flush=True)
            # Fallback to force click
            locator.click(force=True)
    
    def _human_type(self, locator, text: str, description: str = "field"):
        """
        Type text with human-like behavior:
        - Variable typing speed
        - Occasional pauses
        """
        import random
        
        try:
            locator.click()
            self._human_delay(200, 400)
            
            # Clear existing text
            locator.fill("")
            self._human_delay(100, 200)
            
            # Type with variable speed
            for i, char in enumerate(text):
                locator.type(char, delay=random.randint(10, 50))
                
                # Occasional longer pause (like thinking)
                if random.random() < 0.02:
                    time.sleep(random.uniform(0.2, 0.5))
            
            print(f"[Flow] Human-typed into: {description} ({len(text)} chars)", flush=True)
            
        except Exception as e:
            print(f"[Flow] Human type failed for {description}: {e}, using fill()", flush=True)
            locator.fill(text)
    
    def _scroll_into_view(self, locator):
        """Scroll element into view with human-like behavior"""
        import random
        try:
            locator.scroll_into_view_if_needed()
            self._human_delay(200, 400)
        except Exception:
            pass
    
    def stop(self):
        """Stop the browser"""
        print("[Flow] Stopping browser...", flush=True)
        
        if self._context:
            self._context.close()
            self._context = None
        
        if self._browser:
            self._browser.close()
            self._browser = None
        
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        
        print("[Flow] Browser stopped", flush=True)
    
    def cancel(self):
        """Cancel current operation"""
        self._cancelled = True
    
    def _get_storage_state(self) -> Optional[dict]:
        """Get Playwright storage state for authentication"""
        # Try local file first
        if self.storage_state_path and os.path.exists(self.storage_state_path):
            print(f"[Flow] Loading storage state from: {self.storage_state_path}", flush=True)
            with open(self.storage_state_path, 'r') as f:
                return json.load(f)
        
        # Try S3/R2
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
    
    def _screenshot(self, name: str, upload: bool = True) -> Optional[str]:
        """
        Take a screenshot and optionally upload to R2.
        
        Args:
            name: Screenshot name (without extension)
            upload: Whether to upload to R2 and return URL
            
        Returns:
            Presigned URL if uploaded, local path otherwise
        """
        local_path = f"/tmp/flow_{name}.png"
        
        try:
            self._page.screenshot(path=local_path)
            print(f"[Flow] Screenshot saved: {local_path}", flush=True)
        except Exception as e:
            print(f"[Flow] Failed to take screenshot: {e}", flush=True)
            return None
        
        if not upload:
            return local_path
        
        # Upload to R2 and get presigned URL
        try:
            from .storage import is_storage_configured, get_storage
            
            if not is_storage_configured():
                print(f"[Flow] Storage not configured - screenshot only saved locally", flush=True)
                return local_path
            
            storage = get_storage()
            
            # Create unique key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            remote_key = f"debug/screenshots/{timestamp}_{name}.png"
            
            # Upload
            storage.upload_file(local_path, remote_key, content_type="image/png")
            
            # Get presigned URL (valid for 24 hours)
            url = storage.get_presigned_url(remote_key, expires_in=86400)
            
            print(f"[Flow] 📸 Screenshot uploaded: {url}", flush=True)
            return url
            
        except Exception as e:
            print(f"[Flow] Failed to upload screenshot: {e}", flush=True)
            return local_path
    
    def _check_login_required(self) -> bool:
        """Check if Google login is required"""
        current_url = self._page.url.lower()
        
        login_indicators = [
            "accounts.google.com",
            "identifier",
            "signin",
        ]
        
        is_login_page = any(indicator in current_url for indicator in login_indicators)
        
        # Also check for login elements
        if not is_login_page:
            try:
                sign_in_text = self._page.locator("text=Sign in").count() > 0
                email_input = self._page.locator("input[type='email']").count() > 0
                is_login_page = sign_in_text and email_input
            except Exception:
                pass
        
        return is_login_page
    
    def _wait_for_login(self, timeout: int = 300) -> bool:
        """
        Wait for login to complete (for interactive mode).
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if login completed, False if timeout
        """
        if not self._check_login_required():
            return True
        
        print("[Flow] Login required - waiting for manual login...", flush=True)
        self._needs_auth = True
        
        # In headless mode, we can't do manual login
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
    
    def _monitor_generation(self, job: 'FlowJob', total_wait_seconds: int = 120):
        """
        Monitor generation progress with periodic screenshots and error checking.
        
        Args:
            job: The job being processed
            total_wait_seconds: Total time to wait for generation
        """
        screenshot_interval = 30  # Take screenshot every 30 seconds
        check_interval = 10  # Check for errors every 10 seconds
        elapsed = 0
        screenshot_count = 0
        
        print(f"[Flow] 🔍 Starting generation monitoring (total wait: {total_wait_seconds}s)", flush=True)
        
        while elapsed < total_wait_seconds:
            time.sleep(check_interval)
            elapsed += check_interval
            
            # Check for and dismiss any popups
            self._check_and_dismiss_popup()
            
            # Check for error messages
            error_found = self._check_for_generation_errors()
            if error_found:
                print(f"[Flow] ⚠️ Error detected at {elapsed}s - taking screenshot", flush=True)
                self._screenshot(f"generation_error_{elapsed}s")
            
            # Check generation progress
            progress_info = self._check_generation_progress()
            if progress_info:
                print(f"[Flow] 📊 Progress at {elapsed}s: {progress_info}", flush=True)
            
            # Take periodic screenshots
            if elapsed % screenshot_interval == 0:
                screenshot_count += 1
                print(f"[Flow] 📸 Taking periodic screenshot #{screenshot_count} at {elapsed}s", flush=True)
                self._screenshot(f"generation_progress_{elapsed}s")
            
            # Log status
            if elapsed % 30 == 0:
                print(f"[Flow] ⏳ Waiting... {elapsed}/{total_wait_seconds}s elapsed", flush=True)
        
        # Final screenshot
        print(f"[Flow] 📸 Taking final screenshot after {total_wait_seconds}s wait", flush=True)
        self._screenshot("generation_complete")
        
        print(f"[Flow] ✅ Generation monitoring complete", flush=True)
    
    def _check_for_generation_errors(self) -> bool:
        """
        Check for error messages or popups during generation.
        
        Returns:
            True if an error was found
        """
        error_selectors = [
            "text=Error",
            "text=Failed",
            "text=Something went wrong",
            "text=limit reached",
            "text=quota exceeded",
            "text=try again",
            "text=couldn't generate",
            "text=generation failed",
            "[role='alert']",
            ".error-message",
            ".error",
        ]
        
        for selector in error_selectors:
            try:
                error_el = self._page.locator(selector).first
                if error_el.count() > 0 and error_el.is_visible():
                    try:
                        error_text = error_el.text_content()[:100]
                        print(f"[Flow] ❌ Error found: {error_text}", flush=True)
                    except Exception:
                        print(f"[Flow] ❌ Error element found with selector: {selector}", flush=True)
                    return True
            except Exception:
                pass
        
        # Also check for unexpected dialogs/modals
        modal_selectors = [
            "[role='dialog']",
            "[role='alertdialog']",
            ".modal",
            "[aria-modal='true']",
        ]
        
        for selector in modal_selectors:
            try:
                modal = self._page.locator(selector).first
                if modal.count() > 0 and modal.is_visible():
                    try:
                        modal_text = modal.text_content()[:200]
                        # Only flag if it looks like an error
                        if any(word in modal_text.lower() for word in ['error', 'fail', 'wrong', 'sorry', 'limit']):
                            print(f"[Flow] ⚠️ Error modal found: {modal_text[:100]}...", flush=True)
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
        
        return False
    
    def _check_generation_progress(self) -> Optional[str]:
        """
        Check the current generation progress.
        
        Returns:
            Progress info string, or None if no progress indicators found
        """
        progress_info = []
        
        # Check for video elements (completed generations)
        try:
            video_count = self._page.locator("video").count()
            if video_count > 0:
                progress_info.append(f"{video_count} video(s)")
        except Exception:
            pass
        
        # Check for progress percentage
        try:
            # Look for percentage text (e.g., "45%", "Generating 67%")
            page_text = self._page.locator("body").text_content()
            import re
            percentages = re.findall(r'(\d{1,3})%', page_text)
            if percentages:
                # Get unique percentages
                unique_pcts = list(set(percentages))
                if unique_pcts and unique_pcts != ['100']:
                    progress_info.append(f"progress: {', '.join(unique_pcts)}%")
        except Exception:
            pass
        
        # Check for "Generating" or "Queued" indicators
        try:
            generating = self._page.locator("text=Generating").count()
            queued = self._page.locator("text=Queued").count()
            if generating > 0:
                progress_info.append(f"{generating} generating")
            if queued > 0:
                progress_info.append(f"{queued} queued")
        except Exception:
            pass
        
        return ", ".join(progress_info) if progress_info else None
    
    def export_auth_state(self, output_path: str = None, upload_to_s3: bool = False) -> Optional[str]:
        """
        Export current browser authentication state.
        
        This should be run locally with headless=False to capture login cookies.
        
        Args:
            output_path: Local file path to save state
            upload_to_s3: Also upload to object storage
            
        Returns:
            Path to saved state file, or None if failed
        """
        if not self._context:
            raise RuntimeError("Browser not started. Call start() first.")
        
        # Navigate to Flow and wait for login (with proxy-compatible timeout)
        self._page.goto(FLOW_HOME_URL, timeout=60000, wait_until="load")
        time.sleep(5)
        
        if not self._wait_for_login(timeout=300):
            print("[Flow] Failed to complete login for auth export", flush=True)
            return None
        
        # Get storage state
        storage_state = self._context.storage_state()
        
        # Save locally
        output_path = output_path or "flow_storage_state.json"
        with open(output_path, 'w') as f:
            json.dump(storage_state, f, indent=2)
        
        print(f"[Flow] Auth state exported to: {output_path}", flush=True)
        
        # Optionally upload to S3
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
        """
        Create a new Flow project.
        
        Returns:
            Project URL
        """
        print("[Flow] Creating new project...", flush=True)
        
        # Navigate with longer timeout for proxy - use 'load' instead of 'networkidle'
        # Increase timeout to 90 seconds for slow proxy connections
        try:
            print("[Flow] Navigating to Flow homepage...", flush=True)
            self._page.goto(FLOW_HOME_URL, timeout=90000, wait_until="load")
            print("[Flow] ✓ Page loaded successfully", flush=True)
        except Exception as e:
            print(f"[Flow] ✗ Navigation failed: {e}", flush=True)
            self._screenshot("navigation_failed")
            raise
        
        # Give the page time to initialize JavaScript
        print("[Flow] Waiting for JS initialization...", flush=True)
        time.sleep(5)
        
        # Take screenshot to see what loaded
        self._screenshot("after_navigation")
        
        # Check for login
        if self._check_login_required():
            if not self._wait_for_login():
                raise RuntimeError("Login required but could not complete")
        
        self._check_and_dismiss_popup()
        
        # Click "New project" button - use 'load' instead of 'networkidle' for proxy compatibility
        # Also increase timeout for slow proxy connections
        try:
            with self._page.expect_navigation(wait_until="load", timeout=60000):
                self._page.locator("button:has-text('New project')").click(force=True)
                print("[Flow] Clicked New project button", flush=True)
        except Exception as e:
            # Navigation might not trigger if already on project page or SPA navigation
            print(f"[Flow] Navigation wait completed or timed out: {e}", flush=True)
        
        # Wait for page to stabilize (important for proxy connections)
        time.sleep(5)
        
        # Wait for the prompt input area to be ready - this indicates page is interactive
        print("[Flow] Waiting for project UI to be ready...", flush=True)
        try:
            self._page.wait_for_selector("#PINHOLE_TEXT_AREA_ELEMENT_ID", timeout=30000)
            print("[Flow] Prompt textarea found - page is ready", flush=True)
        except Exception as e:
            print(f"[Flow] Warning: Prompt textarea not found: {e}", flush=True)
            # Take screenshot to debug
            self._screenshot("project_not_ready")
        
        time.sleep(2)
        
        project_url = self._page.url
        print(f"[Flow] Created project: {project_url}", flush=True)
        
        # Verify we got a valid project URL
        if "/project/" not in project_url:
            print("[Flow] Warning: Project URL may be invalid, waiting more...", flush=True)
            time.sleep(5)
            project_url = self._page.url
        
        # Take screenshot to verify project is ready
        self._screenshot("project_created")
        
        return project_url
    
    def _upload_frame_with_button(
        self, 
        image_path: str, 
        button_selector: str, 
        is_first: bool = True,
        frame_name: str = "frame"
    ):
        """
        Click a button and upload a frame, handling both:
        - Direct file chooser popup
        - Modal with separate Upload button
        
        Args:
            image_path: Path to the image file
            button_selector: CSS selector for the add frame button
            is_first: Whether this is the first (start) or last (end) button
            frame_name: Name for logging
        """
        print(f"[Flow] Uploading {frame_name}: {image_path}", flush=True)
        
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"[Flow] Image file exists, size: {os.path.getsize(image_path)} bytes", flush=True)
        
        self._check_and_dismiss_popup()
        
        # Find the button
        if is_first:
            btn = self._page.locator(button_selector).first
        else:
            btn = self._page.locator(button_selector).last
        
        if btn.count() == 0:
            print(f"[Flow] WARNING: Button not found with selector: {button_selector}", flush=True)
            # Take screenshot
            self._screenshot(f"button_not_found_{frame_name.replace(' ', '_')}")
            raise RuntimeError(f"Add frame button not found")
        
        # Try to capture file chooser when clicking the button
        # This handles the case where the button directly opens file picker
        file_chooser = None
        
        try:
            print(f"[Flow] Clicking {frame_name} button (expecting file chooser)...", flush=True)
            
            # Set up file chooser listener BEFORE clicking
            with self._page.expect_file_chooser(timeout=5000) as fc_info:
                btn.click(force=True)
                print(f"[Flow] Clicked {frame_name} button", flush=True)
            
            # If we get here, the button directly opened a file chooser
            file_chooser = fc_info.value
            print(f"[Flow] File chooser opened directly from button click", flush=True)
            
        except Exception as e:
            # File chooser didn't open directly - might need to click "Upload" in a modal
            print(f"[Flow] No direct file chooser (this is normal): {e}", flush=True)
            print(f"[Flow] Looking for Upload button in modal...", flush=True)
            
            time.sleep(2)
            self._check_and_dismiss_popup()
            
            # Look for an Upload button
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
        
        # Upload the file
        if file_chooser:
            file_chooser.set_files(image_path)
            print(f"[Flow] File selected: {os.path.basename(image_path)}", flush=True)
        else:
            raise RuntimeError("No file chooser available")
        
        time.sleep(3)
        self._check_and_dismiss_popup()
        
        # Wait for and handle crop dialog
        print(f"[Flow] Waiting for crop dialog...", flush=True)
        try:
            self._page.wait_for_selector("text=Crop and Save", timeout=15000)
            print(f"[Flow] Crop dialog opened for {frame_name}", flush=True)
        except Exception as e:
            # Maybe no crop dialog needed, or it auto-cropped
            print(f"[Flow] Crop dialog not found (may not be needed): {e}", flush=True)
            # Check if image was already accepted
            time.sleep(2)
            return
        
        time.sleep(1)
        self._check_and_dismiss_popup()
        
        # Try to select Portrait orientation
        print(f"[Flow] Selecting orientation...", flush=True)
        try:
            # Look for orientation dropdown
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
                        
                        # Look for Portrait option
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
        
        # Click Crop and Save
        try:
            self._page.click("text=Crop and Save")
            print(f"[Flow] Clicked Crop and Save for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Could not click Crop and Save: {e}", flush=True)
            # Try alternative
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
        
        # Verify the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"[Flow] Image file exists, size: {os.path.getsize(image_path)} bytes", flush=True)
        
        self._check_and_dismiss_popup()
        
        try:
            # Try to find and click the Upload button with file chooser
            print("[Flow] Waiting for file chooser...", flush=True)
            with self._page.expect_file_chooser(timeout=10000) as fc_info:
                # Try multiple selectors for the upload button
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
            # Take screenshot for debugging
            self._screenshot(f"upload_error_{frame_name.replace(' ', '_')}")
            raise
        
        time.sleep(3)
        
        self._check_and_dismiss_popup()
        
        # Wait for crop dialog
        print("[Flow] Waiting for crop dialog...", flush=True)
        try:
            self._page.wait_for_selector("text=Crop and Save", timeout=15000)
            print(f"[Flow] Crop dialog opened for {frame_name}", flush=True)
        except Exception as e:
            print(f"[Flow] Crop dialog not found: {e}", flush=True)
            # Take screenshot
            self._screenshot(f"crop_error_{frame_name.replace(' ', '_')}")
            raise
        
        time.sleep(1)
        
        self._check_and_dismiss_popup()
        
        # Select Portrait orientation
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
            # Continue anyway - might already be in portrait mode
        
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
        """
        Submit a single clip for generation.
        
        Args:
            clip: The clip to generate
            is_first_clip: Whether this is the first clip in the project
            has_new_frames: Whether new frames should be uploaded
            language: Language for the prompt
            
        Returns:
            True if submitted successfully
        """
        if self._cancelled:
            return False
        
        # Use pre-built prompt from API engine if available, otherwise use simple fallback
        if clip.prompt:
            # Clean the API prompt for Flow (remove === markers and voice profile formatting)
            prompt = clean_prompt_for_flow(clip.prompt, clip.dialogue_text, language)
            print(f"[Flow] Cleaned API prompt for Flow ({len(prompt)} chars)", flush=True)
            print(f"[Flow] Prompt preview: {prompt[:150]}...", flush=True)
        else:
            prompt = get_prompt(clip.dialogue_text, language)
            print(f"[Flow] Using fallback prompt ({len(prompt)} chars)", flush=True)
        
        try:
            if is_first_clip:
                # First clip setup
                print(f"[Flow] Setting up first clip with frames...", flush=True)
                
                # Take screenshot before mode selection
                self._screenshot("before_mode_select")
                
                # Select Frames to Video mode - with retries
                print("[Flow] Selecting 'Frames to Video' mode...", flush=True)
                
                mode_changed = False
                for attempt in range(3):
                    print(f"[Flow] Mode selection attempt {attempt + 1}/3", flush=True)
                    
                    # Look for the mode dropdown button
                    # The dropdown shows current mode (e.g., "Text to Video")
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
                        # Try clicking on the area where the dropdown usually is
                        self._screenshot(f"no_dropdown_attempt_{attempt}")
                        time.sleep(2)
                        continue
                    
                    # Click to open the dropdown
                    mode_button.click()
                    print("[Flow] Clicked mode dropdown", flush=True)
                    time.sleep(1.5)  # Wait for dropdown animation
                    
                    # Wait for dropdown options to appear
                    try:
                        self._page.wait_for_selector("text=Frames to Video", timeout=5000)
                        print("[Flow] Dropdown options visible", flush=True)
                    except Exception:
                        print("[Flow] Dropdown options not visible, retrying...", flush=True)
                        self._page.keyboard.press("Escape")  # Close dropdown if stuck
                        time.sleep(1)
                        continue
                    
                    # Click "Frames to Video" option
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
                    
                    # Verify mode changed by checking for frame upload buttons
                    time.sleep(1)
                    frame_buttons = self._page.locator("button.sc-d02e9a37-1.hvUQuN")
                    if frame_buttons.count() > 0:
                        print(f"[Flow] SUCCESS: Mode changed! Found {frame_buttons.count()} frame button(s)", flush=True)
                        mode_changed = True
                        break
                    else:
                        print("[Flow] Frame buttons not found, mode may not have changed", flush=True)
                        # Take debug screenshot
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
                # Subsequent clip with new frames
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
                
                # Enter prompt with verification
                print(f"[Flow] Entering prompt ({len(prompt)} chars)...", flush=True)
                print(f"[Flow] Prompt preview: {prompt[:100]}...", flush=True)
                
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                
                # Make sure textarea is visible and interactable
                try:
                    textarea.wait_for(state="visible", timeout=5000)
                except Exception as e:
                    print(f"[Flow] Textarea not immediately visible: {e}", flush=True)
                    self._screenshot("no_textarea")
                
                # Click to focus
                textarea.click()
                time.sleep(0.5)
                
                # Clear any existing content
                textarea.fill("")
                time.sleep(0.3)
                
                # Enter the prompt using fill
                textarea.fill(prompt)
                print(f"[Flow] Filled prompt into textarea", flush=True)
                time.sleep(1)
                
                # Trigger input event in case fill didn't
                try:
                    textarea.evaluate("el => el.dispatchEvent(new Event('input', { bubbles: true }))")
                    textarea.evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
                except Exception:
                    pass
                
                # Verify the prompt was entered
                time.sleep(1)
                entered_text = textarea.input_value()
                if entered_text and len(entered_text) > 50:
                    print(f"[Flow] ✓ Prompt verified ({len(entered_text)} chars in textarea)", flush=True)
                else:
                    print(f"[Flow] ⚠ Prompt may not have been entered. Trying press_sequentially...", flush=True)
                    # Clear and try typing character by character (slower but more reliable)
                    textarea.fill("")
                    # Type just the first part to trigger the UI
                    textarea.press_sequentially(prompt[:200], delay=10)
                    time.sleep(0.5)
                    # Then fill the rest
                    textarea.fill(prompt)
                    time.sleep(1)
                
                # Take screenshot showing prompt entered
                self._screenshot("prompt_entered")
                
                # Brief wait for UI to process
                time.sleep(2)
                
            else:
                # Reuse frames, just change prompt
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
                
                # Click generate for reuse mode
                self._page.click("i:text('arrow_forward')", force=True)
                print(f"[Flow] Clip {clip.clip_index + 1}: Generation started (reuse)", flush=True)
                time.sleep(3)
                
                clip.status = "generating"
                return True
            
            # === VERIFY PROMPT WAS ENTERED ===
            print("[Flow] Verifying prompt was entered...", flush=True)
            time.sleep(1)
            
            try:
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                current_text = textarea.input_value()
                if current_text and len(current_text) > 10:
                    print(f"[Flow] ✓ Prompt verified in textarea ({len(current_text)} chars)", flush=True)
                else:
                    print(f"[Flow] ⚠ Textarea seems empty or short: '{current_text[:50] if current_text else 'EMPTY'}'", flush=True)
                    # Try entering prompt again
                    textarea.click()
                    time.sleep(0.5)
                    textarea.fill(prompt)
                    print("[Flow] Re-entered prompt", flush=True)
                    time.sleep(2)
            except Exception as e:
                print(f"[Flow] Could not verify prompt: {e}", flush=True)
            
            # Take screenshot before Generate
            self._screenshot("before_generate")
            
            # Add human-like delay before generating
            self._human_delay(1000, 2000)
            
            # === CLICK GENERATE BUTTON ===
            # The Generate button is the arrow icon (→) at the bottom right of the prompt area
            # It uses Material Icons with text 'arrow_forward'
            print("[Flow] Looking for Generate button (arrow icon)...", flush=True)
            
            generate_clicked = False
            
            # Method 1: Click the arrow_forward icon directly with human behavior
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
            
            # Method 2: Try clicking the parent button of the arrow icon
            if not generate_clicked:
                try:
                    # Find button containing the arrow icon
                    arrow_btn = self._page.locator("button:has(i:text('arrow_forward'))").first
                    if arrow_btn.count() > 0:
                        print("[Flow] Found button with arrow icon, human-clicking...", flush=True)
                        self._human_click(arrow_btn, "button with arrow icon")
                        generate_clicked = True
                        print("[Flow] ✓ Human-clicked button with arrow icon", flush=True)
                except Exception as e:
                    print(f"[Flow] Button with arrow icon failed: {e}", flush=True)
            
            # Method 3: Try the original CSS selector
            if not generate_clicked:
                try:
                    btn = self._page.locator("div.sc-408537d4-1.eiHkev > button").first
                    if btn.count() > 0 and btn.is_visible():
                        print("[Flow] Trying CSS class selector...", flush=True)
                        self._human_click(btn, "CSS selector button")
                        generate_clicked = True
                        print("[Flow] ✓ Clicked via CSS selector", flush=True)
                except Exception as e:
                    print(f"[Flow] CSS selector failed: {e}", flush=True)
            
            # Method 4: Find any button near "Expand" text
            if not generate_clicked:
                try:
                    # The generate button is right of "Expand"
                    expand_area = self._page.locator("text=Expand")
                    if expand_area.count() > 0:
                        # Get the parent container and find buttons in it
                        all_buttons = self._page.locator("button")
                        for i in range(all_buttons.count()):
                            btn = all_buttons.nth(i)
                            try:
                                # Look for a circular button (likely the submit button)
                                if btn.is_visible():
                                    inner_html = btn.inner_html()
                                    if 'arrow' in inner_html.lower() or 'svg' in inner_html.lower():
                                        print(f"[Flow] Found button {i} with arrow/svg, clicking...", flush=True)
                                        self._human_click(btn, f"button {i}")
                                        generate_clicked = True
                                        print(f"[Flow] ✓ Clicked button {i}", flush=True)
                                        break
                            except Exception:
                                pass
                except Exception as e:
                    print(f"[Flow] Expand-area search failed: {e}", flush=True)
            
            # Method 5: Use keyboard shortcut (Ctrl+Enter or Enter on focused button)
            if not generate_clicked:
                try:
                    print("[Flow] Trying keyboard shortcut...", flush=True)
                    # Focus on the textarea and try Ctrl+Enter
                    textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                    textarea.focus()
                    time.sleep(0.3)
                    self._page.keyboard.press("Control+Enter")
                    time.sleep(1)
                    # Or try Tab then Enter to focus the button
                    self._page.keyboard.press("Tab")
                    time.sleep(0.3)
                    self._page.keyboard.press("Enter")
                    generate_clicked = True
                    print("[Flow] ✓ Used keyboard shortcut", flush=True)
                except Exception as e:
                    print(f"[Flow] Keyboard shortcut failed: {e}", flush=True)
            
            if not generate_clicked:
                print("[Flow] ERROR: All Generate button methods failed!", flush=True)
                self._screenshot("generate_failed")
                raise RuntimeError("Could not click Generate button")
            
            # Wait for generation to start
            print("[Flow] Waiting for generation to start...", flush=True)
            time.sleep(5)
            
            # Take screenshot after clicking
            self._screenshot("after_generate")
            
            # Check for error messages
            error_messages = [
                "text=Error",
                "text=Failed",
                "text=limit",
                "text=quota",
                ".error",
                "[role='alert']",
            ]
            
            for error_sel in error_messages:
                try:
                    error_el = self._page.locator(error_sel).first
                    if error_el.count() > 0 and error_el.is_visible():
                        error_text = error_el.text_content()
                        print(f"[Flow] ⚠ Found error message: {error_text}", flush=True)
                except Exception:
                    pass
            
            # Look for signs that generation started
            generation_started = False
            
            # Check for generation indicators
            generation_indicators = [
                "text=Generating",
                "text=Queued",
                "text=Processing",
                "text=in progress",
                ".generating",
                "[data-generating='true']",
            ]
            
            for indicator in generation_indicators:
                try:
                    if self._page.locator(indicator).count() > 0:
                        print(f"[Flow] ✓ Found generation indicator: {indicator}", flush=True)
                        generation_started = True
                        break
                except Exception:
                    pass
            
            # Check if video elements appeared
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
            clip.status = "failed"
            clip.error_message = str(e)
            return False
    
    def _download_clip(
        self,
        clip: FlowClip,
        project_url: str,
        line_mapping: Dict[str, int]
    ) -> bool:
        """
        Download a generated clip.
        
        Args:
            clip: The clip to download
            project_url: Project URL to navigate to
            line_mapping: Mapping of dialogue text to line numbers
            
        Returns:
            True if downloaded successfully
        """
        try:
            # Navigate to project with extended timeout for proxy
            print(f"[Flow] Navigating to project for download: {project_url}", flush=True)
            self._page.goto(project_url, timeout=60000, wait_until="load")
            time.sleep(5)  # Extra time for proxy
            
            if self._check_login_required():
                if not self._wait_for_login():
                    raise RuntimeError("Login required for download")
            
            self._check_and_dismiss_popup()
            time.sleep(2)
            
            # Wait for clips to appear - look for various possible selectors
            print("[Flow] Waiting for clips to load...", flush=True)
            video_found = False
            
            for attempt in range(60):  # Wait up to 60 seconds
                # Try multiple selectors for video elements
                video_count = self._page.locator("video").count()
                
                # Also check for generation-in-progress indicators
                generating = self._page.locator("text=Generating").count()
                queued = self._page.locator("text=Queued").count()
                
                if video_count > 0:
                    print(f"[Flow] Found {video_count} video element(s)", flush=True)
                    video_found = True
                    break
                
                if attempt % 10 == 0:
                    print(f"[Flow] Still waiting... videos={video_count}, generating={generating}, queued={queued}", flush=True)
                    
                    # Debug: log what's visible on the page
                    if attempt == 30:
                        self._screenshot(f"debug_clip_{clip.clip_index}")
                
                time.sleep(1)
            
            if not video_found:
                print(f"[Flow] No video elements found after 60s wait", flush=True)
                # Mark as "generating" - the video may still be processing
                clip.status = "generating"
                clip.error_message = "Video still generating - check project URL manually"
                return False
            
            # Try multiple approaches to find the clip
            downloaded = False
            
            # Approach 1: Look for data-index container
            container = self._page.locator("div[data-index='0']")
            if container.count() > 0:
                print("[Flow] Found data-index container", flush=True)
                
                # Try to find video within container
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
                        
                        # Look for download button with various selectors
                        download_btn = None
                        for selector in [
                            "i:text('download')",
                            "[aria-label='Download']",
                            "button:has-text('download')",
                            ".download-button",
                            "i.material-icons:text('download')"
                        ]:
                            btn = container.locator(selector).first
                            if btn.count() > 0:
                                download_btn = btn
                                print(f"[Flow] Found download button with selector: {selector}", flush=True)
                                break
                        
                        if download_btn:
                            download_btn.click(force=True)
                            time.sleep(2)
                            
                            # Try to click download option
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
                        else:
                            print("[Flow] Could not find download button", flush=True)
                            
                    except Exception as e:
                        print(f"[Flow] Error interacting with video: {e}", flush=True)
            
            # Approach 2: If data-index failed, try finding any video on page
            if not downloaded:
                print("[Flow] Trying alternative video detection...", flush=True)
                all_videos = self._page.locator("video")
                if all_videos.count() > 0:
                    print(f"[Flow] Found {all_videos.count()} videos on page", flush=True)
                    
                    # Video exists but we couldn't download - mark as partially complete
                    clip.status = "generating"  # Will be marked completed by flow_worker
                    clip.error_message = "Video generated but download failed - check project URL"
                    return False
            
            if downloaded:
                return True
            
            print(f"[Flow] Could not download clip {clip.clip_index + 1}", flush=True)
            clip.status = "generating"  # Still mark as generating since submission worked
            return False
            
        except Exception as e:
            print(f"[Flow] Error downloading clip {clip.clip_index + 1}: {e}", flush=True)
            clip.error_message = str(e)
            clip.status = "generating"  # Submission worked, download failed
            return False
    
    def process_job(
        self,
        job: FlowJob,
        language: str = "English",
        wait_for_generation: bool = True
    ) -> bool:
        """
        Process a complete job (submit all clips).
        
        Args:
            job: The job to process
            language: Language for prompts
            wait_for_generation: Whether to wait and download after submitting
            
        Returns:
            True if all clips submitted successfully
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        print(f"[Flow] Processing job {job.job_id} with {len(job.clips)} clips", flush=True)
        
        try:
            # Create or navigate to project
            if job.project_url and "/project/" in job.project_url:
                print(f"[Flow] Resuming project: {job.project_url}", flush=True)
                self._page.goto(job.project_url, timeout=60000, wait_until="load")
                time.sleep(5)  # Extra time for proxy
                
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
            
            # Optionally wait and download
            if wait_for_generation:
                print(f"[Flow] Monitoring generation for {DEFAULT_WAIT_AFTER_SUBMIT}s...", flush=True)
                
                # Monitor generation with periodic screenshots and error checks
                self._monitor_generation(job, DEFAULT_WAIT_AFTER_SUBMIT)
                
                # Build line mapping for download matching
                line_mapping = {}
                for clip in job.clips:
                    dialogue = clip.dialogue_text.strip().strip('"').strip("'")
                    line_mapping[dialogue] = clip.clip_index + 1
                
                # Download each clip
                for clip in job.clips:
                    if clip.status == "generating":
                        self._download_clip(clip, job.project_url, line_mapping)
            
            return True
            
        except Exception as e:
            print(f"[Flow] Error processing job: {e}", flush=True)
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
    """
    Create a FlowJob from database data.
    
    Args:
        job_id: Job ID
        clips_data: List of clip dicts with dialogue_text, start_frame, end_frame, prompt
        project_url: Existing project URL if resuming
        
    Returns:
        FlowJob instance
    """
    clips = []
    for i, clip_data in enumerate(clips_data):
        clips.append(FlowClip(
            clip_index=i,
            dialogue_text=clip_data.get("dialogue_text", ""),
            start_frame_path=clip_data.get("start_frame"),
            end_frame_path=clip_data.get("end_frame"),
            prompt=clip_data.get("prompt"),  # Pre-built prompt from API engine
        ))
    
    return FlowJob(
        job_id=job_id,
        clips=clips,
        project_url=project_url
    )


def export_auth_state_command():
    """
    CLI command to export auth state interactively.
    
    Usage: python -m backends.flow_backend --export-auth
    """
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
