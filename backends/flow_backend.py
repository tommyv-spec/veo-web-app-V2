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
        temp_dir: Optional[str] = None
    ):
        """
        Initialize Flow backend.
        
        Args:
            storage_state_path: Local path to Playwright storage state JSON
            storage_state_url: S3 URL/key to download storage state from
            headless: Whether to run browser headlessly
            download_dir: Directory for downloads
            temp_dir: Directory for temporary files
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
        """Start the browser"""
        print("[Flow] Starting browser...", flush=True)
        
        self._playwright = sync_playwright().start()
        
        # Use persistent context for better session handling
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--disable-gpu",
            ]
        )
        
        # Get storage state
        storage_state = self._get_storage_state()
        
        # Create context with storage state if available
        context_options = {
            "accept_downloads": True,
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        if storage_state:
            context_options["storage_state"] = storage_state
        
        self._context = self._browser.new_context(**context_options)
        self._page = self._context.new_page()
        
        # Add anti-detection scripts
        self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)
        
        print("[Flow] Browser started", flush=True)
    
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
        
        # Navigate to Flow and wait for login
        self._page.goto(FLOW_HOME_URL)
        self._page.wait_for_load_state("networkidle")
        time.sleep(2)
        
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
        
        self._page.goto(FLOW_HOME_URL)
        self._page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Check for login
        if self._check_login_required():
            if not self._wait_for_login():
                raise RuntimeError("Login required but could not complete")
        
        self._check_and_dismiss_popup()
        
        # Click "New project" button
        with self._page.expect_navigation(wait_until="networkidle"):
            self._page.locator("button:has-text('New project')").click(force=True)
            print("[Flow] Clicked New project button", flush=True)
        
        # Wait longer for project page to fully load
        time.sleep(5)
        
        # Wait for the prompt input area to be ready - this indicates page is interactive
        print("[Flow] Waiting for project UI to be ready...", flush=True)
        try:
            self._page.wait_for_selector("#PINHOLE_TEXT_AREA_ELEMENT_ID", timeout=15000)
            print("[Flow] Prompt textarea found - page is ready", flush=True)
        except Exception as e:
            print(f"[Flow] Warning: Prompt textarea not found: {e}", flush=True)
        
        time.sleep(2)
        
        project_url = self._page.url
        print(f"[Flow] Created project: {project_url}", flush=True)
        
        # Verify we got a valid project URL
        if "/project/" not in project_url:
            print("[Flow] Warning: Project URL may be invalid, waiting more...", flush=True)
            time.sleep(5)
            project_url = self._page.url
        
        # Take screenshot to verify project is ready
        try:
            self._page.screenshot(path="/tmp/flow_project_created.png")
            print("[Flow] Screenshot saved: /tmp/flow_project_created.png", flush=True)
        except Exception:
            pass
        
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
            self._page.screenshot(path=f"/tmp/flow_button_not_found_{frame_name}.png")
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
                    self._page.screenshot(path=f"/tmp/flow_upload_failed_{frame_name}.png")
                    raise
            else:
                print(f"[Flow] No Upload button found", flush=True)
                self._page.screenshot(path=f"/tmp/flow_no_upload_btn_{frame_name}.png")
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
            try:
                screenshot_path = f"/tmp/flow_upload_error_{frame_name}.png"
                self._page.screenshot(path=screenshot_path)
                print(f"[Flow] Debug screenshot saved: {screenshot_path}", flush=True)
            except Exception:
                pass
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
            try:
                self._page.screenshot(path=f"/tmp/flow_crop_error_{frame_name}.png")
            except Exception:
                pass
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
            prompt = clip.prompt
            print(f"[Flow] Using API-generated prompt ({len(prompt)} chars)", flush=True)
        else:
            prompt = get_prompt(clip.dialogue_text, language)
            print(f"[Flow] Using fallback prompt ({len(prompt)} chars)", flush=True)
        
        try:
            if is_first_clip:
                # First clip setup
                print(f"[Flow] Setting up first clip with frames...", flush=True)
                
                # Take screenshot before mode selection
                try:
                    self._page.screenshot(path="/tmp/flow_before_mode_select.png")
                except Exception:
                    pass
                
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
                        try:
                            self._page.screenshot(path=f"/tmp/flow_no_dropdown_attempt_{attempt}.png")
                        except Exception:
                            pass
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
                        try:
                            self._page.screenshot(path=f"/tmp/flow_mode_attempt_{attempt}.png")
                        except Exception:
                            pass
                
                if not mode_changed:
                    print("[Flow] ERROR: Failed to switch to 'Frames to Video' mode after 3 attempts!", flush=True)
                    self._page.screenshot(path="/tmp/flow_mode_failed.png")
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
                
                # Enter prompt
                textarea = self._page.locator("#PINHOLE_TEXT_AREA_ELEMENT_ID")
                textarea.click()
                time.sleep(0.5)
                textarea.fill("")
                time.sleep(0.3)
                textarea.fill(prompt)
                print(f"[Flow] Entered prompt: {clip.dialogue_text[:50]}...", flush=True)
                time.sleep(10)
                
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
            
            # Wait for Generate button and click it properly
            print("[Flow] Looking for Generate button...", flush=True)
            
            # Take screenshot before trying to click Generate
            try:
                self._page.screenshot(path="/tmp/flow_before_generate.png")
                print("[Flow] Screenshot saved: /tmp/flow_before_generate.png", flush=True)
            except Exception:
                pass
            
            # Try multiple selectors for the Generate button
            generate_btn = None
            generate_selectors = [
                "div.sc-408537d4-1.eiHkev > button",  # Original selector
                "button:has-text('Generate')",
                "[aria-label='Generate']",
                "button[type='submit']",
                "button.generate-button",
                # Look for the arrow/send icon button at the bottom right
                "button:has(svg)",
                "button:near(:text('Expand'))",
            ]
            
            for selector in generate_selectors:
                try:
                    btn = self._page.locator(selector).first
                    if btn.count() > 0 and btn.is_visible():
                        # Check if it's not disabled
                        if not btn.is_disabled():
                            generate_btn = btn
                            print(f"[Flow] Found Generate button with selector: {selector}", flush=True)
                            break
                        else:
                            print(f"[Flow] Button found but disabled: {selector}", flush=True)
                except Exception as e:
                    pass
            
            if not generate_btn:
                # Try finding by the arrow icon (common in chat-like interfaces)
                try:
                    # The generate button might be the arrow at bottom right of the prompt box
                    arrow_btn = self._page.locator("button").filter(has=self._page.locator("svg")).last
                    if arrow_btn.count() > 0 and arrow_btn.is_visible():
                        generate_btn = arrow_btn
                        print("[Flow] Found Generate button by arrow icon", flush=True)
                except Exception:
                    pass
            
            if not generate_btn:
                print("[Flow] ERROR: Could not find Generate button!", flush=True)
                self._page.screenshot(path="/tmp/flow_no_generate_btn.png")
                raise RuntimeError("Generate button not found")
            
            # Wait for button to be enabled
            print("[Flow] Waiting for Generate button to be enabled...", flush=True)
            for attempt in range(30):
                try:
                    if not generate_btn.is_disabled():
                        print("[Flow] Generate button is enabled!", flush=True)
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            
            # Click the button (not keyboard Enter!)
            print("[Flow] Clicking Generate button...", flush=True)
            time.sleep(1)
            
            try:
                generate_btn.click(force=True)
                print("[Flow] Clicked Generate button", flush=True)
            except Exception as e:
                print(f"[Flow] Direct click failed: {e}, trying JavaScript click...", flush=True)
                try:
                    generate_btn.evaluate("el => el.click()")
                    print("[Flow] JavaScript click succeeded", flush=True)
                except Exception as e2:
                    print(f"[Flow] JavaScript click also failed: {e2}", flush=True)
                    # Last resort: keyboard
                    generate_btn.focus()
                    self._page.keyboard.press("Enter")
                    print("[Flow] Used keyboard Enter as fallback", flush=True)
            
            # Wait and verify something is happening
            time.sleep(3)
            
            # Take screenshot after clicking to verify
            try:
                self._page.screenshot(path="/tmp/flow_after_generate.png")
                print("[Flow] Screenshot saved: /tmp/flow_after_generate.png", flush=True)
            except Exception:
                pass
            
            # Look for signs that generation started
            generation_started = False
            
            # Check for loading indicators or status changes
            loading_indicators = [
                "text=Generating",
                "text=Loading",
                "text=Processing",
                ".loading",
                ".spinner",
                "[role='progressbar']",
                "text=Queued",
            ]
            
            for indicator in loading_indicators:
                try:
                    if self._page.locator(indicator).count() > 0:
                        print(f"[Flow] Found generation indicator: {indicator}", flush=True)
                        generation_started = True
                        break
                except Exception:
                    pass
            
            # Also check if a video element or clip container appeared
            try:
                video_count = self._page.locator("video").count()
                clip_containers = self._page.locator("[data-index]").count()
                if video_count > 0 or clip_containers > 0:
                    print(f"[Flow] Found {video_count} videos, {clip_containers} clip containers", flush=True)
                    generation_started = True
            except Exception:
                pass
            
            if not generation_started:
                print("[Flow] WARNING: Could not verify generation started - continuing anyway", flush=True)
            
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
            # Navigate to project
            print(f"[Flow] Navigating to project for download: {project_url}", flush=True)
            self._page.goto(project_url)
            self._page.wait_for_load_state("networkidle")
            time.sleep(3)
            
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
                        try:
                            # Take screenshot for debugging
                            screenshot_path = os.path.join(self.download_dir, f"debug_clip_{clip.clip_index}.png")
                            self._page.screenshot(path=screenshot_path)
                            print(f"[Flow] Debug screenshot saved: {screenshot_path}", flush=True)
                        except Exception:
                            pass
                
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
                self._page.goto(job.project_url)
                self._page.wait_for_load_state("networkidle")
                time.sleep(3)
                
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
                print(f"[Flow] Waiting {DEFAULT_WAIT_AFTER_SUBMIT}s for generation...", flush=True)
                time.sleep(DEFAULT_WAIT_AFTER_SUBMIT)
                
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
