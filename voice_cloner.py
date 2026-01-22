"""
Voice Cloning Module using Modal-hosted OpenVoice v2

Uses OpenVoice for voice-to-voice conversion (tone color transfer)
- Takes source audio + target voice
- Outputs source speech with target voice characteristics
- Preserves timing/prosody from source

Setup:
1. Deploy OpenVoice to Modal (see openvoice_modal.py)
2. Set OPENVOICE_MODAL_URL environment variable
"""

import os
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional
import logging
import httpx

logger = logging.getLogger(__name__)

# Modal endpoint URL
OPENVOICE_MODAL_URL = os.environ.get(
    "OPENVOICE_MODAL_URL", 
    "https://kaveno-biz--openvoice-v2-openvoiceconverter-convert-endpoint.modal.run"
)

# Timeout settings
REQUEST_TIMEOUT = 300  # 5 minutes for full audio conversion

# Model info
MODEL_NAME = "OpenVoice v2"
MODEL_COST = "~$0.01"  # Modal T4 cost per conversion


def voice_convert_sync(
    source_audio_path: Path,
    target_voice_path: Path,
    output_path: Path,
    tau: float = 0.3,
    pitch_normalize: float = 0.0
) -> bool:
    """
    Convert voice using Modal-hosted OpenVoice v2.
    
    Args:
        source_audio_path: Path to source audio (speech to convert)
        target_voice_path: Path to target voice sample (voice to clone)
        output_path: Path to save output audio
        tau: Conversion temperature (0.1-0.5, lower = more similar to target)
        pitch_normalize: Pitch compression (0.0-1.0, 0 = off)
    
    Returns:
        True if successful
    """
    logger.info("[OpenVoice] Starting voice-to-voice conversion...")
    logger.info(f"  Source: {source_audio_path}")
    logger.info(f"  Target: {target_voice_path}")
    logger.info(f"  tau={tau}, pitch_normalize={pitch_normalize}")
    logger.info(f"  Endpoint: {OPENVOICE_MODAL_URL}")
    
    # Read and encode audio files
    try:
        with open(source_audio_path, "rb") as f:
            source_bytes = f.read()
        with open(target_voice_path, "rb") as f:
            target_bytes = f.read()
        
        source_b64 = base64.b64encode(source_bytes).decode('utf-8')
        target_b64 = base64.b64encode(target_bytes).decode('utf-8')
        
        logger.info(f"[OpenVoice] Encoded: source={len(source_bytes)}B, target={len(target_bytes)}B")
        
    except Exception as e:
        logger.error(f"[OpenVoice] Failed to read audio files: {e}")
        return False
    
    # Call Modal endpoint
    start_time = time.time()
    try:
        logger.info("[OpenVoice] Calling Modal endpoint...")
        
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                OPENVOICE_MODAL_URL,
                json={
                    "source_base64": source_b64,
                    "target_base64": target_b64,
                    "tau": tau,
                    "pitch_normalize": pitch_normalize
                }
            )
            response.raise_for_status()
            result = response.json()
        
        elapsed = time.time() - start_time
        logger.info(f"[OpenVoice] Response in {elapsed:.1f}s")
        
    except httpx.TimeoutException:
        logger.error(f"[OpenVoice] Request timed out after {REQUEST_TIMEOUT}s")
        return False
    except httpx.HTTPError as e:
        logger.error(f"[OpenVoice] HTTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"[OpenVoice] Request error: {e}")
        return False
    
    # Process response
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        logger.error(f"[OpenVoice] Conversion failed: {error}")
        return False
    
    output_b64 = result.get("output_base64")
    if not output_b64:
        logger.error("[OpenVoice] No output in response")
        return False
    
    # Decode and save output
    try:
        output_bytes = base64.b64decode(output_b64)
        with open(output_path, "wb") as f:
            f.write(output_bytes)
        
        logger.info(f"[OpenVoice] ✅ Saved to {output_path} ({len(output_bytes)} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"[OpenVoice] Failed to save output: {e}")
        return False


def voice_swap_video_sync(
    video_path: Path,
    reference_voice_path: Path,
    output_path: Path,
    progress_callback=None,
    tau: float = 0.3,
    pitch_normalize: float = 0.0
) -> dict:
    """
    Complete voice swap pipeline for video using OpenVoice.
    
    Args:
        video_path: Input video file
        reference_voice_path: Reference voice sample (the voice to clone)
        output_path: Output video with swapped voice
        progress_callback: Optional callback for progress updates
        tau: Voice similarity (0.1-0.5, lower = more similar to target voice)
        pitch_normalize: Pitch compression (0.0-1.0, 0 = off, 0.5 = moderate)
    
    Returns:
        dict with processing stats
    """
    from audio_processor import extract_audio, replace_audio
    
    stats = {
        "success": False,
        "model": MODEL_NAME,
        "cost_estimate": MODEL_COST
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Extract audio from video
        if progress_callback:
            progress_callback("Extracting audio...")
        
        source_audio = temp_path / "source_audio.wav"
        if not extract_audio(video_path, source_audio):
            return {"success": False, "error": "Failed to extract audio from video"}
        
        logger.info(f"[VoiceSwap] Extracted source audio: {source_audio.stat().st_size} bytes")
        
        # Step 2: Convert voice using OpenVoice
        progress_msg = "Converting voice (30-90 seconds)..."
        if pitch_normalize > 0:
            progress_msg = f"Normalizing pitch + converting voice (60-120 seconds)..."
        if progress_callback:
            progress_callback(progress_msg)
        
        converted_audio = temp_path / "converted_audio.wav"
        
        try:
            success = voice_convert_sync(
                source_audio_path=source_audio,
                target_voice_path=reference_voice_path,
                output_path=converted_audio,
                tau=tau,
                pitch_normalize=pitch_normalize
            )
            
            if not success:
                return {"success": False, "error": "Voice conversion returned no output"}
                
        except Exception as e:
            logger.error(f"[VoiceSwap] Conversion error: {e}")
            return {"success": False, "error": f"Voice conversion failed: {e}"}
        
        # Step 3: Replace audio in video
        if progress_callback:
            progress_callback("Creating final video...")
        
        if not replace_audio(video_path, converted_audio, output_path):
            return {"success": False, "error": "Failed to create output video"}
        
        stats["success"] = True
        logger.info(f"[VoiceSwap] Complete! Output: {output_path}")
    
    if progress_callback:
        progress_callback("Voice swap complete!")
    
    return stats


def check_openvoice_available() -> dict:
    """Check if OpenVoice Modal endpoint is available (quick check, 30s timeout)"""
    health_url = OPENVOICE_MODAL_URL.replace("convert-endpoint", "health")
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(health_url)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                return {
                    "available": True,
                    "message": "OpenVoice Modal endpoint ready",
                    "model": data.get("model", MODEL_NAME),
                    "cost": MODEL_COST,
                    "device": data.get("device", "unknown")
                }
            else:
                return {
                    "available": False,
                    "message": "OpenVoice endpoint returned unexpected status",
                    "model": MODEL_NAME,
                    "cost": MODEL_COST
                }
                
    except httpx.TimeoutException:
        return {
            "available": False,
            "message": "OpenVoice endpoint timed out (may be cold starting)",
            "model": MODEL_NAME,
            "cost": MODEL_COST
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"OpenVoice endpoint error: {str(e)}",
            "model": MODEL_NAME,
            "cost": MODEL_COST
        }


def warmup_openvoice_sync(timeout: int = 120) -> dict:
    """
    Warmup OpenVoice Modal endpoint with longer timeout.
    
    This function waits for the container to fully start (cold start can take 60-90s).
    Use this for proactive warmup, not quick availability checks.
    
    Args:
        timeout: How long to wait for warmup (default 120s for cold starts)
    
    Returns:
        dict with 'available', 'message', 'warmup_time' keys
    """
    import time
    health_url = OPENVOICE_MODAL_URL.replace("convert-endpoint", "health")
    
    start_time = time.time()
    logger.info(f"[OpenVoice Warmup] Starting warmup (timeout={timeout}s)...")
    logger.info(f"[OpenVoice Warmup] Health URL: {health_url}")
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(health_url)
            response.raise_for_status()
            data = response.json()
            
            elapsed = time.time() - start_time
            
            if data.get("status") == "ok":
                logger.info(f"[OpenVoice Warmup] ✅ Ready in {elapsed:.1f}s (device: {data.get('device', 'unknown')})")
                return {
                    "available": True,
                    "message": f"OpenVoice ready in {elapsed:.1f}s",
                    "model": data.get("model", MODEL_NAME),
                    "device": data.get("device", "unknown"),
                    "warmup_time": elapsed
                }
            else:
                logger.warning(f"[OpenVoice Warmup] Unexpected status after {elapsed:.1f}s")
                return {
                    "available": False,
                    "message": "Unexpected status from endpoint",
                    "warmup_time": elapsed
                }
                
    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        logger.warning(f"[OpenVoice Warmup] ⚠️ Timed out after {elapsed:.1f}s (cold start too slow?)")
        return {
            "available": False,
            "message": f"Warmup timed out after {elapsed:.1f}s - container may still be starting",
            "warmup_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[OpenVoice Warmup] ❌ Error after {elapsed:.1f}s: {e}")
        return {
            "available": False,
            "message": f"Warmup error: {str(e)}",
            "warmup_time": elapsed
        }


# Backwards compatibility aliases
check_replicate_available = check_openvoice_available
