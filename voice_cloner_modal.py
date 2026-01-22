"""
Voice Cloner using Modal-hosted OpenVoice v2
Converts Veo's TTS output to sound like a cloned voice.
"""

import os
import httpx
import base64
import tempfile
from pathlib import Path
from typing import Optional
import time

# Modal endpoint URL - set this after deploying
OPENVOICE_MODAL_URL = os.environ.get(
    "OPENVOICE_MODAL_URL", 
    "https://your-username--openvoice-v2-openvoiceconverter-convert-endpoint.modal.run"
)

# Timeout settings
REQUEST_TIMEOUT = 120  # OpenVoice can take 30-60 seconds


def convert_voice_with_modal(
    source_audio_path: Path,
    target_voice_path: Path,
    output_path: Optional[Path] = None,
    tau: float = 0.3
) -> Optional[Path]:
    """
    Convert source audio to sound like target voice using Modal-hosted OpenVoice.
    
    Args:
        source_audio_path: Path to audio file to convert (Veo's TTS output)
        target_voice_path: Path to voice sample to clone
        output_path: Where to save the converted audio (optional)
        tau: Conversion temperature (0.1-0.5, lower = more similar to target)
    
    Returns:
        Path to converted audio file, or None if failed
    """
    try:
        # Read audio files
        with open(source_audio_path, "rb") as f:
            source_bytes = f.read()
        with open(target_voice_path, "rb") as f:
            target_bytes = f.read()
        
        print(f"[VoiceCloner] Source: {len(source_bytes)}B, Target: {len(target_bytes)}B")
        
        # For Modal, we need to either:
        # 1. Upload files to a temporary hosting and send URLs
        # 2. Send base64 encoded data directly
        
        # Option 2: Send as base64 (simpler, works for files < 10MB)
        source_b64 = base64.b64encode(source_bytes).decode('utf-8')
        target_b64 = base64.b64encode(target_bytes).decode('utf-8')
        
        # Call Modal endpoint
        start_time = time.time()
        print(f"[VoiceCloner] Calling Modal endpoint: {OPENVOICE_MODAL_URL}")
        
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                OPENVOICE_MODAL_URL,
                json={
                    "source_base64": source_b64,
                    "target_base64": target_b64,
                    "tau": tau
                }
            )
            response.raise_for_status()
            result = response.json()
        
        elapsed = time.time() - start_time
        print(f"[VoiceCloner] Modal response in {elapsed:.1f}s")
        
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            print(f"[VoiceCloner] ❌ Modal error: {error}")
            return None
        
        # Decode output
        output_b64 = result.get("output_base64")
        if not output_b64:
            print("[VoiceCloner] ❌ No output in response")
            return None
        
        output_bytes = base64.b64decode(output_b64)
        
        # Save to file
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        
        with open(output_path, "wb") as f:
            f.write(output_bytes)
        
        print(f"[VoiceCloner] ✅ Saved converted audio: {output_path} ({len(output_bytes)}B)")
        return output_path
        
    except httpx.HTTPError as e:
        print(f"[VoiceCloner] ❌ HTTP error: {e}")
        return None
    except Exception as e:
        print(f"[VoiceCloner] ❌ Error: {e}")
        return None


def clone_voice_for_video(
    video_path: Path,
    voice_sample_path: Path,
    output_video_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Extract audio from video, convert voice, and mux back.
    
    Args:
        video_path: Path to video file with audio to convert
        voice_sample_path: Path to voice sample to clone
        output_video_path: Where to save the result (optional)
    
    Returns:
        Path to video with converted voice, or None if failed
    """
    import subprocess
    
    try:
        work_dir = Path(tempfile.mkdtemp())
        
        # Extract audio from video
        extracted_audio = work_dir / "extracted.wav"
        print(f"[VoiceCloner] Extracting audio from {video_path}")
        
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            str(extracted_audio)
        ], check=True, capture_output=True)
        
        # Convert voice
        converted_audio = work_dir / "converted.wav"
        result = convert_voice_with_modal(
            extracted_audio,
            voice_sample_path,
            converted_audio
        )
        
        if result is None:
            print("[VoiceCloner] ❌ Voice conversion failed")
            return None
        
        # Mux converted audio back into video
        if output_video_path is None:
            output_video_path = video_path.with_stem(video_path.stem + "_cloned")
        
        print(f"[VoiceCloner] Muxing audio back into video")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(converted_audio),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_video_path)
        ], check=True, capture_output=True)
        
        print(f"[VoiceCloner] ✅ Output video: {output_video_path}")
        return output_video_path
        
    except subprocess.CalledProcessError as e:
        print(f"[VoiceCloner] ❌ FFmpeg error: {e.stderr.decode() if e.stderr else e}")
        return None
    except Exception as e:
        print(f"[VoiceCloner] ❌ Error: {e}")
        return None


# =============================================================================
# Alternative: Direct Modal Python client (faster, no HTTP overhead)
# =============================================================================

def convert_voice_modal_direct(
    source_audio_path: Path,
    target_voice_path: Path,
    output_path: Optional[Path] = None,
    tau: float = 0.3
) -> Optional[Path]:
    """
    Convert voice using Modal's Python client directly (faster than HTTP).
    Requires: pip install modal
    """
    try:
        import modal
        
        # Get the deployed function
        convert_fn = modal.Function.lookup("openvoice-v2", "convert_voice_simple")
        
        # Read files
        with open(source_audio_path, "rb") as f:
            source_bytes = f.read()
        with open(target_voice_path, "rb") as f:
            target_bytes = f.read()
        
        print(f"[VoiceCloner] Calling Modal function directly...")
        start_time = time.time()
        
        # Call the function
        output_bytes = convert_fn.remote(source_bytes, target_bytes, tau)
        
        elapsed = time.time() - start_time
        print(f"[VoiceCloner] Modal completed in {elapsed:.1f}s")
        
        # Save output
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        
        with open(output_path, "wb") as f:
            f.write(output_bytes)
        
        print(f"[VoiceCloner] ✅ Saved: {output_path}")
        return output_path
        
    except ImportError:
        print("[VoiceCloner] Modal not installed, falling back to HTTP")
        return convert_voice_with_modal(source_audio_path, target_voice_path, output_path, tau)
    except Exception as e:
        print(f"[VoiceCloner] ❌ Modal direct error: {e}")
        return None


# =============================================================================
# Main entry point for Veo integration
# =============================================================================

def process_voice_cloning(
    source_audio_path: str,
    target_voice_path: str,
    output_path: Optional[str] = None,
    use_direct: bool = False
) -> Optional[str]:
    """
    Main entry point for voice cloning.
    
    Args:
        source_audio_path: Audio to convert
        target_voice_path: Voice to clone
        output_path: Where to save result
        use_direct: Use Modal Python client (faster) vs HTTP
    
    Returns:
        Path to converted audio, or None
    """
    source = Path(source_audio_path)
    target = Path(target_voice_path)
    output = Path(output_path) if output_path else None
    
    if use_direct:
        result = convert_voice_modal_direct(source, target, output)
    else:
        result = convert_voice_with_modal(source, target, output)
    
    return str(result) if result else None


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python voice_cloner_modal.py <source_audio> <target_voice> [output]")
        sys.exit(1)
    
    source = sys.argv[1]
    target = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = process_voice_cloning(source, target, output)
    
    if result:
        print(f"✅ Success: {result}")
    else:
        print("❌ Failed")
        sys.exit(1)