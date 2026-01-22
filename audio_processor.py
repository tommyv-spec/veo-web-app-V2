"""
Audio Processing Module for Veo Web App
- Voice-optimized enhancement (safe for Veo3 voices)
- Noise reduction: DeepFilterNet (if torch installed) → noisereduce → FFmpeg
- Two-pass loudness normalization
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Voice chain (safe for Veo3: avoids heavy denoise artifacts)
VOICE_FILTER = (
    "highpass=f=80,"
    "acompressor=threshold=-20dB:ratio=3:attack=15:release=150:makeup=5,"
    "agate=threshold=-40dB:ratio=4:attack=5:release=150,"
    "dynaudnorm=f=150:g=3:p=0.9,"
    "alimiter=limit=-1.2dB"
)

# Loudness target for spoken mono voice
LOUDNORM_I = -16     # LUFS
LOUDNORM_TP = -1.5   # dBTP
LOUDNORM_LRA = 11    # LU

# Output audio codec inside MP4
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"


def run_cmd(cmd: list, timeout: int = 300) -> tuple:
    """Run command and return (returncode, stdout, stderr)"""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except Exception as e:
        return -1, "", str(e)


def parse_loudnorm_json(stderr_text: str) -> dict:
    """Parse loudnorm JSON from ffmpeg stderr output."""
    matches = re.findall(r"\{[\s\S]*?\}", stderr_text)
    if not matches:
        raise RuntimeError("Could not parse loudnorm JSON from ffmpeg output.")
    
    for cand in reversed(matches):
        try:
            data = json.loads(cand)
            if all(k in data for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
                return data
        except json.JSONDecodeError:
            pass
    
    # Fallback: try last
    return json.loads(matches[-1])


def try_deepfilternet(in_wav: Path, out_wav: Path) -> bool:
    """
    DeepFilterNet - high quality neural network noise reduction.
    Requires torch, torchaudio, deepfilternet packages.
    """
    try:
        import torch
        import torchaudio
        import soundfile as sf
        from df.enhance import init_df, enhance
        import numpy as np

        model, state, _ = init_df()

        # Load audio
        wav, sr = torchaudio.load(str(in_wav))

        # Get model sample rate
        sr_attr = getattr(state, "sr", None)
        if callable(sr_attr):
            target_sr = sr_attr()
        elif sr_attr is not None:
            target_sr = sr_attr
        else:
            target_sr = 48000

        # Resample if needed
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr

        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Enhance
        enhanced = enhance(model, state, wav)

        # Convert to numpy
        if isinstance(enhanced, torch.Tensor):
            enhanced_np = enhanced.detach().cpu().numpy()
        else:
            enhanced_np = np.asarray(enhanced)

        enhanced_np = np.squeeze(enhanced_np)
        sf.write(str(out_wav), enhanced_np.astype(np.float32), sr)

        logger.info("[DeepFilterNet] Denoise applied successfully")
        return True

    except ImportError:
        logger.debug("[DeepFilterNet] Not available (missing torch/torchaudio/deepfilternet)")
        return False
    except Exception as e:
        logger.warning(f"[DeepFilterNet] Failed: {e}")
        return False


def try_noisereduce(in_wav: Path, out_wav: Path, strength: float = 0.75) -> bool:
    """
    Noise reduction using noisereduce library.
    Removes background noise, laughter, etc. while preserving voice.
    
    Args:
        in_wav: Input WAV file
        out_wav: Output WAV file  
        strength: Noise reduction strength (0.0-1.0, default 0.75)
    """
    try:
        import numpy as np
        import soundfile as sf
        import noisereduce as nr
        
        print(f"[NoiseReduce] Loading audio from {in_wav}...", flush=True)
        
        # Load audio
        audio, sr = sf.read(str(in_wav))
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        print(f"[NoiseReduce] Applying noise reduction (strength={strength}, stationary=False)...", flush=True)
        
        # Apply noise reduction
        # prop_decrease controls how much noise is reduced (0.0-1.0)
        # stationary=False helps with non-stationary noise like laughter
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=strength,
            stationary=False,  # Better for varying background noise
            n_fft=2048,
            hop_length=512,
        )
        
        # Save result
        sf.write(str(out_wav), reduced.astype(np.float32), sr)
        
        print(f"[NoiseReduce] ✅ Success! Saved to {out_wav}", flush=True)
        logger.info(f"[NoiseReduce] Applied with strength={strength}")
        return True
        
    except ImportError as e:
        print(f"[NoiseReduce] ❌ Not installed: {e}", flush=True)
        logger.debug(f"[NoiseReduce] Not available: {e}")
        return False
    except Exception as e:
        print(f"[NoiseReduce] ❌ Failed: {e}", flush=True)
        logger.warning(f"[NoiseReduce] Failed: {e}")
        return False


def try_ffmpeg_denoise(in_wav: Path, out_wav: Path, noise_floor: int = -30) -> bool:
    """
    FFmpeg-based noise reduction using afftdn filter.
    Fallback if noisereduce is not available.
    
    Args:
        in_wav: Input WAV file
        out_wav: Output WAV file
        noise_floor: Noise floor in dB (default -30)
    """
    try:
        # afftdn = Adaptive FFT Denoiser
        # nr = noise reduction amount (0-97, default 12)
        # nf = noise floor in dB
        # tn = enable noise tracking
        denoise_filter = f"afftdn=nf={noise_floor}:nr=20:tn=1"
        
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(in_wav),
            "-af", denoise_filter,
            "-c:a", "pcm_s24le",
            str(out_wav)
        ]
        code, _, err = run_cmd(cmd)
        
        if code != 0:
            logger.warning(f"[FFmpeg Denoise] Failed: {err}")
            return False
            
        logger.info(f"[FFmpeg Denoise] Applied with noise_floor={noise_floor}dB")
        return True
        
    except Exception as e:
        logger.warning(f"[FFmpeg Denoise] Failed: {e}")
        return False


def try_deepfilter_modal(in_wav: Path, out_wav: Path) -> bool:
    """
    Call DeepFilterNet via Modal serverless function.
    Highest quality AI-powered noise reduction.
    
    Uses DEEPFILTER_MODAL_URL env var, or falls back to default endpoint.
    """
    import os
    import base64
    import requests
    
    # Default Modal endpoint (same pattern as OpenVoice)
    modal_url = os.environ.get(
        "DEEPFILTER_MODAL_URL",
        "https://kaveno-biz--deepfilter-denoiser-denoise-endpoint.modal.run"  # Default
    )
    
    print(f"[DeepFilter Modal] Using endpoint: {modal_url[:60]}...", flush=True)
    
    try:
        # Read audio file
        with open(in_wav, "rb") as f:
            audio_bytes = f.read()
        
        # Encode as base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"[DeepFilter Modal] Sending {len(audio_bytes)} bytes to Modal...", flush=True)
        
        # Call Modal endpoint
        response = requests.post(
            modal_url,
            json={"audio_base64": audio_base64, "sample_rate": 48000},
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code != 200:
            print(f"[DeepFilter Modal] HTTP {response.status_code}: {response.text[:200]}", flush=True)
            return False
        
        result = response.json()
        
        if not result.get("success"):
            print(f"[DeepFilter Modal] Failed: {result.get('error')}", flush=True)
            return False
        
        # Decode and save output
        output_bytes = base64.b64decode(result["audio_base64"])
        with open(out_wav, "wb") as f:
            f.write(output_bytes)
        
        print(f"[DeepFilter Modal] ✅ Success! {result.get('input_size')} -> {result.get('output_size')} bytes", flush=True)
        return True
        
    except requests.exceptions.Timeout:
        print("[DeepFilter Modal] Request timed out", flush=True)
        return False
    except Exception as e:
        print(f"[DeepFilter Modal] Error: {e}", flush=True)
        return False


def try_elevenlabs_voice_isolator(in_wav: Path, out_wav: Path) -> bool:
    """
    Call ElevenLabs Voice Isolator API.
    Isolates ONLY speech - removes ALL non-speech sounds including laughter, music, ambient.
    
    Uses ELEVENLABS_API_KEY env var.
    Cost: ~1000 characters per minute of audio (~$0.01/min)
    """
    import os
    import requests
    
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("[ElevenLabs Voice Isolator] No ELEVENLABS_API_KEY set", flush=True)
        return False
    
    print("[ElevenLabs Voice Isolator] Calling API...", flush=True)
    
    try:
        # Read audio file
        with open(in_wav, "rb") as f:
            audio_bytes = f.read()
        
        print(f"[ElevenLabs Voice Isolator] Sending {len(audio_bytes)} bytes...", flush=True)
        
        # Call ElevenLabs audio-isolation endpoint
        response = requests.post(
            "https://api.elevenlabs.io/v1/audio-isolation",
            headers={
                "xi-api-key": api_key
            },
            files={
                "audio": ("audio.wav", audio_bytes, "audio/wav")
            },
            timeout=180  # 3 minute timeout for longer files
        )
        
        if response.status_code != 200:
            print(f"[ElevenLabs Voice Isolator] HTTP {response.status_code}: {response.text[:200]}", flush=True)
            return False
        
        # Response is the isolated audio directly
        output_bytes = response.content
        
        if len(output_bytes) < 1000:
            print(f"[ElevenLabs Voice Isolator] Response too small ({len(output_bytes)} bytes), likely error", flush=True)
            return False
        
        with open(out_wav, "wb") as f:
            f.write(output_bytes)
        
        print(f"[ElevenLabs Voice Isolator] ✅ Success! {len(audio_bytes)} -> {len(output_bytes)} bytes", flush=True)
        return True
        
    except requests.exceptions.Timeout:
        print("[ElevenLabs Voice Isolator] Request timed out", flush=True)
        return False
    except Exception as e:
        print(f"[ElevenLabs Voice Isolator] Error: {e}", flush=True)
        return False


def apply_denoise(in_wav: Path, out_wav: Path, strength: float = 0.75, aggressive: bool = False) -> bool:
    """
    Apply best available noise reduction.
    
    Args:
        in_wav: Input WAV file
        out_wav: Output WAV file
        strength: Noise reduction strength for fallback methods (0.0-1.0)
        aggressive: If True, run DeepFilterNet twice + noisereduce for stubborn sounds like laughter
    
    Priority:
    1. DeepFilterNet via Modal (highest quality, if DEEPFILTER_MODAL_URL is set)
    2. noisereduce library (good quality, local)
    3. FFmpeg afftdn (basic, fallback)
    """
    import tempfile
    import shutil
    
    if aggressive:
        # Aggressive mode: DeepFilterNet (pass 1) → DeepFilterNet (pass 2) → noisereduce
        print("[Denoise] Aggressive mode: multiple passes for stubborn noise/laughter", flush=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pass1_out = tmp_path / "pass1.wav"
            pass2_out = tmp_path / "pass2.wav"
            
            # Pass 1: DeepFilterNet
            if try_deepfilter_modal(in_wav, pass1_out):
                print("[Denoise] Pass 1 (DeepFilterNet) complete", flush=True)
                
                # Pass 2: DeepFilterNet again
                if try_deepfilter_modal(pass1_out, pass2_out):
                    print("[Denoise] Pass 2 (DeepFilterNet) complete", flush=True)
                    
                    # Pass 3: noisereduce for any remaining artifacts
                    if try_noisereduce(pass2_out, out_wav, strength=0.5):
                        print("[Denoise] Pass 3 (noisereduce) complete", flush=True)
                        return True
                    else:
                        # Use pass 2 result if noisereduce fails
                        shutil.copy(pass2_out, out_wav)
                        return True
                else:
                    # Use pass 1 result + noisereduce
                    if try_noisereduce(pass1_out, out_wav, strength):
                        return True
                    shutil.copy(pass1_out, out_wav)
                    return True
            else:
                # DeepFilterNet not available, use aggressive noisereduce
                print("[Denoise] DeepFilterNet unavailable, using aggressive noisereduce", flush=True)
                return try_noisereduce(in_wav, out_wav, strength=min(strength + 0.2, 1.0))
    
    # Normal mode: single pass
    # Try DeepFilterNet via Modal first (best quality)
    if try_deepfilter_modal(in_wav, out_wav):
        return True
    
    # Fall back to noisereduce library (good quality, lightweight)
    if try_noisereduce(in_wav, out_wav, strength):
        return True
    
    # Last resort: FFmpeg afftdn filter
    if try_ffmpeg_denoise(in_wav, out_wav):
        return True
    
    return False


def extract_audio(video_path: Path, audio_path: Path, mono: bool = True, sample_rate: int = 48000, format: str = "wav") -> bool:
    """
    Extract audio from video.
    
    Args:
        video_path: Input video file
        audio_path: Output audio file
        mono: If True, convert to mono
        sample_rate: Output sample rate
        format: Output format - "wav" (24-bit PCM) or "mp3" (192k)
    """
    if format == "mp3":
        cmd = [
            FFMPEG_BIN, "-y", "-i", str(video_path),
            "-vn",
            "-ac", "1" if mono else "2",
            "-ar", str(sample_rate),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            str(audio_path)
        ]
    else:
        cmd = [
            FFMPEG_BIN, "-y", "-i", str(video_path),
            "-vn",
            "-ac", "1" if mono else "2",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s24le",
            str(audio_path)
        ]
    code, _, err = run_cmd(cmd)
    if code != 0:
        logger.error(f"Failed to extract audio: {err}")
        return False
    return True


def replace_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Replace audio in video with new audio"""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(output_path)
    ]
    code, _, err = run_cmd(cmd)
    if code != 0:
        logger.error(f"Failed to replace audio: {err}")
        return False
    return True


def enhance_audio(
    video_path: Path,
    output_path: Path,
    remove_laughter: bool = True,
    denoise_strength: float = 0.75,
    apply_deepfilter: bool = True,
    apply_voice_filter: bool = False,
    apply_loudnorm: bool = True,
    progress_callback=None
) -> dict:
    """
    Audio enhancement pipeline with individual toggles.
    
    Pipeline (each step optional):
        1. Extract audio (mono, 48kHz)
        2. noisereduce (removes laughter) - if remove_laughter=True
        3. DeepFilterNet (removes hiss/static) - if apply_deepfilter=True
        4. Voice filter (compressor, gate, limiter) - if apply_voice_filter=True
        5. Two-pass loudnorm (EBU R128 -16 LUFS) - if apply_loudnorm=True
        6. Mux back into video
    
    Args:
        video_path: Input video file
        output_path: Output video file with enhanced audio
        remove_laughter: Apply noisereduce to remove laughter
        denoise_strength: Strength for noisereduce (0.0-1.0)
        apply_deepfilter: Apply DeepFilterNet (removes hiss/static)
        apply_voice_filter: Apply voice filter chain (can sound robotic)
        apply_loudnorm: Apply loudness normalization
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing stats
    """
    stats = {
        "enhanced": False,
        "noisereduce_applied": False,
        "deepfilter_applied": False,
        "voice_filter_applied": False,
        "loudnorm_applied": False,
        "measured_loudness": None
    }
    
    if not Path(video_path).exists():
        logger.error(f"Input video not found: {video_path}")
        return {"enhanced": False, "reason": "Input not found"}
    
    workdir = Path(tempfile.mkdtemp(prefix="veo3_audio_"))
    
    try:
        extracted_wav = workdir / "01_extracted.wav"
        noisereduce_wav = workdir / "02_noisereduce.wav"
        deepfilter_wav = workdir / "03_deepfilter.wav"
        voicefilter_wav = workdir / "04_voicefilter.wav"
        final_wav = workdir / "05_final.wav"
        
        # Step 1: Extract audio (mono, 48kHz)
        if progress_callback:
            progress_callback("Extracting audio...")
        
        if not extract_audio(video_path, extracted_wav):
            shutil.copy(video_path, output_path)
            return {"enhanced": False, "reason": "Failed to extract audio"}
        
        current_wav = extracted_wav
        
        # Step 2: noisereduce (removes laughter)
        if remove_laughter:
            if progress_callback:
                progress_callback("Removing laughter...")
            
            print(f"[Audio Enhancement] noisereduce (strength={denoise_strength})", flush=True)
            if try_noisereduce(current_wav, noisereduce_wav, strength=denoise_strength):
                stats["noisereduce_applied"] = True
                current_wav = noisereduce_wav
            else:
                print("[Audio Enhancement] noisereduce failed, continuing", flush=True)
        
        # Step 3: DeepFilterNet (removes hiss/static)
        if apply_deepfilter:
            if progress_callback:
                progress_callback("Removing hiss (DeepFilterNet)...")
            
            print("[Audio Enhancement] DeepFilterNet (removes hiss/static)", flush=True)
            if try_deepfilter_modal(current_wav, deepfilter_wav):
                stats["deepfilter_applied"] = True
                current_wav = deepfilter_wav
            else:
                print("[Audio Enhancement] DeepFilterNet unavailable, continuing", flush=True)
        
        # Step 4: Voice filter (compressor, gate, limiter)
        if apply_voice_filter:
            if progress_callback:
                progress_callback("Applying voice filter...")
            
            print("[Audio Enhancement] Voice filter (compressor, gate, limiter)", flush=True)
            cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(current_wav),
                "-af", VOICE_FILTER,
                "-c:a", "pcm_s24le",
                str(voicefilter_wav)
            ]
            code, _, err = run_cmd(cmd)
            if code == 0:
                stats["voice_filter_applied"] = True
                current_wav = voicefilter_wav
            else:
                print(f"[Audio Enhancement] Voice filter failed: {err}", flush=True)
        
        # Step 5: Loudness normalization (2-pass)
        if apply_loudnorm:
            if progress_callback:
                progress_callback("Normalizing loudness...")
            
            print("[Audio Enhancement] Loudnorm (-16 LUFS)", flush=True)
            
            # Pass 1: Analyze
            loudnorm_analyze = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
            cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(current_wav),
                "-af", loudnorm_analyze,
                "-f", "null", "-"
            ]
            code, _, err = run_cmd(cmd)
            
            if code == 0:
                try:
                    data = parse_loudnorm_json(err)
                    stats["measured_loudness"] = data.get("input_i")
                    
                    # Pass 2: Apply correction
                    loudnorm_apply = (
                        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:"
                        f"measured_I={data['input_i']}:"
                        f"measured_TP={data['input_tp']}:"
                        f"measured_LRA={data['input_lra']}:"
                        f"measured_thresh={data['input_thresh']}:"
                        f"offset={data['target_offset']}:"
                        "linear=true:print_format=summary"
                    )
                    
                    cmd = [
                        FFMPEG_BIN, "-y",
                        "-i", str(current_wav),
                        "-af", loudnorm_apply,
                        "-c:a", "pcm_s24le",
                        str(final_wav)
                    ]
                    code, _, err = run_cmd(cmd)
                    if code == 0:
                        stats["loudnorm_applied"] = True
                        current_wav = final_wav
                except Exception as e:
                    print(f"[Audio Enhancement] Loudnorm failed: {e}", flush=True)
        
        # Mux back into video
        if progress_callback:
            progress_callback("Finalizing...")
        
        if not replace_audio(video_path, current_wav, output_path):
            shutil.copy(video_path, output_path)
            return {"enhanced": False, "reason": "Failed to replace audio"}
        
        stats["enhanced"] = True
        
        if progress_callback:
            progress_callback("Audio enhancement complete!")
        
        applied = [k for k, v in stats.items() if v is True and k != "enhanced"]
        logger.info(f"[AudioEnhance] Complete. Applied: {applied}")
        
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    
    return stats


def enhance_audio_for_voice_clone(
    video_path: Path,
    output_audio_path: Path,
    denoise: bool = True,
    denoise_strength: float = 0.75,
    progress_callback=None
) -> dict:
    """
    Enhance audio and export as WAV for voice cloning source.
    Same pipeline as enhance_audio but outputs WAV instead of muxing back.
    
    Args:
        video_path: Input video file
        output_audio_path: Output WAV file (enhanced)
        denoise: Whether to apply noise reduction
        denoise_strength: Noise reduction strength (0.0-1.0)
        progress_callback: Optional callback
    
    Returns:
        dict with processing stats
    """
    stats = {
        "enhanced": False,
        "denoise_applied": False,
        "voice_filter_applied": False,
        "loudnorm_applied": False
    }
    
    if not Path(video_path).exists():
        logger.error(f"Input video not found: {video_path}")
        return {"enhanced": False, "reason": "Input not found"}
    
    workdir = Path(tempfile.mkdtemp(prefix="veo3_voice_"))
    
    try:
        extracted_wav = workdir / "01_extracted.wav"
        denoised_wav = workdir / "02_denoised.wav"
        processed_wav = workdir / "03_processed.wav"
        
        # Step 1: Extract audio
        if progress_callback:
            progress_callback("Extracting audio...")
        
        if not extract_audio(video_path, extracted_wav):
            return {"enhanced": False, "reason": "Failed to extract audio"}
        
        # Step 2: Noise reduction
        if progress_callback:
            progress_callback("Applying noise reduction...")
        
        if denoise:
            stats["denoise_applied"] = apply_denoise(extracted_wav, denoised_wav, denoise_strength)
        
        src_for_processing = denoised_wav if stats["denoise_applied"] else extracted_wav
        
        # Step 3: Voice stabilization chain
        if progress_callback:
            progress_callback("Applying voice enhancement...")
        
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(src_for_processing),
            "-af", VOICE_FILTER,
            "-c:a", "pcm_s24le",
            str(processed_wav)
        ]
        code, _, err = run_cmd(cmd)
        if code != 0:
            logger.error(f"Voice filter failed: {err}")
            # Fallback: just extract without enhancement
            shutil.copy(extracted_wav, output_audio_path)
            return {"enhanced": False, "reason": "Voice filter failed"}
        
        stats["voice_filter_applied"] = True
        
        # Step 4: Two-pass loudnorm directly to output
        if progress_callback:
            progress_callback("Normalizing loudness...")
        
        # Pass 1: Analyze
        loudnorm_analyze = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(processed_wav),
            "-af", loudnorm_analyze,
            "-f", "null", "-"
        ]
        code, _, err = run_cmd(cmd)
        
        if code != 0:
            # Fallback: use processed without loudnorm
            shutil.copy(processed_wav, output_audio_path)
            stats["enhanced"] = True
            return stats
        
        try:
            data = parse_loudnorm_json(err)
        except Exception:
            data = {
                "input_i": "-24", "input_tp": "-2", "input_lra": "7",
                "input_thresh": "-34", "target_offset": "0"
            }
        
        # Pass 2: Apply to output
        loudnorm_apply = (
            f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:"
            f"measured_I={data['input_i']}:"
            f"measured_TP={data['input_tp']}:"
            f"measured_LRA={data['input_lra']}:"
            f"measured_thresh={data['input_thresh']}:"
            f"offset={data['target_offset']}:"
            "linear=true"
        )
        
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(processed_wav),
            "-af", loudnorm_apply,
            "-c:a", "pcm_s24le",
            str(output_audio_path)
        ]
        code, _, err = run_cmd(cmd)
        
        if code != 0:
            shutil.copy(processed_wav, output_audio_path)
        else:
            stats["loudnorm_applied"] = True
        
        stats["enhanced"] = True
        
        if progress_callback:
            progress_callback("Audio enhancement complete!")
        
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    
    return stats


def concatenate_audio_files(audio_files: list, output_path: Path, enhance: bool = True) -> bool:
    """
    Concatenate multiple audio files into one.
    Used for combining multiple clips' audio as voice reference.
    
    Args:
        audio_files: List of audio file paths
        output_path: Output concatenated audio file
        enhance: Whether to apply voice enhancement to result
    
    Returns:
        True if successful
    """
    if not audio_files:
        return False
    
    if len(audio_files) == 1:
        if enhance:
            # Enhance single file
            workdir = Path(tempfile.mkdtemp(prefix="veo3_concat_"))
            try:
                temp_out = workdir / "enhanced.wav"
                # Run voice filter + loudnorm on the single file
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-i", str(audio_files[0]),
                    "-af", VOICE_FILTER,
                    "-c:a", "pcm_s24le",
                    str(temp_out)
                ]
                code, _, _ = run_cmd(cmd)
                if code == 0:
                    shutil.copy(temp_out, output_path)
                    return True
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        
        shutil.copy(audio_files[0], output_path)
        return True
    
    try:
        import numpy as np
        import soundfile as sf
        
        all_data = []
        target_rate = None
        
        for audio_file in audio_files:
            if not Path(audio_file).exists():
                continue
            data, rate = sf.read(str(audio_file))
            
            if target_rate is None:
                target_rate = rate
            
            # Convert to mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            all_data.append(data)
        
        if not all_data:
            return False
        
        # Concatenate
        combined = np.concatenate(all_data)
        
        if enhance:
            # Save temp, enhance, then copy to output
            workdir = Path(tempfile.mkdtemp(prefix="veo3_concat_"))
            try:
                temp_raw = workdir / "raw.wav"
                temp_enhanced = workdir / "enhanced.wav"
                
                sf.write(str(temp_raw), combined, target_rate)
                
                # Apply voice filter
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-i", str(temp_raw),
                    "-af", VOICE_FILTER,
                    "-c:a", "pcm_s24le",
                    str(temp_enhanced)
                ]
                code, _, _ = run_cmd(cmd)
                
                if code == 0:
                    shutil.copy(temp_enhanced, output_path)
                else:
                    sf.write(str(output_path), combined, target_rate)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        else:
            sf.write(str(output_path), combined, target_rate)
        
        logger.info(f"Concatenated {len(all_data)} audio files ({len(combined)/target_rate:.1f}s total)")
        return True
        
    except ImportError:
        # Fallback: use ffmpeg concat
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
                concat_list = f.name
            
            cmd = [
                FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                str(output_path)
            ]
            code, _, err = run_cmd(cmd)
            os.unlink(concat_list)
            
            if code != 0:
                logger.error(f"FFmpeg concat failed: {err}")
                return False
            return True
        except Exception as e:
            logger.error(f"FFmpeg fallback failed: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to concatenate audio: {e}")
        return False


# Legacy function names for compatibility
def enhance_audio_basic(video_path, output_path, **kwargs):
    """Legacy wrapper - redirects to enhance_audio without denoise"""
    return enhance_audio(video_path, output_path, denoise=False)


def enhance_audio_professional(video_path, output_path, mode="voice", **kwargs):
    """Legacy wrapper - redirects to enhance_audio with denoise"""
    return enhance_audio(video_path, output_path, denoise=True, denoise_strength=0.75)


def export_audio_only(video_path: Path, output_path: Path, enhance: bool = True) -> bool:
    """Export audio from video as WAV file."""
    if enhance:
        result = enhance_audio_for_voice_clone(video_path, output_path)
        return result.get("enhanced", False)
    return extract_audio(video_path, output_path)


def import_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Import external audio into video."""
    return replace_audio(video_path, audio_path, output_path)