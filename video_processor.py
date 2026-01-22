"""
Video Processing Module for Final Export
- Trim frames from start/end of clips
- Concatenate multiple clips
- Optional Voice Activity Detection (VAD) to remove silence
"""

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# FFmpeg binary path (will use system PATH by default)
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")


def run(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=600)  # 10 minute timeout
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        p.kill()
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def ffprobe_json(path: Path) -> dict:
    """Get video metadata as JSON."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path)
    ]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {err}")
    return json.loads(out)


def get_fps(info: dict) -> float:
    """Extract FPS from ffprobe info."""
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise RuntimeError("No video stream found.")
    fr = vstreams[0].get("avg_frame_rate") or vstreams[0].get("r_frame_rate") or "30/1"
    num, den = fr.split("/")
    num, den = float(num), float(den)
    return num / den if den != 0 else 30.0


def get_duration(info: dict) -> float:
    """Extract duration from ffprobe info."""
    fmt = info.get("format", {})
    if "duration" in fmt and fmt["duration"] not in ("N/A", None):
        return float(fmt["duration"])
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and "duration" in s and s["duration"] not in ("N/A", None):
            return float(s["duration"])
    return 8.0  # Default assumption for Veo clips


def check_vad_available() -> bool:
    """Check if Silero VAD dependencies are available."""
    try:
        import torch
        import numpy as np
        return True
    except ImportError:
        return False


def load_vad_model():
    """Load Silero VAD model."""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    return model, utils


def detect_speech_segments(
    video_path: Path,
    threshold: float = 0.5,
    min_silence_duration: float = 1.0,
    padding_before: float = 0.1,
    padding_after: float = 0.2
) -> List[Tuple[float, float]]:
    """
    Detect speech segments using Silero VAD.
    Returns list of (start_time, end_time) tuples for segments WITH speech.
    """
    import torch
    import numpy as np
    import wave
    
    # Extract audio from video
    with tempfile.TemporaryDirectory() as td:
        audio_path = Path(td) / "audio.wav"
        cmd = [
            FFMPEG_BIN, "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ]
        code, _, err = run(cmd)
        if code != 0:
            raise RuntimeError(f"Failed to extract audio: {err}")
        
        # Load VAD model
        model, utils = load_vad_model()
        (get_speech_timestamps, *_) = utils
        
        # Read audio from WAV file
        with wave.open(str(audio_path), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Convert to torch tensor
        wav = torch.from_numpy(audio_np)
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=100,
            min_silence_duration_ms=int(min_silence_duration * 1000),
            window_size_samples=512,
            speech_pad_ms=0
        )
        
        # Convert to seconds and add padding
        speech_segments = []
        for ts in speech_timestamps:
            start = max(0, ts['start'] / 16000 - padding_before)
            end = ts['end'] / 16000 + padding_after
            speech_segments.append((start, end))
        
        return speech_segments


def apply_vad(
    src: Path,
    out: Path,
    threshold: float = 0.5,
    min_gap_duration: float = 1.0,
    padding_before: float = 0.1,
    padding_after: float = 0.2,
    progress_callback=None
) -> dict:
    """
    Remove non-dialogue segments using Voice Activity Detection.
    Returns stats about the processing.
    """
    info = ffprobe_json(src)
    original_duration = get_duration(info)
    
    if progress_callback:
        progress_callback("Analyzing audio for speech...")
    
    # Detect speech segments
    speech_segments = detect_speech_segments(
        src,
        threshold=threshold,
        min_silence_duration=min_gap_duration,
        padding_before=padding_before,
        padding_after=padding_after
    )
    
    if not speech_segments:
        # No speech detected - just copy the file
        logger.warning("No speech detected in video")
        import shutil
        shutil.copy(src, out)
        return {
            "original_duration": original_duration,
            "final_duration": original_duration,
            "segments_found": 0,
            "silence_removed": 0
        }
    
    # Merge overlapping segments
    merged = []
    for start, end in sorted(speech_segments):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    total_speech = sum(end - start for start, end in merged)
    
    if progress_callback:
        progress_callback(f"Found {len(merged)} speech segments ({total_speech:.1f}s)")
    
    # Extract and concatenate speech segments
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        segment_files = []
        
        for idx, (start, end) in enumerate(merged, 1):
            if progress_callback:
                progress_callback(f"Extracting segment {idx}/{len(merged)}...")
            
            segment_file = temp_path / f"segment_{idx:04d}.mp4"
            duration_seg = end - start
            
            cmd = [
                FFMPEG_BIN, "-y",
                "-ss", f"{start:.6f}",
                "-i", str(src),
                "-t", f"{duration_seg:.6f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(segment_file)
            ]
            
            code, _, err = run(cmd)
            if code != 0:
                raise RuntimeError(f"Failed to extract segment {idx}: {err}")
            
            segment_files.append(segment_file)
        
        if progress_callback:
            progress_callback("Joining segments...")
        
        # Concatenate all segments
        concat_file = temp_path / "concat_list.txt"
        with concat_file.open("w", encoding="utf-8") as f:
            for seg_file in segment_files:
                f.write(f"file {shlex.quote(str(seg_file))}\n")
        
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(out)
        ]
        
        code, _, err = run(cmd)
        if code != 0:
            raise RuntimeError(f"Failed to concatenate segments: {err}")
    
    # Get final duration
    final_info = ffprobe_json(out)
    final_duration = get_duration(final_info)
    
    return {
        "original_duration": original_duration,
        "final_duration": final_duration,
        "segments_found": len(merged),
        "silence_removed": original_duration - final_duration
    }


def trim_video(
    src: Path,
    out: Path,
    frames_start: int = 0,
    frames_end: int = 0
) -> None:
    """Trim frames from start and end of video.
    
    Always re-encodes to ensure frame-accurate cutting.
    Uses ultrafast preset and memory-optimized settings for Render.
    """
    print(f"[VideoProcessor] trim_video: {src} -> {out}")
    print(f"[VideoProcessor]   frames_start={frames_start}, frames_end={frames_end}")
    
    if not src.exists():
        raise RuntimeError(f"Source file does not exist: {src}")
    
    info = ffprobe_json(src)
    fps = get_fps(info)
    duration = get_duration(info)
    
    print(f"[VideoProcessor]   fps={fps}, duration={duration}")
    
    cut_start_seconds = frames_start / fps
    cut_end_seconds = frames_end / fps
    target_duration = max(0.1, duration - cut_start_seconds - cut_end_seconds)
    
    print(f"[VideoProcessor]   cut_start={cut_start_seconds:.6f}s, cut_end={cut_end_seconds:.6f}s, target_duration={target_duration:.6f}s")
    
    # Re-encode with memory-optimized settings for Render (512MB limit)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{cut_start_seconds:.6f}",
        "-i", str(src),
        "-t", f"{target_duration:.6f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",  # ultrafast uses less memory
        "-pix_fmt", "yuv420p",
        "-threads", "1",  # Single thread to limit memory
        "-c:a", "aac", "-b:a", "128k",
        "-max_muxing_queue_size", "512",  # Limit muxing buffer
        "-avoid_negative_ts", "make_zero",
        str(out)
    ]
    
    print(f"[VideoProcessor]   Running ffmpeg (ultrafast, low memory)...")
    code, _, err = run(cmd)
    if code != 0:
        print(f"[VideoProcessor]   ERROR: {err}")
        raise RuntimeError(f"Failed to trim video: {err}")
    print(f"[VideoProcessor]   trim_video completed")


def concat_videos(files: List[Path], output: Path) -> None:
    """Concatenate multiple videos into one."""
    print(f"[VideoProcessor] concat_videos: {len(files)} files -> {output}")
    
    with tempfile.TemporaryDirectory() as td:
        listfile = Path(td) / "inputs.txt"
        with listfile.open("w", encoding="utf-8") as f:
            for p in files:
                f.write(f"file {shlex.quote(str(p))}\n")
        
        # Try stream copy first (faster)
        cmd_copy = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-c", "copy",
            str(output)
        ]
        print(f"[VideoProcessor]   Trying stream copy...")
        code, _, _ = run(cmd_copy)
        if code == 0:
            print(f"[VideoProcessor]   concat_videos completed (stream copy)")
            return
        
        # Fall back to re-encoding
        print(f"[VideoProcessor]   Stream copy failed, re-encoding...")
        cmd_re = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-threads", "1",  # Limit threads to reduce memory
            "-c:a", "aac", "-b:a", "128k",
            "-max_muxing_queue_size", "1024",
            str(output)
        ]
        code, _, err = run(cmd_re)
        if code != 0:
            print(f"[VideoProcessor]   ERROR: {err}")
            raise RuntimeError(f"Failed to concatenate videos: {err}")
        print(f"[VideoProcessor]   concat_videos completed (re-encoded)")


def export_final_video(
    clip_info: List[dict],
    output_path: Path,
    frames_to_cut_start: int = 0,   # Default: no start trim
    frames_to_cut_end: int = 7,     # Default: trim 7 frames from end (removes morph artifacts)
    remove_silence: bool = False,
    vad_threshold: float = 0.5,
    vad_min_gap: float = 1.0,
    vad_pad_before: float = 0.1,
    vad_pad_after: float = 0.2,
    progress_callback=None
) -> dict:
    """
    Main export function: trim, concat, and optionally apply VAD.
    
    Args:
        clip_info: List of dicts with keys:
            - path: Path to video file
            - clip_index: Index of clip
            - skip_start_trim: Whether to skip trimming start frames
        output_path: Where to save the final video
        frames_to_cut_start: Frames to trim from start (default 0)
        frames_to_cut_end: Frames to trim from end (default 7 - removes morph artifacts)
        remove_silence: Whether to apply VAD
        vad_threshold: VAD sensitivity (0-1, higher = more aggressive)
        vad_min_gap: Minimum silence duration to remove (seconds)
        vad_pad_before: Padding before speech (seconds)
        vad_pad_after: Padding after speech (seconds)
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing stats
    """
    print(f"[VideoProcessor] export_final_video called")
    print(f"[VideoProcessor] clip_info count: {len(clip_info)}")
    print(f"[VideoProcessor] output_path: {output_path}")
    
    if not clip_info:
        raise ValueError("No clips provided")
    
    # Check if any trimming is needed
    needs_trimming = frames_to_cut_start > 0 or frames_to_cut_end > 0
    
    stats = {
        "clips_processed": len(clip_info),
        "frames_trimmed_start": frames_to_cut_start,
        "frames_trimmed_end": frames_to_cut_end,
        "vad_applied": remove_silence,
        "clips_with_start_trim_skipped": sum(1 for c in clip_info if c.get("skip_start_trim", False)),
        "pre_trimmed": not needs_trimming
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Determine which files to concatenate
        if needs_trimming:
            # Additional trimming requested - trim each clip
            print(f"[VideoProcessor] Additional trimming requested: start={frames_to_cut_start}, end={frames_to_cut_end}")
            files_to_concat = []
            
            for idx, info in enumerate(clip_info, 1):
                clip_path = info["path"]
                skip_start = info.get("skip_start_trim", False)
                
                if progress_callback:
                    progress_callback(f"Trimming clip {idx}/{len(clip_info)}...")
                
                trimmed_file = temp_path / f"trimmed_{idx:04d}.mp4"
                actual_start_trim = 0 if skip_start else frames_to_cut_start
                
                logger.info(f"Clip {info.get('clip_index', idx)}: start_trim={actual_start_trim}, end_trim={frames_to_cut_end}")
                
                trim_video(clip_path, trimmed_file, actual_start_trim, frames_to_cut_end)
                files_to_concat.append(trimmed_file)
        else:
            # Clips are pre-trimmed - just use them directly (FAST PATH)
            print(f"[VideoProcessor] Using pre-trimmed clips (fast concat)")
            files_to_concat = [Path(info["path"]) for info in clip_info]
        
        # Concatenate
        if progress_callback:
            progress_callback("Finalizing video...")
        
        if remove_silence:
            concat_output = temp_path / "concatenated.mp4"
        else:
            concat_output = output_path
        
        concat_videos(files_to_concat, concat_output)
        
        # Step 3: Apply VAD (if enabled)
        if remove_silence:
            if not check_vad_available():
                raise RuntimeError(
                    "VAD requires torch and numpy. "
                    "Install with: pip install torch numpy"
                )
            
            if progress_callback:
                progress_callback("Applying Voice Activity Detection...")
            
            vad_stats = apply_vad(
                concat_output,
                output_path,
                threshold=vad_threshold,
                min_gap_duration=vad_min_gap,
                padding_before=vad_pad_before,
                padding_after=vad_pad_after,
                progress_callback=progress_callback
            )
            stats.update(vad_stats)
        else:
            # Get duration of final video
            info = ffprobe_json(concat_output)
            stats["final_duration"] = get_duration(info)
    
    if progress_callback:
        progress_callback("Export complete!")
    
    return stats