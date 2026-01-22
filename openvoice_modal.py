"""
OpenVoice Voice Conversion API on Modal
Deploy: modal deploy openvoice_modal.py
Test locally: modal serve openvoice_modal.py
"""

import modal
import os

# Define the Modal app
app = modal.App("openvoice-v2")

# Create image with minimal dependencies for voice conversion only
openvoice_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", 
        "ffmpeg", 
        "libsndfile1",
        "wget",
        # Required for PyAV build
        "pkg-config",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libswscale-dev",
        "libswresample-dev",
    )
    .pip_install(
        # Core dependencies
        "torch==2.1.0",
        "torchaudio==2.1.0", 
        "numpy<2",
        "librosa==0.9.2",
        "scipy",
        "httpx",
        "pydub",
        "soundfile",
        "fastapi",  # Required for web endpoints
        # PyAV for faster-whisper (use binary wheel)
        "av>=11.0.0",
        # Pitch manipulation
        "praat-parselmouth",
        # OpenVoice specific deps for tone color converter
        "wavmark==0.0.3",
        "eng_to_ipa==0.0.2",
        "inflect==7.0.0",
        "unidecode==1.3.7",
        "pypinyin==0.50.0",
        "cn2an==0.5.22",
        "jieba==0.42.1",
        "langid==1.1.6",
        # Whisper for speaker embedding extraction  
        "openai-whisper",
        "faster-whisper>=1.0.0",
        "whisper-timestamped",  # Required by se_extractor
    )
    .run_commands(
        # Clone OpenVoice repo (we'll add to path, not install)
        "git clone https://github.com/myshell-ai/OpenVoice.git /openvoice",
    )
)

# Volume for model checkpoints (persistent storage)
volume = modal.Volume.from_name("openvoice-checkpoints", create_if_missing=True)
CHECKPOINT_PATH = "/checkpoints"


def download_checkpoints():
    """Download OpenVoice v2 checkpoints if not present"""
    import subprocess
    
    converter_config = f"{CHECKPOINT_PATH}/checkpoints_v2/converter/config.json"
    
    if os.path.exists(converter_config):
        print("✅ Checkpoints already present")
        return
    
    print("📥 Downloading OpenVoice v2 checkpoints...")
    os.makedirs(f"{CHECKPOINT_PATH}/checkpoints_v2/converter", exist_ok=True)
    
    # Download converter checkpoint from HuggingFace
    subprocess.run([
        "wget", "-q", "--show-progress",
        "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth",
        "-O", f"{CHECKPOINT_PATH}/checkpoints_v2/converter/checkpoint.pth"
    ], check=True)
    
    subprocess.run([
        "wget", "-q",
        "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json",
        "-O", f"{CHECKPOINT_PATH}/checkpoints_v2/converter/config.json"
    ], check=True)
    
    print("✅ Checkpoints downloaded!")


@app.cls(
    image=openvoice_image,
    gpu="T4",  # Cheapest GPU, good enough for OpenVoice
    timeout=600,  # 10 minutes for longer audio
    container_idle_timeout=120,  # Keep warm for 2 min
    volumes={CHECKPOINT_PATH: volume},
)
class OpenVoiceConverter:
    """OpenVoice v2 Voice Conversion Service"""
    
    @modal.enter()
    def load_model(self):
        """Load model once when container starts"""
        import sys
        sys.path.insert(0, "/openvoice")
        
        # Ensure checkpoints exist
        download_checkpoints()
        volume.commit()
        
        from openvoice.api import ToneColorConverter
        
        ckpt_converter = f"{CHECKPOINT_PATH}/checkpoints_v2/converter"
        
        self.device = "cuda:0"
        self.converter = ToneColorConverter(
            f'{ckpt_converter}/config.json', 
            device=self.device
        )
        self.converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        print("✅ OpenVoice model loaded!")
    
    def normalize_pitch(self, audio_path: str, output_path: str, compression: float = 0.5):
        """
        Compress pitch range to reduce variation while keeping natural inflection.
        
        Args:
            audio_path: Input audio file path
            output_path: Output audio file path  
            compression: 0.0 = no change, 1.0 = completely flat (monotone)
                        0.5 = reduce variation by 50% (recommended)
        
        Returns:
            True if successful
        """
        import parselmouth
        from parselmouth.praat import call
        import numpy as np
        import soundfile as sf
        
        try:
            # Load audio
            sound = parselmouth.Sound(audio_path)
            
            # Extract pitch
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            # Get pitch values
            pitch_values = pitch.selected_array['frequency']
            voiced_frames = pitch_values > 0
            
            if not np.any(voiced_frames):
                print("[PitchNorm] No voiced frames detected, skipping normalization")
                # Just copy the file
                import shutil
                shutil.copy(audio_path, output_path)
                return True
            
            # Calculate median pitch (more robust than mean)
            median_pitch = np.median(pitch_values[voiced_frames])
            
            # Create manipulation object
            manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
            pitch_tier = call(manipulation, "Extract pitch tier")
            
            # Get all pitch points
            num_points = call(pitch_tier, "Get number of points")
            
            if num_points > 0:
                # Compress each pitch point towards median
                for i in range(1, num_points + 1):
                    time = call(pitch_tier, "Get time from index", i)
                    value = call(pitch_tier, "Get value at index", i)
                    
                    if value > 0:
                        # Compress towards median
                        new_value = median_pitch + (value - median_pitch) * (1 - compression)
                        call(pitch_tier, "Remove point", i)
                        call(pitch_tier, "Add point", time, new_value)
                
                # Replace pitch tier in manipulation
                call([manipulation, pitch_tier], "Replace pitch tier")
                
                # Resynthesize
                new_sound = call(manipulation, "Get resynthesis (overlap-add)")
                
                # Save output
                call(new_sound, "Save as WAV file", output_path)
                print(f"[PitchNorm] Compressed pitch by {compression*100:.0f}% (median: {median_pitch:.1f}Hz)")
            else:
                # No pitch points, just copy
                import shutil
                shutil.copy(audio_path, output_path)
            
            return True
            
        except Exception as e:
            print(f"[PitchNorm] Error: {e}, using original audio")
            import shutil
            shutil.copy(audio_path, output_path)
            return True
    
    def convert_voice(
        self, 
        source_audio_bytes: bytes, 
        target_voice_bytes: bytes,
        tau: float = 0.3,
        pitch_normalize: float = 0.0
    ) -> bytes:
        """
        Convert source audio to sound like target voice.
        
        Args:
            source_audio_bytes: Audio to convert (the speech content)
            target_voice_bytes: Voice sample to clone (the voice style)
            tau: Temperature for conversion (0.1-0.5, lower = more similar)
            pitch_normalize: Pitch compression (0.0 = off, 0.5 = moderate, 1.0 = monotone)
        
        Returns:
            Converted audio as bytes (WAV format)
        """
        import sys
        sys.path.insert(0, "/openvoice")
        
        import tempfile
        import uuid
        from openvoice import se_extractor
        
        work_dir = tempfile.mkdtemp()
        source_path = f"{work_dir}/source_{uuid.uuid4().hex}.wav"
        source_normalized = f"{work_dir}/source_norm_{uuid.uuid4().hex}.wav"
        target_path = f"{work_dir}/target_{uuid.uuid4().hex}.wav"
        output_path = f"{work_dir}/output_{uuid.uuid4().hex}.wav"
        
        # Write input files
        with open(source_path, "wb") as f:
            f.write(source_audio_bytes)
        with open(target_path, "wb") as f:
            f.write(target_voice_bytes)
        
        print(f"Processing: source={len(source_audio_bytes)}B, target={len(target_voice_bytes)}B, pitch_norm={pitch_normalize}")
        
        # Apply pitch normalization if requested
        if pitch_normalize > 0:
            print(f"[PitchNorm] Applying {pitch_normalize*100:.0f}% compression...")
            self.normalize_pitch(source_path, source_normalized, compression=pitch_normalize)
            source_for_conversion = source_normalized
        else:
            source_for_conversion = source_path
        
        # Extract speaker embeddings
        target_se, _ = se_extractor.get_se(
            target_path, 
            self.converter, 
            vad=True
        )
        source_se, _ = se_extractor.get_se(
            source_for_conversion, 
            self.converter, 
            vad=True
        )
        
        # Convert voice
        self.converter.convert(
            audio_src_path=source_for_conversion,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            tau=tau,
        )
        
        # Read and return output
        with open(output_path, "rb") as f:
            output_bytes = f.read()
        
        print(f"✅ Conversion complete: {len(output_bytes)}B output")
        return output_bytes
    
    @modal.web_endpoint(method="POST", docs=True)
    async def convert_endpoint(self, request: dict):
        """
        HTTP endpoint for voice conversion.
        
        Accepts either URLs or base64 encoded audio:
        
        Option 1 - URLs:
        - source_url: URL to audio file to convert
        - target_url: URL to voice sample to clone
        
        Option 2 - Base64:
        - source_base64: Base64 encoded source audio
        - target_base64: Base64 encoded target voice
        
        Optional:
        - tau: conversion temperature, default 0.3 (0.1-0.5, lower = more similar)
        - pitch_normalize: pitch compression, default 0.0 (0.0 = off, 0.5 = moderate, 1.0 = flat)
        
        Returns JSON with:
        - success: bool
        - output_base64: Base64 encoded WAV audio
        - output_size: size in bytes
        """
        import httpx
        import base64
        
        tau = request.get("tau", 0.3)
        pitch_normalize = request.get("pitch_normalize", 0.0)
        
        # Check for base64 input (preferred - no download needed)
        source_b64 = request.get("source_base64")
        target_b64 = request.get("target_base64")
        
        if source_b64 and target_b64:
            try:
                source_bytes = base64.b64decode(source_b64)
                target_bytes = base64.b64decode(target_b64)
                print(f"Received base64: source={len(source_bytes)}B, target={len(target_bytes)}B")
            except Exception as e:
                return {"success": False, "error": f"Invalid base64: {str(e)}"}
        else:
            # Fall back to URL download
            source_url = request.get("source_url")
            target_url = request.get("target_url")
            
            if not source_url or not target_url:
                return {
                    "success": False, 
                    "error": "Provide either (source_base64, target_base64) or (source_url, target_url)"
                }
            
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    print(f"Downloading source: {source_url}")
                    source_resp = await client.get(source_url)
                    source_resp.raise_for_status()
                    source_bytes = source_resp.content
                    
                    print(f"Downloading target: {target_url}")
                    target_resp = await client.get(target_url)
                    target_resp.raise_for_status()
                    target_bytes = target_resp.content
            except httpx.HTTPError as e:
                return {"success": False, "error": f"Failed to download audio: {str(e)}"}
        
        try:
            # Convert
            output_bytes = self.convert_voice(source_bytes, target_bytes, tau, pitch_normalize)
            
            # Return as base64
            output_b64 = base64.b64encode(output_bytes).decode('utf-8')
            
            return {
                "success": True,
                "output_base64": output_b64,
                "output_size": len(output_bytes)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Conversion failed: {str(e)}"}
    
    @modal.web_endpoint(method="GET", docs=True)
    async def health(self):
        """Health check endpoint"""
        return {
            "status": "ok",
            "model": "OpenVoice v2",
            "device": self.device
        }


# CLI test
@app.local_entrypoint()
def main():
    """Test the deployment locally"""
    print("Testing OpenVoice Modal deployment...")
    
    # Create a simple test
    converter = OpenVoiceConverter()
    
    # Check health
    result = converter.health.remote()
    print(f"Health check: {result}")
    
    print("\n✅ Deployment ready!")
    print("Endpoints:")
    print("  - POST /convert_endpoint - Voice conversion")
    print("  - GET /health - Health check")