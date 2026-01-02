"""
DeepFilterNet Audio Denoiser - Modal Deployment
High-quality AI-powered noise reduction for voice audio.

Deploy: modal deploy modal_deepfilter.py
Test: modal run modal_deepfilter.py

Cost: ~$0.001-0.002 per audio file (CPU-only, very fast)
"""

import modal

# Create Modal app
app = modal.App("deepfilter-denoiser")

# Build image with DeepFilterNet dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")  # git needed by deepfilternet
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0", 
        "deepfilternet>=0.5.6",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "fastapi",  # Required for web endpoints
    )
)


@app.function(
    image=image,
    timeout=300,
    cpu=2.0,  # CPU is sufficient, DeepFilterNet is fast
    memory=4096,  # 4GB RAM
)
def denoise_audio(audio_bytes: bytes, sample_rate: int = 48000) -> bytes:
    """
    Denoise audio using DeepFilterNet.
    
    Args:
        audio_bytes: Raw audio bytes (WAV format)
        sample_rate: Expected sample rate (default 48kHz)
    
    Returns:
        Denoised audio bytes (WAV format)
    """
    import io
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np
    from df.enhance import init_df, enhance
    
    print(f"[DeepFilter] Received {len(audio_bytes)} bytes")
    
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(audio_io)
    print(f"[DeepFilter] Loaded audio: {wav.shape}, sr={sr}")
    
    # Initialize DeepFilterNet model
    model, state, _ = init_df()
    
    # Get model's expected sample rate
    sr_attr = getattr(state, "sr", None)
    if callable(sr_attr):
        target_sr = sr_attr()
    elif sr_attr is not None:
        target_sr = sr_attr
    else:
        target_sr = 48000
    
    print(f"[DeepFilter] Model expects sr={target_sr}")
    
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
        print(f"[DeepFilter] Resampled to {sr}")
    
    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        print("[DeepFilter] Converted to mono")
    
    # Enhance (denoise)
    print("[DeepFilter] Running enhancement...")
    enhanced = enhance(model, state, wav)
    
    # Convert to numpy
    if isinstance(enhanced, torch.Tensor):
        enhanced_np = enhanced.detach().cpu().numpy()
    else:
        enhanced_np = np.asarray(enhanced)
    
    enhanced_np = np.squeeze(enhanced_np)
    print(f"[DeepFilter] Enhanced shape: {enhanced_np.shape}")
    
    # Write to bytes
    output_io = io.BytesIO()
    sf.write(output_io, enhanced_np.astype(np.float32), sr, format='WAV', subtype='PCM_24')
    output_bytes = output_io.getvalue()
    
    print(f"[DeepFilter] Output: {len(output_bytes)} bytes")
    return output_bytes


@app.local_entrypoint()
def main():
    """Test the denoiser locally"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: modal run modal_deepfilter.py <input.wav> [output.wav]")
        print("\nTest with a sample audio file to verify deployment.")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "denoised_output.wav"
    
    print(f"Reading: {input_path}")
    with open(input_path, "rb") as f:
        audio_bytes = f.read()
    
    print(f"Sending {len(audio_bytes)} bytes to Modal...")
    result = denoise_audio.remote(audio_bytes)
    
    print(f"Writing: {output_path}")
    with open(output_path, "wb") as f:
        f.write(result)
    
    print(f"Done! Denoised audio saved to {output_path}")


# Web endpoint for HTTP access
@app.function(
    image=image,
    timeout=300,
    cpu=2.0,
    memory=4096,
)
@modal.web_endpoint(method="POST")
def denoise_endpoint(request: dict) -> dict:
    """
    HTTP endpoint for denoising audio.
    
    POST /denoise_endpoint
    Body: {"audio_base64": "<base64-encoded-wav>", "sample_rate": 48000}
    
    Returns: {"audio_base64": "<base64-encoded-denoised-wav>", "success": true}
    """
    import base64
    import io
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np
    from df.enhance import init_df, enhance
    
    try:
        # Extract from request body
        audio_base64 = request.get("audio_base64")
        sample_rate = request.get("sample_rate", 48000)
        
        if not audio_base64:
            return {"success": False, "error": "audio_base64 is required"}
        
        # Decode input
        audio_bytes = base64.b64decode(audio_base64)
        print(f"[DeepFilter] Received {len(audio_bytes)} bytes via HTTP")
        
        # Load audio
        audio_io = io.BytesIO(audio_bytes)
        wav, sr = torchaudio.load(audio_io)
        
        # Initialize model
        model, state, _ = init_df()
        
        # Get target sample rate
        sr_attr = getattr(state, "sr", None)
        target_sr = sr_attr() if callable(sr_attr) else (sr_attr or 48000)
        
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
        
        # Write to bytes
        output_io = io.BytesIO()
        sf.write(output_io, enhanced_np.astype(np.float32), sr, format='WAV', subtype='PCM_24')
        output_bytes = output_io.getvalue()
        
        # Encode output
        output_base64 = base64.b64encode(output_bytes).decode('utf-8')
        
        return {
            "success": True,
            "audio_base64": output_base64,
            "input_size": len(audio_bytes),
            "output_size": len(output_bytes)
        }
        
    except Exception as e:
        print(f"[DeepFilter] Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }