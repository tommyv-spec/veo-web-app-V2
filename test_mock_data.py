#!/usr/bin/env python3
"""
Test script - creates mock job data to test the UI without real API calls.

Run this AFTER starting the server:
    python main.py &
    python test_mock_data.py
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


def create_test_images():
    """Create dummy test images"""
    test_dir = Path("uploads") / "test-job"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple 1x1 pixel PNGs
    for i in range(5):
        img_path = test_dir / f"frame_{i:02d}.png"
        # Minimal valid PNG (1x1 red pixel)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        with open(img_path, "wb") as f:
            f.write(png_data)
    
    print(f"✓ Created test images in {test_dir}")
    return str(test_dir)


def create_mock_job():
    """Create a mock job directly in the database"""
    from models import init_db, get_db, Job, Clip
    from config import JobStatus, ClipStatus
    
    init_db()
    
    job_id = str(uuid.uuid4())
    images_dir = create_test_images()
    output_dir = f"outputs/{job_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dialogue_lines = [
        {"id": 1, "text": "Questo potrebbe costarmi caro, ma devo dirlo."},
        {"id": 2, "text": "Quello che hai è un'infiammazione cronica."},
        {"id": 3, "text": "Andiamo dal medico con una lista di disturbi."},
        {"id": 4, "text": "Non è l'età che avanza, è un problema infiammatorio."},
        {"id": 5, "text": "C'è una soluzione naturale che riattiva il fegato."},
    ]
    
    config = {
        "aspect_ratio": "9:16",
        "resolution": "720p",
        "duration": "8",
        "language": "Italian",
        "use_interpolation": True,
    }
    
    with get_db() as db:
        # Create job
        job = Job(
            id=job_id,
            status=JobStatus.COMPLETED.value,
            config_json=json.dumps(config),
            dialogue_json=json.dumps(dialogue_lines),
            api_keys_json=json.dumps({"gemini_keys": ["fake-key"]}),
            images_dir=images_dir,
            output_dir=output_dir,
            total_clips=len(dialogue_lines),
            completed_clips=len(dialogue_lines),
            progress_percent=100.0,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        
        # Create clips with various states for testing
        states = [
            # (approval_status, generation_attempt, has_output)
            ("pending_review", 1, True),   # Fresh, needs review
            ("approved", 1, True),          # Already approved
            ("pending_review", 2, True),    # Redo'd once, pending
            ("pending_review", 3, True),    # Last attempt, pending
            ("max_attempts", 3, True),      # Maxed out
        ]
        
        for i, line in enumerate(dialogue_lines):
            approval, attempt, has_output = states[i]
            
            clip = Clip(
                job_id=job_id,
                clip_index=i,
                dialogue_id=line["id"],
                dialogue_text=line["text"],
                status=ClipStatus.COMPLETED.value,
                start_frame=f"frame_{i:02d}.png",
                end_frame=f"frame_{(i+1) % 5:02d}.png",
                output_filename=f"clip_{i+1}.mp4" if has_output else None,
                approval_status=approval,
                generation_attempt=attempt,
                versions_json=json.dumps([
                    {"attempt": j, "filename": f"clip_{i+1}_v{j}.mp4", "approved": False}
                    for j in range(1, attempt)
                ]) if attempt > 1 else "[]",
            )
            db.add(clip)
        
        db.commit()
    
    print(f"✓ Created mock job: {job_id}")
    print(f"  - 5 clips with different review states")
    print(f"  - Open http://localhost:8000 and click on the job")
    
    return job_id


def create_fake_videos(job_id: str):
    """Create fake MP4 files for testing video playback"""
    output_dir = Path(f"outputs/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Minimal valid MP4 (won't actually play but browser won't error)
    # For real testing, copy any small .mp4 file here
    for i in range(1, 6):
        mp4_path = output_dir / f"clip_{i}.mp4"
        # Create empty file (video player will show error but UI works)
        mp4_path.touch()
        
        # Also create version files
        for v in range(1, 3):
            version_path = output_dir / f"clip_{i}_v{v}.mp4"
            version_path.touch()
    
    print(f"✓ Created placeholder video files in {output_dir}")
    print(f"  (Replace with real .mp4 files to test video playback)")


if __name__ == "__main__":
    print("\n🧪 VEO WEB APP - TEST DATA GENERATOR\n")
    
    job_id = create_mock_job()
    create_fake_videos(job_id)
    
    print("\n" + "="*50)
    print("TEST DATA READY!")
    print("="*50)
    print(f"\n1. Make sure server is running: python main.py")
    print(f"2. Open: http://localhost:8000")
    print(f"3. You should see a completed job with 5 clips")
    print(f"\nClip states to test:")
    print(f"  • Clip 1: Fresh, needs review (2 redos left)")
    print(f"  • Clip 2: Already approved")
    print(f"  • Clip 3: Redo'd once, pending (1 redo left)")
    print(f"  • Clip 4: Last attempt, pending (0 redos left)")
    print(f"  • Clip 5: Maxed out - shows 'Contact Support'")
    print()
