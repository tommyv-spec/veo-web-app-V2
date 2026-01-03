# -*- coding: utf-8 -*-
"""
Flow Worker Service for Veo Web App

Background worker that processes Flow jobs (browser automation).
Runs as a separate Render background worker service.

Features:
- Redis/Valkey queue consumption
- Playwright browser automation
- Object storage integration
- Auth state management
- Error recovery and retry logic
"""

import os
import sys
import json
import time
import signal
import traceback
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import JobStatus, ClipStatus


# === Queue Configuration ===

def get_redis_client():
    """Get Redis/Valkey client for queue operations"""
    redis_url = os.environ.get("KEYVALUE_URL") or os.environ.get("REDIS_URL")
    
    if not redis_url:
        print("[FlowWorker] WARNING: No KEYVALUE_URL or REDIS_URL configured", flush=True)
        return None
    
    try:
        import redis
        return redis.from_url(redis_url, decode_responses=True)
    except ImportError:
        print("[FlowWorker] WARNING: redis package not installed", flush=True)
        return None
    except Exception as e:
        print(f"[FlowWorker] ERROR connecting to Redis: {e}", flush=True)
        return None


# Queue names
FLOW_QUEUE_NAME = "flow:jobs"
FLOW_QUEUE_PROCESSING = "flow:processing"
FLOW_QUEUE_FAILED = "flow:failed"


class FlowWorker:
    """
    Background worker for Flow (browser automation) jobs.
    
    Consumes jobs from Redis queue, processes them with Playwright,
    and updates the database with results.
    """
    
    def __init__(
        self,
        max_concurrent: int = 1,
        poll_interval: float = 5.0,
        job_timeout: int = 600  # 10 minutes per job
    ):
        """
        Initialize Flow worker.
        
        Args:
            max_concurrent: Maximum concurrent jobs (usually 1 for browser automation)
            poll_interval: Seconds between queue polls
            job_timeout: Maximum seconds per job before timeout
        """
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        
        self._redis = None
        self._shutdown_event = threading.Event()
        self._current_jobs: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        
        # Temp directories
        self._temp_dir = tempfile.mkdtemp(prefix="flow_worker_")
        self._download_dir = os.path.join(self._temp_dir, "downloads")
        os.makedirs(self._download_dir, exist_ok=True)
        
        print(f"[FlowWorker] Initialized with max_concurrent={max_concurrent}", flush=True)
        print(f"[FlowWorker] Temp dir: {self._temp_dir}", flush=True)
    
    def start(self):
        """Start the worker"""
        print("[FlowWorker] Starting...", flush=True)
        
        # Connect to Redis
        self._redis = get_redis_client()
        if not self._redis:
            print("[FlowWorker] ERROR: Could not connect to Redis. Exiting.", flush=True)
            return False
        
        print("[FlowWorker] Connected to Redis", flush=True)
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Main loop
        self._main_loop()
        
        return True
    
    def stop(self):
        """Stop the worker gracefully"""
        print("[FlowWorker] Shutting down...", flush=True)
        self._shutdown_event.set()
        
        # Wait for current jobs to complete (with timeout)
        with self._lock:
            for job_id, thread in list(self._current_jobs.items()):
                print(f"[FlowWorker] Waiting for job {job_id[:8]}...", flush=True)
                thread.join(timeout=30)
        
        print("[FlowWorker] Shutdown complete", flush=True)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        print(f"[FlowWorker] Received signal {signum}", flush=True)
        self.stop()
    
    def _main_loop(self):
        """Main job processing loop"""
        print("[FlowWorker] Entering main loop", flush=True)
        
        while not self._shutdown_event.is_set():
            try:
                # Check if we can accept more jobs
                with self._lock:
                    current_count = len(self._current_jobs)
                
                if current_count >= self.max_concurrent:
                    time.sleep(self.poll_interval)
                    continue
                
                # Try to claim a job from queue
                job_data = self._claim_job()
                
                if job_data:
                    # Start processing in background thread
                    job_id = job_data.get("job_id")
                    thread = threading.Thread(
                        target=self._process_job,
                        args=(job_data,),
                        daemon=True
                    )
                    
                    with self._lock:
                        self._current_jobs[job_id] = thread
                    
                    thread.start()
                else:
                    # No jobs available, sleep
                    time.sleep(self.poll_interval)
                
                # Clean up completed threads
                self._cleanup_threads()
                
            except Exception as e:
                print(f"[FlowWorker] Error in main loop: {e}", flush=True)
                traceback.print_exc()
                time.sleep(self.poll_interval)
        
        print("[FlowWorker] Exited main loop", flush=True)
    
    def _claim_job(self) -> Optional[Dict[str, Any]]:
        """
        Claim a job from the queue.
        
        Returns:
            Job data dict or None if no jobs available
        """
        if not self._redis:
            return None
        
        try:
            # Move job from queue to processing
            # Using BRPOPLPUSH for atomic operation
            job_json = self._redis.brpoplpush(
                FLOW_QUEUE_NAME,
                FLOW_QUEUE_PROCESSING,
                timeout=1
            )
            
            if job_json:
                job_data = json.loads(job_json)
                job_id = job_data.get("job_id", "unknown")
                print(f"[FlowWorker] Claimed job: {job_id[:8]}...", flush=True)
                return job_data
            
        except Exception as e:
            print(f"[FlowWorker] Error claiming job: {e}", flush=True)
        
        return None
    
    def _complete_job(self, job_id: str, success: bool, error: str = None):
        """
        Mark job as complete in queue.
        
        Args:
            job_id: Job ID
            success: Whether job completed successfully
            error: Error message if failed
        """
        if not self._redis:
            return
        
        try:
            # Find and remove from processing queue
            processing_jobs = self._redis.lrange(FLOW_QUEUE_PROCESSING, 0, -1)
            for job_json in processing_jobs:
                job_data = json.loads(job_json)
                if job_data.get("job_id") == job_id:
                    self._redis.lrem(FLOW_QUEUE_PROCESSING, 1, job_json)
                    
                    if not success:
                        # Move to failed queue
                        job_data["error"] = error
                        job_data["failed_at"] = datetime.now(timezone.utc).isoformat()
                        self._redis.lpush(FLOW_QUEUE_FAILED, json.dumps(job_data))
                    
                    break
            
        except Exception as e:
            print(f"[FlowWorker] Error completing job in queue: {e}", flush=True)
    
    def _cleanup_threads(self):
        """Clean up completed job threads"""
        with self._lock:
            completed = []
            for job_id, thread in self._current_jobs.items():
                if not thread.is_alive():
                    completed.append(job_id)
            
            for job_id in completed:
                del self._current_jobs[job_id]
    
    def _process_job(self, job_data: Dict[str, Any]):
        """
        Process a single job.
        
        Args:
            job_data: Job data from queue
        """
        job_id = job_data.get("job_id", "unknown")
        
        print(f"[FlowWorker] Processing job: {job_id}", flush=True)
        
        try:
            # Import database models
            from models import init_db, get_db, Job, Clip, add_job_log, update_job_progress
            
            # Initialize database
            init_db()
            
            with get_db() as db:
                # Get job from database
                job = db.query(Job).filter(Job.id == job_id).first()
                
                if not job:
                    print(f"[FlowWorker] Job {job_id} not found in database", flush=True)
                    self._complete_job(job_id, False, "Job not found")
                    return
                
                # Update job status
                job.status = JobStatus.RUNNING.value
                job.started_at = datetime.now(timezone.utc)
                db.commit()
                
                add_job_log(db, job_id, "Flow worker started processing", "INFO", "flow")
                
                # Get clips
                clips = db.query(Clip).filter(
                    Clip.job_id == job_id
                ).order_by(Clip.clip_index.asc()).all()
                
                # If no clips exist, create them from dialogue_json
                if not clips:
                    print(f"[FlowWorker] No clips found, creating from dialogue_json", flush=True)
                    dialogue_data = json.loads(job.dialogue_json) if job.dialogue_json else {}
                    dialogue_lines = dialogue_data.get("lines", [])
                    
                    if not dialogue_lines:
                        raise ValueError("No dialogue lines found in job")
                    
                    for i, line in enumerate(dialogue_lines):
                        clip = Clip(
                            job_id=job_id,
                            clip_index=i,
                            dialogue_id=line.get("id", i + 1),
                            dialogue_text=line.get("text", ""),
                            status=ClipStatus.PENDING.value,
                            start_frame=None,  # Flow will handle frames
                            end_frame=None,
                        )
                        db.add(clip)
                    
                    db.commit()
                    
                    # Re-fetch clips
                    clips = db.query(Clip).filter(
                        Clip.job_id == job_id
                    ).order_by(Clip.clip_index.asc()).all()
                    
                    add_job_log(db, job_id, f"Created {len(clips)} clips from dialogue", "INFO", "flow")
                
                if not clips:
                    raise ValueError("No clips found for job")
                
                # Parse job config
                config_data = json.loads(job.config_json) if job.config_json else {}
                language = config_data.get("language", "English")
                
                # Download frames from object storage if needed
                frames_dir = self._prepare_frames(job, clips)
                
                # Process with Flow backend
                success = self._run_flow_automation(
                    db, job, clips, frames_dir, language
                )
                
                if success:
                    # Update job status
                    job.status = JobStatus.COMPLETED.value
                    job.completed_at = datetime.now(timezone.utc)
                    update_job_progress(db, job_id)
                    db.commit()
                    
                    add_job_log(db, job_id, "Flow processing completed", "INFO", "flow")
                    self._complete_job(job_id, True)
                else:
                    raise RuntimeError("Flow automation failed")
                
        except Exception as e:
            error_msg = str(e)
            print(f"[FlowWorker] Error processing job {job_id}: {error_msg}", flush=True)
            traceback.print_exc()
            
            # Update job status in database
            try:
                with get_db() as db:
                    job = db.query(Job).filter(Job.id == job_id).first()
                    if job:
                        job.status = JobStatus.FAILED.value
                        db.commit()
                        
                        add_job_log(
                            db, job_id, 
                            f"Flow processing failed: {error_msg}", 
                            "ERROR", "flow"
                        )
            except Exception:
                pass
            
            self._complete_job(job_id, False, error_msg)
        
        finally:
            # Clean up from current jobs
            with self._lock:
                if job_id in self._current_jobs:
                    del self._current_jobs[job_id]
    
    def _prepare_frames(self, job, clips) -> str:
        """
        Download frames from object storage to local temp directory.
        Also assigns frames to clips if not already assigned.
        
        Args:
            job: Job database object
            clips: List of Clip database objects
            
        Returns:
            Local directory path containing frames
        """
        frames_dir = os.path.join(self._temp_dir, f"job_{job.id[:8]}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Check if we need to download from object storage
        from backends.storage import is_storage_configured, get_storage
        
        if is_storage_configured():
            storage = get_storage()
            
            for clip in clips:
                # Download start frame if stored in S3
                if clip.start_frame and clip.start_frame.startswith("jobs/"):
                    local_path = os.path.join(frames_dir, os.path.basename(clip.start_frame))
                    try:
                        storage.download_file(clip.start_frame, local_path)
                        clip.start_frame = local_path
                        print(f"[FlowWorker] Downloaded start frame for clip {clip.clip_index}: {local_path}", flush=True)
                    except Exception as e:
                        print(f"[FlowWorker] Error downloading start frame: {e}", flush=True)
                
                # Download end frame if stored in S3
                if clip.end_frame and clip.end_frame.startswith("jobs/"):
                    local_path = os.path.join(frames_dir, os.path.basename(clip.end_frame))
                    try:
                        storage.download_file(clip.end_frame, local_path)
                        clip.end_frame = local_path
                        print(f"[FlowWorker] Downloaded end frame for clip {clip.clip_index}: {local_path}", flush=True)
                    except Exception as e:
                        print(f"[FlowWorker] Error downloading end frame: {e}", flush=True)
        
        # Collect available images from job's images directory
        available_images = []
        if job.images_dir and os.path.exists(job.images_dir):
            import shutil
            # Get sorted list of image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
            for f in sorted(os.listdir(job.images_dir)):
                if os.path.splitext(f.lower())[1] in image_extensions:
                    src = os.path.join(job.images_dir, f)
                    dst = os.path.join(frames_dir, f)
                    if os.path.isfile(src):
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                        available_images.append(dst)
            
            print(f"[FlowWorker] Found {len(available_images)} images in job directory", flush=True)
            for img in available_images:
                print(f"[FlowWorker]   - {os.path.basename(img)}", flush=True)
        
        # Assign frames to clips if they don't have any
        # For single-image mode: use first image as start_frame for all clips
        # For storyboard mode: each clip gets its assigned image
        if available_images:
            for i, clip in enumerate(clips):
                if not clip.start_frame:
                    # Try to match image by index or use first available
                    if i < len(available_images):
                        clip.start_frame = available_images[i]
                        print(f"[FlowWorker] Assigned image {i+1} to clip {clip.clip_index}: {os.path.basename(available_images[i])}", flush=True)
                    elif available_images:
                        # Fallback to first image for all clips (single image mode)
                        clip.start_frame = available_images[0]
                        print(f"[FlowWorker] Assigned first image to clip {clip.clip_index}: {os.path.basename(available_images[0])}", flush=True)
        else:
            print(f"[FlowWorker] WARNING: No images found for job!", flush=True)
        
        return frames_dir
    
    def _run_flow_automation(
        self,
        db,
        job,
        clips,
        frames_dir: str,
        language: str
    ) -> bool:
        """
        Run Flow browser automation.
        
        Args:
            db: Database session
            job: Job database object
            clips: List of Clip database objects
            frames_dir: Directory containing frame images
            language: Language for prompts
            
        Returns:
            True if successful
        """
        from backends.flow_backend import FlowBackend, FlowJob, FlowClip, create_flow_job_from_db
        from backends.storage import is_storage_configured, get_storage
        from models import add_job_log, get_db, Clip
        
        # Create FlowJob from database clips
        clips_data = []
        for clip in clips:
            # Find actual frame paths
            start_frame = None
            end_frame = None
            
            if clip.start_frame:
                # Check temp dir first
                temp_path = os.path.join(frames_dir, os.path.basename(clip.start_frame))
                if os.path.exists(temp_path):
                    start_frame = temp_path
                elif os.path.exists(clip.start_frame):
                    start_frame = clip.start_frame
            
            if clip.end_frame:
                temp_path = os.path.join(frames_dir, os.path.basename(clip.end_frame))
                if os.path.exists(temp_path):
                    end_frame = temp_path
                elif os.path.exists(clip.end_frame):
                    end_frame = clip.end_frame
            
            clips_data.append({
                "dialogue_text": clip.dialogue_text,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
        
        # Get existing project URL if resuming
        project_url = None
        try:
            state_data = json.loads(job.flow_state_json) if hasattr(job, 'flow_state_json') and job.flow_state_json else {}
            project_url = state_data.get("project_url") or getattr(job, 'flow_project_url', None)
        except Exception:
            pass
        
        flow_job = create_flow_job_from_db(job.id, clips_data, project_url)
        
        # Set up progress callback
        def on_progress(clip_index, status, message):
            try:
                with get_db() as progress_db:
                    clip = progress_db.query(Clip).filter(
                        Clip.job_id == job.id,
                        Clip.clip_index == clip_index
                    ).first()
                    
                    if clip:
                        if status == "generating":
                            clip.status = ClipStatus.GENERATING.value
                        elif status == "completed":
                            clip.status = ClipStatus.COMPLETED.value
                        
                        progress_db.commit()
                    
                    add_job_log(progress_db, job.id, message, "INFO", "flow", clip_index)
            except Exception as e:
                print(f"[FlowWorker] Progress callback error: {e}", flush=True)
        
        flow_job.on_progress = on_progress
        
        # Run automation
        with FlowBackend(
            headless=True,
            download_dir=self._download_dir,
            temp_dir=self._temp_dir
        ) as flow:
            # Check if auth is valid
            if flow.needs_auth:
                # Mark job as needing auth
                add_job_log(db, job.id, "Flow authentication required - pausing job", "WARNING", "flow")
                job.status = "paused_needs_flow_auth"
                db.commit()
                
                # Send alert
                self._send_auth_alert(job.id)
                return False
            
            # Process the job
            success = flow.process_job(
                flow_job,
                language=language,
                wait_for_generation=True
            )
            
            if success:
                # Save project URL to database
                if flow_job.project_url:
                    if hasattr(job, 'flow_project_url'):
                        job.flow_project_url = flow_job.project_url
                    
                    # Also save to state JSON
                    state = {"project_url": flow_job.project_url}
                    if hasattr(job, 'flow_state_json'):
                        job.flow_state_json = json.dumps(state)
                
                # Get storage if configured
                storage = None
                if is_storage_configured():
                    storage = get_storage()
                
                # Update ALL clips in database
                for flow_clip in flow_job.clips:
                    db_clip = db.query(Clip).filter(
                        Clip.job_id == job.id,
                        Clip.clip_index == flow_clip.clip_index
                    ).first()
                    
                    if db_clip:
                        print(f"[FlowWorker] Clip {flow_clip.clip_index}: status={flow_clip.status}, output={flow_clip.output_url}", flush=True)
                        
                        # Upload to storage if available and clip has output
                        if storage and flow_clip.status == "completed" and flow_clip.output_url:
                            try:
                                output_key = storage.upload_job_output(
                                    job.id,
                                    os.path.basename(flow_clip.output_url),
                                    flow_clip.output_url
                                )
                                if hasattr(db_clip, 'output_url'):
                                    db_clip.output_url = output_key
                                db_clip.output_filename = os.path.basename(flow_clip.output_url)
                            except Exception as e:
                                print(f"[FlowWorker] Error uploading clip {flow_clip.clip_index}: {e}", flush=True)
                        
                        # Always update clip status based on flow_clip status
                        if flow_clip.status == "completed":
                            db_clip.status = ClipStatus.COMPLETED.value
                            db_clip.completed_at = datetime.now(timezone.utc)
                        elif flow_clip.status == "generating":
                            # Mark as completed anyway since flow.process_job returned success
                            # This means generation was submitted but download may have failed
                            db_clip.status = ClipStatus.COMPLETED.value
                            db_clip.completed_at = datetime.now(timezone.utc)
                            add_job_log(db, job.id, f"Clip {flow_clip.clip_index + 1} generated (download pending)", "INFO", "flow")
                        elif flow_clip.status == "failed":
                            db_clip.status = ClipStatus.FAILED.value
                
                db.commit()
                
                # Log summary with project URL
                completed = sum(1 for c in flow_job.clips if c.status in ("completed", "generating"))
                downloaded = sum(1 for c in flow_job.clips if c.status == "completed" and c.output_url)
                
                add_job_log(db, job.id, f"Flow processed {completed}/{len(flow_job.clips)} clips", "INFO", "flow")
                
                # If download failed, provide project URL for manual access
                if flow_job.project_url:
                    if downloaded < completed:
                        add_job_log(
                            db, job.id, 
                            f"📎 Videos available at: {flow_job.project_url}", 
                            "INFO", "flow"
                        )
                    else:
                        add_job_log(db, job.id, f"✅ All clips downloaded successfully", "INFO", "flow")
            
            return success
    
    def _send_auth_alert(self, job_id: str):
        """Send alert that auth is needed"""
        print(f"[FlowWorker] ALERT: Authentication required for Flow backend!", flush=True)
        print(f"[FlowWorker] Job {job_id} paused until auth is refreshed", flush=True)
        
        # TODO: Implement email/Slack notification
        # For now just log it


# === Queue Helper Functions ===

def enqueue_flow_job(job_id: str, priority: int = 0) -> bool:
    """
    Add a job to the Flow queue.
    
    Args:
        job_id: Job ID to process
        priority: Job priority (higher = sooner)
        
    Returns:
        True if queued successfully
    """
    redis_client = get_redis_client()
    if not redis_client:
        print(f"[FlowQueue] ERROR: Redis not available", flush=True)
        return False
    
    try:
        job_data = {
            "job_id": job_id,
            "priority": priority,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        
        redis_client.lpush(FLOW_QUEUE_NAME, json.dumps(job_data))
        print(f"[FlowQueue] Queued job: {job_id[:8]}...", flush=True)
        return True
        
    except Exception as e:
        print(f"[FlowQueue] Error queuing job: {e}", flush=True)
        return False


def get_queue_status() -> Dict[str, Any]:
    """
    Get Flow queue status.
    
    Returns:
        Queue status dict
    """
    redis_client = get_redis_client()
    if not redis_client:
        return {"error": "Redis not available"}
    
    try:
        return {
            "pending": redis_client.llen(FLOW_QUEUE_NAME),
            "processing": redis_client.llen(FLOW_QUEUE_PROCESSING),
            "failed": redis_client.llen(FLOW_QUEUE_FAILED),
        }
    except Exception as e:
        return {"error": str(e)}


# === CLI Entry Point ===

def main():
    """Main entry point for Flow worker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flow Worker Service")
    parser.add_argument(
        "--export-auth",
        action="store_true",
        help="Export Flow authentication state (run locally with GUI)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent jobs (default: 1)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Queue poll interval in seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    if args.export_auth:
        # Run auth export command
        from backends.flow_backend import export_auth_state_command
        export_auth_state_command()
        return
    
    # Start worker
    print("=" * 50)
    print("FLOW WORKER SERVICE")
    print("=" * 50)
    print(f"Max concurrent jobs: {args.max_concurrent}")
    print(f"Poll interval: {args.poll_interval}s")
    print("")
    
    worker = FlowWorker(
        max_concurrent=args.max_concurrent,
        poll_interval=args.poll_interval
    )
    
    worker.start()


if __name__ == "__main__":
    main()
