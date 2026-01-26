# -*- coding: utf-8 -*-
"""
Object Storage module for Veo Web App

Provides S3/R2 compatible storage for:
- Input frames/images (uploaded by user)
- Generated video outputs
- Flow auth state (storage_state.json)

This is required because Render persistent disks cannot be shared
between services (web and worker).
"""

import os
import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional, BinaryIO, Union
from datetime import datetime, timedelta
import hashlib


class ObjectStorage:
    """
    S3/R2 compatible object storage wrapper.
    
    Supports:
    - AWS S3
    - Cloudflare R2
    - MinIO
    - Any S3-compatible storage
    """
    
    def __init__(
        self,
        endpoint_url: str = None,
        bucket_name: str = None,
        access_key: str = None,
        secret_key: str = None,
        region: str = "auto"
    ):
        """
        Initialize storage client.
        
        Args:
            endpoint_url: S3 endpoint (e.g., https://xxx.r2.cloudflarestorage.com)
            bucket_name: Bucket name
            access_key: Access key ID
            secret_key: Secret access key
            region: AWS region (use "auto" for R2)
        """
        self.endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT")
        self.bucket_name = bucket_name or os.environ.get("S3_BUCKET", "veo-studio")
        self.access_key = access_key or os.environ.get("S3_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("S3_SECRET_KEY")
        self.region = region or os.environ.get("S3_REGION", "auto")
        
        self._client = None
    
    @property
    def client(self):
        """Lazy-init S3 client"""
        if self._client is None:
            if not self.endpoint_url or not self.access_key or not self.secret_key:
                raise ValueError(
                    "Object storage not configured. Set S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY env vars."
                )
            
            self._client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=Config(
                    signature_version='s3v4',
                    retries={'max_attempts': 3, 'mode': 'adaptive'}
                )
            )
        
        return self._client
    
    def is_configured(self) -> bool:
        """Check if storage is properly configured"""
        return bool(
            self.endpoint_url and 
            self.access_key and 
            self.secret_key and 
            self.bucket_name
        )
    
    def _get_key(self, path: str, prefix: str = None) -> str:
        """
        Get the full object key for a path.
        
        Args:
            path: File path or name
            prefix: Optional prefix (e.g., "jobs/abc123/")
            
        Returns:
            Full object key
        """
        # Remove leading slashes
        key = path.lstrip("/")
        
        if prefix:
            prefix = prefix.strip("/")
            key = f"{prefix}/{key}"
        
        return key
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_key: str,
        content_type: str = None,
        metadata: dict = None
    ) -> str:
        """
        Upload a file to object storage.
        
        Args:
            local_path: Local file path
            remote_key: Object key in bucket
            content_type: MIME type (auto-detected if not provided)
            metadata: Optional metadata dict
            
        Returns:
            Object key
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        # Auto-detect content type
        if not content_type:
            ext = local_path.suffix.lower()
            content_type_map = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".json": "application/json",
            }
            content_type = content_type_map.get(ext, "application/octet-stream")
        
        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata
        
        self.client.upload_file(
            str(local_path),
            self.bucket_name,
            remote_key,
            ExtraArgs=extra_args
        )
        
        print(f"[Storage] Uploaded: {local_path.name} → {remote_key}", flush=True)
        return remote_key
    
    def upload_bytes(
        self,
        data: bytes,
        remote_key: str,
        content_type: str = "application/octet-stream",
        metadata: dict = None
    ) -> str:
        """
        Upload bytes to object storage.
        
        Args:
            data: Bytes to upload
            remote_key: Object key in bucket
            content_type: MIME type
            metadata: Optional metadata dict
            
        Returns:
            Object key
        """
        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata
        
        from io import BytesIO
        self.client.upload_fileobj(
            BytesIO(data),
            self.bucket_name,
            remote_key,
            ExtraArgs=extra_args
        )
        
        print(f"[Storage] Uploaded bytes: {len(data)} bytes → {remote_key}", flush=True)
        return remote_key
    
    def download_file(
        self,
        remote_key: str,
        local_path: Union[str, Path]
    ) -> Path:
        """
        Download a file from object storage.
        
        Args:
            remote_key: Object key in bucket
            local_path: Local file path to save to
            
        Returns:
            Local path
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client.download_file(
            self.bucket_name,
            remote_key,
            str(local_path)
        )
        
        print(f"[Storage] Downloaded: {remote_key} → {local_path}", flush=True)
        return local_path
    
    def download_bytes(self, remote_key: str) -> bytes:
        """
        Download bytes from object storage.
        
        Args:
            remote_key: Object key in bucket
            
        Returns:
            File contents as bytes
        """
        from io import BytesIO
        buffer = BytesIO()
        
        self.client.download_fileobj(
            self.bucket_name,
            remote_key,
            buffer
        )
        
        buffer.seek(0)
        return buffer.read()
    
    def get_presigned_url(
        self,
        remote_key: str,
        expires_in: int = 3600,
        method: str = "get_object"
    ) -> str:
        """
        Generate a presigned URL for an object.
        
        Args:
            remote_key: Object key in bucket
            expires_in: URL expiration in seconds (default 1 hour)
            method: S3 method (get_object, put_object)
            
        Returns:
            Presigned URL
        """
        return self.client.generate_presigned_url(
            method,
            Params={
                "Bucket": self.bucket_name,
                "Key": remote_key
            },
            ExpiresIn=expires_in
        )
    
    def exists(self, remote_key: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            remote_key: Object key in bucket
            
        Returns:
            True if exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def delete(self, remote_key: str) -> bool:
        """
        Delete an object.
        
        Args:
            remote_key: Object key in bucket
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            print(f"[Storage] Deleted: {remote_key}", flush=True)
            return True
        except ClientError:
            return False
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list:
        """
        List objects with a prefix.
        
        Args:
            prefix: Key prefix to filter
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object keys
        """
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        return [obj['Key'] for obj in response.get('Contents', [])]
    
    # === Job-specific helpers ===
    
    def upload_job_frame(
        self,
        job_id: str,
        frame_name: str,
        local_path: Union[str, Path]
    ) -> str:
        """
        Upload a frame image for a job.
        
        Args:
            job_id: Job ID
            frame_name: Frame filename
            local_path: Local file path
            
        Returns:
            Object key
        """
        key = f"jobs/{job_id}/frames/{frame_name}"
        return self.upload_file(local_path, key)
    
    def upload_job_output(
        self,
        job_id: str,
        output_name: str,
        local_path: Union[str, Path]
    ) -> str:
        """
        Upload a generated output for a job.
        
        Args:
            job_id: Job ID
            output_name: Output filename
            local_path: Local file path
            
        Returns:
            Object key
        """
        key = f"jobs/{job_id}/outputs/{output_name}"
        return self.upload_file(local_path, key, content_type="video/mp4")
    
    def download_job_frame(
        self,
        job_id: str,
        frame_name: str,
        local_dir: Union[str, Path]
    ) -> Path:
        """
        Download a frame for a job to local directory.
        
        Args:
            job_id: Job ID
            frame_name: Frame filename
            local_dir: Local directory to save to
            
        Returns:
            Local file path
        """
        key = f"jobs/{job_id}/frames/{frame_name}"
        local_path = Path(local_dir) / frame_name
        return self.download_file(key, local_path)
    
    def get_job_output_url(
        self,
        job_id: str,
        output_name: str,
        expires_in: int = 86400  # 24 hours
    ) -> str:
        """
        Get a presigned URL for a job output.
        
        Args:
            job_id: Job ID
            output_name: Output filename
            expires_in: URL expiration in seconds
            
        Returns:
            Presigned URL
        """
        key = f"jobs/{job_id}/outputs/{output_name}"
        return self.get_presigned_url(key, expires_in)
    
    # === Flow auth state helpers ===
    
    def upload_flow_auth_state(self, storage_state: dict, account_name: str = "default") -> str:
        """
        Upload Flow authentication state.
        
        Args:
            storage_state: Playwright storage state dict
            account_name: Account identifier
            
        Returns:
            Object key
        """
        key = f"flow/auth/{account_name}/storage_state.json"
        data = json.dumps(storage_state, indent=2).encode('utf-8')
        return self.upload_bytes(data, key, content_type="application/json")
    
    def download_flow_auth_state(self, account_name: str = "default") -> Optional[dict]:
        """
        Download Flow authentication state.
        
        Args:
            account_name: Account identifier
            
        Returns:
            Storage state dict or None if not found
        """
        # First try the environment variable path
        env_key = os.environ.get("FLOW_STORAGE_STATE_URL")
        if env_key:
            print(f"[Storage] Trying to download auth state from: {env_key}", flush=True)
            if self.exists(env_key):
                data = self.download_bytes(env_key)
                print(f"[Storage] Successfully downloaded auth state from: {env_key}", flush=True)
                return json.loads(data.decode('utf-8'))
            else:
                print(f"[Storage] Auth state not found at: {env_key}", flush=True)
        
        # Fall back to default path
        key = f"flow/auth/{account_name}/storage_state.json"
        print(f"[Storage] Trying default path: {key}", flush=True)
        
        if not self.exists(key):
            print(f"[Storage] Auth state not found at default path either", flush=True)
            return None
        
        data = self.download_bytes(key)
        return json.loads(data.decode('utf-8'))


# Singleton storage instance
_storage: Optional[ObjectStorage] = None


def get_storage() -> ObjectStorage:
    """
    Get the singleton storage instance.
    
    Returns:
        ObjectStorage instance
    """
    global _storage
    
    if _storage is None:
        _storage = ObjectStorage()
    
    return _storage


def is_storage_configured() -> bool:
    """
    Check if object storage is configured.
    
    Returns:
        True if configured, False otherwise
    """
    storage = get_storage()
    configured = storage.is_configured()
    
    # Debug logging to help diagnose configuration issues
    if not configured:
        missing = []
        if not storage.endpoint_url:
            missing.append("S3_ENDPOINT")
        if not storage.access_key:
            missing.append("S3_ACCESS_KEY")
        if not storage.secret_key:
            missing.append("S3_SECRET_KEY")
        if not storage.bucket_name:
            missing.append("S3_BUCKET")
        if missing:
            print(f"[Storage] NOT configured - missing env vars: {', '.join(missing)}", flush=True)
        else:
            print(f"[Storage] NOT configured - values are empty strings", flush=True)
    
    return configured


def get_storage_status() -> dict:
    """
    Get detailed storage configuration status for diagnostics.
    
    Returns:
        Dict with configuration status details
    """
    storage = get_storage()
    return {
        "configured": storage.is_configured(),
        "has_endpoint": bool(storage.endpoint_url),
        "has_access_key": bool(storage.access_key),
        "has_secret_key": bool(storage.secret_key),
        "has_bucket": bool(storage.bucket_name),
        "bucket_name": storage.bucket_name if storage.bucket_name else None,
        "endpoint_domain": storage.endpoint_url.split("//")[-1].split("/")[0] if storage.endpoint_url else None,
    }