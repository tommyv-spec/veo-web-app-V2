# -*- coding: utf-8 -*-
"""
Comprehensive Error Handling for Veo Web App

Provides:
- Error classification and codes
- Detailed error context capture
- User-friendly error messages
- Recovery suggestions
- Logging integration
"""

import traceback
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import ErrorCode


@dataclass
class VeoError:
    """Structured error information"""
    code: ErrorCode
    message: str
    user_message: str
    details: Dict[str, Any]
    recoverable: bool
    suggestion: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat(),
        }


class ErrorHandler:
    """
    Centralized error handling for all Veo operations.
    
    Usage:
        handler = ErrorHandler()
        error = handler.classify_exception(exception, context={"clip_index": 1})
        if error.recoverable:
            # retry logic
        else:
            # fail permanently
    """
    
    # Error patterns for classification
    RATE_LIMIT_PATTERNS = [
        r"429",
        r"RESOURCE_EXHAUSTED",
        r"rate.?limit",
        r"quota.?exceeded",
        r"too.?many.?requests",
    ]
    
    # Transient errors - model overloaded, should retry with same key
    TRANSIENT_PATTERNS = [
        r"overloaded",
        r"'code':\s*14",
        r"code:\s*14",
        r"UNAVAILABLE",
        r"temporarily",
        r"try.?again.?later",
    ]
    
    CELEBRITY_PATTERNS = [
        r"celebrity",
        r"likenesses",
        r"rai_media_filtered",
        r"filtered_reasons",
        r"public.?figure",
    ]
    
    CONTENT_POLICY_PATTERNS = [
        r"content.?policy",
        r"safety.?filter",
        r"harmful.?content",
        r"blocked.?content",
        r"violat",
    ]
    
    NETWORK_PATTERNS = [
        r"connection.?error",
        r"timeout",
        r"network.?unreachable",
        r"dns.?error",
        r"ssl.?error",
        r"connection.?reset",
    ]
    
    # More specific patterns for Gemini/Google API auth errors
    # Avoid catching OpenAI or other service auth errors
    AUTH_PATTERNS = [
        r"api.?key.?invalid",
        r"invalid.?api.?key",
        r"gemini.*401",
        r"gemini.*403",
        r"gemini.*unauthorized",
        r"gemini.*authentication",
        r"google.*401",
        r"google.*403", 
        r"google.*unauthorized",
        r"google.*authentication",
        r"genai.*401",
        r"genai.*403",
        r"permission.?denied.*gemini",
        r"permission.?denied.*google",
        r"api_key_invalid",  # Google's specific error code
    ]
    
    # OpenAI-specific auth errors - EXPANDED to catch common error messages
    # These MUST be checked BEFORE AUTH_PATTERNS to avoid misclassification
    # OpenAI errors often say "Invalid API key" without mentioning "openai"
    OPENAI_AUTH_PATTERNS = [
        r"openai.*401",
        r"openai.*authentication",
        r"openai.*invalid.*key",
        r"incorrect.?api.?key.*openai",
        # Common OpenAI error messages that don't mention "openai" explicitly
        r"invalid.*api.*key.*provided",  # "Invalid API key provided"
        r"error.*code.*401.*invalid.*api",  # "Error code: 401 - Invalid API key"
        r"authenticationerror.*invalid.*api",  # "AuthenticationError: Invalid API key"
        r"platform\.openai\.com",  # Any error mentioning OpenAI platform URL
        r"you.*can.*find.*your.*api.*key.*at",  # Common OpenAI error message suffix
        r"sk-[a-zA-Z0-9]{20,}",  # OpenAI key format in error message (sk-xxx...)
        # Additional patterns to catch generic OpenAI errors
        r"authenticationerror",  # OpenAI's AuthenticationError class name
        r"openai\.authenticationerror",  # Full class path
        r"api\.openai\.com",  # OpenAI API domain
    ]
    
    def __init__(self):
        self.error_counts: Dict[ErrorCode, int] = {}
    
    def classify_exception(
        self, 
        exception: Exception, 
        context: Dict[str, Any] = None
    ) -> VeoError:
        """
        Classify an exception into a structured VeoError.
        
        Args:
            exception: The caught exception
            context: Additional context (clip_index, image_path, etc.)
        
        Returns:
            VeoError with classification and suggestions
        """
        if context is None:
            context = {}
        
        error_str = str(exception).lower()
        exception_type = type(exception).__name__
        
        # Get full traceback for details
        tb = traceback.format_exc()
        
        # Build base details
        details = {
            "exception_type": exception_type,
            "exception_message": str(exception),
            "traceback": tb,
            **context
        }
        
        # CRITICAL: Check if exception is from OpenAI FIRST (by module)
        # This MUST happen before pattern matching because OpenAI errors
        # often contain "Invalid API key" which would match AUTH_PATTERNS
        exception_module = type(exception).__module__ if hasattr(type(exception), '__module__') else ''
        if 'openai' in exception_module.lower():
            error = VeoError(
                code=ErrorCode.UNKNOWN,  # Don't use API_KEY_INVALID for OpenAI
                message=f"OpenAI error: {exception_type}",
                user_message="OpenAI API key error. This affects prompt tuning but videos can still be generated.",
                details=details,
                recoverable=True,  # OpenAI is optional, job can continue
                suggestion="Check your OPENAI_API_KEY environment variable, or disable prompt tuning."
            )
            self._increment_count(error.code)
            return error
        
        # Try to classify by patterns
        error = self._classify_by_patterns(error_str, details)
        if error:
            self._increment_count(error.code)
            return error
        
        # Check for specific exception types
        error = self._classify_by_type(exception, details)
        if error:
            self._increment_count(error.code)
            return error
        
        # Default to unknown error
        error = VeoError(
            code=ErrorCode.UNKNOWN,
            message=f"Unknown error: {exception_type}: {str(exception)[:200]}",
            user_message="An unexpected error occurred. Please try again or contact support.",
            details=details,
            recoverable=True,  # Assume recoverable for unknown errors
            suggestion="Try again. If the problem persists, check the logs for details."
        )
        self._increment_count(error.code)
        return error
    
    def _classify_by_patterns(
        self, 
        error_str: str, 
        details: Dict
    ) -> Optional[VeoError]:
        """Classify error by string pattern matching"""
        
        # Transient errors (model overloaded) - should retry with SAME key
        if self._matches_patterns(error_str, self.TRANSIENT_PATTERNS):
            return VeoError(
                code=ErrorCode.RATE_LIMIT,  # Use RATE_LIMIT code but different message
                message="Model temporarily overloaded (code 14)",
                user_message="The model is temporarily overloaded. Retrying...",
                details={**details, "transient": True},
                recoverable=True,
                suggestion="Wait a moment - this is a temporary service issue."
            )
        
        # Rate limit
        if self._matches_patterns(error_str, self.RATE_LIMIT_PATTERNS):
            return VeoError(
                code=ErrorCode.RATE_LIMIT,
                message="API rate limit exceeded (429)",
                user_message="The API is temporarily overloaded. Retrying with a different key...",
                details=details,
                recoverable=True,
                suggestion="Wait a moment or add more API keys for rotation."
            )
        
        # Celebrity filter
        if self._matches_patterns(error_str, self.CELEBRITY_PATTERNS):
            return VeoError(
                code=ErrorCode.CELEBRITY_FILTER,
                message="Celebrity/RAI filter triggered",
                user_message="The image triggered a celebrity detection filter. Trying different image...",
                details=details,
                recoverable=True,
                suggestion="Use different images or modify the source images to avoid detection."
            )
        
        # Content policy
        if self._matches_patterns(error_str, self.CONTENT_POLICY_PATTERNS):
            return VeoError(
                code=ErrorCode.CONTENT_POLICY,
                message="Content policy violation detected",
                user_message="The content triggered a safety filter. Please review your images and dialogue.",
                details=details,
                recoverable=False,
                suggestion="Review and modify the content to comply with content policies."
            )
        
        # Network errors
        if self._matches_patterns(error_str, self.NETWORK_PATTERNS):
            return VeoError(
                code=ErrorCode.API_NETWORK_ERROR,
                message="Network error during API call",
                user_message="Network connection issue. Retrying...",
                details=details,
                recoverable=True,
                suggestion="Check your internet connection. The system will retry automatically."
            )
        
        # OpenAI auth errors (separate from Gemini - clearer message)
        if self._matches_patterns(error_str, self.OPENAI_AUTH_PATTERNS):
            return VeoError(
                code=ErrorCode.UNKNOWN,  # Don't use API_KEY_INVALID for OpenAI
                message="OpenAI API key issue (not Gemini)",
                user_message="OpenAI API key error. This affects prompt tuning but videos can still be generated.",
                details=details,
                recoverable=True,  # OpenAI is optional, job can continue
                suggestion="Check your OPENAI_API_KEY environment variable, or disable prompt tuning."
            )
        
        # Gemini/Google Auth errors
        if self._matches_patterns(error_str, self.AUTH_PATTERNS):
            return VeoError(
                code=ErrorCode.API_KEY_INVALID,
                message="API key authentication failed",
                user_message="API key is invalid or expired. Please check your API keys.",
                details=details,
                recoverable=False,
                suggestion="Verify your API keys are correct and have the necessary permissions."
            )
        
        return None
    
    def _classify_by_type(
        self, 
        exception: Exception, 
        details: Dict
    ) -> Optional[VeoError]:
        """Classify error by exception type"""
        
        exception_type = type(exception).__name__
        
        # Check if exception is from OpenAI library (by module name)
        # This catches OpenAI exceptions even if the error message doesn't match patterns
        exception_module = type(exception).__module__ if hasattr(type(exception), '__module__') else ''
        if 'openai' in exception_module.lower():
            return VeoError(
                code=ErrorCode.UNKNOWN,  # Don't use API_KEY_INVALID for OpenAI
                message=f"OpenAI error: {exception_type}",
                user_message="OpenAI API error. This affects prompt tuning but videos can still be generated.",
                details=details,
                recoverable=True,  # OpenAI is optional, job can continue
                suggestion="Check your OPENAI_API_KEY environment variable, or disable prompt tuning."
            )
        
        # File errors - check if it's an images/uploads directory issue
        if exception_type in ("FileNotFoundError", "IOError"):
            error_str = str(exception).lower()
            # Check if this is an images/uploads directory issue
            if "uploads" in error_str or "images" in error_str:
                return VeoError(
                    code=ErrorCode.IMAGE_NOT_FOUND,
                    message=f"Original images unavailable: {str(exception)}",
                    user_message="Original images are no longer available. The files may have been deleted from temporary storage.",
                    details=details,
                    recoverable=False,
                    suggestion="Please create a new job with re-uploaded images. To prevent this in the future, ensure cloud storage (R2) is configured."
                )
            return VeoError(
                code=ErrorCode.IMAGE_NOT_FOUND,
                message=f"File not found: {str(exception)}",
                user_message="Required file could not be found.",
                details=details,
                recoverable=False,
                suggestion="Ensure all uploaded files are accessible."
            )
        
        # Permission errors
        if exception_type == "PermissionError":
            return VeoError(
                code=ErrorCode.FILE_WRITE_ERROR,
                message=f"Permission denied: {str(exception)}",
                user_message="Could not write output file due to permissions.",
                details=details,
                recoverable=False,
                suggestion="Check file system permissions for the output directory."
            )
        
        # Timeout
        if exception_type in ("TimeoutError", "asyncio.TimeoutError"):
            return VeoError(
                code=ErrorCode.API_TIMEOUT,
                message="API request timed out",
                user_message="The request took too long. Retrying...",
                details=details,
                recoverable=True,
                suggestion="The API may be slow. The system will retry automatically."
            )
        
        # JSON decode errors (bad API response)
        if exception_type == "JSONDecodeError":
            return VeoError(
                code=ErrorCode.API_NETWORK_ERROR,
                message="Invalid API response (not JSON)",
                user_message="Received invalid response from API. Retrying...",
                details=details,
                recoverable=True,
                suggestion="This is usually temporary. The system will retry."
            )
        
        # Value/Type errors in config
        if exception_type in ("ValueError", "TypeError"):
            if "config" in str(exception).lower() or "invalid" in str(exception).lower():
                return VeoError(
                    code=ErrorCode.INVALID_CONFIG,
                    message=f"Configuration error: {str(exception)}",
                    user_message="Invalid configuration. Please check your settings.",
                    details=details,
                    recoverable=False,
                    suggestion="Review your configuration settings and correct any invalid values."
                )
        
        return None
    
    def _matches_patterns(self, text: str, patterns: list) -> bool:
        """Check if text matches any of the patterns"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _increment_count(self, code: ErrorCode):
        """Track error occurrences"""
        self.error_counts[code] = self.error_counts.get(code, 0) + 1
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts"""
        return {code.value: count for code, count in self.error_counts.items()}
    
    def classify_veo_operation(
        self, 
        operation,
        context: Dict[str, Any] = None
    ) -> Optional[VeoError]:
        """
        Classify error from a Veo API operation response.
        
        Args:
            operation: The Veo API operation result
            context: Additional context
        
        Returns:
            VeoError if there's an error, None if successful
        """
        if context is None:
            context = {}
        
        details = {"context": context}
        
        try:
            # Check for direct error
            error = getattr(operation, "error", None)
            if error:
                error_msg = getattr(error, "message", str(error))
                details["error_message"] = error_msg
                return self.classify_exception(Exception(error_msg), details)
            
            # Check metadata for errors
            metadata = getattr(operation, "metadata", None)
            if metadata:
                state = getattr(metadata, "state", None)
                blocked_reason = getattr(metadata, "blockedReason", None)
                
                if blocked_reason:
                    details["blocked_reason"] = blocked_reason
                    return VeoError(
                        code=ErrorCode.CONTENT_POLICY,
                        message=f"Content blocked: {blocked_reason}",
                        user_message=f"Content was blocked: {blocked_reason}",
                        details=details,
                        recoverable=False,
                        suggestion="Modify your content to comply with policies."
                    )
            
            # Check response for RAI filtering
            response = getattr(operation, "response", None)
            if response:
                filtered_reasons = getattr(response, "rai_media_filtered_reasons", None)
                filtered_count = getattr(response, "rai_media_filtered_count", 0)
                
                if filtered_reasons and len(filtered_reasons) > 0:
                    details["filtered_reasons"] = str(filtered_reasons)
                    
                    # Check if it's celebrity filter
                    reasons_str = str(filtered_reasons).lower()
                    if "celebrity" in reasons_str or "likeness" in reasons_str:
                        return VeoError(
                            code=ErrorCode.CELEBRITY_FILTER,
                            message=f"Celebrity filter triggered: {filtered_reasons}",
                            user_message="The image triggered celebrity detection. Trying different image...",
                            details=details,
                            recoverable=True,
                            suggestion="Use different images to avoid celebrity detection."
                        )
                    else:
                        return VeoError(
                            code=ErrorCode.SAFETY_FILTER,
                            message=f"Safety filter triggered: {filtered_reasons}",
                            user_message="Content was filtered for safety reasons.",
                            details=details,
                            recoverable=True,
                            suggestion="Try with different images or content."
                        )
                
                if filtered_count and filtered_count > 0:
                    details["filtered_count"] = filtered_count
                    return VeoError(
                        code=ErrorCode.SAFETY_FILTER,
                        message=f"Content filtered ({filtered_count} items)",
                        user_message="Some content was filtered.",
                        details=details,
                        recoverable=True,
                        suggestion="Try with different images."
                    )
                
                # Check for generated videos
                generated_videos = getattr(response, "generated_videos", None)
                if not generated_videos or len(generated_videos) == 0:
                    return VeoError(
                        code=ErrorCode.VIDEO_GENERATION_FAILED,
                        message="No videos generated in response",
                        user_message="Video generation failed - no output received.",
                        details=details,
                        recoverable=True,
                        suggestion="Try again with different parameters or images."
                    )
            
        except Exception as e:
            # Error while checking operation - classify that error
            return self.classify_exception(e, {"operation_check_failed": True, **context})
        
        # No error found
        return None


# Singleton error handler
error_handler = ErrorHandler()


def format_error_for_user(error: VeoError) -> str:
    """Format error for user-friendly display"""
    lines = [
        f"❌ {error.user_message}",
        f"",
        f"Error Code: {error.code.value}",
    ]
    
    if error.recoverable:
        lines.append(f"Status: Retrying automatically...")
    else:
        lines.append(f"Status: Manual intervention required")
    
    lines.append(f"")
    lines.append(f"💡 Suggestion: {error.suggestion}")
    
    return "\n".join(lines)


def format_error_for_log(error: VeoError) -> str:
    """Format error for detailed logging"""
    lines = [
        f"[{error.code.value}] {error.message}",
        f"Recoverable: {error.recoverable}",
        f"Details: {error.details}",
    ]
    
    if "traceback" in error.details:
        lines.append(f"Traceback:\n{error.details['traceback']}")
    
    return "\n".join(lines)


def is_openai_error(exception: Exception) -> bool:
    """
    Check if an exception is from OpenAI (should NOT fail the job).
    
    OpenAI is optional - used for prompt tuning. If it fails, the job
    should continue with default prompts, not fail entirely.
    
    Returns True if the exception is from OpenAI (by module or pattern match).
    """
    error_str = str(exception).lower()
    
    # Check exception module - most reliable method
    exception_module = type(exception).__module__ if hasattr(type(exception), '__module__') else ''
    if 'openai' in exception_module.lower():
        return True
    
    # Check error message patterns
    patterns = ErrorHandler.OPENAI_AUTH_PATTERNS
    for pattern in patterns:
        if re.search(pattern, error_str):
            return True
    
    return False