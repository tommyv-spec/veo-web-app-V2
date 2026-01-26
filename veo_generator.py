# -*- coding: utf-8 -*-
"""
Veo 3.1 Professional Generator
Implements "Enrichment -> Translation -> Routing" Workflow

Architecture:
1. ENRICHMENT: Expand brief user context into forensic details
2. TRANSLATION: Rewrite visual description with context as primary anchor
3. ROUTING: Map enriched details to specific JSON Blueprint slots

This ensures user context is the DOMINANT driver of all generation.
"""

import os
import re
import time
import random
import hashlib
import mimetypes
import base64
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Any, Callable
from datetime import datetime

def vlog(msg):
    """Log with immediate flush"""
    print(msg, flush=True)

# Google GenAI - Optional import
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Video generation disabled.")


def describe_subject_for_continuity(image_path: str) -> str:
    """Describe subject for continuity across clips (stub for compatibility)"""
    return ""


from config import (
    VideoConfig, APIKeysConfig, DialogueLine,
    VEO_MODEL, OPENAI_MODEL, SUPPORTED_IMAGE_FORMATS,
    BASE_PROMPT, NO_TEXT_INSTRUCTION, AUDIO_TIMING_INSTRUCTION,
    AUDIO_QUALITY_INSTRUCTION, PRONUNCIATION_TEMPLATE,
    ErrorCode, ClipStatus
)
from error_handler import ErrorHandler, VeoError, error_handler

# ===================== OPENAI CLIENT =====================

_openai_client = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_openai_client(api_key: Optional[str] = None):
    """Get or create OpenAI client"""
    global _openai_client
    
    if OpenAI is None:
        return None
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    if _openai_client is None:
        try:
            _openai_client = OpenAI(api_key=api_key)
        except Exception:
            return None
    
    return _openai_client


# ===================== STEP 1: ENRICHMENT ENGINE =====================

def process_user_context(
    user_context: str,
    language: str,
    openai_key: Optional[str] = None
) -> dict:
    """
    STEP 1: ENRICHMENT
    
    Takes brief user input (e.g., "he is very angry" or "news anchor reporting") 
    and EXPANDS it into forensic details for each Expert Network (Visual, Audio, Motion).
    
    This is the KEY function - it transforms simple directions into
    specific, actionable instructions that Veo can follow.
    
    Now also extracts SPEAKER ROLE for voice casting.
    """
    if not user_context or not user_context.strip():
        return {}

    client = get_openai_client(openai_key)
    if client is None:
        # Fallback: map raw text to all fields
        return {
            "subject_action": user_context,
            "facial_expression": user_context,
            "voice_tone": user_context,
            "delivery_style": user_context,
            "atmosphere": user_context,
            "body_language": user_context,
            "background_action": "",
            "camera_motion": "",
            "speaker_role": "",
        }

    try:
        system_msg = """You are a Director for Veo 3.1 (Google's AI Video Generator).

Your job is to EXPAND brief user directions into SPECIFIC, REALISTIC details.

The user might say something simple like "he is angry" or "nervous interview" or "fitness coach explaining".
You must expand this into detailed instructions for EACH aspect of the video.

CRITICAL: Be SPECIFIC and REALISTIC. Describe what you would actually SEE in real life.
- Don't say "intense cinematic gaze" - say "narrowed eyes, looking directly at camera"
- Don't say "dramatic tension" - say "shoulders raised, jaw tight"

SPEAKER ROLE EXTRACTION:
Look for any profession, role, or archetype in the user's description:
- "news anchor" → formal, authoritative delivery
- "fitness influencer" → energetic, motivational, upbeat
- "doctor explaining" → calm, reassuring, professional
- "teacher" → clear, patient, educational
- "salesperson" → enthusiastic, persuasive
- "meditation guide" → soft, calm, soothing
If no specific role mentioned, use "natural speaker"

OUTPUT JSON with these fields:
{
  "speaker_role": "The role/profession/archetype (e.g., 'news anchor', 'fitness coach', 'doctor', 'natural speaker')",
  "subject_action": "What the person physically does (e.g., 'leaning forward, pointing finger', 'sitting still, hands folded')",
  "facial_expression": "Realistic facial details (e.g., 'furrowed brow, tight lips, narrowed eyes')",
  "voice_tone": "How the voice sounds - MUST match the speaker_role (e.g., news anchor = 'clear, authoritative, measured', fitness coach = 'energetic, loud, motivational')",
  "delivery_style": "How they speak - MUST match the speaker_role (e.g., news anchor = 'formal pacing, clear enunciation', fitness coach = 'fast-paced, encouraging, punchy')",
  "body_language": "Posture and gestures (e.g., 'arms crossed, leaning back' or 'hands gesturing while talking')",
  "background_action": "What happens in background (e.g., 'nothing, static background')",
  "camera_motion": "Camera movement (e.g., 'static, no movement' or 'slight zoom')",
  "atmosphere": "Lighting (e.g., 'normal indoor lighting' or 'bright daylight')"
}

Keep it REALISTIC and NATURAL. No dramatic or cinematic language."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"USER DIRECTION: {user_context}\nLANGUAGE: {language}\n\nExpand this into detailed video production instructions. Be sure to identify the speaker_role."}
            ],
            temperature=0.7,  # Higher temp for creative expansion
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        vlog(f"[ENRICHMENT] Expanded '{user_context}' into: {json.dumps(result, indent=2)[:500]}...")
        return result

    except Exception as e:
        vlog(f"[ENRICHMENT] Error: {e}")
        return {
            "subject_action": user_context,
            "facial_expression": user_context,
            "voice_tone": user_context,
            "atmosphere": user_context,
            "speaker_role": "",
        }


# ===================== STEP 2: FRAME ANALYSIS =====================

def analyze_frame(image_path: str, openai_key: Optional[str] = None) -> dict:
    """
    COMPREHENSIVE FRAME ANALYSIS
    
    Extracts EVERYTHING needed from the image automatically:
    - Subject: age, gender, appearance, clothing
    - Role: detected profession/archetype based on visual cues
    - Action: what they're doing, interacting with
    - Objects: microphone, props, items they're holding/using
    - Setting: location, environment, context
    - Mood: apparent emotional state from expression/body language
    - Voice suggestion: appropriate voice based on all of the above
    
    This is the DEFAULT - works without any user input.
    User context can ADD to or OVERRIDE any of these.
    
    IMPORTANT: Avoids any mentions of audience, crowd, stage performance context
    to prevent video generation from adding fake audience sounds.
    """
    client = get_openai_client(openai_key)
    
    # Default fallback
    default_result = {
        # Subject
        "subject_age": "adult",
        "subject_gender": "neutral",
        "subject_appearance": "person in frame",
        "subject_clothing": "",
        
        # Role & Action
        "apparent_role": "natural speaker",
        "current_action": "speaking to camera",
        "body_language": "neutral posture",
        "facial_expression": "neutral expression",
        
        # Objects & Interaction
        "objects_in_scene": "",
        "objects_interacting_with": "",
        "props": "",
        
        # Setting & Environment
        "setting_location": "indoor",
        "setting_type": "studio",
        "background_description": "",
        "lighting": "natural lighting",
        "atmosphere": "neutral",
        
        # Voice Suggestions (auto-generated)
        "suggested_voice_tone": "clear, natural, conversational",
        "suggested_delivery": "measured pace, professional",
        "suggested_energy": "moderate",
        
        # Meta
        "visual_description": "",
        "confidence": "low"
    }
    
    if client is None:
        return default_result
    
    path = Path(image_path)
    if not path.exists():
        return default_result
    
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        system_msg = """You are an expert Scene Analyst for video production.

Analyze this frame and extract EVERYTHING needed for video generation.

=== CRITICAL AUDIO INSTRUCTION ===
This is for a SOLO SPEAKER video with COMPLETELY ISOLATED AUDIO.
The final video will have ONLY the speaker's voice - NO OTHER SOUNDS.

NEVER mention, imply, or suggest ANY of these in your analysis:
- Audience, crowd, spectators, attendees, viewers, listeners
- Stage, podium, conference, presentation, keynote, TED talk, seminar
- Live performance, public speaking event, speech, lecture
- Applause, reactions, laughter, cheering, clapping
- Background music, ambient sounds, environmental audio
- Any suggestion that other people are present or listening
- Entertainment setting, talk show, interview set, panel

Instead, ALWAYS describe settings as:
- "conference stage" → "professional studio with backdrop"
- "talk show set" → "professional recording studio"
- "speaking to audience" → "speaking directly to camera"
- "presentation setting" → "solo recording environment"
- "keynote speaker" → "professional speaker" or "subject matter expert"
- "interview setting" → "studio recording setup"

=== WHAT TO ANALYZE ===

1. SUBJECT (the person):
   - Age range (young adult, middle-aged, older adult, etc.)
   - Gender
   - Appearance (hair, facial features, build)
   - Clothing (what they're wearing - this helps identify role)
   - Facial expression (specific: furrowed brow, slight smile, etc.)
   - Body language (posture, hand position, stance)

2. ROLE DETECTION (based on visual cues):
   Look at clothing + setting + props to determine role:
   - Suit + professional backdrop = business professional / executive
   - Suit + desk = corporate professional
   - Scrubs + medical setting = doctor / nurse
   - Workout clothes = fitness instructor / athlete
   - Casual + ring light = content creator / vlogger
   - Uniform = specific profession
   - Casual clothes = everyday person / interviewee
   
3. ACTION & OBJECTS:
   - What are they doing right now? (always "speaking to camera" or similar)
   - What objects are visible? (microphone, desk, equipment, props)
   - What are they interacting with or holding?
   - Any relevant props?

4. SETTING & ENVIRONMENT:
   - Where is this? (studio, office, gym, outdoor, home, etc.)
   - What's in the background? (describe as studio/recording setup, NOT live event)
   - Lighting quality and type
   - Overall atmosphere/mood of the scene

5. VOICE SUGGESTIONS:
   Based on ALL of the above, suggest:
   - Voice tone that would match this person and role
   - Delivery style appropriate for the context
   - Energy level (calm, moderate, high energy)

=== OUTPUT JSON ===
{
  "subject_age": "specific age range",
  "subject_gender": "male / female",
  "subject_appearance": "brief description of how they look",
  "subject_clothing": "what they're wearing",
  "facial_expression": "specific expression details",
  "body_language": "posture and stance",
  
  "apparent_role": "detected role/profession based on visual cues",
  "current_action": "what they appear to be doing - ALWAYS speaking to camera",
  
  "objects_in_scene": "list of visible objects",
  "objects_interacting_with": "what they're holding or using",
  "props": "any notable props",
  
  "setting_location": "indoor / outdoor / studio",
  "setting_type": "specific type (office, gym, studio, etc.) - NEVER conference/stage/event/talk show",
  "background_description": "what's behind them - describe as studio setup, NOT live venue",
  "lighting": "lighting description",
  "atmosphere": "mood/feel of the scene - professional/warm/etc, NOT entertaining/lively",
  
  "suggested_voice_tone": "voice quality that matches this person and role",
  "suggested_delivery": "speaking style appropriate for context",
  "suggested_energy": "low / moderate / high",
  
  "visual_description": "50-word summary - describe as solo recording, NEVER as live event",
  "confidence": "high / medium / low - confidence in role detection"
}

Be SPECIFIC. Don't say "professional" - say "business executive" or "content creator".
Don't say "nice clothes" - say "dark blue suit with red tie".

BANNED WORDS (never use these): audience, crowd, stage, conference, keynote, presentation, applause, spectators, laughter, cheering, talk show, interview set, panel, seminar, lecture, entertainment."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this frame completely. Extract all details for video generation. Remember: describe as a solo recording setup, never as a live event with audience."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        
        # POST-PROCESS: Sanitize any audience/audio-related terms that might have slipped through
        # These terms can trigger Veo to add fake laughter, applause, or background music
        audio_trigger_terms = [
            'audience', 'crowd', 'spectator', 'attendee', 'viewer', 'listener',
            'stage', 'podium', 'conference', 'keynote', 'presentation', 'seminar', 'lecture',
            'ted talk', 'talk show', 'interview set', 'panel', 'forum',
            'applause', 'laughter', 'cheering', 'clapping', 'reactions',
            'live event', 'public speaking', 'entertainment', 'performance',
            'show', 'broadcast', 'program'
        ]
        
        for key, value in result.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for term in audio_trigger_terms:
                    if term in value_lower:
                        # Replace problematic descriptions with neutral alternatives
                        if key == 'setting_type':
                            result[key] = 'professional recording studio'
                        elif key == 'background_description':
                            result[key] = 'clean professional backdrop'
                        elif key == 'current_action':
                            result[key] = 'speaking directly to camera'
                        elif key == 'atmosphere':
                            result[key] = 'professional focused'
                        elif key == 'visual_description':
                            # Clean up the description
                            cleaned = value
                            for t in audio_trigger_terms:
                                cleaned = cleaned.lower().replace(t, 'studio')
                            result[key] = cleaned
                        elif key == 'apparent_role':
                            if 'host' in value_lower or 'presenter' in value_lower:
                                result[key] = 'professional speaker'
                        vlog(f"[FRAME ANALYSIS] Sanitized '{term}' from {key}")
                        break
        
        # Log the analysis
        vlog(f"\n[FRAME ANALYSIS] === Auto-detected from image ===")
        vlog(f"  Subject: {result.get('subject_age', '?')} {result.get('subject_gender', '?')}")
        vlog(f"  Clothing: {result.get('subject_clothing', '?')}")
        vlog(f"  Role: {result.get('apparent_role', '?')} (confidence: {result.get('confidence', '?')})")
        vlog(f"  Action: {result.get('current_action', '?')}")
        vlog(f"  Setting: {result.get('setting_type', '?')} ({result.get('setting_location', '?')})")
        vlog(f"  Objects: {result.get('objects_in_scene', '?')}")
        vlog(f"  Expression: {result.get('facial_expression', '?')}")
        vlog(f"  Voice suggestion: {result.get('suggested_voice_tone', '?')}")
        
        return result
        
    except Exception as e:
        vlog(f"[FRAME ANALYSIS] Error: {e}")
        return default_result


# Legacy function for backward compatibility
@lru_cache(maxsize=512)
def describe_frame(image_path: str, openai_key: Optional[str] = None) -> str:
    """Legacy: Simple frame description. Use analyze_frame() for full analysis."""
    analysis = analyze_frame(image_path, openai_key)
    return analysis.get('visual_description', '')


def analyze_dialogue_for_gestures(dialogue_line: str, language: str, openai_key: Optional[str] = None) -> dict:
    """
    Analyze dialogue to determine appropriate expressions, gestures, and delivery.
    
    Returns dict with:
    - emotion: primary emotion
    - intensity: low, medium, high
    - suggested_expression: facial expression
    - suggested_gestures: hand/body gestures
    - suggested_posture: body posture
    - delivery_style: how to deliver the line
    """
    default_result = {
        "emotion": "neutral",
        "intensity": "medium",
        "suggested_expression": "natural engaged expression",
        "suggested_gestures": "natural hand movements",
        "suggested_posture": "upright attentive posture",
        "delivery_style": "conversational natural delivery"
    }
    
    client = get_openai_client(openai_key)
    if client is None:
        return default_result
    
    try:
        system_msg = """You analyze dialogue lines to determine appropriate non-verbal communication.

Given a line of dialogue, determine:
1. EMOTION: Primary emotion (excited, happy, sad, angry, frustrated, thoughtful, surprised, worried, confident, neutral, empathetic, persuasive, curious, skeptical)
2. INTENSITY: How strongly expressed (low, medium, high)
3. EXPRESSION: Specific facial expression (e.g., "raised eyebrows with slight smile", "furrowed brow")
4. GESTURES: Hand/arm gestures (e.g., "open palms", "pointing for emphasis", "hands together")
5. POSTURE: Body posture (e.g., "leaning forward", "relaxed shoulders", "slight head tilt")
6. DELIVERY: How to speak it (e.g., "slow and deliberate", "energetic", "soft and warm")

Respond ONLY with valid JSON."""

        user_msg = f"""Dialogue line ({language}): "{dialogue_line}"

Respond with JSON:
{{
  "emotion": "...",
  "intensity": "...",
  "suggested_expression": "...",
  "suggested_gestures": "...",
  "suggested_posture": "...",
  "delivery_style": "..."
}}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        
        result_text = resp.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        vlog(f"[DIALOGUE ANALYSIS] Emotion: {result.get('emotion')}, Intensity: {result.get('intensity')}")
        
        return {
            "emotion": result.get("emotion", "neutral"),
            "intensity": result.get("intensity", "medium"),
            "suggested_expression": result.get("suggested_expression", default_result["suggested_expression"]),
            "suggested_gestures": result.get("suggested_gestures", default_result["suggested_gestures"]),
            "suggested_posture": result.get("suggested_posture", default_result["suggested_posture"]),
            "delivery_style": result.get("delivery_style", default_result["delivery_style"])
        }
        
    except Exception as e:
        vlog(f"[DIALOGUE ANALYSIS] Error: {e}")
        return default_result


# ===================== STEP 3: VISUAL TRANSLATION =====================

def build_visual_description(
    base_prompt: str,
    frame_desc: str,
    enriched_context: dict,
    dialogue_line: str,
    language: str,
    openai_key: Optional[str] = None
) -> str:
    """
    STEP 2: TRANSLATION
    
    Rewrites the visual description to ensure USER CONTEXT is the
    DOMINANT anchor. Focuses on REALISTIC, NATURAL output.
    """
    client = get_openai_client(openai_key)
    if client is None:
        # Fallback: combine base with context
        action = enriched_context.get('subject_action', '')
        expression = enriched_context.get('facial_expression', '')
        return f"{base_prompt}. {action}. {expression}."
    
    try:
        system_msg = """You are writing a shot description for Veo 3.1 video generation.

Write a SINGLE paragraph (50-70 words) describing what happens in the video.

CRITICAL RULES:
1. Write for REALISTIC, NATURAL output - NOT cinematic or dramatic
2. Describe exactly what you see: the person, their expression, their action
3. Use simple, direct language - no film terminology
4. Focus on: What the person looks like, what they're doing, their expression
5. The ENRICHED CONTEXT details MUST be included

DO NOT use words like: cinematic, dramatic, atmospheric, moody, artistic
DO use words like: natural, realistic, authentic, genuine, real

Example good output:
"A middle-aged man in a blue shirt sits at a desk. His brow is furrowed and jaw clenched, showing frustration. He speaks directly to camera with an intense expression, gesturing with his hands occasionally."

Example bad output:
"A dramatic close-up captures the raw intensity of a weathered businessman, shadows dancing across his chiseled features as emotion pours from his soul."
"""

        user_msg = f"""BASE DESCRIPTION: {base_prompt}
FRAME: {frame_desc}
DIALOGUE: "{dialogue_line}"

=== ENRICHED CONTEXT (MUST INCLUDE) ===
ACTION: {enriched_context.get('subject_action', 'Speaking naturally')}
EXPRESSION: {enriched_context.get('facial_expression', 'Natural')}
BODY LANGUAGE: {enriched_context.get('body_language', 'Natural')}

Write a realistic, natural description. No cinematic language."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=200,
        )
        
        result = resp.choices[0].message.content.strip()
        vlog(f"[TRANSLATION] Visual description: {result[:200]}...")
        return result
        
    except Exception as e:
        vlog(f"[TRANSLATION] Error: {e}")
        return base_prompt


# ===================== STEP 4: VOICE PROFILE =====================

def generate_voice_profile(
    frame_analysis: dict,
    language: str,
    user_context_enriched: dict,
    openai_key: Optional[str] = None
) -> str:
    """
    Generate a CONCISE voice profile with only what Veo needs to generate voice.
    
    Veo only needs:
    - Gender/Age (affects pitch)
    - Pitch & Timbre (core voice quality)
    - Texture (raspy, smooth, etc.)
    - Tone (warm, authoritative, etc.)
    - Pacing (fast, slow, etc.)
    - Accent (if any)
    
    NO "consistency" instructions - Veo doesn't know about other clips.
    """
    client = get_openai_client(openai_key)
    
    # === EXTRACT FROM FRAME ANALYSIS (defaults) ===
    auto_age = frame_analysis.get('subject_age', 'adult')
    auto_gender = frame_analysis.get('subject_gender', 'neutral')
    auto_role = frame_analysis.get('apparent_role', 'natural speaker')
    auto_voice_tone = frame_analysis.get('suggested_voice_tone', 'clear, natural')
    auto_delivery = frame_analysis.get('suggested_delivery', 'measured pace')
    detection_confidence = frame_analysis.get('confidence', 'low')
    
    # === EXTRACT FROM USER CONTEXT (overrides) ===
    user_role = user_context_enriched.get('speaker_role', '')
    user_voice_tone = user_context_enriched.get('voice_tone', '')
    user_delivery = user_context_enriched.get('delivery_style', '')
    user_accent = user_context_enriched.get('accent', '')
    
    # === MERGE: User context overrides auto-detected ===
    final_role = user_role if user_role else auto_role
    final_voice_tone = user_voice_tone if user_voice_tone else auto_voice_tone
    final_delivery = user_delivery if user_delivery else auto_delivery
    
    vlog(f"[VOICE CASTING] Role: {final_role}, Accent: {user_accent if user_accent else 'NONE'}")
    
    if client is None:
        return build_voice_profile_template(
            age=auto_age,
            gender=auto_gender,
            language=language,
            role=final_role,
            tone=final_voice_tone,
            delivery=final_delivery,
            user_accent=user_accent
        )
    
    try:
        system_msg = f"""You create CONCISE voice profiles for AI video generation.

OUTPUT FORMAT (use exactly this structure):
---
Gender: [male/female]
Age: [specific like "mid-40s" or "early 30s"]
Pitch: [low/medium/high, e.g. "low-medium, ~150Hz"]
Timbre: [warm/bright/dark/rich] with [chest/head/balanced] resonance
Texture: [smooth/raspy/breathy/crisp/velvety] - [any unique quality]
Tone: [confident/warm/authoritative/friendly/calm]
Pacing: [slow/moderate/fast], [steady/varied] rhythm
Accent: {user_accent if user_accent else "Neutral, no regional accent"}
---

RULES:
1. Keep it SHORT - only voice characteristics
2. NO instructions about "consistency" or "all clips" - the AI doesn't know about other clips
3. NO audio quality notes - that's handled separately
4. Be SPECIFIC with descriptors"""

        user_msg = f"""Create a voice profile for:
- Detected: {auto_age} {auto_gender}
- Role: {final_role}
- Tone preference: {final_voice_tone}
- Delivery: {final_delivery}
- Language: {language}
- Accent: {user_accent if user_accent else "None - neutral"}

Output the voice profile in the exact format specified."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception as e:
        vlog(f"[VOICE PROFILE] Error: {e}")
        return build_voice_profile_template(
            age=auto_age,
            gender=auto_gender,
            language=language,
            role=final_role,
            tone=final_voice_tone,
            delivery=final_delivery,
            user_accent=user_accent
        )


def build_voice_profile_template(
    age: str,
    gender: str,
    language: str,
    role: str,
    tone: str,
    delivery: str,
    user_accent: str = ""
) -> str:
    """Build a CONCISE voice profile - only what Veo needs."""
    
    accent = user_accent if user_accent else "Neutral, no regional accent"
    
    return f"""---
Gender: {gender}
Age: {age}
Pitch: Medium, warm
Timbre: Clear with balanced resonance
Texture: Smooth, professional
Tone: {tone}
Pacing: Moderate, {delivery}
Accent: {accent}
---"""


def get_default_voice_profile(language: str, user_context: str = "") -> str:
    """Default CONCISE voice profile."""
    
    return f"""---
Gender: Match the person in image
Age: Adult
Pitch: Medium
Timbre: Warm and clear
Texture: Smooth, natural
Tone: Professional, confident
Pacing: Moderate, conversational
Accent: Neutral {language}, no regional accent
---"""


# ===================== STEP 5: PROMPT ASSEMBLY (ROUTING) =====================

def build_prompt(
    dialogue_line: str,
    start_frame_path: Path,
    end_frame_path: Optional[Path],
    clip_index: int,
    language: str,
    voice_profile: str,
    config: VideoConfig,
    openai_key: Optional[str] = None,
    frame_analysis: Optional[dict] = None,
    user_context_override: Optional[dict] = None,
    redo_feedback: Optional[str] = None,  # User's feedback for redo
    override_duration: Optional[str] = None,  # Override duration for last clip
) -> str:
    """
    VEO 3.1 PROMPT BUILDER
    
    KEY PRINCIPLES:
    1. 🚫 NO VISUAL REDESCRIPTION: The image locks appearance
    2. ✅ RAW/DOCUMENTARY STYLE: Not "cinematic" - prevents AI glossy look
    3. ✅ STATIC CAMERA: For talking heads, locked-off camera preserves lip-sync
    4. ✅ VOICE PROFILE: Extract and pass voice traits correctly
    5. ✅ "Character says:" syntax for Veo lip-sync engine
    
    Priority: Redo feedback > User context > Dialogue analysis > Frame analysis > Defaults
    """
    vlog(f"[ROUTING] Building prompt for clip {clip_index}...")
    if redo_feedback:
        vlog(f"[ROUTING] REDO FEEDBACK: {redo_feedback}")

    # === 1. GET DEFAULTS FROM FRAME ANALYSIS ===
    if frame_analysis is None:
        frame_analysis = {}
    
    if user_context_override is None:
        user_context_override = {}
    
    # === 2. ANALYZE DIALOGUE FOR GESTURES/EXPRESSIONS ===
    dialogue_analysis = {}
    if config.use_openai_prompt_tuning:
        dialogue_analysis = analyze_dialogue_for_gestures(dialogue_line, language, openai_key)
    
    # === 3. MERGE: User context > Dialogue analysis > Frame analysis > Defaults ===
    def get_value(key, frame_key=None, dialogue_key=None, default=""):
        if frame_key is None:
            frame_key = key
        if dialogue_key is None:
            dialogue_key = key
            
        user_val = user_context_override.get(key, "")
        if user_val:
            return user_val
            
        dialogue_val = dialogue_analysis.get(dialogue_key, "")
        if dialogue_val:
            return dialogue_val
            
        frame_val = frame_analysis.get(frame_key, "")
        if frame_val:
            return frame_val
            
        return default
    
    # === 4. EXTRACT VALUES ===
    facial_expression = get_value("facial_expression", "facial_expression", "suggested_expression", "natural engaged expression")
    body_language = get_value("body_language", "body_language", "suggested_posture", "natural posture")
    suggested_gestures = dialogue_analysis.get("suggested_gestures", "natural hand movements")
    
    # Emotion and delivery
    emotion = dialogue_analysis.get("emotion", "neutral")
    intensity = dialogue_analysis.get("intensity", "medium")
    delivery_style = get_value("delivery_style", "suggested_delivery", "delivery_style", "natural conversational")
    
    # === 5. EXTRACT VOICE PROFILE PROPERLY ===
    # Parse the voice profile to extract key traits for Veo
    voice_texture = ""
    voice_tone = ""
    voice_accent = ""
    voice_signature = ""
    
    if voice_profile:
        lines = voice_profile.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            line_clean = line.strip()
            
            # Extract texture (raspy, smooth, gravelly, etc.)
            if 'texture:' in line_lower:
                voice_texture = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            elif 'quality:' in line_lower and not voice_texture:
                voice_texture = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            
            # Extract tone
            if 'tone:' in line_lower:
                voice_tone = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            
            # Extract accent
            if 'accent:' in line_lower:
                accent_val = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
                if accent_val and 'none' not in accent_val.lower() and 'neutral' not in accent_val.lower():
                    voice_accent = accent_val
            
            # Extract signature traits
            if 'signature' in line_lower and 'trait' in line_lower:
                voice_signature = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
    
    # Build consolidated voice instruction for Veo
    voice_parts = []
    if voice_texture:
        voice_parts.append(voice_texture)
    if voice_tone:
        voice_parts.append(voice_tone)
    if voice_signature:
        voice_parts.append(voice_signature)
    if voice_accent:
        voice_parts.append(f"accent: {voice_accent}")
    
    voice_instruction = ", ".join(voice_parts) if voice_parts else "natural authentic voice"
    
    vlog(f"[VOICE] Extracted - Texture: {voice_texture}, Tone: {voice_tone}, Accent: {voice_accent}")
    vlog(f"[VOICE] Final instruction: {voice_instruction}")
    
    # === 6. CALCULATE TIMING ===
    if override_duration:
        duration = float(override_duration)
    else:
        duration = float(config.duration.value if hasattr(config.duration, 'value') else config.duration)
    speech_end_time = duration - 1.0
    
    # === 7. BUILD THE PROMPT IN VEO 3.1 OFFICIAL FORMAT ===
    # Google's formula: [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
    # Dialogue format: The character says, "text here"
    # Audio format: SFX: description, Ambient noise: description
    
    # Simplified voice instruction - extract key trait only
    short_voice = ""
    if voice_texture:
        short_voice = voice_texture
    elif voice_tone:
        short_voice = voice_tone
    elif voice_signature:
        short_voice = voice_signature
    if voice_accent:
        short_voice = f"{short_voice}, {voice_accent}" if short_voice else voice_accent
    if not short_voice:
        short_voice = "natural voice"
    
    # Build prompt following Veo 3.1 official structure
    final_prompt = f"""=== VOICE PROFILE ===
{voice_profile}
===

Medium shot, static locked-off camera, sharp focus on subject.

The subject in the frame speaks directly to camera with {facial_expression}, {body_language}.

The character says in {language}, "{dialogue_line}"

Voice: {short_voice}. {delivery_style}, {emotion} emotion.

Ambient noise: Complete silence, professional recording booth, no room ambiance.

Style: Raw realistic footage, natural lighting, photorealistic. Speech timing: 0s to {speech_end_time:.1f}s, then silence.

No subtitles, no text overlays, no captions, no watermarks. No background music, no laughter, no applause, no crowd sounds, no ambient noise. No morphing, no face distortion, no jerky movements. Only the speaker's isolated voice.

(no subtitles)"""

    # Add redo feedback at the top if present
    if redo_feedback:
        final_prompt = f"""=== PRIORITY ===
{redo_feedback}
===

{final_prompt}"""
    
    # Log the FULL prompt (this will be sent to websocket)
    vlog(f"\n{'='*80}")
    vlog(f"[FULL PROMPT] CLIP {clip_index}")
    vlog(f"{'='*80}")
    vlog(final_prompt)
    vlog(f"{'='*80}\n")
    
    return final_prompt


# ===================== HELPERS =====================

def list_images(images_dir: Path, config: VideoConfig) -> List[Path]:
    """List and sort images in directory.
    
    IMPORTANT: Callers should use safe_images_dir() helper before calling this
    to avoid passing Path(".") which would search the current directory.
    """
    # Safety check: Detect if someone passed Path("") which becomes Path(".")
    if str(images_dir) == "." or str(images_dir) == "":
        raise ValueError(
            f"Invalid images_dir: '{images_dir}' - this appears to be an empty path. "
            "Flow jobs have images in R2, not local disk. Use safe_images_dir() helper."
        )
    
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")
    
    # Wrap in try/except to handle race condition where directory is deleted
    # between exists() check and iterdir() call
    try:
        files = [
            p for p in images_dir.iterdir() 
            if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ]
    except FileNotFoundError:
        # Directory was deleted between exists() check and iterdir()
        raise ValueError(
            f"Images directory was deleted: {images_dir}. "
            "Original files are no longer available. Please create a new job with re-uploaded images."
        )
    except OSError as e:
        # Handle other OS errors (permission denied, etc.)
        raise ValueError(f"Cannot access images directory {images_dir}: {e}")
    
    # Also wrap sorting in try/except in case files are deleted during sort
    try:
        if config.images_sort_key == "date":
            files.sort(key=lambda p: p.stat().st_mtime, reverse=config.images_sort_reverse)
        else:
            files.sort(key=lambda p: p.name.lower(), reverse=config.images_sort_reverse)
    except FileNotFoundError:
        # A file was deleted during sorting - re-filter the list
        files = [f for f in files if f.exists()]
        if config.images_sort_key == "date":
            files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=config.images_sort_reverse)
        else:
            files.sort(key=lambda p: p.name.lower(), reverse=config.images_sort_reverse)
    
    return files


def get_mime_type(path: Path) -> str:
    """Get MIME type for image"""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error, no-keys-available error, or transient overload error"""
    s = str(exception).lower()
    # Standard rate limit errors
    if "429" in s or "resource_exhausted" in s:
        return True
    # Our custom "no keys available" error from _get_client()
    if "no api keys available" in s or "all keys are rate-limited" in s:
        return True
    # Model overloaded / service unavailable (gRPC code 14 = UNAVAILABLE)
    if "overloaded" in s or "'code': 14" in s or "code: 14" in s:
        return True
    # Other transient errors that should trigger retry
    if "unavailable" in s or "temporarily" in s:
        return True
    return False


def is_transient_error(exception: Exception) -> bool:
    """Check if exception is a transient error that should trigger retry (not key rotation)"""
    s = str(exception).lower()
    # Model overloaded - retry with same key after backoff
    if "overloaded" in s or "'code': 14" in s or "code: 14" in s:
        return True
    if "unavailable" in s or "temporarily" in s:
        return True
    return False


def is_celebrity_error(operation) -> bool:
    """Check if operation failed due to celebrity filter"""
    try:
        op_str = str(operation).lower()
        keywords = ["celebrity", "likenesses", "rai_media_filtered", "filtered_reasons"]
        
        if any(kw in op_str for kw in keywords):
            return True
        
        resp = getattr(operation, "response", None)
        if resp:
            if getattr(resp, "rai_media_filtered_count", 0) > 0:
                return True
            if getattr(resp, "rai_media_filtered_reasons", None):
                return True
    except Exception:
        pass
    
    return False


def modify_image_for_celebrity_bypass(image_path: Path, api_keys: List[str], attempt: int = 1) -> Optional[Path]:
    """
    Modify an image using PIL to bypass celebrity filter false positives.
    
    Strategy: Apply simple image transformations that change how the face is perceived
    by the filter while preserving visual quality. Uses PIL instead of AI for speed
    and reliability.
    
    IMPORTANT: Always works from the ORIGINAL image (without _celeb_bypass in name)
    to prevent cumulative modifications.
    
    Args:
        image_path: Path to the image (may be original or already modified)
        api_keys: List of Gemini API keys (unused, kept for API compatibility)
        attempt: Which bypass attempt this is (affects modification strategy)
    
    Returns:
        Path to modified image, or None if modification failed
    """
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import io
        
        # CRITICAL: Always find and use the ORIGINAL image
        # This prevents infinite modification chains like _celeb_bypass_1_celeb_bypass_1...
        original_path = image_path
        original_stem = image_path.stem
        
        # Strip any existing _celeb_bypass_N suffixes to find original
        while '_celeb_bypass_' in original_stem:
            # Find the original by removing the last _celeb_bypass_N suffix
            parts = original_stem.rsplit('_celeb_bypass_', 1)
            original_stem = parts[0]
        
        # Look for original file with various extensions
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            potential_original = image_path.parent / f"{original_stem}{ext}"
            if potential_original.exists():
                original_path = potential_original
                break
        
        vlog(f"[Celebrity Bypass] Attempt {attempt}: Using original image {original_path.name}")
        
        # Load the ORIGINAL image
        img = Image.open(original_path)
        original_mode = img.mode
        original_size = img.size
        
        # Convert to RGB if necessary for processing
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparency
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Different modification strategies based on attempt number
        # Each strategy makes subtle changes that may affect face detection heuristics
        
        if attempt == 1:
            # Strategy 1: Slight crop (2% from each edge) + minor brightness adjustment
            vlog("[Celebrity Bypass] Strategy 1: Slight crop + brightness")
            w, h = img.size
            crop_px_w = int(w * 0.02)
            crop_px_h = int(h * 0.02)
            img = img.crop((crop_px_w, crop_px_h, w - crop_px_w, h - crop_px_h))
            img = img.resize(original_size, Image.Resampling.LANCZOS)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.03)  # 3% brighter
            
        elif attempt == 2:
            # Strategy 2: Slight rotation + color temperature shift
            vlog("[Celebrity Bypass] Strategy 2: Micro-rotation + warmth")
            img = img.rotate(0.5, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(128, 128, 128))
            # Warm up the image slightly (add red, reduce blue)
            r, g, b = img.split()
            r = r.point(lambda x: min(255, int(x * 1.02)))
            b = b.point(lambda x: int(x * 0.98))
            img = Image.merge('RGB', (r, g, b))
            
        elif attempt == 3:
            # Strategy 3: Subtle contrast + slight blur edge + JPEG compression
            vlog("[Celebrity Bypass] Strategy 3: Contrast + edge softening + recompress")
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.97)  # Slightly reduce contrast
            # Very subtle sharpening (opposite effect to blur, but changes pixel patterns)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
            
        elif attempt == 4:
            # Strategy 4: Mirror + unmirror (changes JPEG compression artifacts)
            vlog("[Celebrity Bypass] Strategy 4: Flip transform + saturation")
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Back to normal
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.95)  # Slightly desaturate
            
        else:
            # Strategy 5+: Combination with random seed-based noise
            vlog(f"[Celebrity Bypass] Strategy {attempt}: Combined adjustments")
            # Small crop
            w, h = img.size
            crop_px = int(min(w, h) * 0.01 * attempt)
            crop_px = min(crop_px, 20)  # Max 20px
            if crop_px > 0:
                img = img.crop((crop_px, crop_px, w - crop_px, h - crop_px))
                img = img.resize(original_size, Image.Resampling.LANCZOS)
            # Slight gamma adjustment
            factor = 1.0 + (0.02 * (attempt % 3 - 1))  # -2% to +2%
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # Save modified image with controlled JPEG quality to change compression artifacts
        modified_path = original_path.parent / f"{original_stem}_celeb_bypass_{attempt}.jpg"
        
        # Use JPEG with specific quality to introduce different compression artifacts
        quality = 92 - (attempt * 2)  # 90, 88, 86, 84...
        quality = max(quality, 80)  # Don't go below 80
        
        img.save(modified_path, 'JPEG', quality=quality, optimize=True)
        vlog(f"[Celebrity Bypass] Modified image saved: {modified_path.name} (JPEG quality={quality})")
        
        return modified_path
        
    except ImportError:
        vlog("[Celebrity Bypass] PIL/Pillow not installed, cannot modify image")
        return None
    except Exception as e:
        vlog(f"[Celebrity Bypass] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_next_clean_image(
    current_index: int,
    images_list: List[Path],
    blacklist: Set[Path],
    max_attempts: int = 10
) -> Optional[Tuple[int, Path]]:
    """Find next non-blacklisted image"""
    if not images_list:
        return None
    
    total = len(images_list)
    
    for offset in range(1, min(max_attempts + 1, total + 1)):
        new_index = (current_index + offset) % total
        candidate = images_list[new_index]
        
        if candidate not in blacklist:
            return (new_index, candidate)
    
    return None


def generate_output_filename(
    idx: int, 
    start_img: Path, 
    end_img: Optional[Path],
    timestamp: str = ""
) -> str:
    """Generate safe output filename"""
    def slugify(s: str) -> str:
        return re.sub(r"[^\w\-.]+", "_", s).strip("._")
    
    def short_stem(p: Path, n: int = 40) -> str:
        return slugify(p.stem)[:n]
    
    idx_str = str(idx)
    s1 = short_stem(start_img, 40)
    s2 = short_stem(end_img, 40) if end_img else ""
    
    base = f"{idx_str}_{s1}" + (f"_to_{s2}" if s2 else "")
    if timestamp:
        base = f"{base}_{timestamp}"
    
    base = slugify(base)
    if len(base) > 120:
        h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        base = f"{idx_str}_{s1[:40]}" + (f"_to_{s2[:40]}" if s2 else "") + f"_{h}"
    
    return f"{base}.mp4"


# ===================== CALLBACK TYPE =====================

ProgressCallback = Callable[[int, str, str, Optional[Dict]], None]


# ===================== MAIN GENERATOR CLASS =====================

class VeoGenerator:
    """
    Video generator with Enrichment -> Translation -> Routing workflow.
    
    Usage:
        generator = VeoGenerator(config, api_keys)
        generator.on_progress = my_callback_function
        result = generator.generate_single_clip(...)
    """
    
    def __init__(
        self,
        config: VideoConfig,
        api_keys: APIKeysConfig,
        openai_key: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        self.config = config
        self.api_keys = api_keys
        self.openai_key = openai_key
        self.job_id = job_id or f"job_{id(self)}"
        
        # Key pool integration - NEW DYNAMIC APPROACH
        # No static key reservations - all keys are shared dynamically
        from config import key_pool
        self.key_pool = key_pool
        self.reserved_keys: List[int] = []  # Legacy - kept for compatibility
        self._current_key_index: Optional[int] = None
        self._is_key_owner = True  # Always owner in dynamic mode
        
        # Check pool status at start (for logging/debugging)
        pool_status = key_pool.get_pool_status_summary(api_keys)
        if pool_status["available"] == 0:
            self._no_keys_at_start = True
            vlog(f"[VeoGenerator] WARNING: No keys available at start ({pool_status['rate_limited']} rate-limited, {pool_status['invalid']} invalid)")
        else:
            self._no_keys_at_start = False
            vlog(f"[VeoGenerator] Key pool: {pool_status['available']} available, {pool_status['rate_limited']} rate-limited, {pool_status['total']} total")
        
        self.blacklist: Set[Path] = set()  # Shared blacklist (used directly in sequential, as hint in parallel)
        self.celebrity_hints: Set[Path] = set()  # Global hint for celebrity-filtered images (parallel mode)
        self.voice_profile: Optional[str] = None
        self.voice_profile_id: Optional[str] = None  # Short ID for logging
        self.frame_analysis: Optional[dict] = None  # Auto-detected from frame (age, gender, role)
        self.enriched_context: Optional[dict] = None  # From user context (override)
        self.client: Optional[genai.Client] = None
        
        # Per-scene frame analysis cache (keyed by image path string)
        self.scene_frame_analyses: Dict[str, dict] = {}
        
        # Callbacks
        self.on_progress: Optional[ProgressCallback] = None
        self.on_error: Optional[Callable[[VeoError], None]] = None
        
        # State
        self.cancelled = False
        self.paused = False
    
    def cleanup(self):
        """Cleanup when job is done. With dynamic keys, just clear state."""
        # In dynamic mode, we don't need to release keys since they're not reserved
        # Just clear our state
        self.reserved_keys = []
        self._current_key_index = None
        vlog(f"[VeoGenerator] Cleanup complete for job {self.job_id[:8] if self.job_id else 'unknown'}")
    
    def get_frame_analysis_for_image(self, image_path: Path) -> dict:
        """
        Get frame analysis for a specific image, using cache if available.
        This ensures each scene's image is analyzed separately.
        """
        path_key = str(image_path)
        
        if path_key in self.scene_frame_analyses:
            vlog(f"[VeoGenerator] Using cached frame analysis for {image_path.name}")
            return self.scene_frame_analyses[path_key]
        
        # Analyze this image
        vlog(f"[VeoGenerator] Analyzing new scene image: {image_path.name}")
        if self.config.use_openai_prompt_tuning:
            analysis = analyze_frame(str(image_path), self.openai_key)
        else:
            analysis = {
                "subject_age": "adult",
                "subject_gender": "neutral", 
                "apparent_role": "natural speaker",
                "suggested_voice_tone": "natural, conversational",
                "suggested_delivery": "clear, measured",
                "visual_description": "",
                "confidence": "low"
            }
        
        # Cache it
        self.scene_frame_analyses[path_key] = analysis
        return analysis
    
    def initialize_voice_profile(self, reference_frame: Path) -> str:
        """
        Initialize voice profile ONCE per job. Must be called before generating clips.
        Returns a voice profile ID for tracking.
        """
        import hashlib
        
        user_context = getattr(self.config, 'user_context', '') or ''
        
        # === STEP 1: ANALYZE FRAME (auto-detect age, gender, role) ===
        # This is the DEFAULT - works even without user context
        if self.config.use_openai_prompt_tuning:
            self.frame_analysis = analyze_frame(str(reference_frame), self.openai_key)
        else:
            self.frame_analysis = {
                "subject_age": "adult",
                "subject_gender": "neutral", 
                "apparent_role": "natural speaker",
                "suggested_voice_tone": "natural, conversational",
                "suggested_delivery": "clear, measured",
                "visual_description": "",
                "confidence": "low"
            }
        
        # === STEP 2: ENRICH USER CONTEXT (if provided - this is the OVERRIDE) ===
        if user_context:
            self.enriched_context = process_user_context(
                user_context, self.config.language, self.openai_key
            )
        else:
            self.enriched_context = {}
        
        # === STEP 3: GENERATE VOICE PROFILE (frame analysis + user override) ===
        if self.config.use_openai_prompt_tuning:
            self.voice_profile = generate_voice_profile(
                self.frame_analysis,  # Auto-detected defaults
                self.config.language,
                self.enriched_context,  # User overrides (if any)
                self.openai_key
            )
        else:
            self.voice_profile = get_default_voice_profile(self.config.language, user_context)
        
        # Create a short ID based on the profile content
        profile_hash = hashlib.md5(self.voice_profile.encode()).hexdigest()[:8]
        self.voice_profile_id = f"VP-{profile_hash.upper()}"
        
        # === LOG EVERYTHING ===
        auto_role = self.frame_analysis.get('apparent_role', 'unknown')
        user_role = self.enriched_context.get('speaker_role', '')
        final_role = user_role if user_role else auto_role
        
        vlog(f"\n{'='*60}")
        vlog(f"[VOICE PROFILE INITIALIZED]")
        vlog(f"{'='*60}")
        vlog(f"Voice ID: {self.voice_profile_id}")
        vlog(f"")
        vlog(f"=== FRAME ANALYSIS (auto-detected) ===")
        vlog(f"  Subject: {self.frame_analysis.get('subject_age', '?')} {self.frame_analysis.get('subject_gender', '?')}")
        vlog(f"  Auto Role: {auto_role} (confidence: {self.frame_analysis.get('confidence', '?')})")
        vlog(f"  Auto Voice: {self.frame_analysis.get('suggested_voice_tone', '?')}")
        vlog(f"  Auto Delivery: {self.frame_analysis.get('suggested_delivery', '?')}")
        vlog(f"")
        if user_context:
            vlog(f"=== USER CONTEXT (override) ===")
            vlog(f"  Raw: '{user_context}'")
            vlog(f"  User Role: {user_role or '(not specified)'}")
            vlog(f"  User Voice: {self.enriched_context.get('voice_tone', '(not specified)')}")
            vlog(f"  User Delivery: {self.enriched_context.get('delivery_style', '(not specified)')}")
            vlog(f"")
        else:
            vlog(f"=== USER CONTEXT: None (using auto-detected) ===")
            vlog(f"")
        vlog(f"=== FINAL VOICE CASTING ===")
        vlog(f"  Final Role: {final_role}")
        vlog(f"")
        vlog(f"Generated Profile:")
        vlog(f"  {self.voice_profile}")
        vlog(f"{'='*60}\n")
        
        return self.voice_profile_id
    
    def _get_client(self) -> 'genai.Client':
        """Get or create Gemini client using DYNAMIC key selection.
        
        NEW: Uses fully dynamic key pool - any available key can be used by any job.
        This ensures keys are efficiently shared between parallel jobs.
        """
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-genai package not installed.")
        
        # DYNAMIC approach: Get any available key from the pool
        # This allows all jobs to share keys efficiently
        result = self.key_pool.get_any_available_key(self.api_keys)
        if result:
            key_index, api_key = result
            self._current_key_index = key_index
            key_suffix = api_key[-8:]
            vlog(f"[VeoGenerator] Using key {key_index + 1} (...{key_suffix})")
            return genai.Client(api_key=api_key)
        
        # No keys available - all are rate-limited or invalid
        raise ValueError("No API keys available - all keys are rate-limited or invalid")
    
    def _rotate_key(self, block_current: bool = True):
        """Mark current key as rate-limited in KeyPoolManager.
        
        The old APIKeysConfig system is NOT used - all key management goes through KeyPoolManager.
        """
        if self._current_key_index is not None and block_current:
            # Mark this key as rate-limited for 300 seconds (5 minutes)
            # Google's rate limits typically last 1-5 minutes
            self.key_pool.mark_key_rate_limited(self._current_key_index, duration_seconds=300)
            # Safe access with bounds check
            if self._current_key_index < len(self.api_keys.gemini_api_keys):
                key_suffix = self.api_keys.gemini_api_keys[self._current_key_index][-8:]
                vlog(f"[VeoGenerator] 🚫 Key {self._current_key_index + 1} (...{key_suffix}) rate-limited for 300s")
            else:
                vlog(f"[VeoGenerator] 🚫 Key {self._current_key_index + 1} rate-limited for 300s (key index out of range for suffix)")
        
        # Clear cached client so next _get_client() gets a fresh key
        self.client = None
    
    def _get_pool_status(self) -> tuple:
        """Get current status from KeyPoolManager - DYNAMIC (all keys, not just reserved)."""
        status = self.key_pool.get_pool_status_summary(self.api_keys)
        total = status["total"]
        rate_limited = status["rate_limited"]
        available = status["available"]
        return available, rate_limited, total
    
    def _emit_progress(self, clip_index: int, status: str, message: str, details: Dict = None):
        """Emit progress update"""
        if self.on_progress:
            self.on_progress(clip_index, status, message, details)
    
    def _emit_error(self, error: VeoError):
        """Emit error"""
        if self.on_error:
            self.on_error(error)
    
    def generate_single_clip(
        self,
        start_frame: Path,
        end_frame: Optional[Path],
        dialogue_line: str,
        dialogue_id: int,
        clip_index: int,
        output_dir: Path,
        images_list: List[Path],
        current_end_index: int,
        scene_image: Optional[Path] = None,  # Original scene image for analysis (may differ from start_frame in CONTINUE mode)
        redo_feedback: Optional[str] = None,  # User's feedback for redo - what should be different
        override_duration: Optional[str] = None,  # Override duration for this specific clip (e.g., "4" for last clip)
        generation_mode: str = "parallel",  # "parallel" or "sequential" - affects blacklist sharing
        on_frames_locked: Optional[Callable[[int, Path, Optional[Path]], None]] = None,  # Callback when frames are locked for generation
        frames_locked: bool = False,  # If True, frames cannot be swapped (Phase 2 staggered mode)
    ) -> Dict[str, Any]:
        """Generate a single video clip with retry logic.
        
        In PARALLEL mode: Uses local blacklist (per-clip) to avoid cross-clip interference
        In SEQUENTIAL mode: Uses shared blacklist for proper frame chaining
        
        on_frames_locked: Called with (clip_index, start_frame, end_frame) right before API call
        frames_locked: If True, prevents frame swapping (used in staggered Phase 2 where frames are confirmed)
        """
        
        vlog(f"[VeoGenerator] Generating clip {clip_index}: '{dialogue_line[:50]}...'")
        if redo_feedback:
            vlog(f"[VeoGenerator] Redo feedback: '{redo_feedback}'")
        if override_duration:
            vlog(f"[VeoGenerator] Using override duration: {override_duration}s")
        
        # In PARALLEL mode, use LOCAL blacklist to prevent cross-clip interference
        # But also check global celebrity_hints to avoid wasted API calls
        # In SEQUENTIAL mode, use shared blacklist for proper chaining
        if generation_mode == "parallel":
            # Start with known celebrity-filtered images as hints
            local_blacklist: Set[Path] = set(self.celebrity_hints)
            vlog(f"[Clip {clip_index}] Using LOCAL blacklist (parallel mode) with {len(self.celebrity_hints)} celebrity hints")
        else:
            local_blacklist = self.blacklist  # Reference to shared blacklist
            vlog(f"[Clip {clip_index}] Using SHARED blacklist (sequential mode)")
        
        result = {
            "success": False,
            "output_path": None,
            "end_frame_used": None,
            "end_index": current_end_index,
            "error": None,
            "prompt_text": None,
        }
        
        if not GENAI_AVAILABLE:
            result["error"] = VeoError(
                code=ErrorCode.UNKNOWN,
                message="google-genai not installed",
                user_message="Video generation library not available",
                details={},
                recoverable=False,
                suggestion="Contact administrator to install google-genai"
            )
            return result
        
        # Store redo feedback for use in prompt building
        self._current_redo_feedback = redo_feedback
        
        # Initialize voice profile if not already done
        # This should be called by the worker before generating clips, but fallback here
        if self.voice_profile is None:
            vlog(f"[WARNING] Voice profile not pre-initialized, initializing now for clip {clip_index}")
            self.initialize_voice_profile(start_frame)
        
        # Get frame analysis for THIS scene's image (not the start_frame which may be extracted)
        # scene_image is the original uploaded image for this scene
        # start_frame may be an extracted frame from previous clip (in CONTINUE mode)
        analysis_image = scene_image if scene_image else start_frame
        clip_frame_analysis = self.get_frame_analysis_for_image(analysis_image)
        
        # Log voice ID for this clip
        vlog(f"[Clip {clip_index}] Using Voice ID: {self.voice_profile_id}, Scene image: {analysis_image.name if hasattr(analysis_image, 'name') else 'unknown'}")
        
        # Check if any reserved keys are available BEFORE starting attempts
        available, rate_limited, total = self._get_pool_status()
        
        # Log pool status at clip start
        vlog(f"[Clip {clip_index}] Key pool: {available} working, {rate_limited} rate-limited (of {total} total)")
        
        if available == 0 and rate_limited > 0:
            vlog(f"[VeoGenerator] ⚠️ All {total} keys are rate-limited. Will wait for recovery...")
            # Don't fail immediately - get_any_available_key will handle this
        elif total == 0:
            vlog(f"[VeoGenerator] ❌ No API keys configured!")
            result["error"] = VeoError(
                code=ErrorCode.RATE_LIMIT,
                message="No API keys configured",
                user_message="No API keys are configured.",
                details={"total_keys": total},
                recoverable=False,
                suggestion="Check API key configuration"
            )
            result["no_keys"] = True
            self._emit_progress(
                clip_index, "error",
                f"No API keys configured.",
                {"total_keys": total}
            )
            return result
        elif available == 0:
            # All keys are invalid (not rate-limited, but not available either)
            vlog(f"[VeoGenerator] ❌ All {total} keys are invalid!")
            result["error"] = VeoError(
                code=ErrorCode.API_KEY_INVALID,
                message="All API keys are invalid",
                user_message="All API keys have been marked as invalid.",
                details={"total_keys": total, "should_pause": True},
                recoverable=True,
                suggestion="Check your API keys in the settings"
            )
            result["no_keys"] = True
            result["should_pause"] = True
            return result
        
        failed_end_frames = []
        attempts = 0
        rate_limit_retries = 0  # Separate counter for rate limit retries (don't count toward real attempts)
        max_rate_limit_retries = 3  # Max times to retry due to rate limits before pausing job
        current_attempt_end_index = current_end_index
        is_celebrity_retry = False  # Track if we're retrying due to celebrity filter
        is_rate_limit_retry = False  # Track if we're retrying due to rate limit exhaustion
        
        # Calculate current_start_index from start_frame (needed for celebrity filter start frame swapping)
        current_start_index = 0
        for i, img in enumerate(images_list):
            if img == start_frame or img.name == start_frame.name:
                current_start_index = i
                break
        
        while attempts < self.config.max_retries_per_clip:
            # Safety limit for rate limit retries
            if rate_limit_retries >= max_rate_limit_retries:
                vlog(f"[Clip {clip_index}] Exceeded max rate limit retries ({max_rate_limit_retries}) - job should pause")
                result["error"] = VeoError(
                    code=ErrorCode.RATE_LIMIT,
                    message=f"Rate limited {max_rate_limit_retries} times - keys exhausted",
                    user_message="API keys exhausted. Job will pause - resume later when quota resets.",
                    details={"rate_limit_retries": rate_limit_retries, "should_pause": True},
                    recoverable=True,
                    suggestion="Wait 5 minutes for quota to reset, then resume the job"
                )
                return result
            
            if self.cancelled:
                result["error"] = VeoError(
                    code=ErrorCode.UNKNOWN, 
                    message="Cancelled", 
                    user_message="Generation was cancelled",
                    details={},
                    recoverable=False,
                    suggestion="Start a new job"
                )
                return result
            
            while self.paused:
                time.sleep(1)
                if self.cancelled:
                    return result
            
            # Only increment attempts for real failures (not celebrity filter retries or rate limit retries)
            if not is_celebrity_retry and not is_rate_limit_retry:
                attempts += 1
                vlog(f"[Clip {clip_index}] Starting attempt {attempts}/{self.config.max_retries_per_clip}")
            elif is_rate_limit_retry:
                vlog(f"[Clip {clip_index}] Retrying after rate limit (not counting as attempt, still at {attempts}/{self.config.max_retries_per_clip})")
            is_celebrity_retry = False  # Reset for this iteration
            is_rate_limit_retry = False  # Reset for this iteration
            
            # Determine end frame - ALWAYS use the assigned end_frame if provided
            # Only try alternate end frames if the original is blacklisted
            # EXCEPTION: If frames_locked=True (Phase 2 staggered), we MUST use the locked end_frame
            # regardless of blacklist because it's the confirmed start of the next clip
            
            if end_frame:
                # PREVENT SAME-FRAME GENERATION
                # EXCEPTION: Single image mode with interpolation - same frame is intentional
                single_image_mode = getattr(self.config, 'single_image_mode', False)
                use_interpolation = getattr(self.config, 'use_interpolation', True)
                
                if end_frame == start_frame or (hasattr(end_frame, 'name') and hasattr(start_frame, 'name') and end_frame.name == start_frame.name):
                    if single_image_mode and use_interpolation:
                        # Single image mode: same frame is intentional for smoother motion
                        vlog(f"[Clip {clip_index}] Single image mode: using same frame for interpolation")
                        actual_end_frame = end_frame
                        actual_end_index = current_end_index
                    else:
                        vlog(f"[Clip {clip_index}] WARNING: end_frame same as start_frame, finding different end frame")
                        next_result = get_next_clean_image(
                            current_end_index, images_list, local_blacklist | {start_frame}, self.config.max_image_attempts
                        )
                        if next_result is None:
                            vlog(f"[Clip {clip_index}] No different end frame available - cannot generate with same start/end")
                            result["error"] = VeoError(
                                code=ErrorCode.VIDEO_GENERATION_FAILED,
                                message="No different end frame available",
                                user_message="Cannot generate clip - start and end frames are the same and no alternatives available.",
                                details={"start_frame": start_frame.name},
                                recoverable=False,
                                suggestion="Add more images or use single-image mode."
                            )
                            return result
                        actual_end_index, actual_end_frame = next_result
                        current_attempt_end_index = actual_end_index
                        vlog(f"[Clip {clip_index}] Using different end frame: {actual_end_frame.name}")
                elif frames_locked or end_frame not in local_blacklist:
                    # If frames_locked, ALWAYS use the locked end_frame (it's confirmed from Phase 1)
                    # Otherwise, use it if not blacklisted
                    if frames_locked and end_frame in local_blacklist:
                        vlog(f"[Clip {clip_index}] FRAMES LOCKED: Using locked end_frame {end_frame.name} despite blacklist")
                        # Remove from local blacklist for this clip (it was confirmed working in Phase 1)
                        local_blacklist.discard(end_frame)
                    actual_end_frame = end_frame
                    actual_end_index = current_end_index
                else:
                    # End frame is blacklisted, try to find alternative
                    next_result = get_next_clean_image(
                        current_end_index, images_list, local_blacklist | {start_frame}, self.config.max_image_attempts
                    )
                    
                    if next_result is None:
                        if len(failed_end_frames) >= 2:
                            local_blacklist.add(start_frame)
                        return result
                    
                    actual_end_index, actual_end_frame = next_result
                    current_attempt_end_index = actual_end_index
            else:
                # No end frame specified (single image mode or no interpolation)
                actual_end_frame = None
                actual_end_index = current_end_index
            
            # Build prompt using per-scene frame analysis + user context override
            try:
                prompt_text = build_prompt(
                    dialogue_line, start_frame, actual_end_frame, clip_index,
                    self.config.language, self.voice_profile, self.config, self.openai_key,
                    frame_analysis=clip_frame_analysis,  # Per-scene analysis (based on scene_image)
                    user_context_override=self.enriched_context,  # User overrides (if any)
                    redo_feedback=self._current_redo_feedback,  # User's feedback for redo
                    override_duration=override_duration,  # Override duration for last clip
                )
                result["prompt_text"] = prompt_text
            except Exception as e:
                error = error_handler.classify_exception(e, {"stage": "prompt_building"})
                self._emit_error(error)
                result["error"] = error
                return result
            
            # Prepare images
            try:
                with open(start_frame, "rb") as f:
                    start_bytes = f.read()
                start_image = types.Image(
                    image_bytes=start_bytes,
                    mime_type=get_mime_type(start_frame)
                )
                
                end_image = None
                if actual_end_frame and self.config.use_interpolation:
                    with open(actual_end_frame, "rb") as f:
                        end_bytes = f.read()
                    end_image = types.Image(
                        image_bytes=end_bytes,
                        mime_type=get_mime_type(actual_end_frame)
                    )
            except Exception as e:
                error = error_handler.classify_exception(e, {"stage": "image_loading"})
                result["error"] = error
                return result
            
            self._emit_progress(
                clip_index, "generating",
                f"Generating: {start_frame.name} → {actual_end_frame.name if actual_end_frame else 'none'}",
                {"start": start_frame.name, "end": actual_end_frame.name if actual_end_frame else None}
            )
            
            # Submit to Veo API
            operation = None
            submit_client = None  # Track the client used for submission - MUST use same for poll/download
            for submit_attempt in range(1, self.config.max_retries_submit + 1):
                try:
                    # Get client (KeyPoolManager will log which key is being used)
                    submit_client = self._get_client()
                    
                    aspect = self.config.aspect_ratio.value if hasattr(self.config.aspect_ratio, 'value') else self.config.aspect_ratio
                    res = self.config.resolution.value if hasattr(self.config.resolution, 'value') else self.config.resolution
                    # Use override duration if provided (for last clip dynamic duration)
                    if override_duration:
                        dur = override_duration
                    else:
                        dur = self.config.duration.value if hasattr(self.config.duration, 'value') else self.config.duration
                    
                    cfg = types.GenerateVideosConfig(
                        aspect_ratio=aspect,
                        resolution=res,
                        duration_seconds=dur,
                    )
                    
                    if end_image is not None and hasattr(cfg, "last_frame"):
                        cfg.last_frame = end_image
                    
                    try:
                        pg = self.config.person_generation.value if hasattr(self.config.person_generation, 'value') else self.config.person_generation
                        cfg.person_generation = pg
                    except Exception:
                        pass
                    
                    vlog(f"[VeoGenerator] Submitting to Veo (attempt {submit_attempt})...")
                    operation = submit_client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                        image=start_image,
                        config=cfg,
                    )
                    vlog(f"[VeoGenerator] Submit OK!")
                    break
                    
                except Exception as e:
                    vlog(f"[VeoGenerator] Submit error: {str(e)[:200]}")
                    
                    # Check for transient errors (model overloaded) - retry with SAME key
                    if is_transient_error(e) and submit_attempt < self.config.max_retries_submit:
                        # Model overloaded - NOT a key issue, don't rotate keys
                        backoff_time = min(5 * (submit_attempt + 1), 30)  # 5s, 10s, 15s... cap at 30s
                        vlog(f"[VeoGenerator] Model overloaded, waiting {backoff_time}s before retry (attempt {submit_attempt + 1})...")
                        self._emit_progress(
                            clip_index, "retrying",
                            f"Model temporarily overloaded, retrying in {backoff_time}s...",
                            {"attempt": submit_attempt + 1, "backoff": backoff_time}
                        )
                        time.sleep(backoff_time)
                        continue
                    
                    # Check for rate limit errors - rotate keys
                    if is_rate_limit_error(e) and not is_transient_error(e) and submit_attempt < self.config.max_retries_submit:
                        # Mark current key as rate-limited in KeyPoolManager
                        self._rotate_key(block_current=True)
                        
                        # Check pool status (NOT old system)
                        available, rate_limited, total = self._get_pool_status()
                        
                        if available == 0:
                            # ALL keys are rate-limited - don't retry, immediately signal pause
                            # Retrying with 30s sleeps when NO keys exist is wasteful
                            vlog(f"[VeoGenerator] All {total} reserved keys rate-limited - signaling immediate pause")
                            result["error"] = VeoError(
                                code=ErrorCode.RATE_LIMIT, 
                                message="All reserved keys rate-limited", 
                                user_message="All API keys are temporarily blocked",
                                details={"rate_limited": rate_limited, "total": total, "should_pause": True},
                                recoverable=True,
                                suggestion="Wait for keys to recover (~5 minutes)"
                            )
                            result["should_pause"] = True  # Also at top level for easy checking
                            result["no_keys"] = True
                            return result  # Immediate return, no retry loops
                        
                        self._emit_progress(
                            clip_index, "rate_limited",
                            f"Key rate-limited, switching... ({available} working, {rate_limited} rate-limited)",
                            {"available": available, "rate_limited": rate_limited, "total": total}
                        )
                        
                        # Exponential backoff: wait before retrying (2s, 4s, 8s, 16s...)
                        backoff_time = min(2 ** submit_attempt, 30)  # Cap at 30 seconds
                        vlog(f"[VeoGenerator] Rate limited, waiting {backoff_time}s before retry...")
                        time.sleep(backoff_time)
                        continue
                    
                    # Check if we exhausted all submit retries due to rate limits
                    # Don't count this as a real attempt failure
                    if is_rate_limit_error(e) and submit_attempt >= self.config.max_retries_submit - 1:
                        # Check if all keys are now rate-limited
                        available, rate_limited, total = self._get_pool_status()
                        if available == 0:
                            # No keys left - immediately return should_pause
                            vlog(f"[VeoGenerator] All {total} keys exhausted after {submit_attempt} submit retries - signaling immediate pause")
                            result["error"] = VeoError(
                                code=ErrorCode.RATE_LIMIT, 
                                message="All keys exhausted during submit retries", 
                                user_message="All API keys are temporarily blocked",
                                details={"rate_limited": rate_limited, "total": total, "should_pause": True},
                                recoverable=True,
                                suggestion="Wait for keys to recover (~5 minutes)"
                            )
                            result["should_pause"] = True  # Also at top level for easy checking
                            result["no_keys"] = True
                            return result
                        
                        is_rate_limit_retry = True
                        rate_limit_retries += 1
                        vlog(f"[VeoGenerator] Rate limit exhausted all {self.config.max_retries_submit} submit retries - will retry (rate limit retry {rate_limit_retries}/{max_rate_limit_retries})")
                        # Reduced wait - just 5 seconds since we have keys cycling
                        time.sleep(5)
                        break
                    
                    failed_end_frames.append(actual_end_frame)
                    break
            
            if operation is None:
                vlog(f"[Clip {clip_index}] No operation after submit attempts - continuing to next attempt")
                continue
            
            if submit_client is None:
                vlog(f"[Clip {clip_index}] No submit client available - continuing to next attempt")
                continue
            
            # Poll for completion with progress logging
            # CRITICAL: Must use same client that submitted - file ownership is tied to API key!
            try:
                poll_start = time.time()
                last_log_time = poll_start
                
                while not operation.done:
                    if self.cancelled:
                        return result
                    
                    time.sleep(self.config.poll_interval_sec)
                    operation = submit_client.operations.get(operation)
                    
                    # Log progress every 60 seconds
                    elapsed = time.time() - poll_start
                    if time.time() - last_log_time >= 60:
                        last_log_time = time.time()
                        if elapsed > 180:  # 3 minutes
                            vlog(f"[VeoGenerator] ⚠️ Clip {clip_index} taking longer than usual... ({elapsed:.0f}s elapsed)")
                            self._emit_progress(clip_index, "generating", f"Taking longer than usual ({int(elapsed)}s)...")
                        else:
                            vlog(f"[VeoGenerator] Still waiting for clip {clip_index}... ({elapsed:.0f}s elapsed)")
                    
            except Exception as e:
                error_str = str(e)
                # Check for transient errors (model overloaded) - wait and retry
                if is_transient_error(e):
                    vlog(f"[Clip {clip_index}] Model overloaded during poll - waiting 10s before retry...")
                    self._emit_progress(
                        clip_index, "poll_retry",
                        f"Model temporarily overloaded, retrying...",
                        {"error": error_str[:100]}
                    )
                    time.sleep(10)
                    # Try polling again with same key
                    try:
                        while not operation.done:
                            if self.cancelled:
                                return result
                            time.sleep(self.config.poll_interval_sec)
                            operation = submit_client.operations.get(operation)
                    except Exception as retry_e:
                        vlog(f"[Clip {clip_index}] Poll retry also failed: {type(retry_e).__name__}: {retry_e}")
                        failed_end_frames.append(actual_end_frame)
                        continue
                # Check if it's a rate limit error during polling - wait and retry with same key
                elif is_rate_limit_error(e):
                    vlog(f"[Clip {clip_index}] Rate limit during poll - waiting 30s before retry...")
                    self._emit_progress(
                        clip_index, "poll_rate_limit",
                        f"Rate limited during poll, waiting...",
                        {"error": error_str[:100]}
                    )
                    time.sleep(30)
                    # Try polling again with same key (we can't switch keys - operation is tied to this key)
                    try:
                        while not operation.done:
                            if self.cancelled:
                                return result
                            time.sleep(self.config.poll_interval_sec)
                            operation = submit_client.operations.get(operation)
                    except Exception as retry_e:
                        vlog(f"[Clip {clip_index}] Poll retry also failed: {type(retry_e).__name__}: {retry_e}")
                        failed_end_frames.append(actual_end_frame)
                        continue
                else:
                    vlog(f"[Clip {clip_index}] Poll/wait error: {type(e).__name__}: {e}")
                    self._emit_progress(
                        clip_index, "poll_error",
                        f"Poll error: {type(e).__name__}",
                        {"error": str(e)[:200]}
                    )
                    failed_end_frames.append(actual_end_frame)
                    continue
            
            poll_duration = time.time() - poll_start
            vlog(f"[Clip {clip_index}] Poll complete after {poll_duration:.1f}s, checking result...")
            
            # Check for errors
            veo_error = error_handler.classify_veo_operation(
                operation, {"clip_index": clip_index}
            )
            
            if veo_error:
                self._emit_error(veo_error)
                
                # Special handling for celebrity filter - immediately blacklist end frame and try next
                if veo_error.code == ErrorCode.CELEBRITY_FILTER:
                    # Check if we should skip immediately (storyboard mode)
                    if getattr(self.config, 'skip_on_celebrity_filter', False):
                        vlog(f"[Clip {clip_index}] Celebrity filter triggered - SKIPPING (skip_on_celebrity_filter=True)")
                        self._emit_progress(
                            clip_index, "celebrity_skipped",
                            f"Celebrity filter triggered. Clip skipped (will be logged for manual generation).",
                            {
                                "start_frame": start_frame.name if start_frame else None,
                                "end_frame": actual_end_frame.name if actual_end_frame else None,
                                "prompt": result.get("prompt_text"),
                            }
                        )
                        result["skipped"] = True
                        result["skip_reason"] = "celebrity_filter"
                        result["error"] = VeoError(
                            code=ErrorCode.CELEBRITY_FILTER,
                            message="Celebrity filter - clip skipped",
                            user_message="Celebrity filter triggered. Clip skipped for reimbursement.",
                            details={
                                "start_frame": start_frame.name if start_frame else None,
                                "end_frame": actual_end_frame.name if actual_end_frame else None,
                            },
                            recoverable=False,
                            suggestion="You can try generating this clip manually in Google AI Studio."
                        )
                        return result
                    
                    if actual_end_frame:
                        # Blacklist the end frame (it's the new one in the cycle)
                        local_blacklist.add(actual_end_frame)
                        # Also add to global celebrity hints for parallel mode (other clips can benefit)
                        self.celebrity_hints.add(actual_end_frame)
                        failed_end_frames.append(actual_end_frame)
                        vlog(f"[Clip {clip_index}] Celebrity filter - blacklisted end frame: {actual_end_frame.name}")
                        vlog(f"[Clip {clip_index}] Failed end frames so far: {len(failed_end_frames)}, global hints: {len(self.celebrity_hints)}")
                        
                        # If 2+ different end frames failed with the same start frame,
                        # the START frame is likely the problem - try swapping it
                        # EXCEPTION: If frames_locked=True, both frames are locked - we cannot swap
                        if len(failed_end_frames) >= 2:
                            # If frames are locked (Phase 2 staggered), cannot swap
                            if frames_locked:
                                vlog(f"[Clip {clip_index}] FRAMES LOCKED: Cannot swap frames. Failing clip.")
                                self._emit_progress(
                                    clip_index, "celebrity_failed",
                                    f"Celebrity filter triggered with locked frames. Cannot swap.",
                                    {"start": start_frame.name, "end": actual_end_frame.name if actual_end_frame else None}
                                )
                                result["error"] = VeoError(
                                    code=ErrorCode.CELEBRITY_FILTER,
                                    message="Celebrity filter with locked frames",
                                    user_message="Celebrity filter triggered. Frames are locked - try with different images.",
                                    details={"start_frame": start_frame.name, "end_frame": actual_end_frame.name if actual_end_frame else None},
                                    recoverable=False,
                                    suggestion="Use different source images or try sequential mode."
                                )
                                return result
                            
                            vlog(f"[Clip {clip_index}] 2+ end frames failed - START frame {start_frame.name} is likely the problem")
                            self._emit_progress(
                                clip_index, "celebrity_retry",
                                f"Start frame {start_frame.name} appears problematic. Trying different start frame...",
                                {"blacklisted_start": start_frame.name}
                            )
                            
                            # Blacklist the start frame (local and global)
                            local_blacklist.add(start_frame)
                            self.celebrity_hints.add(start_frame)
                            
                            # Find a new start frame
                            new_start_result = get_next_clean_image(
                                current_start_index, images_list, local_blacklist, len(images_list)
                            )
                            
                            if new_start_result:
                                new_start_index, new_start_frame = new_start_result
                                vlog(f"[Clip {clip_index}] Switching to new start frame: {new_start_frame.name}")
                                
                                # Update start frame
                                start_frame = new_start_frame
                                current_start_index = new_start_index
                                
                                # Clear failed_end_frames since we have a new start
                                failed_end_frames = []
                                
                                # Find a clean end frame (reset to check from new start)
                                # Exclude new start frame from end frame candidates
                                next_result = get_next_clean_image(
                                    new_start_index, images_list, local_blacklist | {new_start_frame}, len(images_list)
                                )
                                
                                if next_result:
                                    current_attempt_end_index, new_end_frame = next_result
                                    end_frame = new_end_frame
                                    vlog(f"[Clip {clip_index}] New end frame: {new_end_frame.name}")
                                    is_celebrity_retry = True
                                    time.sleep(1)
                                    continue
                                else:
                                    # Found new start but no clean end frame
                                    vlog(f"[Clip {clip_index}] Found new start {new_start_frame.name} but no clean end frames")
                                    self._emit_progress(
                                        clip_index, "celebrity_failed",
                                        "All end frames triggered celebrity filter.",
                                        {"total_blacklisted": len(local_blacklist)}
                                    )
                                    result["error"] = VeoError(
                                        code=ErrorCode.CELEBRITY_FILTER,
                                        message="All images blacklisted by celebrity filter",
                                        user_message="All available end frames triggered celebrity detection.",
                                        details={"blacklisted_count": len(local_blacklist)},
                                        recoverable=False,
                                        suggestion="Try with different source images."
                                    )
                                    return result
                            else:
                                # No clean start frames available
                                vlog(f"[Clip {clip_index}] No clean start frames available")
                                self._emit_progress(
                                    clip_index, "celebrity_failed",
                                    "All start frames triggered celebrity filter.",
                                    {"total_blacklisted": len(local_blacklist)}
                                )
                                result["error"] = VeoError(
                                    code=ErrorCode.CELEBRITY_FILTER,
                                    message="All images blacklisted by celebrity filter",
                                    user_message="All available start frames triggered celebrity detection.",
                                    details={"blacklisted_count": len(local_blacklist)},
                                    recoverable=False,
                                    suggestion="Try with different source images."
                                )
                                return result
                        
                        self._emit_progress(
                            clip_index, "celebrity_retry",
                            f"Celebrity filter on {actual_end_frame.name}. Trying next image...",
                            {"blacklisted": actual_end_frame.name}
                        )
                        
                        # Find next clean end frame (exclude start frame)
                        next_result = get_next_clean_image(
                            actual_end_index, images_list, local_blacklist | {start_frame}, len(images_list)
                        )
                        
                        if next_result:
                            current_attempt_end_index, new_end_frame = next_result
                            vlog(f"[Clip {clip_index}] Switching to new end frame: {new_end_frame.name}")
                            # Update end_frame for next iteration
                            end_frame = new_end_frame
                            is_celebrity_retry = True  # Don't count toward main attempts
                            time.sleep(1)  # Brief delay
                            continue
                        else:
                            # All images exhausted
                            vlog(f"[Clip {clip_index}] All images blacklisted - no clean frames left")
                            self._emit_progress(
                                clip_index, "celebrity_failed",
                                "All images triggered celebrity filter. Clip cannot be generated.",
                                {"total_blacklisted": len(local_blacklist)}
                            )
                            result["error"] = VeoError(
                                code=ErrorCode.CELEBRITY_FILTER,
                                message="All images blacklisted by celebrity filter",
                                user_message="All available images triggered celebrity detection.",
                                details={"blacklisted_count": len(local_blacklist)},
                                recoverable=False,
                                suggestion="Try with different source images."
                            )
                            return result
                    else:
                        # Single image mode - can't swap, just fail
                        vlog(f"[Clip {clip_index}] Celebrity filter in single image mode - cannot swap")
                else:
                    # Non-celebrity error - log it
                    vlog(f"[Clip {clip_index}] Veo API error: {veo_error.code} - {veo_error.message}")
                    self._emit_progress(
                        clip_index, "api_error",
                        f"API error: {veo_error.message}",
                        {"error_code": str(veo_error.code), "error_message": veo_error.message}
                    )
                
                failed_end_frames.append(actual_end_frame)
                continue
            
            # SUCCESS! Frames passed celebrity filter
            # Notify that frames are confirmed - next clip can start with these frames
            if on_frames_locked:
                vlog(f"[Clip {clip_index}] Frames CONFIRMED (passed celebrity filter): {start_frame.name} → {actual_end_frame.name if actual_end_frame else 'None'}")
                on_frames_locked(clip_index, start_frame, actual_end_frame)
            
            # Download video
            video = None  # Initialize for safe retry check in except block
            output_path = None
            output_filename = None
            try:
                resp = getattr(operation, "response", None)
                vids = getattr(resp, "generated_videos", None) if resp else None
                
                if not vids:
                    vlog(f"[Clip {clip_index}] No videos returned from API. resp={resp}")
                    self._emit_progress(
                        clip_index, "api_error",
                        f"No video returned from API",
                        {"error": "empty_response"}
                    )
                    failed_end_frames.append(actual_end_frame)
                    continue
                
                video = vids[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.config.timestamp_names else ""
                output_filename = generate_output_filename(dialogue_id, start_frame, actual_end_frame, timestamp)
                output_path = output_dir / output_filename
                
                # CRITICAL: Must use same client that submitted - file ownership is tied to API key!
                submit_client.files.download(file=video.video)
                video.video.save(str(output_path))
                
                result["success"] = True
                result["output_path"] = output_path
                result["end_frame_used"] = actual_end_frame
                result["end_index"] = actual_end_index
                
                self._emit_progress(clip_index, "completed", f"Saved: {output_filename}", {"output": output_filename})
                return result
                
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error during download - wait and retry with same key
                # But only if we got far enough to have video and output_path defined
                if is_rate_limit_error(e) and video is not None and output_path is not None and output_filename is not None:
                    vlog(f"[Clip {clip_index}] Rate limit during download - waiting 30s before retry...")
                    self._emit_progress(
                        clip_index, "download_rate_limit",
                        f"Rate limited during download, waiting...",
                        {"error": error_str[:100]}
                    )
                    time.sleep(30)
                    # Try download again with same key
                    try:
                        submit_client.files.download(file=video.video)
                        video.video.save(str(output_path))
                        
                        result["success"] = True
                        result["output_path"] = output_path
                        result["end_frame_used"] = actual_end_frame
                        result["end_index"] = actual_end_index
                        
                        self._emit_progress(clip_index, "completed", f"Saved: {output_filename}", {"output": output_filename})
                        return result
                    except Exception as retry_e:
                        vlog(f"[Clip {clip_index}] Download retry also failed: {type(retry_e).__name__}: {retry_e}")
                        failed_end_frames.append(actual_end_frame)
                        continue
                else:
                    vlog(f"[Clip {clip_index}] Download/save error: {type(e).__name__}: {e}")
                    self._emit_progress(
                        clip_index, "download_error",
                        f"Download error: {type(e).__name__}",
                        {"error": str(e)[:200]}
                    )
                    failed_end_frames.append(actual_end_frame)
                    continue
        
        # Exhausted retries
        if len(failed_end_frames) >= 2:
            local_blacklist.add(start_frame)
        
        result["error"] = VeoError(
            code=ErrorCode.VIDEO_GENERATION_FAILED,
            message=f"Failed after {attempts} attempts ({rate_limit_retries} rate limit retries)",
            user_message="Video generation failed after multiple attempts",
            details={
                "attempts": attempts, 
                "max_attempts": self.config.max_retries_per_clip,
                "rate_limit_retries": rate_limit_retries,
                "failed_frames": len(failed_end_frames)
            },
            recoverable=True,
            suggestion="Try regenerating the clip or use a different image"
        )
        
        return result
    
    def cancel(self):
        """Cancel generation"""
        self.cancelled = True
    
    def pause(self):
        """Pause generation"""
        self.paused = True
    
    def resume(self):
        """Resume generation"""
        self.paused = False