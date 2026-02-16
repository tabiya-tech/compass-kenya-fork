"""
Language Detection Module for Milestone 3

Detects the language of user input to enable runtime locale switching
between English and Swahili (including mixed/code-switched input).

Detection Strategy:
- Keyword matching using curated Swahili high-signal word lists
- Ratio-based classification (Swahili token ratio determines language)
- Session stickiness to prevent flapping on ambiguous short messages

DetectedLanguage:
- ENGLISH: Input is predominantly English
- SWAHILI: Input is predominantly Swahili
- MIXED: Input contains significant amounts of both languages
"""

import re
from enum import Enum
from typing import Optional
from app.i18n.types import Locale


class DetectedLanguage(Enum):
    """Classification of the detected input language."""
    ENGLISH = "english"
    SWAHILI = "swahili"
    MIXED = "mixed"


# High-signal Swahili words that are unlikely to appear in English text.
# These include greetings, common verbs, connectors, and job-related nouns.
SWAHILI_SIGNAL_WORDS = [
    # Greetings and common phrases
    "habari", "jambo", "karibu", "asante", "tafadhali", "ndio", "ndiyo",
    "hapana", "sawa", "pole", "shikamoo", "marahaba", "hujambo", "sijambo",
    # Common verbs and verb forms
    "nilifanya", "ninafanya", "nitafanya", "nilisaidia", "nilikuwa",
    "nimekuwa", "tunafanya", "wanafanya", "kufanya", "kusaidia",
    "kuuza", "kununua", "kupika", "kulima", "kujenga", "kushona",
    "kufundisha", "kusafisha", "kupanga", "kutengeneza",
    # Connectors and prepositions
    "na", "ya", "wa", "kwa", "katika", "kama", "lakini", "pia",
    "au", "hata", "baada", "kabla", "tangu", "mpaka",
    # Job/work related nouns
    "kazi", "biashara", "soko", "duka", "shamba", "nyumba",
    "ofisi", "shule", "hospitali", "barabara",
    "mfanyakazi", "muuzaji", "mkulima", "fundi", "dereva",
    "mwalimu", "daktari", "askari", "seremala", "mshonaji",
    # Common nouns
    "mtu", "watu", "watoto", "mama", "baba", "familia", "jirani",
    "pesa", "mshahara", "fedha", "siku", "mwezi", "mwaka",
    # Time and place
    "asubuhi", "mchana", "jioni", "usiku", "leo", "jana", "kesho",
    # Informal/slang terms (Sheng)
    "vibarua", "hustle", "watchie", "boda",
]

# High-signal English words that help distinguish English from Swahili.
# These are common English words that have no Swahili meaning.
ENGLISH_SIGNAL_WORDS = [
    # Common verbs
    "worked", "working", "helped", "managed", "did", "was", "were",
    "have", "had", "been", "started", "finished", "sold", "built",
    # Job-related
    "job", "work", "company", "business", "office", "position",
    "title", "role", "department", "responsibilities", "salary",
    "employed", "employment", "experience", "years",
    # Common words
    "the", "and", "for", "with", "from", "this", "that", "what",
    "when", "where", "how", "who", "which", "because", "about",
    "also", "very", "just", "then", "than", "before", "after",
    # Pronouns
    "my", "your", "his", "her", "our", "their",
]

# Swahili threshold ratios
_SWAHILI_THRESHOLD = 0.40  # >40% Swahili tokens => SWAHILI
_MIXED_THRESHOLD = 0.10    # 10-40% Swahili tokens => MIXED
# <10% Swahili tokens => ENGLISH

# Session stickiness: how many consecutive turns of the same language
# before we consider the session "locked" to that language.
_STICKINESS_TURNS = 3


def detect_language(
    user_message: str,
    conversation_history: Optional[list[str]] = None,
    previous_detections: Optional[list[DetectedLanguage]] = None,
) -> DetectedLanguage:
    """
    Detect the language of the user's message.

    Args:
        user_message: The current user message to analyze.
        conversation_history: Optional list of previous user messages (unused for now,
                              reserved for future context-based detection).
        previous_detections: Optional list of previous detection results for session stickiness.

    Returns:
        DetectedLanguage.ENGLISH, DetectedLanguage.SWAHILI, or DetectedLanguage.MIXED
    """
    if not user_message or not user_message.strip():
        # Empty message: fall back to session history or default to ENGLISH
        if previous_detections and len(previous_detections) > 0:
            return previous_detections[-1]
        return DetectedLanguage.ENGLISH

    text = user_message.lower().strip()

    # Tokenize by whitespace and basic punctuation
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)

    if not tokens:
        # No alphabetic tokens (e.g., just numbers/punctuation)
        if previous_detections and len(previous_detections) > 0:
            return previous_detections[-1]
        return DetectedLanguage.ENGLISH

    # Count Swahili and English signal matches
    sw_matches = 0
    en_matches = 0

    for token in tokens:
        token_lower = token.lower()
        if token_lower in _SWAHILI_SIGNAL_SET:
            sw_matches += 1
        if token_lower in _ENGLISH_SIGNAL_SET:
            en_matches += 1

    total_signal = sw_matches + en_matches
    if total_signal == 0:
        # No signal words found -- use session stickiness or default
        if previous_detections and len(previous_detections) > 0:
            return previous_detections[-1]
        return DetectedLanguage.ENGLISH

    sw_ratio = sw_matches / total_signal

    # Classify based on ratio
    if sw_ratio > _SWAHILI_THRESHOLD:
        raw_detection = DetectedLanguage.SWAHILI
    elif sw_ratio > _MIXED_THRESHOLD:
        raw_detection = DetectedLanguage.MIXED
    else:
        raw_detection = DetectedLanguage.ENGLISH

    # Apply session stickiness: if the last N turns were all the same language,
    # only switch if there's a strong signal (not MIXED).
    if previous_detections and len(previous_detections) >= _STICKINESS_TURNS:
        recent = previous_detections[-_STICKINESS_TURNS:]
        if all(d == recent[0] for d in recent) and recent[0] != DetectedLanguage.MIXED:
            sticky_lang = recent[0]
            # Only override stickiness with a clear, non-MIXED signal
            if raw_detection == DetectedLanguage.MIXED:
                return sticky_lang

    return raw_detection


def get_locale_for_detected_language(detected: DetectedLanguage, default_locale_str: str = "en-US") -> str:
    """
    Map a DetectedLanguage to a locale string.

    For MIXED input, we default to Swahili locale (respond in Swahili)
    since the user has shown they understand Swahili.
    """
    if detected == DetectedLanguage.SWAHILI:
        return "sw-KE"
    elif detected == DetectedLanguage.MIXED:
        return "sw-KE"
    else:
        return default_locale_str


def get_detected_language_for_locale(locale: "Locale") -> str:
    """
    Map a Locale to the detected language value for context vars.
    Used when locale is pre-configured (e.g. E2E tests, golden transcripts)
    rather than detected from user input.
    """
    if locale == Locale.SW_KE:
        return DetectedLanguage.SWAHILI.value
    return DetectedLanguage.ENGLISH.value


# Pre-compute sets for fast lookup
_SWAHILI_SIGNAL_SET = frozenset(w.lower() for w in SWAHILI_SIGNAL_WORDS)
_ENGLISH_SIGNAL_SET = frozenset(w.lower() for w in ENGLISH_SIGNAL_WORDS)
