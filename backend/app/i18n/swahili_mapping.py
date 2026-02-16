"""
Swahili Term Mapping Service for Milestone 3

Normalizes Swahili and code-switched job/skill terms to English taxonomy-aligned
terms before they hit the ESCO search pipeline.

The mapping data is loaded from a single JSON file (swahili_terms.json) which
serves as the single source of truth for both:
- Term normalization (mapping service)
- RAG glossary injection (prompt context)
"""

import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the terms file (single source of truth)
_TERMS_FILE = Path(__file__).parent / "resources" / "swahili_terms.json"


@dataclass
class NormalizedResult:
    """Result of a term normalization attempt."""
    original: str
    normalized: str
    matched: bool
    taxonomy_type: Optional[str] = None
    variant_type: Optional[str] = None
    confidence: float = 0.0


@dataclass
class TextNormalizationResult:
    """Result of normalizing a full text (with code-switch handling)."""
    original_text: str
    normalized_text: str
    terms_matched: list[NormalizedResult]
    match_count: int


class SwahiliMappingService:
    """
    Service to normalize Swahili and code-switched terms to taxonomy-aligned English terms.

    Loads mapping data from swahili_terms.json at init and caches in memory.
    Provides:
    - normalize(term): Single-term normalization
    - normalize_text(text): Full-text normalization with code-switch handling
    - get_glossary_sample(n, pin_terms): Random term sampling for RAG prompt injection
    """

    _instance: Optional["SwahiliMappingService"] = None

    def __init__(self):
        self._terms: list[dict] = []
        self._lookup: dict[str, dict] = {}  # normalized_key -> term entry
        self._load_terms()

    @classmethod
    def get_instance(cls) -> "SwahiliMappingService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def _load_terms(self):
        """Load and index terms from the JSON file."""
        try:
            with open(_TERMS_FILE, "r", encoding="utf-8") as f:
                self._terms = json.load(f)

            # Build lookup index (lowercase, stripped)
            for entry in self._terms:
                key = self._normalize_key(entry["term"])
                self._lookup[key] = entry

            logger.info("Loaded %d Swahili term mappings from %s", len(self._terms), _TERMS_FILE)
        except FileNotFoundError:
            logger.warning("Swahili terms file not found at %s, mapping service will return no matches", _TERMS_FILE)
            self._terms = []
            self._lookup = {}
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Swahili terms file: %s", e)
            self._terms = []
            self._lookup = {}

    @staticmethod
    def _normalize_key(term: str) -> str:
        """Normalize a term for lookup: lowercase, strip, remove diacritics."""
        term = term.lower().strip()
        # Remove diacritics (e.g., accented characters)
        term = unicodedata.normalize("NFD", term)
        term = "".join(c for c in term if unicodedata.category(c) != "Mn")
        return term

    def normalize(self, term: str) -> NormalizedResult:
        """
        Normalize a single Swahili term to its English taxonomy equivalent.

        Args:
            term: The Swahili term to normalize.

        Returns:
            NormalizedResult with matched=True if found, False otherwise.
            When not found, normalized=original (passthrough).
        """
        if not term or not term.strip():
            return NormalizedResult(original=term, normalized=term, matched=False)

        key = self._normalize_key(term)

        # Direct lookup
        if key in self._lookup:
            entry = self._lookup[key]
            return NormalizedResult(
                original=term,
                normalized=entry["normalized"],
                matched=True,
                taxonomy_type=entry.get("taxonomy_type"),
                variant_type=entry.get("variant_type"),
                confidence=1.0,
            )

        # Try matching multi-word terms that might be substrings
        # (e.g., "mama mboga" in "nilikuwa mama mboga")
        for lookup_key, entry in self._lookup.items():
            if " " in lookup_key and lookup_key in key:
                return NormalizedResult(
                    original=term,
                    normalized=entry["normalized"],
                    matched=True,
                    taxonomy_type=entry.get("taxonomy_type"),
                    variant_type=entry.get("variant_type"),
                    confidence=0.8,
                )

        # No match -- return original as fallback
        return NormalizedResult(original=term, normalized=term, matched=False)

    def normalize_text(self, text: str) -> TextNormalizationResult:
        """
        Normalize a full text by performing phrase-level mapping.
        Handles code-switched (mixed English + Swahili) input.

        Strategy:
        1. Try multi-word phrases first (longest match).
        2. Then single-word tokens.
        3. English tokens pass through unchanged.
        4. Returns both original and normalized text.

        Args:
            text: The full input text (may be mixed English + Swahili).

        Returns:
            TextNormalizationResult with original and normalized text.
        """
        if not text or not text.strip():
            return TextNormalizationResult(
                original_text=text,
                normalized_text=text,
                terms_matched=[],
                match_count=0,
            )

        text_lower = text.lower().strip()
        normalized_parts = []
        terms_matched = []
        i = 0
        words = text_lower.split()

        while i < len(words):
            matched = False

            # Try longest multi-word match first (up to 4 words)
            for window in range(min(4, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i + window])
                key = self._normalize_key(phrase)

                if key in self._lookup:
                    entry = self._lookup[key]
                    result = NormalizedResult(
                        original=phrase,
                        normalized=entry["normalized"],
                        matched=True,
                        taxonomy_type=entry.get("taxonomy_type"),
                        variant_type=entry.get("variant_type"),
                        confidence=1.0 if window == 1 else 0.9,
                    )
                    terms_matched.append(result)
                    normalized_parts.append(entry["normalized"])
                    i += window
                    matched = True
                    break

            if not matched:
                # No mapping found -- pass through unchanged
                normalized_parts.append(words[i])
                i += 1

        normalized_text = " ".join(normalized_parts)
        return TextNormalizationResult(
            original_text=text,
            normalized_text=normalized_text,
            terms_matched=terms_matched,
            match_count=len(terms_matched),
        )

    def get_glossary_sample(
        self,
        n: int = 10,
        pin_terms: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> list[dict]:
        """
        Get a random sample of terms for RAG glossary injection into prompts.

        Args:
            n: Number of terms to sample (default 10).
            pin_terms: Optional list of Swahili terms to always include
                       (e.g., terms the user mentioned in current turn).
            seed: Optional random seed for reproducibility (e.g., session_id).

        Returns:
            List of term entries (dicts with 'term' and 'normalized' keys).
        """
        if not self._terms:
            return []

        pinned = []
        remaining = list(self._terms)

        # Pin requested terms
        if pin_terms:
            pin_keys = {self._normalize_key(t) for t in pin_terms}
            pinned = [entry for entry in self._terms if self._normalize_key(entry["term"]) in pin_keys]
            remaining = [entry for entry in self._terms if self._normalize_key(entry["term"]) not in pin_keys]

        # Calculate how many random picks we need
        random_count = max(0, n - len(pinned))

        # Sample from remaining (seeded for reproducibility, not crypto)
        rng = random.Random(seed)  # nosec B311
        if random_count >= len(remaining):
            sample = remaining
        else:
            sample = rng.sample(remaining, random_count)

        return pinned + sample

    def get_all_terms(self) -> list[dict]:
        """Return all loaded term entries."""
        return list(self._terms)

    @property
    def term_count(self) -> int:
        """Number of loaded terms."""
        return len(self._terms)
