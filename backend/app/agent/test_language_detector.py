"""
Tests for language detection, focused on the structured-message guard.

The BWS best/worst card in the frontend submits a JSON payload
({"type": "bws_response", ...}) as the user message. That payload carries no
linguistic signal and, because it contains tokens like "best"/"worst", is
misread as English. These tests lock in that such payloads are recognised as
structured and therefore excluded from language detection, so they cannot flip
an in-progress Swahili conversation back to English.
"""

import pytest

from app.agent.language_detector import (
    detect_language,
    is_structured_message,
    DetectedLanguage,
    get_locale_for_detected_language,
)


BWS_PAYLOAD = '{"type": "bws_response", "task_id": "0", "best": "4.A.4.b.4", "worst": "4.A.2.b.1"}'


class TestIsStructuredMessage:
    def test_recognises_bws_payload(self):
        assert is_structured_message(BWS_PAYLOAD) is True

    def test_unknown_type_not_treated_as_structured(self):
        # Allowlist behaviour: a typed JSON object whose type is NOT known is left
        # to normal detection rather than silently skipped.
        assert is_structured_message('{"type": "something_else", "x": 1}') is False

    @pytest.mark.parametrize("msg", [
        "Nilifanya kazi ya kuuza sokoni",
        "I worked as a teacher",
        "Most: B, Least: D",           # prose that looks BWS-ish but is natural language
        "",
        None,
        "{not valid json",
        '{"no_type_field": true}',      # JSON object without a "type" key
        '["a", "list"]',                # JSON but not an object
        '42',                           # bare JSON scalar
    ])
    def test_rejects_natural_language_and_non_typed_json(self, msg):
        assert is_structured_message(msg) is False


class TestStructuredMessagesDoNotFlipLocale:
    """Reproduces the reported bug: BWS JSON turns must not switch SW -> EN."""

    def _resolve(self, history, current):
        # Mirrors the detection block in ConversationService.send()
        natural = [m for m in history if not is_structured_message(m)]
        prev = [detect_language(m) for m in natural[-3:]] if natural else []
        if is_structured_message(current):
            detected = prev[-1] if prev else DetectedLanguage.ENGLISH
        else:
            detected = detect_language(current, previous_detections=prev or None)
        return get_locale_for_detected_language(detected)

    def test_swahili_conversation_survives_consecutive_bws_turns(self):
        history = [
            "Habari, ninataka kazi",
            "Nilifanya kazi ya kuuza sokoni",
            "Ndio, ninapenda kufundisha",
        ]
        for _ in range(5):
            assert self._resolve(history, BWS_PAYLOAD) == "sw-KE"
            history.append(BWS_PAYLOAD)

    def test_english_conversation_stays_english_through_bws(self):
        history = [
            "Hi, I worked as a teacher",
            "I really enjoyed helping students",
            "Yes that is correct",
        ]
        for _ in range(5):
            assert self._resolve(history, BWS_PAYLOAD) == "en-US"
            history.append(BWS_PAYLOAD)
