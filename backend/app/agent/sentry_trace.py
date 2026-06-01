"""
Tagged Sentry trace events for the preference-elicitation save journey and
the recommender-agent matching service call.

These are temporary observability hooks added during the investigation into
"users not getting preferences saved / not getting recommendations". Every
event:

- Carries the prefix "[PREF_AGENT_TRACE]" in its message so it stands out in the
  Sentry UI versus regular logger.error events.
- Sets a `trace_point` tag so events can be filtered by milestone
  (e.g. `trace_point:preference.bayesian_update.failed`).
- Optionally attaches a `trace_data` context dict with the small payload
  needed to interpret the event.
- Inherits `session_id` / `user_id` / `client_id` tags automatically via
  the `attach_ticket_info` before_send hook in app/sentry_init.py.

No-op when Sentry isn't initialised (BACKEND_ENABLE_SENTRY=False) — the
sentry_sdk capture functions are safe to call in that case.

The "[PREF_AGENT_TRACE]" prefix is deliberate so these can be excluded from alerting
rules and easily distinguished from genuine error events; once the
investigation closes, the call sites can be removed in one pass.
"""
from typing import Any, Optional

import sentry_sdk


_TRACE_PREFIX = "[PREF_AGENT_TRACE]"


def trace(point: str, level: str = "info", **extras: Any) -> None:
    """
    Emit a tagged Sentry event for a known trace point.

    Args:
        point: short dotted identifier, e.g. "preference.bayesian_update.failed".
            Becomes the `trace_point` tag for filtering.
        level: Sentry event level. Use "info" for milestones, "warning" for
            unexpected-but-handled, "error" for things that should pager-alert
            once filters are tuned. Defaults to "info".
        **extras: small payload (ints, strings, bools, short lists). Stored
            under the `trace_data` context on the event.
    """
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("trace_point", point)
            if extras:
                # Only set context if we actually have data; keeps lean events lean.
                scope.set_context("trace_data", _sanitise(extras))
            sentry_sdk.capture_message(f"{_TRACE_PREFIX} {point}", level=level)
    except Exception:
        # Observability must never break the request path.
        pass


def trace_exception(point: str, exc: BaseException, **extras: Any) -> None:
    """
    Emit a tagged Sentry event for an exception caught inside an otherwise
    silent try/except block. Forwards the full traceback as the event payload
    (via capture_exception) rather than a plain message.

    Use this in handlers that catch + log + return — i.e. the silent-fail
    funnels — so the exception is visible in Sentry even though the caller
    never sees it.
    """
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("trace_point", point)
            scope.set_tag("trace_kind", "swallowed_exception")
            if extras:
                scope.set_context("trace_data", _sanitise(extras))
            # capture_exception uses the live exception if exc is None;
            # we pass exc explicitly so this works even when called from
            # outside an `except` block.
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def _sanitise(extras: dict) -> dict:
    """Truncate long strings / lists so trace events stay small."""
    out: dict[str, Any] = {}
    for k, v in extras.items():
        if isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + f"... (+{len(v) - 500} chars)"
        elif isinstance(v, list) and len(v) > 20:
            out[k] = list(v[:20]) + [f"... (+{len(v) - 20} more)"]
        else:
            out[k] = v
    return out
