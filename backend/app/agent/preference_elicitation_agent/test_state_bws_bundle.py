"""
Unit tests for PreferenceElicitationAgentState.bws_bundle_for_matching().

The bundle is what gets forwarded to the matching service as
preference_vector.bws_scores / top_10_bws. HB posterior means must be carried
under the bws_scores name (no rename) when HB succeeded; counts are only a fallback.

Run with:
    poetry run pytest app/agent/preference_elicitation_agent/test_state_bws_bundle.py -v
"""

from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState


def _state(**kwargs) -> PreferenceElicitationAgentState:
    return PreferenceElicitationAgentState(session_id=1, **kwargs)


def _hb_entry(mean: float) -> dict:
    return {"mean": mean, "sd": 0.3, "ci_low": mean - 0.6, "ci_high": mean + 0.6, "rank": 1}


def test_hb_present_with_ranking_forwards_hb_means_not_counts():
    s = _state(
        hb_scores={"4.A.2.b.1": _hb_entry(2.1), "4.A.3.a.1": _hb_entry(-1.4)},
        hb_ranking=["4.A.2.b.1", "4.A.3.a.1"],
        bws_scores={"4.A.2.b.1": 2.0, "4.A.3.a.1": -2.0},  # counts must be ignored
    )
    bws, top = s.bws_bundle_for_matching()
    assert bws == {"4.A.2.b.1": 2.1, "4.A.3.a.1": -1.4}
    assert top == ["4.A.2.b.1", "4.A.3.a.1"]


def test_hb_present_without_ranking_derives_ranking_from_means():
    # Legacy session: hb_scores populated (HB succeeded) but hb_ranking empty.
    s = _state(
        hb_scores={"a": _hb_entry(-1.0), "b": _hb_entry(2.5), "c": _hb_entry(0.4)},
        hb_ranking=[],
        bws_scores={"a": 1.0},
    )
    bws, top = s.bws_bundle_for_matching()
    assert bws == {"a": -1.0, "b": 2.5, "c": 0.4}
    assert top == ["b", "c", "a"]  # sorted by mean, descending


def test_falls_back_to_counts_when_hb_absent():
    s = _state(hb_scores=None, bws_scores={"x": 2.0, "y": -1.0}, top_10_bws=["x", "y"])
    bws, top = s.bws_bundle_for_matching()
    assert bws == {"x": 2.0, "y": -1.0}
    assert top == ["x", "y"]


def test_count_fallback_with_empty_top10_returns_none():
    s = _state(hb_scores=None, bws_scores={"x": 1.0}, top_10_bws=[])
    bws, top = s.bws_bundle_for_matching()
    assert bws == {"x": 1.0}
    assert top is None


def test_no_bws_data_returns_none_none():
    assert _state().bws_bundle_for_matching() == (None, None)
