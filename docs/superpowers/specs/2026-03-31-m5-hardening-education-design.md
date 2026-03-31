# M5: Hardening + Education Collection — Design Spec

**Date**: 2026-03-31
**Branch**: `feat/m4-cv-uploads` (continuing into M5)
**Priority order**: 5.3 → 5.4 → 5.6 → 5.7 → 5.1 → 5.2 → 5.5 → 5.8 → 5.9

---

## 1. Education Collection in CollectExperiencesAgent (Task 5.3)

### Goal

Add a dedicated education phase as the first question on `first_time_visit`, before the work type loop. Education experiences are stored as `CollectedData` with `source="education"` and `work_type=None`.

### Design Decision: No New WorkType

Adding a new `WorkType` enum has blast radius across 6+ functions with if/elif chains that raise `ValueError` on unknown types, prompt text referencing "four work types", the transition decision tool, and serialized state backward compatibility. Instead, education is a special phase controlled by a boolean flag.

### State Changes

**`CollectExperiencesAgentState`** (`collect_experiences_agent.py`):
- Add field: `education_phase_done: bool = False`
- Pydantic default ensures backward compatibility with existing serialized states

### Flow

```
first_time_visit=True AND education_phase_done=False
  → Education prompt: "Before we talk about work, have you completed any post-secondary education?"
  → If yes: collect course/programme name (experience_title), institution (company), dates, location
  → Support multiple entries
  → When done or user says no: set education_phase_done=True
  → Proceed to normal work type loop (first_time_visit stays True for work type intro)

first_time_visit=True AND education_phase_done=True
  → Normal work type loop begins (existing behavior)
```

### execute() Logic Change

In `collect_experiences_agent.py`, the `execute()` method currently starts with data extraction and conversation LLM. The education phase inserts before the work type loop:

```python
# In execute(), after data extraction but before conversation LLM:
if self._state.first_time_visit and not self._state.education_phase_done:
    # Use education-specific conversation prompt
    # When transition decision says END_WORKTYPE during education phase,
    # set education_phase_done=True instead of popping from unexplored_types
```

The transition decision tool continues to work as-is — it sees `exploring_type=None` during education phase and will trigger `END_WORKTYPE` when the user indicates they're done with education.

### _conversation_llm.py Changes (Collect Experiences)

**New function**: `_get_education_phase_prompt(country_of_user, persona_type)`
- Returns prompt asking about post-secondary education (university, TVET, college, vocational training)
- Framed as introductory: "Before we explore your work experiences..."

**System instructions variant for education phase**:
- Same field collection rules as existing system instructions
- Relabeled fields: "course/programme name" instead of "job title", "institution" instead of "company/employer"
- `experience_title` = course/programme name
- `company` = institution name
- `work_type` = None (not set by conversation LLM during education phase)

### Data Extraction During Education Phase

The existing `_DataExtractionLLM` handles education entries naturally:
- The temporal classifier will return `work_type=None` for education titles — this is correct
- `CollectedData` is created with `source="education"` (set explicitly in the education phase code path)
- No changes needed to `_dataextraction_llm.py` or `_types.py`

### i18n Keys to Add

In `app/i18n/locales/{locale}/messages.json`:
- `collectExperiences.education.askAboutEducation` — "Before we explore your work experiences, have you completed any post-secondary education — for example university, TVET, college, or vocational training?"
- `collectExperiences.education.fields.courseTitle` — "course or programme name"
- `collectExperiences.education.fields.institution` — "institution"
- `collectExperiences.education.transitionToWork` — transition text moving from education to work types

Locales: `en-US`, `sw-KE` (at minimum)

### Files Modified

| File | Change |
|------|--------|
| `collect_experiences_agent.py` | Add `education_phase_done` to state, education phase logic in `execute()` |
| `_conversation_llm.py` (collect) | Add `_get_education_phase_prompt()`, education system instructions variant |
| `app/i18n/locales/en-US/messages.json` | Education translation keys |
| `app/i18n/locales/sw-KE/messages.json` | Education translation keys (Swahili) |

### Files NOT Modified

| File | Reason |
|------|--------|
| `work_type.py` | No new enum value |
| `_transition_decision_tool.py` | Education phase uses boolean flag, not transition LLM work type logic |
| `_dataextraction_llm.py` | `work_type=None` already flows through |
| `_types.py` | `CollectedData.source` and `work_type=None` already supported |
| `_get_experience_type()` | Not called during education phase |
| `_ask_experience_type_question()` | Not called during education phase |
| `_get_excluding_experiences()` | Not called during education phase |

### Acceptance Criteria

- Agent asks about post-secondary education before work types on first visit
- Education experiences collected with course name, institution, dates
- Multiple education entries supported
- Education entries stored with `work_type=None`, `source="education"`
- Normal work type loop proceeds after education phase
- No changes to `WorkType` enum or work-type-dependent functions
- Existing conversations not disrupted (backward compat via Pydantic default)

---

## 2. Education-Aware Skills Explorer Prompts (Task 5.4)

### Goal

When the skills explorer encounters an experience with `source="education"`, adapt prompts to ask about applied skills from coursework rather than day-to-day work responsibilities.

### Current State

`skill_explorer_agent.py:172` already branches on `source == "cv"` to pass CV responsibilities. This is the insertion point for education-specific behavior.

### skill_explorer_agent.py Changes

Line 172 area — extend source handling:
```python
# Current:
cv_responsibilities = ... if source == "cv" else None

# New: also pass source to conversation LLM
# education experiences won't have cv_responsibilities, but need source-aware prompts
```

Pass `source` as a new parameter to the conversation LLM's `execute()` and `create_first_time_generative_prompt()`.

### _conversation_llm.py Changes (Skill Explorer)

**`create_first_time_generative_prompt()`** — add `source` parameter:
- When `source == "education"`: "What tasks are you now able to complete because of what you learned in [course/programme]?"
- When `source == "cv"`: existing CV-aware prompt (unchanged)
- Default: existing "describe a typical day" prompt (unchanged)

**`_create_conversation_system_instructions()`** — add `source` parameter:
- When `source == "education"`:
  - Turn 1: "What you can now do because of this course" (replaces "typical day and key responsibilities")
  - Turn 3: Education-specific question (replaces career growth/business question)
  - Role description: "reflect on what you learned" (replaces "reflect on my experience as [title]")
- Default: existing behavior (unchanged)

**`_get_question_c()`** — add education case:
- Currently returns `""` for `None`/unrecognized work types
- Add: when `source == "education"`, return "What area of your studies are you most confident applying in a work setting?"
- This requires passing `source` into this function (currently only takes `work_type`)

### Data Flow

```
ExperienceEntity(source="education", work_type=None)
  → explore_experiences_agent_director.py (passes ExperienceEntity as-is)
    → skill_explorer_agent.py (reads source from experience_entity)
      → _conversation_llm.py (receives source, adapts prompts)
```

`explore_experiences_agent_director.py` needs no changes — it already passes the full `ExperienceEntity` through.

### Files Modified

| File | Change |
|------|--------|
| `skill_explorer_agent.py` | Pass `source` to conversation LLM |
| `_conversation_llm.py` (skill explorer) | Branch on `source == "education"` in 3 methods |

### Files NOT Modified

| File | Reason |
|------|--------|
| `explore_experiences_agent_director.py` | Already passes ExperienceEntity with source |
| `_ResponsibilitiesExtractionTool` | Extracts from conversation text regardless of source |
| Linking/ranking pipeline | Processes all experiences the same; `work_type=None` supported |

### Acceptance Criteria

- Skills explorer uses education-specific prompts when `source="education"`
- First question asks about applied skills/capabilities from the course
- Follow-ups focus on practical projects, tools, techniques
- Skills extracted from education experiences ranked alongside work-derived skills
- Non-education experiences completely unaffected

---

## 3. E2E Turn-Count Validation (Task 5.6)

### Goal

Validate that Persona 2 with CV completes in fewer turns than without.

### Approach

Add evaluation test cases in `evaluation_tests/` following the existing `ScriptedUserEvaluationTestCase` pattern:
- Persona 2 with CV upload: scripted user with formal employment background + CV data pre-loaded
- Persona 2 without CV: same persona, no CV
- Compare turn counts

### Acceptance Criteria

- E2E test: Persona 2 with CV upload completes in <=15 turns
- E2E test: Persona 2 with CV completes in fewer turns than without CV

---

## 4. Persona 2 Flow Hardening (Task 5.7)

### Goal

Error handling, retry logic, and graceful degradation for the full Persona 2 pipeline.

### Areas

- Structured extraction LLM call failures: retry with backoff, fall back to bullet extraction
- Agent state corruption: validation on load, safe defaults
- GCS upload failures: retry, clear error state for user
- Concurrent CV upload + conversation race conditions

### Files Modified

| File | Change |
|------|--------|
| `cv/service.py` | Retry logic for LLM calls, GCS upload retries |
| `cv/utils/structured_extractor.py` | Handle malformed LLM responses, fallback to bullets |

### Acceptance Criteria

- Persona 2 flow hardened with error handling and retries
- Graceful degradation when LLM extraction fails
- No data loss on transient failures

---

## 5. CV Upload Edge Cases (Task 5.1)

### Goal

Harden CV upload pipeline against real-world edge cases.

### Test Scenarios

- Empty/corrupt PDF upload → graceful error with `CVUploadErrorCode`
- CV with no extractable experiences (fresh graduate with only education)
- CV with 10+ experiences → verify deduplication and agent state limits
- CV upload during active conversation (mid-flow injection)
- Duplicate CV re-upload → `DuplicateCVUploadError` handled gracefully
- Very large CV (50+ pages) → timeout handling via `call_with_timeout`
- CV in Swahili → structured extraction still works

### Files Modified

| File | Change |
|------|--------|
| `cv/service.py` | Defensive checks for edge cases |
| `cv/utils/structured_extractor.py` | Handle malformed LLM responses |
| `evaluation_tests/` | Edge case E2E tests |

---

## 6. Conversation Flow Edge Cases (Task 5.2)

### Goal

Test agent behavior when CV data is incomplete, contradictory, or missing fields.

### Test Scenarios

- CV experience with no responsibilities → agent asks naturally
- CV experience with no dates → agent asks for timeline
- User contradicts CV data during conversation → graceful handling
- User uploads CV after conversation has already started
- Multiple CV uploads by same user → latest extraction used

---

## 7. Frontend Experience Cards (Task 5.5, P1)

### Goal

Display structured experience cards in the chat UI for user review/editing after CV extraction.

### Deferred from M4 Task 3.4. Frontend (React) work.

---

## 8. Swahili Flow Hardening (Task 5.8, P1)

### Goal

Ensure Swahili conversations degrade gracefully.

### Areas

- Language drift detection
- Code-switching handling (English/Swahili mix)
- Swahili CV extraction quality

---

## 9. Handover Package (Task 5.9, P1)

### Goal

Single documentation deliverable: architecture diagram, decision log, operational runbook, known limitations, support paths.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Education phase breaks existing first_time_visit flow | Education phase is gated by `education_phase_done=False` AND `first_time_visit=True` — existing conversations (where `first_time_visit` is already `False`) skip it entirely |
| `_DataExtractionLLM` misclassifies education as work | Education entries have `source="education"` set explicitly in the collection phase — even if the temporal classifier assigns a work type, source remains "education" |
| Transition decision tool confused by education phase | During education, `exploring_type=None` — the tool sees no active work type and will suggest `END_WORKTYPE` when appropriate |
| Skills explorer crashes on `work_type=None` | Already handles `None` — `_get_question_c()` returns `""`, `work_type_short()` returns `""` |
| Backward compatibility with serialized state | Pydantic default `education_phase_done=False` handles missing field in old documents |
