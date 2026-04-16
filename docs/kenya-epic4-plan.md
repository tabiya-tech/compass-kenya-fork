# Kenya Epic 4: Conversation Flow Optimization & Swahili Enablement

## Quick Summary

**Goal**: Make skills elicitation faster (20% time reduction), less repetitive (30% reduction), and more natural while maintaining quality (85%+ skill overlap). Enable Swahili language support.

**Key Personas**: 
- Persona 1: Informal worker (no CV, speaks to tasks/years)
- Persona 2: Formal/mixed worker (has CV, responsibilities documented or explained)

---

# MILESTONE 1: Baseline, Harness & Design Locks

**Objective**: Establish measurable baselines and unblock parallel workstreams.

---

## A1: Technical Work Plan & Dependency Map ✓

**Status**: COMPLETE (this document)

---

## A2: Evaluation Harness + Baseline Runs

### Task: Implement & Integrate Metrics Collector ✓

**What**: Automated metrics collection in E2E tests.

**Files to Create**:
- `backend/evaluation_tests/baseline_metrics_collector.py`

**Files to Modify**:
- `backend/evaluation_tests/e2e_chat_executor.py` - Add metrics_collector parameter, call hooks
- `backend/evaluation_tests/app_conversation_e2e_test.py` - Initialize collector, export metrics
- `backend/evaluation_tests/evaluation_metrics.py` - Add baseline columns to CSV

**Metrics Captured**:
- Turn count, conversation time (total + by phase + by agent)
- LLM calls count and duration
- Experiences discovered/explored, skills per experience
- Repetition rate (semantic similarity > 0.75)
- Skill overlap percentage

**Baseline Test Runs**:
```bash
cd backend
pytest -m "evaluation_test" --repeat 3 \
  -k "single_experience_specific_and_concise_user_e2e or golden_simple_formal_employment" \
  evaluation_tests/app_conversation_e2e_test.py -s
```

**Post-Processing**:
- Calculate mean, median, std dev, 95% CI
- Document metrics output as benchmarks

**Acceptance Criteria**:
- [x] Metrics collector implemented and integrated
- [x] 6 baseline runs completed (2 personas × 3 repititions)
- [x] Metrics exported to JSON/CSV per test
- [x] Statistics calculated and documented

---

## A3: Observability Plan

### Task: Add Correlation IDs & Logging Fields

**Files to Create**:
- `backend/app/middleware/correlation_id_middleware.py`
- `docs/observability-sensitive-data-checklist.md`

**Files to Modify**:
- `backend/app/context_vars.py` - Add `correlation_id: ContextVar[str]`
- `backend/app/server.py` - Register middleware
- `backend/app/conversations/service.py` - Add session_id, turn_index to logs
- `backend/app/agent/agent_director/llm_agent_director.py` - Add agent_type, phase to logs
- `backend/common_libs/llm/llm_caller.py` - Add llm_call_duration_ms to logs

**Logging Fields**:
- `correlation_id`, `session_id`, `turn_index`, `agent_type`, `llm_call_duration_ms`, `phase`

**Sensitive Data Checklist**:
- ❌ NEVER: User PII, full conversation text, raw input before PII filter, API keys
- ✅ SAFE: Session ID (numeric), UUIDs, timing metrics, agent types, aggregated stats

**Acceptance Criteria**:
- [x] Correlation ID middleware implemented
- [x] All 6 logging fields added to relevant code
- [x] Code review confirms no PII logged


## C1: Swahili Model Assessment - FINAL VERDICT IS GEMINI 2.5 (https://docs.cloud.google.com/gemini/docs/codeassist/supported-languages)

**What**: Evaluation framework for Swahili language support.

**Content**:
- Evaluation criteria: Performance, Quality, Cost, Integration, Localization
- Candidate models for language support: Gemini 2.5 Flas
- Shortlist 2-3 models with pros/cons
- Collect 20+ Swahili job terms

**Acceptance Criteria**:
- [x] 2-3 models shortlisted - Chosen the best to be Gemini 2.5

<!-- **New Taxonomy Introoduced For Swahili**:

10 Formal Jobs Added:
- Muuguzi - Nurse
- Dokta - Doctor
- Mhasibu - Accountant / Bookkeeper
- Karani - Clerk / Office worker
- Mwal - Teacher
- Makani - Engineer
- Rubani - Pilot / Driver (can also mean captain)
- Kiongozi - Leader / Manager
- Mzoefu - Trainer / Coach
- Muabiria - Passenger attendant / Tour guide

10 Informal Jobs Added:
- Mchapa kazi - Laborer / General worker
- Msukule kazi - Handyman / Odd jobs person
- Muuzaji - Salesperson / Street vendor
- Mwenye Duka - Small shop owner
- Msee wa Mjengo - Builder / Mason (informal construction)
- Mshonaji - Tailor / Seamstress
- Watchie - Watchman / Security guard
- Seremala - Carpenter
- Mwanamuziki - Musician
- Mchezaji - Player / Athlete / Performer
- Mchukuaji mizigo - Porter / Loader  -->

## Success Criteria

**Quantitative Baselines**:
- [x] Median turn count with confidence interval
- [x] Average conversation time by phase and agent
- [x] Repetition rate calculated
- [x] Skill overlap percentage
- [x] LLM call count and duration

**Infrastructure**:
- [x] Evaluation harness runs automatically
- [x] Metrics exported in JSON/CSV
- [x] Correlation IDs in logs
- [x] Sensitive data checklist reviewed

**Documentation**:
- [x] `baseline_metrics_collector.py` committed
- [x] Baseline metrics documented
- [x] Milestone 2 implementation plan documented (see M2 section below)

---

# MILESTONE 2: Refactor Skills Flow + Persona-Aware Probing

**Objective**: Deliver measurable improvements in flow quality for both personas.

**Baseline Metrics** (from M1):
- Avg turns: 32.4 | LLM calls: 251 | Repetition rate: 11% | Starter diversity: 15.4%
- Test case variance: 16 turns (best) to 70 turns (worst - formal verbose style)
- Critical issue: FAREWELL_AGENT consuming 83% of processing time (64 LLM calls post-conversation)

---

## B1: Refactored Skills Elicitation Flow

**Task 1.1: Debug FAREWELL_AGENT Performance Issue (P0)**
- Investigate why FAREWELL_AGENT makes 64 LLM calls after conversation ends
- Determine if user-facing or backend processing (job matching, skill extraction)
- Fix or separate metrics for accurate timing data
- Files: `llm_agent_director.py`, `farewell_agent.py`, `conversations/service.py`

**Task 1.2: Reduce Starter Phrase Repetition (P0)**
- Problem: "Okay" used in 27% of questions; diversity only 15.4%
- Target: Top starter <15%, diversity >35%
- Add varied acknowledgment phrases to prompts
- Files: `collect_experiences_prompt.py`, `explore_skills_prompt.py`

**Task 1.3: Increase Achievement Question Rate (P1)**
- Problem: Only 1.9% achievement questions (target: >8%)
- Add prompts for accomplishments, challenges overcome, improvements
- Files: `explore_skills_prompt.py`

**Task 1.4: Optimize Skills Exploration (P0)**
- Reduce from 6 turns to 4 turns per experience
- Consolidate questions, add exit criteria (8-12 skills OR 4 turns)
- Files: `explore_skills_agent.py`, `explore_skills_prompt.py`

**Task 1.5: Early Exit for Concise Users (P2)**
- Detect rich, detailed responses and skip redundant follow-ups
- Target: Concise users complete in <18 turns
- Files: `llm_agent_director.py`

---

## B2: Persona-Aware Flow Implementation

**Important**: CV upload integration deferred to Milestone 4. Persona detection is verbal-only for M2.

**Task 2.1: Implement Persona Detection (P0)**
- Detect Persona 2 (Formal) via verbal cues: "title", "position", "department", "responsibilities"
- Detect Persona 1 (Informal) via: "tasks", "daily work", "what I did"
- Default to Persona 1 (safer for informal workers)
- Create: `backend/app/agent/persona_detector.py`
- Modify: `conversations/service.py`, `llm_agent_director.py`

**Task 2.2: Persona 1 (Informal) Optimization (P1)**
- Target: 18-22 turns (simple), ≤35 turns (multi-experience)
- Use simpler language, more examples/scaffolding
- Focus on "what did you do daily" → skills mapping
- Files: `collect_experiences_prompt.py`, `explore_skills_prompt.py`

**Task 2.3: Persona 2 (Formal) Optimization (P0 - Highest Impact)**
- Problem: Formal verbose descriptions take 70 turns (!)
- Target: ≤35 turns (down from 70)
- Acknowledge formal info upfront, avoid redundant questions
- Track information completeness per experience
- Files: `collect_experiences_agent.py`, prompt files

**Task 2.4: Multi-Experience Optimization (P1)**
- Problem: 49 turns for 3+ experiences
- Target: ≤35 turns for 3+ experiences
- First experience: Full exploration (4-5 turns)
- Subsequent: Focused exploration (3 turns)
- Files: `llm_agent_director.py`, `conversations/service.py`

---

## Golden Transcripts (English) + CI Gating

**Task 3.1: Create Golden Transcripts (Based on Refactored Flow)**
- Timing: Create AFTER B1 + B2 refactoring complete
- 6 transcripts total (3 per persona):
  - Persona 1: Simple single exp (18-20 turns), Multi-exp (30-35), Process questioner (20-25)
  - Persona 2: Simple formal (20-25), Formal verbose (30-35), Career progression (35-40)
- Create: `backend/evaluation_tests/golden_transcripts/persona_1/*.json`
- Create: `backend/evaluation_tests/golden_transcripts/persona_2/*.json`

**Task 3.2: Implement CI Test Integration (P0)**
- Metrics to Gate (Block PR): Turn count ±2, Repetition ≤8%, Skill overlap ≥85%
- Metrics to Warn: Achievement Q rate ≥5%, Starter diversity ≥35%
- Create: `golden_transcript_runner.py`, `check_metrics_thresholds.py`
- Create: `.github/workflows/golden_transcript_tests.yml`

---

## C1: Swahili Model Documentation

**Task 4.1: Document Gemini 2.5 Flash Selection**
- Model comparison: Gemini 2.5 Flash vs GPT-4o vs Claude 3.5
- Criteria: Swahili performance, cost, latency, integration
- Selection rationale and cost analysis
- Create: `docs/swahili-model-selection.md`

**Task 4.2: Gemini Integration Preparation**
- API setup checklist for M3
- Environment variables, rate limits, pricing
- Create: `docs/gemini-integration-checklist.md`

---

## Success Criteria

**Performance Improvements** (vs Baseline: 32.4 turns, 11% repetition, 251 LLM calls):
- [x] Turn count reduced to ≤27 (17%+ reduction)
- [x] Repetition rate reduced to ≤8% (27%+ reduction)
- [x] Starter diversity increased to ≥35% (from 15.4%)
- [x] Achievement question rate ≥8% (from 1.9%)
- [x] LLM calls reduced to ≤200 (20%+ reduction)

**Quality Maintained**:
- [x] Skill overlap maintained at 85%+
- [x] Experience completeness maintained at 95%+
- [x] No regression in occupation accuracy

**Persona-Aware Flows**:
- [x] Persona detection implemented (verbal-only, >90% accuracy)
- [x] Persona 1 (Informal): 18-22 turns simple, ≤35 multi-experience
- [x] Persona 2 (Formal): 20-25 turns simple, ≤35 turns verbose (down from 70!)
- [x] Flow adapts based on detected persona type

**CI/CD Integration**:
- [x] 6 golden transcripts created (3 per persona)
- [x] Automated tests run on every PR with metric thresholds
- [x] Clear failure messages when quality gates violated

**Swahili Preparation**:
- [x] Gemini 2.5 Flash selection documented with rationale
- [x] Integration checklist ready for M3 (no blockers)

---

# MILESTONE 3: Swahili Enablement + Localization

**Objective**: Deliver Swahili flows with mapping parity and regression protection.

## C2: Localization/Synonym Mapping Module

**Objective**: Normalize Swahili and code-switched inputs and map them to taxonomy terms.

**Tasks**:
- Build a Swahili term dictionary (50+ terms) including informal slang and code-switch variants.
- Define a normalized mapping format (JSON/CSV) and load it at runtime.
- Implement a mapping service to resolve Swahili terms before retrieval / skill extraction.
- Add coverage + accuracy checks (mapping hit rate, false positives).
- Document mapping sources and update process.

**How RAG helps**:
- Use a Swahili glossary + taxonomy snippets as retrieval context to disambiguate slang and code-switched terms.
- Retrieve localized examples for prompts so the agent uses consistent Swahili phrasing and domain terms.
- Support fallback mapping when exact synonym matches are missing, without overfitting prompts.

## C3: Swahili-Enabled Flows End-to-End

**Objective**: Enable the full flow in Swahili with quality parity.

**Tasks**:
- Add Swahili locale to backend config and frontend supported locales.
- Provide Swahili translations for core system messages and prompts.
- Enforce Swahili responses in LLM prompt templates (no language drift).
- Create Swahili E2E test cases (Persona 1 + Persona 2).
- Create Swahili golden transcripts and integrate into CI.
- Compare Swahili skill discovery accuracy to English baseline (≥80% parity).

---

## Success Criteria

**Swahili Language Support**:
- [x] Skills elicitation flow works end-to-end in Swahili
- [x] Preference flow functional in Swahili
- [x] Language switching implemented
- [x] Swahili responses maintain correct tone and grammar

**Localization/Mapping**:
- [x] Synonym mapping module created and tested
- [x] 50+ Swahili job terms mapped to taxonomy
- [x] Code-switched terms handled
- [x] Regional variations documented

**Quality Parity**:
- [x] Swahili skill discovery accuracy at 80%+ of English baseline
- [x] Occupation matching works for Swahili inputs
- [x] Same structured output as English flows

**Testing & Regression**:
- [x] Swahili test cases created for both personas
- [x] Automated tests integrated into CI
- [x] Regression protection for English flows

---

# MILESTONE 4: CV Integration + Qualifications Extraction

**Objective**: Make Persona 2 experience coherent and add qualifications affecting eligibility. This should support basic CV file uploads in the data extraction layer.

**Current State (What Already Exists)**:
- CV upload pipeline: `backend/app/users/cv/service.py` — file upload → markdown conversion → LLM extraction → GCS storage
- CV extraction: `CVExperienceExtractor` produces `list[str]` (plain experience bullets), **not** structured entities
- Frontend: `Chat.tsx` / `ChatMessageField.tsx` handles file upload, polls status, displays bullets
- MongoDB: `user_cv_uploads` collection stores upload records + experience bullets
- Feature flag: `GLOBAL_ENABLE_CV_UPLOAD` gates the feature
- **Gap**: CV data never flows into the conversation agents — the extracted bullets are dead-end data

---

## B3: CV Integration → Merged Profile

**Objective**: Bridge the existing CV upload pipeline with the conversational experience collection so that Persona 2 users who upload a CV get a faster, less repetitive flow. The agent should acknowledge what the CV already says and only ask for supplementary details.

### Task 3.1: Structured CV Extraction (P0)

**Problem**: `CVExperienceExtractor` returns `list[str]` — unstructured sentences. The `CollectExperiencesAgent` needs `CollectedData` objects with typed fields (title, company, timeline, work_type).

**What**: Enhance the CV extraction LLM to produce structured experience data that maps directly to `CollectedData`.

**Files to Create**:
- `backend/app/users/cv/utils/structured_extractor.py` — New `CVStructuredExtractor` class
- `backend/app/users/cv/types.py` — Add `CVExtractedExperience` Pydantic model

**Files to Modify**:
- `backend/app/users/cv/service.py` — Add structured extraction step after bullet extraction
- `backend/app/users/cv/repository.py` — Persist structured experiences alongside bullets

**`CVExtractedExperience` Model** (new type in `types.py`):
```python
class CVExtractedExperience(BaseModel):
    experience_title: str
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    work_type: Optional[str] = None  # maps to WorkType enum key
    responsibilities: list[str] = Field(default_factory=list)
    source: str = "cv"  # provenance marker
```

**LLM Prompt Design**: The `CVStructuredExtractor` system instructions should ask Gemini to return:
```json
{
  "experiences": [
    {
      "experience_title": "Project Manager",
      "company": "University of Oxford",
      "location": "Oxford, UK",
      "start_date": "2018",
      "end_date": "2020",
      "work_type": "FORMAL_SECTOR_WAGED_EMPLOYMENT",
      "responsibilities": ["Led 5-person team", "Managed £200k budget"]
    }
  ],
  "qualifications": [...]  // (extracted in B4)
}
```

**Pipeline Update** (in `CVUploadService._pipeline`):
1. CONVERTING → markdown
2. EXTRACTING → bullet extraction (existing, kept for backward compat)
3. **NEW**: STRUCTURING → structured extraction (produces `list[CVExtractedExperience]`)
4. UPLOADING_TO_GCS → storage
5. COMPLETED

**Acceptance Criteria**:
- [x] `CVStructuredExtractor` produces `CVExtractedExperience` objects from CV markdown
- [x] Pipeline stores structured experiences in `user_cv_uploads` record
- [x] Existing bullet extraction is preserved (backward compat)
- [x] Unit tests: 3+ real CV markdowns → verified structured output

---

### Task 3.2: CV-to-Agent State Mapper (P0)

**Problem**: Even with structured CV data, there's no bridge to the agent's `CollectedData` state. When a conversation starts, the agent has no knowledge of the uploaded CV.

**What**: Create a mapper that converts `CVExtractedExperience` → `CollectedData` and a loader that pre-populates the `CollectExperiencesAgent` state.

**Files to Create**:
- `backend/app/users/cv/cv_to_agent_mapper.py` — Mapping logic + deduplication

**Files to Modify**:
- `backend/app/conversations/service.py` — Before first conversation turn, check for completed CV uploads and pre-populate agent state
- `backend/app/agent/collect_experiences_agent/_types.py` — Add `source: Optional[str] = None` field to `CollectedData` (provenance: "cv" | "conversation" | None)
- `backend/app/agent/collect_experiences_agent/collect_experiences_agent.py` — Add `set_cv_experiences()` method

**Mapping Logic** (`cv_to_agent_mapper.py`):
```python
def map_cv_to_collected_data(cv_experiences: list[CVExtractedExperience],
                              existing_data: list[CollectedData]) -> list[CollectedData]:
    """
    Convert CV experiences to CollectedData, deduplicating against
    any already-collected conversational data.
    """
```

- Map `work_type` string → `WorkType` enum key, defaulting to `FORMAL_SECTOR_WAGED_EMPLOYMENT`
- Mark fields from CV as populated (not None), so agent won't re-ask
- Set `source="cv"` for provenance tracking
- Deduplicate using `CollectedData.compare_relaxed()` against existing data

**Integration in `ConversationService.send()`**:
```python
# On first turn, check for completed CV uploads
if is_first_turn and cv_upload_exists:
    cv_experiences = await cv_repository.get_structured_experiences(user_id)
    mapped = map_cv_to_collected_data(cv_experiences, state.collected_data)
    state.collected_data.extend(mapped)
    # Mark relevant work types as partially explored
```

**Acceptance Criteria**:
- [x] CV experiences appear in `CollectExperiencesAgent` state on first turn
- [x] Deduplication prevents double-counting
- [x] Provenance field tracks data source ("cv" vs "conversation")
- [x] Agent state is serializable/deserializable with new field

---

### Task 3.3: Persona 2 Conversational Flow Adaptation (P0)

**Problem**: When CV data is pre-populated, the `CollectExperiencesAgent` must behave differently — it should acknowledge the CV, confirm extracted info, and only probe for missing details rather than asking everything from scratch.

**What**: Modify prompts and transition logic so the agent recognizes pre-populated CV data.

**Files to Modify**:
- `backend/app/agent/collect_experiences_agent/_conversation_llm.py` — Add CV-aware prompt variation
- `backend/app/agent/collect_experiences_agent/collect_experiences_prompt.py` — New prompt section for CV-seeded flow
- `backend/app/agent/collect_experiences_agent/_transition_decision_tool.py` — Adjust transition thresholds when CV data present

**Prompt Changes** (when CV data detected):
- Opening: "I see from your CV that you've worked as [title] at [company]. Can you tell me more about what you did day-to-day?" instead of "What jobs have you had?"
- Skip basic info questions (title, company, dates) for CV-sourced experiences
- Focus on responsibilities, achievements, and skills not captured in CV
- Still ask about experience types not found in CV (e.g., volunteer work, informal work)

**Transition Logic Changes**:
- If all CV experiences have confirmed titles + at least one responsibility, the `END_WORKTYPE` threshold should trigger sooner for CV-covered work types
- Still explore `unexplored_types` not represented in CV data (e.g., if CV only has formal employment, still ask about self-employment, volunteer work, unpaid work)

**Acceptance Criteria**:
- [x] Agent acknowledges CV data in opening turn
- [x] Agent skips redundant questions for CV-populated fields
- [x] Agent still explores work types not found in CV
- [ ] Turn count for Persona 2 with CV: ≤15 turns (down from 20-25 without CV)
- [ ] E2E test: Persona 2 with CV upload completes in fewer turns than without

---

### Task 3.4: CV Confirmation & Edit UI Flow (P1)

**Problem**: Users should be able to review, correct, and supplement CV-extracted experiences before the agent proceeds with skills exploration.

**What**: After CV extraction completes, present a summary to the user in the chat and allow inline corrections.

**Files to Modify**:
- `frontend-new/src/chat/Chat.tsx` — After CV upload completes, display structured experience cards
- `backend/app/conversations/experience/routes.py` — Existing PATCH endpoint works for CV-sourced experiences (no change needed, but verify)
- `backend/app/users/cv/routes.py` — Add `GET /users/{user_id}/cv/{upload_id}/structured` endpoint for structured data

**Flow**:
1. User uploads CV → polling → extraction completes
2. Frontend displays: "I found N experiences in your CV:" with structured cards
3. Each card shows: title, company, dates, location
4. User can edit inline (uses existing experience PATCH endpoint)
5. User confirms → conversation proceeds with supplementary questions only

**Acceptance Criteria**:
- [x] Structured CV data available via API
- [ ] Frontend displays experience cards from CV
- [ ] User can edit CV-extracted experience details
- [ ] Edits persist and are reflected in agent state

---

## B4: CV Qualifications Extraction

**Objective**: Extract qualifications (certifications, diplomas, trade licenses) from CVs as part of the structured extraction pipeline. Qualifications are captured in the `structured_extraction` field alongside experiences.

> **Note**: A separate qualifications module (entity model, repository, routes, conversational detection, job matching integration) was originally planned but descoped after review. Qualification data is captured in the `structured_extraction` field on `user_cv_uploads` instead. Dedicated qualifications features (Kenya-specific recognition, conversational detection, job matching integration) are deferred to M5.

### Task 4.2: CV Qualifications Extraction (P0)

**What**: The structured extraction prompt (Task 3.1) extracts both experiences and qualifications in a single LLM call, stored in the `structured_extraction` field.

**Acceptance Criteria**:
- [x] CV extraction returns both experiences and qualifications in structured format
- [x] Qualifications stored in `user_cv_uploads` record (`structured_extraction` field)
- [x] Unit tests: CVs with qualifications → verified extraction output

---

## C4: Swahili Tests + Evaluation Scripts

**Objective**: Ensure Swahili language flows maintain quality parity with English and are protected by regression tests.

### Task 5.1: Swahili Golden Transcripts (P0)

**What**: Create golden test transcripts for Swahili conversations across both personas.

**Files to Create**:
- `backend/evaluation_tests/golden_transcripts/swahili/persona_1_simple.json`
- `backend/evaluation_tests/golden_transcripts/swahili/persona_1_multi_experience.json`
- `backend/evaluation_tests/golden_transcripts/swahili/persona_1_artisan.json`
- `backend/evaluation_tests/golden_transcripts/swahili/persona_2_formal.json`
- `backend/evaluation_tests/golden_transcripts/swahili/persona_2_with_cv.json`
- `backend/evaluation_tests/golden_transcripts/swahili/persona_2_code_switched.json`

**Transcript Design**:
- Persona 1 Simple: Informal worker, single experience, pure Swahili (18-22 turns)
- Persona 1 Multi: Multiple informal experiences, Swahili with some Sheng (30-35 turns)
- Persona 1 Artisan: Trade worker with NITA qualification, Swahili (20-25 turns)
- Persona 2 Formal: Formal employment, Swahili, references CV (20-25 turns)
- Persona 2 CV: Formal with CV upload, Swahili (12-18 turns with CV pre-population)
- Persona 2 Code-Switched: English-Swahili code-switching throughout (25-30 turns)

### Task 5.2: Evaluation Scripts (P0)

**What**: Build automated evaluation scripts that measure Swahili-specific metrics.

**Files to Create**:
- `backend/evaluation_tests/swahili_evaluation_runner.py`
- `backend/evaluation_tests/swahili_metrics.py`

**Metrics to Capture**:
- Skill discovery accuracy vs English baseline (target: ≥80% parity)
- Language drift rate (agent responding in wrong language)
- Swahili synonym mapping hit rate
- Code-switch handling accuracy
- Qualification extraction accuracy in Swahili

### Task 5.3: CI Integration (P1)

**What**: Integrate Swahili tests into the existing CI pipeline.

**Files to Modify**:
- `.github/workflows/golden_transcript_tests.yml` — Add Swahili test jobs
- `backend/evaluation_tests/check_metrics_thresholds.py` — Add Swahili thresholds

**Thresholds**:
- Swahili skill overlap vs English: ≥80%
- Language drift: ≤2% of agent turns
- Turn count: within 20% of English equivalent

**Acceptance Criteria**:
- [ ] 6 Swahili golden transcripts created (3 per persona)
- [ ] Evaluation scripts measure language-specific metrics
- [ ] CI runs Swahili tests alongside English tests
- [ ] Regression protection for both languages
- [ ] Performance benchmarks documented

---

## Deployment Readiness

**Objective**: Ensure the system is deployable with the Gemini 2.5 Flash model provider and all new M4 features are operationally ready.

### Task 6.1: Infrastructure & Config (P0)

**What**: Update deployment configuration for Gemini model provider and new M4 features.

**Files to Modify**:
- `backend/app/app_config.py` — Ensure CV config complete
- `backend/app/server.py` — Ensure CV routes enabled

**New Environment Variables**:
- `BACKEND_CV_STRUCTURED_EXTRACTION_ENABLED` — Feature flag for structured CV extraction
- `BACKEND_GEMINI_API_KEY` — (verify existing) Gemini API key
- `BACKEND_GEMINI_MODEL_ID` — (verify existing) Model identifier

### Task 6.2: Secrets & Security Review (P1)

**What**: Review secrets management and ensure no PII leaks in new features.

**Deliverables**:
- CV content: verify files stored encrypted in GCS, never logged
- LLM prompts: verify no CV text forwarded beyond extraction step
- API endpoints: verify authentication required on all new routes
- Update `docs/observability-sensitive-data-checklist.md` with M4 additions

### Task 6.3: Deployment Documentation (P1)

**Files to Create**:
- `docs/deployment-runbook-m4.md` — Step-by-step deployment guide

**Contents**:
- GCS bucket permissions for CV storage
- Environment variable checklist
- Feature flag rollout order: structured extraction → CV-agent integration
- Rollback procedures for each feature
- Health check endpoints to verify

**Acceptance Criteria**:
- [ ] All new environment variables documented
- [ ] Secrets management reviewed — no PII in logs
- [ ] Feature flags allow incremental rollout
- [ ] Deployment runbook covers rollout + rollback

---

## Success Criteria

**CV Integration (Persona 2)**:
- [x] CV upload → structured extraction pipeline functional
- [x] Structured experiences (`CVExtractedExperience`) extracted with title, company, timeline, work_type, responsibilities
- [x] Conversational flow merges with CV data — agent acknowledges CV content
- [x] Duplicate detection prevents redundant questions (`compare_relaxed` dedup)
- [x] Provenance tracking: each experience/qualification marked as "cv" or "conversation" sourced
- [x] Certifications extracted from CVs via `CVStructuredExtractor` (stored in `structured_extraction`)

**Persistence & Data Quality**:
- [x] All experiences saved to database with provenance ("cv" | "conversation")
- [x] All skills persisted with provenance
- [x] Data validation ensures completeness (no null titles, valid enum values)

---

# MILESTONE 5: Hardening + Handover

**Objective**: Finalize robustness, operational readiness, and transition to support. Pick up deferred M4 items (qualifications integration, frontend experience cards, E2E turn-count validation).

---

## B5: Safety/Edge Case Simulation Suite

### Task 5.1: CV Upload Edge Cases (P0)

**What**: Harden the CV upload → structured extraction → agent pipeline against real-world edge cases.

**Test Scenarios**:
- Empty/corrupt PDF upload → graceful error with `CVUploadErrorCode`
- CV with no extractable experiences (e.g., fresh graduate with only education)
- CV with 10+ experiences → verify deduplication and agent state limits
- CV upload during active conversation (mid-flow injection)
- Duplicate CV re-upload → `DuplicateCVUploadError` handled gracefully
- Very large CV (50+ pages) → timeout handling via `call_with_timeout`
- CV in Swahili → structured extraction still works

**Files to Modify**:
- `backend/app/users/cv/service.py` — Add defensive checks for edge cases
- `backend/app/users/cv/utils/structured_extractor.py` — Handle malformed LLM responses
- `backend/evaluation_tests/` — Add edge case E2E tests

**Acceptance Criteria**:
- [ ] All edge case scenarios tested and handled gracefully
- [ ] No unhandled exceptions in CV pipeline for malformed input
- [ ] Error codes returned for each failure mode

### Task 5.2: Conversation Flow Edge Cases (P0)

**What**: Test agent behavior when CV data is incomplete, contradictory, or missing fields.

**Test Scenarios**:
- CV experience with no responsibilities → agent asks for them naturally
- CV experience with no dates → agent asks for timeline
- User contradicts CV data during conversation → agent handles gracefully
- User uploads CV after conversation has already started
- Multiple CV uploads by same user → latest extraction used

**Acceptance Criteria**:
- [ ] Agent handles partial CV data without crashing or looping
- [ ] Contradictions between CV and conversation are resolved gracefully
- [ ] Late CV upload integrates without disrupting flow

---

## B5.2: Post-Secondary Education Collection

**Objective**: Capture post-secondary education (university, TVET, college) as a distinct experience so that education-derived skills are discovered and ranked alongside work-derived skills.

### Design Decision: No new WorkType

Adding a new `WorkType` enum value has a large blast radius — every if/elif chain on `WorkType` (6+ functions), prompt text referencing "four work types", the transition decision tool, the data extraction LLM classifier, and serialized state backward compatibility would all need updates. Instead:

- Education is collected as a **special phase before the work type loop**, not as a work type itself.
- Education entries are stored as `CollectedData` with `work_type=None` and `source="education"`.
- The skills explorer branches on `source == "education"` for prompt adaptation.
- The linking/ranking pipeline processes education entries like any other experience (`work_type=None` is already handled — it maps the experience title to ESCO occupations, which works reasonably for degree/course names).
- **Zero changes** to `WorkType`, transition decision tool, data extraction LLM, or the `_get_experience_type`/`_ask_experience_type_question`/`_get_excluding_experiences` functions.

---

### Task 5.3: Education Collection in CollectExperiencesAgent (P0)

**What**: Add a dedicated education question as a special phase on `first_time_visit`, before the work type loop begins. Education experiences are collected as `CollectedData` entries with `source="education"`.

**Flow**:
1. On first visit, agent explains the process AND asks: "Before we talk about work, have you completed any post-secondary education — for example university, TVET, college, or vocational training?"
2. If yes → collect: course/programme name (`experience_title`), institution (`company`), dates, location. Support multiple entries.
3. If no → proceed directly to work type loop.
4. Once education collection is done (`education_phase_done=True`), proceed to the normal work type loop exactly as before.

**Files to Modify**:
- `backend/app/agent/collect_experiences_agent/collect_experiences_agent.py`
  - Add `education_phase_done: bool = False` to `CollectExperiencesAgentState`
  - In `execute()`: when `first_time_visit=True` and `education_phase_done=False`, use education prompt instead of first work type prompt
  - When user indicates no more education, set `education_phase_done=True` and transition to work type loop
- `backend/app/agent/collect_experiences_agent/_conversation_llm.py`
  - Add `_get_first_time_education_prompt()` — similar to `_get_first_time_generative_prompt()` but asks about education instead of work type
  - Education prompt: asks for course name, institution, dates (reuses existing field collection logic)
  - System instructions variant for education phase: same field collection rules, but labels adjusted ("course/programme" instead of "job title", "institution" instead of "company")

**What does NOT change**:
- `work_type.py` — no new enum value
- `_transition_decision_tool.py` — education phase uses a simple flag, not the transition LLM
- `_dataextraction_llm.py` — education entries have `work_type=None`, extraction LLM handles this already
- `_types.py` — `CollectedData.source` field already exists, `work_type` already accepts `None`
- `_get_experience_type()`, `_ask_experience_type_question()`, `_get_excluding_experiences()` — untouched

**Backward Compatibility**: `education_phase_done` defaults to `False`. Existing serialized states without this field will deserialize cleanly (Pydantic default). For in-progress conversations, they'll get the education question on their next first-visit turn — harmless since `first_time_visit` is already `False` for active conversations.

**i18n**: Add translation keys for education question and field labels (English + Swahili).

**Acceptance Criteria**:
- [x] Agent asks about post-secondary education before work types on first visit
- [x] Education experiences collected with course name, institution, dates
- [x] Multiple education entries supported
- [x] Education entries stored with `work_type=None`, `source="education"`
- [x] Normal work type loop proceeds after education phase
- [x] No changes to `WorkType` enum or work-type-dependent functions
- [x] Existing conversations not disrupted (backward compat)

---

### Task 5.4: Education-Aware Skills Explorer Prompts (P0)

**What**: When the `SkillsExplorerAgent` encounters an experience with `source="education"`, adapt prompts to ask about applied skills from coursework rather than day-to-day work responsibilities.

**Prompt Adaptations** (in `_conversation_llm.py`):
- **First question** (replaces "describe a typical day"): "What tasks are you now able to complete because of what you learned in that course/programme?"
- **Follow-up**: "What practical projects or assignments did you work on?" / "What tools or techniques did you learn to use?"
- **Achievement question**: "What was your biggest accomplishment or most challenging project during your studies?"
- **Question C** (`_get_question_c`): For education, use something like "What area of your studies are you most confident applying in a work setting?" — the existing function returns `""` for unknown work types, so this is additive not breaking

**System instructions adaptation** (`_create_conversation_system_instructions`):
- Turn flow step 1: "What you can now do because of this course" instead of "Typical day and key responsibilities"
- Turn flow step 3: Education-specific question instead of career growth/business question
- Role description: "reflect on what you learned" instead of "reflect on my experience as [title] (work type)"

**Files to Modify**:
- `backend/app/agent/skill_explorer_agent/_conversation_llm.py`
  - `create_first_time_generative_prompt()` — branch on `source == "education"` for initial question
  - `_create_conversation_system_instructions()` — branch on source for turn flow and role description
  - `_get_question_c()` — add education case (currently returns `""` for non-matching types, so this is safe)
- `backend/app/agent/skill_explorer_agent/skill_explorer_agent.py` — pass `source` to conversation LLM (it already passes `work_type` and `cv_responsibilities`)
- `backend/app/agent/explore_experiences_agent_director.py` — ensure `source` is passed through from `ExperienceEntity` to the skills explorer

**What does NOT change**:
- `_ResponsibilitiesExtractionTool` — extracts responsibilities from conversation text regardless of source
- Linking/ranking pipeline — processes all experiences the same way; `work_type=None` is already supported
- `WorkType.work_type_short()` — returns `""` for `None`, prompts will just skip the work type label

**Acceptance Criteria**:
- [x] Skills explorer uses education-specific prompts when `source="education"`
- [x] First question asks about applied skills/capabilities from the course
- [x] Follow-ups focus on practical projects, tools, techniques
- [x] Skills extracted from education experiences are ranked alongside work-derived skills
- [x] Education experiences appear in the final skill profile
- [x] Non-education experiences completely unaffected (no prompt changes)

---

## B5.3: Deferred M4 Items

### Task 5.5: Frontend Experience Cards (P1 — deferred from M4 Task 3.4)

**What**: After CV extraction completes, display structured experience cards in the chat UI for user review and editing.

**Acceptance Criteria**:
- [ ] Frontend displays experience cards from CV extraction
- [ ] User can edit CV-extracted experience details inline
- [ ] Edits persist and are reflected in agent state

### Task 5.6: E2E Turn-Count Validation (P0 — deferred from M4 Task 3.3)

**What**: Validate that Persona 2 with CV completes in fewer turns than without.

**Acceptance Criteria**:
- [ ] E2E test: Persona 2 with CV upload completes in ≤15 turns
- [ ] E2E test: Persona 2 with CV completes in fewer turns than without CV

---

## Hardening Across Persona 2 + Swahili Flows

### Task 5.7: Persona 2 Flow Hardening (P0)

**What**: Error handling, retry logic, and graceful degradation for the full Persona 2 pipeline.

**Areas**:
- Structured extraction LLM call failures → retry with backoff, fall back to bullet extraction
- Agent state corruption → validation on load, safe defaults
- GCS upload failures → retry, clear error state for user
- Concurrent CV upload + conversation race conditions

**Acceptance Criteria**:
- [ ] Persona 2 flow hardened with error handling and retries
- [ ] Graceful degradation when LLM extraction fails
- [ ] No data loss on transient failures

### Task 5.8: Swahili Flow Hardening (P1)

**What**: Ensure Swahili conversations degrade gracefully and maintain quality.

**Areas**:
- Language drift detection (agent responding in wrong language)
- Code-switching handling (English/Swahili mix)
- Swahili synonym mapping fallbacks
- Swahili CV extraction quality

**Acceptance Criteria**:
- [ ] Swahili flow hardened with fallback mechanisms
- [ ] Language drift rate ≤2% of agent turns
- [ ] Code-switching handled without quality loss

---

## A5: Handover Documentation

### Task 5.9: Handover Package (P1)

**What**: Single documentation deliverable covering architecture, operations, and knowledge transfer for the full Kenya fork.

**Deliverables**:
- Architecture diagram: CV upload → structured extraction → agent state → conversation flow → skills ranking → recommendations
- Decision log: key design decisions (qualifications module descoped, `structured_extraction` approach, education as special phase not WorkType, etc.)
- Operational runbook: key metrics to watch, common failure modes, troubleshooting steps, backup/recovery for MongoDB + GCS
- Known limitations and future enhancement opportunities
- Support escalation paths

**Acceptance Criteria**:
- [ ] Single handover document covering architecture, operations, and known limitations

---

## Success Criteria

**Post-Secondary Education Collection**:
- [x] Agent asks about post-secondary education as first question in collection flow
- [x] Education experiences collected with course name, institution, dates
- [x] Skills explorer uses education-specific prompts ("what can you do because of this course?")
- [x] Education-derived skills ranked alongside work-derived skills

**Safety & Edge Cases**:
- [ ] CV upload edge cases tested (corrupt files, empty CVs, oversized CVs, Swahili CVs)
- [ ] Conversation edge cases tested (partial data, contradictions, late uploads)
- [ ] Graceful degradation for model failures

**Deferred M4 Completions**:
- [ ] Frontend experience cards for CV review/editing
- [ ] E2E turn-count validation for Persona 2 with CV

**Robustness & Hardening**:
- [ ] Persona 2 flow hardened with error handling and retries
- [ ] Swahili flow hardened with fallback mechanisms

**Documentation & Handover**:
- [ ] Single handover document covering architecture, operations, and known limitations

---

## Cross-Milestone Contributions (M1–M5)

Work delivered across the full Epic 4 timeline that spans multiple milestones or was done in parallel with planned tasks.

**Infrastructure & Platform**:
- SSE streaming: conversation response streaming with chunked output, status updates, streaming sink refactor
- Apigee migration: moved from API Gateway to Apigee for SSE support and backend authentication (Pulumi IaC)
- NAT Gateway setup for Cloud Run static outbound IP routing
- Docker/deployment: offline vignette generation, deployment procedures, Pulumi configs

**Epic 2 — Preference Elicitation**:
- BWS (Best-Worst Scaling) card UI, HB scoring, backend message_type signal
- Bayesian inference and adaptive D-optimal selection
- Preference elicitation agent with vignette system
- BWS occupation ranking and offline optimization
- Migration from occupation codes to ONET WA elements
- Ratio-based FIM stopping criterion to prevent adaptive phase skip
- Integration tests and comprehensive E2E test suite

**Epic 3 — Recommender Agent**:
- RecommenderAdvisorAgent integration with seamless handoff from skills exploration
- Matching service integration (Node2Vec output schema)
- Qualitative metadata in advisor agent prompts
- Opportunity recommendation fallback when no occupations available
- Phase transition capabilities, SKILLS_UPGRADE_PIVOT handler
- Career path and job credential DB stubs
- User recommendations service injection into agent director

**Conversation Flow & Agent System**:
- Agent configuration updates (Gemini 2.5 Flash/Flash-Lite model switches)
- Counseling sub-phases for better agent routing
- Experience collection parallelization (parallel LLM calls)
- GATE phase addition and conversation flow reordering
- Skip-to-phase functionality for testing all journey phases
- Normalized experience titles for improved display
- Persona detection and prompt customization
- Province/city on user preferences for recommender agent

**Frontend & UX**:
- Chat history mapping fixes (bws_response JSON hidden from refresh)
- BWS button theming (tabiyaGreen/tabiyaRed palette)
- Keyboard input lag fix during typing
- New session feature flag handling
- ESLint/Prettier CI fixes

**i18n & Localization**:
- Centralized translations in Google Sheets with documented workflow
- Frontend locale synchronization to backend user preferences
- Swahili language context, terminology, and prompt customization
- BWS task message internationalization

**Observability & Quality**:
- Correlation ID middleware
- Logging field additions (session_id, turn_index, agent_type, llm_call_duration_ms)
- Bandit security linting, nosec comments for LLM control tokens
- MongoDB index optimization (compound index on skillId/modelId)
- Skill aggregation pipeline performance optimization
