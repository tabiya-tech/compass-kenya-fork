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
- [ ] `CVStructuredExtractor` produces `CVExtractedExperience` objects from CV markdown
- [ ] Pipeline stores structured experiences in `user_cv_uploads` record
- [ ] Existing bullet extraction is preserved (backward compat)
- [ ] Unit tests: 3+ real CV markdowns → verified structured output

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
- [ ] CV experiences appear in `CollectExperiencesAgent` state on first turn
- [ ] Deduplication prevents double-counting
- [ ] Provenance field tracks data source ("cv" vs "conversation")
- [ ] Agent state is serializable/deserializable with new field

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
- [ ] Agent acknowledges CV data in opening turn
- [ ] Agent skips redundant questions for CV-populated fields
- [ ] Agent still explores work types not found in CV
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
- [ ] Structured CV data available via API
- [ ] Frontend displays experience cards from CV
- [ ] User can edit CV-extracted experience details
- [ ] Edits persist and are reflected in agent state

---

## B4: Qualifications Extraction + Persistence

**Objective**: Extract qualifications (certifications, diplomas, artisan qualifications, trade licenses) from CVs and conversation, persist them in the youth profile, and make them available to the recommendation engine for eligibility filtering.

### Task 4.1: Qualification Entity Model (P0)

**Problem**: `YouthProfile.qualifications` is `list[dict[str, Any]]` — untyped and unused. No service or repository layer exists.

**What**: Create a strongly-typed qualification model, MongoDB collection, and repository.

**Files to Create**:
- `backend/app/qualifications/types.py` — `QualificationEntity` model
- `backend/app/qualifications/repository.py` — `QualificationRepository` (MongoDB CRUD)
- `backend/app/qualifications/service.py` — `QualificationService` (business logic)
- `backend/app/qualifications/routes.py` — REST API endpoints

**`QualificationEntity` Model**:
```python
class QualificationType(str, Enum):
    CERTIFICATE = "CERTIFICATE"          # Professional certificates (e.g., CompTIA, CCNA)
    DIPLOMA = "DIPLOMA"                  # Diplomas (e.g., Diploma in Nursing)
    DEGREE = "DEGREE"                    # University degrees (BSc, MSc, PhD)
    TRADE_LICENSE = "TRADE_LICENSE"       # Artisan/trade licenses (e.g., electrician license)
    PROFESSIONAL_LICENSE = "PROFESSIONAL_LICENSE"  # Professional licenses (e.g., CPA, nursing license)
    TRAINING_COMPLETION = "TRAINING_COMPLETION"    # Training program completions
    OTHER = "OTHER"

class QualificationEntity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    qualification_type: QualificationType
    name: str                             # e.g., "Certificate in Project Management"
    institution: Optional[str] = None     # e.g., "Kenya Institute of Management"
    date_obtained: Optional[str] = None   # ISO date or year string
    expiry_date: Optional[str] = None     # For licenses that expire
    level: Optional[str] = None           # e.g., "Level 3", "Grade I"
    field_of_study: Optional[str] = None  # e.g., "Information Technology"
    source: str = "conversation"          # "cv" | "conversation"
```

**MongoDB Collection**: Add `qualifications` to `Collections` class in `database_collections.py`.

**Repository Methods**:
- `save_qualifications(session_id, qualifications: list[QualificationEntity])`
- `get_qualifications(session_id) -> list[QualificationEntity]`
- `update_qualification(session_id, qualification_uuid, updates)`
- `delete_qualification(session_id, qualification_uuid)`

**API Endpoints** (register in `server.py`):
- `GET /conversations/{session_id}/qualifications`
- `POST /conversations/{session_id}/qualifications`
- `PATCH /conversations/{session_id}/qualifications/{uuid}`
- `DELETE /conversations/{session_id}/qualifications/{uuid}`

**Acceptance Criteria**:
- [ ] `QualificationEntity` model with type enum
- [ ] MongoDB repository with CRUD operations
- [ ] REST API endpoints functional
- [ ] `YouthProfile.qualifications` updated to `list[QualificationEntity]` type

---

### Task 4.2: CV Qualifications Extraction (P0)

**Problem**: The CV extraction pipeline only extracts experiences. Qualifications (degrees, certifications, trade licenses) are ignored.

**What**: Add qualifications extraction to the CV parsing pipeline alongside the structured experience extraction (Task 3.1).

**Files to Modify**:
- `backend/app/users/cv/utils/structured_extractor.py` — Extend to extract qualifications
- `backend/app/users/cv/service.py` — Store extracted qualifications
- `backend/app/users/cv/types.py` — Add `CVExtractedQualification` model

**`CVExtractedQualification` Model**:
```python
class CVExtractedQualification(BaseModel):
    name: str
    qualification_type: str  # maps to QualificationType enum
    institution: Optional[str] = None
    date_obtained: Optional[str] = None
    field_of_study: Optional[str] = None
```

**LLM Prompt Extension** (in `CVStructuredExtractor`):
The structured extraction prompt (Task 3.1) already returns both experiences and qualifications in a single LLM call:
```json
{
  "experiences": [...],
  "qualifications": [
    {
      "name": "Diploma in Business Administration",
      "qualification_type": "DIPLOMA",
      "institution": "Kenya Institute of Management",
      "date_obtained": "2019",
      "field_of_study": "Business Administration"
    },
    {
      "name": "Certified Electrician Grade I",
      "qualification_type": "TRADE_LICENSE",
      "institution": "NITA Kenya",
      "date_obtained": "2021"
    }
  ]
}
```

**Kenya-Specific Qualification Handling**:
- Recognize NITA (National Industrial Training Authority) trade test certificates
- Recognize KNEC (Kenya National Examinations Council) certificates
- Map common Kenyan qualification levels (Grade I/II/III for artisan trades)
- Handle Swahili qualification names (e.g., "Cheti cha..." = Certificate of...)

**Acceptance Criteria**:
- [ ] CV extraction returns both experiences and qualifications in structured format
- [ ] Kenyan qualification types correctly recognized (NITA, KNEC, trade tests)
- [ ] Qualifications stored in `user_cv_uploads` record and `qualifications` collection
- [ ] Unit tests: CVs with qualifications → verified extraction output

---

### Task 4.3: Conversational Qualifications Extraction (P1)

**Problem**: Not all qualifications come from CVs. Persona 1 (informal workers) likely have no CV but may have artisan qualifications, trade licenses, or training completions mentioned verbally.

**What**: Add a lightweight qualifications detection pass to the conversation flow that picks up qualifications mentioned during experience collection.

**Files to Create**:
- `backend/app/qualifications/extraction_llm.py` — `QualificationsDetector` LLM tool

**Files to Modify**:
- `backend/app/agent/collect_experiences_agent/_dataextraction_llm.py` — After experience data extraction, run qualifications detection on the same turn
- `backend/app/agent/collect_experiences_agent/collect_experiences_agent.py` — Store detected qualifications in state
- `backend/app/agent/collect_experiences_agent/_types.py` — Add `detected_qualifications: list[QualificationEntity]` to `CollectExperiencesAgentState`

**Detection Strategy**:
- Run as a secondary extraction on each conversation turn (lightweight — ~100 token prompt)
- Trigger words: "certificate", "diploma", "degree", "license", "qualified", "trained", "certified", "NITA", "Grade I/II/III", "cheti" (Swahili)
- Only extract when confidence is high — don't over-extract
- Deduplicate against already-detected qualifications

**Acceptance Criteria**:
- [ ] Qualifications mentioned in conversation are detected
- [ ] Detection works for both English and Swahili trigger terms
- [ ] Artisan/trade qualifications recognized (e.g., "Grade I electrician")
- [ ] No false positives from casual mentions (e.g., "it was a good experience")
- [ ] Detected qualifications persisted at end of conversation

---

### Task 4.4: Qualifications → Job Matching Integration (P2)

**Problem**: Qualifications should affect which jobs are recommended. Some jobs require specific certifications or minimum education levels.

**What**: Pass qualifications to the recommendation engine for eligibility filtering.

**Files to Modify**:
- `backend/app/conversations/service.py` — When transferring data to recommender, include qualifications
- `backend/app/agent/recommender_advisor_agent/agent.py` — Accept qualifications context
- `backend/app/database_contracts/db6_youth_database/db6_client.py` — Update `YouthProfile.qualifications` to `list[QualificationEntity]`

**Integration Points**:
- When `ExploreExperiencesAgentDirector` finishes and hands off to `RecommenderAdvisorAgent`, include qualifications in the context
- Qualifications act as filters: "requires Grade I trade test" → only recommend if user has it
- Qualifications act as boosters: "prefers diploma holders" → rank higher if user has relevant diploma

**Acceptance Criteria**:
- [ ] Qualifications passed to recommendation engine
- [ ] Jobs requiring specific qualifications are filtered correctly
- [ ] Qualification-based ranking boost works
- [ ] `YouthProfile` persists typed qualifications

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
- `backend/app/app_config.py` — Add qualifications config fields, ensure CV config complete
- `backend/app/server_dependencies/database_collections.py` — Add `qualifications` collection
- `backend/app/server.py` — Register qualification routes, ensure CV routes enabled

**New Environment Variables**:
- `BACKEND_QUALIFICATIONS_ENABLED` — Feature flag for qualifications extraction
- `BACKEND_CV_STRUCTURED_EXTRACTION_ENABLED` — Feature flag for structured CV extraction
- `BACKEND_GEMINI_API_KEY` — (verify existing) Gemini API key
- `BACKEND_GEMINI_MODEL_ID` — (verify existing) Model identifier

### Task 6.2: Secrets & Security Review (P1)

**What**: Review secrets management and ensure no PII leaks in new features.

**Deliverables**:
- CV content: verify files stored encrypted in GCS, never logged
- Qualifications: no PII beyond education details, stored in MongoDB
- LLM prompts: verify no CV text forwarded beyond extraction step
- API endpoints: verify authentication required on all new routes
- Update `docs/observability-sensitive-data-checklist.md` with M4 additions

### Task 6.3: Deployment Documentation (P1)

**Files to Create**:
- `docs/deployment-runbook-m4.md` — Step-by-step deployment guide

**Contents**:
- MongoDB collection creation / index setup for `qualifications`
- GCS bucket permissions for CV storage
- Environment variable checklist
- Feature flag rollout order: structured extraction → qualification extraction → CV-agent integration
- Rollback procedures for each feature
- Health check endpoints to verify

**Acceptance Criteria**:
- [ ] All new environment variables documented
- [ ] MongoDB indexes defined for new collections
- [ ] Secrets management reviewed — no PII in logs
- [ ] Feature flags allow incremental rollout
- [ ] Deployment runbook covers rollout + rollback

---

## Success Criteria

**CV Integration (Persona 2)**:
- [ ] CV upload → structured extraction pipeline functional
- [ ] Structured experiences (`CVExtractedExperience`) extracted with title, company, timeline, work_type, responsibilities
- [ ] Conversational flow merges with CV data — agent acknowledges CV content
- [ ] Duplicate detection prevents redundant questions (`compare_relaxed` dedup)
- [ ] User can edit/confirm CV-extracted information via experience cards
- [ ] Persona 2 with CV completes in ≤15 turns (vs 20-25 without CV)
- [ ] Provenance tracking: each experience/qualification marked as "cv" or "conversation" sourced

**Qualifications Extraction**:
- [ ] `QualificationEntity` model with typed enum (`CERTIFICATE`, `DIPLOMA`, `DEGREE`, `TRADE_LICENSE`, etc.)
- [ ] Certifications extracted from CVs via `CVStructuredExtractor`
- [ ] Artisan qualifications recognized (NITA trade tests, Grade I/II/III)
- [ ] Conversational qualifications detection for Persona 1 (no CV)
- [ ] Kenya-specific qualifications handled (NITA, KNEC)
- [ ] Qualifications stored in MongoDB `qualifications` collection
- [ ] Qualifications affect job matching eligibility (filter + ranking boost)

**Persistence & Data Quality**:
- [ ] All experiences saved to database with provenance ("cv" | "conversation")
- [ ] All skills persisted with provenance
- [ ] All qualifications linked to `YouthProfile`
- [ ] Data validation ensures completeness (no null titles, valid enum values)
- [ ] `DB6Client` implementation persists typed qualifications (not `dict[str, Any]`)

**Swahili Testing**:
- [ ] 6 Swahili golden transcripts created (3 per persona, including artisan + code-switch variants)
- [ ] Evaluation scripts measure skill overlap, language drift, mapping hit rate
- [ ] CI regression tests cover both English and Swahili
- [ ] Swahili skill discovery ≥80% parity with English baseline
- [ ] Performance benchmarks documented for Swahili flows

**Deployment Readiness**:
- [ ] Gemini 2.5 Flash config verified in deployment environment
- [ ] MongoDB `qualifications` collection with indexes
- [ ] GCS bucket configured for CV storage
- [ ] Feature flags enable incremental rollout (structured extraction → qualifications → CV-agent merge)
- [ ] Secrets review: no CV content or PII in logs
- [ ] Deployment runbook with rollout + rollback procedures

---

# MILESTONE 5: Hardening + Handover

**Objective**: Finalize robustness, operational readiness, and transition to support.

## B5: Safety/Edge Case Simulation Suite

**Tasks**: TBD

## Hardening Across Persona 2 + Swahili Flows

**Tasks**: TBD

## A5: Handover/Support Plan

**Tasks**: TBD

---

## Success Criteria

**Safety & Edge Cases**:
- [ ] Safety simulation suite integrated into CI
- [ ] Off-topic detection prevents harmful/sensitive conversations
- [ ] Edge case tests cover empty inputs, very long inputs, code-switching, profanity
- [ ] Graceful degradation for model failures

**Robustness & Hardening**:
- [ ] Persona 2 flow hardened with error handling
- [ ] Swahili flow hardened with fallback mechanisms
- [ ] Performance under load tested
- [ ] Memory leaks and resource issues resolved

**Operational Readiness**:
- [ ] Logging reviewed for completeness and compliance
- [ ] Backup and recovery procedures documented

**Documentation**:
- [ ] Handover plan completed (architecture docs, code walkthrough, knowledge transfer, support escalation)
