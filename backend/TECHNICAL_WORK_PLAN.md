# Technical Work Plan & Dependency Map

**Project:** Compass - Kenya Job Market Taxonomy System  
**Contractor:** Steve Alila  
**Contract Period:** December 1, 2025 - January 26, 2026 (9 weeks)  
**My Epics:** Epic 1 (Taxonomy & Databases) and Epic 4 (Skills Elicitation & Swahili)

---

## Overview

This document outlines my technical approach for delivering Epic 1 (all parts) and Epic 4 (all parts) over the 9-week contract period. It breaks my work into specific milestones, shows dependencies between my epics, and identifies what I need from other contractors (if anything).

---

## My Scope of Work

**Epic 1a: Contextualized Taxonomy (Milestone 1-2)**
- Import ESCO occupations, skills, and relations
- Import and match KeSCO occupations to ESCO
- Build job scraping infrastructure
- Create taxonomy curation tools

**Epic 1b: Additional Databases (Milestone 2)**
- Training opportunities database
- Preference dimensions registry
- Youth profile database with CRUD APIs

**Epic 4a: Skills Elicitation Improvements (Milestone 3)**
- Refactor conversation flow
- Implement persona-aware flows
- Add qualifications extraction
- Integrate CV upload

**Epic 4b: Swahili Implementation (Milestone 4)**
- Evaluate Swahili language models
- Implement localization layer
- Create Swahili conversation flows
- Build Swahili evaluation suite

---

## Milestone Breakdown with Timelines

### MILESTONE 1 (Week 1-2): Due December 12, 2025

**Deliverables:**
1. Technical work plan (this document)
2. Database schemas for taxonomy, labor demand, jobs
3. ESCO/KeSCO taxonomy builder operational
4. Job scraping infrastructure for 6 platforms

**Week 1 (Nov 24-30):**
- Days 1-2: Set up environment, run baseline tests
- Days 3-4: Implement ESCO occupations importer
- Days 5-6: Implement ESCO skills importer
- Day 7: Define all database schemas

**Week 2 (Dec 1-12):**
- Days 1-2: Implement ESCO relations importer
- Days 3-4: Build hierarchical semantic matcher
- Days 5-6: Implement KeSCO importer with matching
- Days 7-10: Complete 5-6 job scrapers, write tests
- Days 11-12: Fix integration issues, finalize documentation

**Dependencies on Others:** None

**What I Provide to Others:**
- Database schemas by Dec 5 (for Epic 2-3 planning)
- Taxonomy structure and query patterns
- Youth profile schema (Epic 2-3 will write/read from it)

---

### MILESTONE 2 (Week 3-4): Due December 26, 2025

**Deliverables:**
1. Epic 1a complete (if any remaining items)
2. Training opportunities database operational
3. Preference dimensions registry operational
4. Youth profile database with CRUD APIs
5. Labor demand data schema (no data yet)

**Week 3 (Dec 13-19):**
- Days 1-2: Research Kenyan training providers
- Days 3-4: Design and implement training opportunities DB
- Days 5-6: Research preference dimensions from literature
- Day 7: Design preference dimensions schema

**Week 4 (Dec 20-26):**
- Days 1-2: Implement preference dimensions registry
- Days 3-4: Design youth profile schema
- Days 5-6: Implement youth profile CRUD APIs
- Day 7: Test all databases, export to CSV, complete documentation

**Dependencies on Others:**
- Need Epic 2 contractor input on preference dimensions (if available)
- Need Epic 2-3 contractor input on youth profile requirements (if available)

**What I Provide to Others:**
- Preference dimensions schema for Epic 2 agent
- Youth profile CRUD APIs for Epic 2 & 3
- Documentation on how to use youth profile

---

### MILESTONE 3 (Week 5-6): Due January 9, 2026

**Deliverables:**
1. Refactored skills elicitation conversation flow
2. Persona-aware conversation logic
3. Qualifications extraction module
4. CV integration with skills extraction

**Week 5 (Dec 27 - Jan 2):**
- Days 1-2: Analyze existing skills elicitation, document problems
- Days 3-4: Design simplified conversation architecture
- Days 5-6: Implement persona detection
- Day 7: Refactor conversation prompts

**Week 6 (Jan 3-9):**
- Days 1-2: Implement qualifications extraction
- Days 3-4: Integrate CV upload with skills extraction
- Days 5-6: Implement safety evaluations
- Day 7: Create test harness, run evaluations, complete documentation

**Dependencies on Others:** None

**What I Provide to Others:**
- Conversation architecture patterns (Epic 2 may reuse)
- Skills data flowing into youth profile for Epic 3 recommender

---

### MILESTONE 4 (Week 7-8): Due January 23, 2026

**Deliverables:**
1. Swahili model evaluation document
2. Localization layer with synonym mapping
3. Swahili-enabled conversation flows
4. Swahili evaluation suite

**Week 7 (Jan 10-16):**
- Days 1-2: Evaluate Swahili models (Gemini, Jacaranda)
- Days 3-4: Write evaluation document with recommendation
- Days 5-6: Implement localization layer
- Day 7: Map Swahili job terms to taxonomy

**Week 8 (Jan 17-23):**
- Days 1-2: Implement Swahili skills elicitation flows
- Days 3-4: Create Swahili test scripts
- Days 5-6: Run Swahili evaluation suite
- Day 7: Complete documentation

**Dependencies on Others:**
- Share Swahili model findings with Epic 2-3 contractor (if coordinating)

**What I Provide to Others:**
- Swahili model selection rationale (Epic 2-3 may use same)
- Swahili synonym mappings (reusable)

---

### MILESTONE 5 (Week 9): Due January 30, 2026

**Deliverables:**
1. All automated tests passing
2. Complete documentation and runbooks
3. Integration testing across all epics
4. Handover session

**Week 9 (Jan 24-30):**
- Days 1-2: Integration environment setup, test Epic 1 → Epic 4 flow
- Days 3-4: Fix integration issues, test database integrations
- Day 5: Run full test suite, ensure ./run-before-merge.sh passes
- Day 6: Documentation review, security review, create runbooks
- Day 7: Handover session and contract completion

**Dependencies on Others:**
- Epic 2-3 contractor for end-to-end integration testing (if available)
- Joint testing: Skills elicitation → Youth profile → Recommendations (if coordinating)

**What I Provide to Others:**
- Working taxonomy and database infrastructure
- Documented APIs and query patterns
- Runbooks for database maintenance

---

## Complete Dependency Map: Epic 1-4

This section shows how Epic 1 (Steve - Taxonomy), Epic 2 (Victor - Preferences), Epic 3 (Victor - Recommender), and Epic 4 (Steve - Skills) depend on each other.

### Epic 1a (Taxonomy) → Epic 2 (Preference Elicitation)

```
Epic 1a: ESCO/KeSCO Taxonomy + Preference Dimensions Registry
        |
        v
Epic 2: Preference Elicitation Agent
        |
        +-- Needs: Preference dimensions schema to know what to elicit
        +-- Needs: Occupation IDs to contextualize preferences
        +-- Needs: Youth profile schema to save preference vectors
```

**What Epic 1 Provides:**
- Preference dimensions registry (DB5) - defines what preferences exist
- Occupation taxonomy - allows preferences to be occupation-specific
- Youth profile database (DB6) - storage for preference vectors

**What Epic 2 Uses:**
- Queries preference dimensions to build vignettes
- References occupations when discussing job preferences
- Writes preference vectors to youth profile

**Timeline:** Epic 1b (DB5, DB6) must complete by Dec 26 before Epic 2 can fully implement

---

### Epic 1a (Taxonomy) → Epic 3 (Recommender)

```
Epic 1a: Taxonomy + Jobs + Training
        |
        v
Epic 3: Recommender Engine
        |
        +-- Needs: Occupations DB to recommend careers
        +-- Needs: Skills DB to match youth skills
        +-- Needs: Jobs DB to recommend actual positions
        +-- Needs: Training DB to recommend upskilling
        +-- Needs: Youth profile to read skills/preferences
```

**What Epic 1 Provides:**
- Occupations taxonomy - the universe of careers to recommend
- Skills taxonomy - enables skills-based matching
- Job listings database - actual jobs to recommend
- Training opportunities database - upskilling options
- Youth profile database - input data for recommendations

**What Epic 3 Uses:**
- Queries occupations to find matches
- Queries jobs filtered by occupation/location
- Queries training programs for skill gaps
- Reads youth profile (skills vector + preference vector)
- Ranks recommendations using taxonomy relationships

**Timeline:** Epic 1a must complete by Dec 12, Epic 1b by Dec 26 for Epic 3 to implement

---

### Epic 1b (Youth Profile) → Epic 2 & Epic 3 Integration

```
Epic 2: Preference Elicitation          Epic 4: Skills Elicitation
        |                                       |
        +------- Both write to ----------------+
                        |
                        v
                Epic 1b: Youth Profile Database
                        |
                        v
                Epic 3: Recommender (reads from youth profile)
```

**Data Flow:**
1. Epic 4 (Steve) extracts skills → writes skills_vector to youth profile
2. Epic 2 (Victor) elicits preferences → writes preference_vector to youth profile
3. Epic 3 (Victor) reads youth profile → generates recommendations

**Critical Dependency:** Youth profile schema (Epic 1b) must accommodate both Epic 2 and Epic 4 writes, and Epic 3 reads.

---

### Epic 4a (Skills Elicitation) → Epic 2 (Preference Elicitation)

```
Epic 4a: Skills Elicitation Architecture
        |
        v
Conversation patterns, state management, error handling
        |
        v
Epic 2: Preference Elicitation (may reuse patterns)
```

**What Epic 4 Provides:**
- Conversation flow architecture
- State management approach
- Safety evaluation framework
- Persona detection logic

**What Epic 2 May Use:**
- Similar conversation patterns (optional)
- State management if compatible
- Error handling approach

**Timeline:** Epic 4a completes Jan 9, Epic 2 may review patterns if beneficial

---

### Epic 1a (Taxonomy) → Epic 4a (Skills Elicitation)

```
Epic 1a: Skills Taxonomy
        |
        v
Epic 4a: Skills Elicitation
        |
        +-- Needs: Skills taxonomy to map extracted skills
        +-- Needs: Occupations taxonomy to suggest careers
```

**What Epic 1 Provides:**
- Skills taxonomy - every skill mentioned gets mapped to taxonomy
- Occupations taxonomy - can suggest careers based on skills

**What Epic 4 Uses:**
- Maps free-text skills to standardized skill IDs
- Validates extracted skills against taxonomy
- Suggests occupations that match skill set

**Timeline:** Epic 1a must complete by Dec 12 before Epic 4a starts Dec 27

---

### Complete Data Flow: All Epics

```
User Interaction
        |
        v
Epic 4a: Skills Elicitation (Steve)
        |
        +-- Uses Epic 1a taxonomy to map skills
        |
        v
Epic 1b: Youth Profile - skills_vector saved
        |
        v
Epic 2: Preference Elicitation (Victor)
        |
        +-- Uses Epic 1b preference dimensions
        |
        v
Epic 1b: Youth Profile - preference_vector saved
        |
        v
Epic 3: Recommender Engine (Victor)
        |
        +-- Reads youth profile (skills + preferences)
        +-- Queries Epic 1a taxonomy, jobs, training
        +-- Ranks using skills match + preference alignment + labor demand
        |
        v
Recommendations to User
```

---

## Dependencies Between My Epics

### Epic 1a → Epic 1b
```
ESCO/KeSCO Taxonomy (1a)
        |
        v
Youth Profile needs occupation/skill IDs to reference (1b)
        |
        v
Preference Dimensions need occupation context (1b)
```

**Dependency:** Epic 1b databases depend on Epic 1a taxonomy being complete. Youth profiles reference occupation IDs, preference dimensions are contextualized to occupations.

**Timeline:** Epic 1a must complete by Dec 7 before Epic 1b starts Dec 8.

---

### Epic 1 → Epic 4a
```
Taxonomy with Skills (Epic 1a)
        |
        v
Skills Elicitation extracts skills and maps to taxonomy (4a)
        |
        v
Extracted skills saved to Youth Profile (Epic 1b)
```

**Dependency:** Epic 4a skills elicitation depends on Epic 1a taxonomy for skills mapping and Epic 1b youth profile for data storage.

**Timeline:** Epic 1a complete Dec 7, Epic 1b complete Dec 21, Epic 4a starts Dec 22.

---

### Epic 4a → Epic 4b
```
Skills Elicitation in English (4a)
        |
        v
Swahili Implementation adapts same flows (4b)
```

**Dependency:** Epic 4b Swahili builds on Epic 4a architecture. Same conversation patterns, same taxonomy mapping, just different language.

**Timeline:** Epic 4a complete Jan 4, Epic 4b starts Jan 5.

---

## Dependencies on Other Contractors

### What I Need from Epic 2-3 Contractor

**By Dec 11:**
- Input on preference dimensions schema
- List of preference dimensions Epic 2 needs to elicit
- Any specific field requirements

**By Dec 15:**
- Input on youth profile schema
- List of fields Epic 2 needs to write (preference vector, etc.)
- List of fields Epic 3 needs to read (for recommendations)

**By Dec 27:**
- Feedback on conversation architecture (optional)
- Whether Epic 2 wants to reuse any patterns

**By Jan 7:**
- Feedback on Swahili model selection (optional)
- Whether Epic 2-3 will use same Swahili model

**During Week 9 (Jan 19-26):**
- Availability for integration testing
- Test: Skills elicitation → Youth profile → Recommendations
- Fix any integration issues jointly

---

## What I Provide to Other Contractors

### To Epic 2 Contractor (Preference Elicitation)

**Dec 5:** Database schemas for review

**Dec 11:** Preference dimensions schema and API
- How to add new dimensions
- How to query dimensions

**Dec 18:** Youth profile CRUD APIs
- `create_youth_profile()`
- `update_youth_profile()`
- `save_preference_vector()`
- Documentation with examples

**Dec 27:** Conversation architecture documentation (optional)
- State management patterns
- Error handling approach
- Safety evaluation framework

---

### To Epic 3 Contractor (Recommender)

**Dec 5:** Database schemas for review

**Dec 18:** Youth profile query APIs
- `get_youth_profile(youth_id)`
- `query_youth_by_skills()`
- `query_youth_by_preferences()`

**Dec 21:** Taxonomy query patterns
- How to query occupations by skills
- How to query skills by occupation
- How to query training by occupation
- Database indexes for performance

**Jan 12:** Database optimization support
- Review Epic 3 recommender queries
- Add indexes if needed
- Optimize slow queries

---

## Clear Ownership Assignments

### Steve Alila Owns

| Epic | Component | Specific Tasks | Deliverable Milestone |
|------|-----------|----------------|----------------------|
| Epic 1a | ESCO Taxonomy Import | Import ESCO occupations, skills, relations | Milestone 1 (Dec 12) |
| Epic 1a | KeSCO Integration | Import KeSCO, match to ESCO, inherit skills | Milestone 1 (Dec 12) |
| Epic 1a | Job Scraping | Build 6 platform scrapers, normalize to taxonomy | Milestone 1 (Dec 12) |
| Epic 1a | Taxonomy Curation | Tools to flag irrelevant occupations, add custom ones | Milestone 1 (Dec 12) |
| Epic 1b | Training Opportunities DB | Research and import Kenyan training programs | Milestone 2 (Dec 26) |
| Epic 1b | Preference Dimensions Registry | Define and populate preference dimensions | Milestone 2 (Dec 26) |
| Epic 1b | Youth Profile Database | Design schema, implement CRUD APIs | Milestone 2 (Dec 26) |
| Epic 1b | Labor Demand Schema | Define schema (data population TBD) | Milestone 2 (Dec 26) |
| Epic 4a | Skills Elicitation Refactor | Simplify conversation flow, reduce repetition | Milestone 3 (Jan 9) |
| Epic 4a | Persona Detection | Detect informal vs formal worker personas | Milestone 3 (Jan 9) |
| Epic 4a | Qualifications Extraction | Extract degrees, diplomas, certificates | Milestone 3 (Jan 9) |
| Epic 4a | CV Integration | Parse uploaded CVs, extract skills/experience | Milestone 3 (Jan 9) |
| Epic 4a | Safety Evaluations | Prevent off-topic, harmful conversations | Milestone 3 (Jan 9) |
| Epic 4b | Swahili Model Evaluation | Compare Gemini vs Jacaranda, recommend best | Milestone 4 (Jan 23) |
| Epic 4b | Localization Layer | Map Swahili job terms to taxonomy | Milestone 4 (Jan 23) |
| Epic 4b | Swahili Skills Elicitation | Implement Swahili conversation flows | Milestone 4 (Jan 23) |
| Epic 4b | Swahili Evaluation Suite | Create golden transcripts, test quality | Milestone 4 (Jan 23) |

### Victor Gitahi Owns (Epic 2-3 Contractor)

| Epic | Component | Specific Tasks | Notes |
|------|-----------|----------------|-------|
| Epic 2 | Preference Elicitation Agent | Design and implement conversation flow | Uses Steve's DB5 |
| Epic 2 | Vignette Design | Create preference elicitation vignettes | Uses Steve's preference dimensions |
| Epic 2 | Preference Vector Construction | Convert conversation to preference vector | Saves to Steve's youth profile DB |
| Epic 2 | Experience-based Questions | Elicit past job experiences and preferences | Integrates with Steve's taxonomy |
| Epic 2 | Swahili Preference Flows | Adapt preference elicitation for Swahili | May use Steve's Swahili model |
| Epic 3 | Recommendation Engine | Build ranking and scoring algorithms | Queries Steve's taxonomy, jobs, training |
| Epic 3 | Advisor Conversation Layer | Generate explanations and discuss trade-offs | Uses Steve's youth profile data |
| Epic 3 | Graph-based Matching | Implement career path recommendations | Uses Steve's taxonomy relationships |
| Epic 3 | RAG Implementation | Retrieve and generate explanations | References Steve's taxonomy |
| Epic 3 | Swahili Recommendations | Generate recommendations in Swahili | May use Steve's Swahili model |

### Shared Integration Work (Milestone 5)

| Task | Steve's Role | Victor's Role | Timeline |
|------|--------------|---------------|----------|
| End-to-end testing | Test Epic 1 → Epic 4 flow | Test Epic 2 → Epic 3 flow | Jan 24-30 |
| Youth profile integration | Verify CRUD APIs work correctly | Verify read/write operations | Jan 24-30 |
| Complete user journey test | Provide taxonomy and databases | Provide agents and recommender | Jan 24-30 |
| Documentation | Document Epic 1 & 4 | Document Epic 2 & 3 | Jan 24-30 |
| Handover session | Present infrastructure and databases | Present agents and recommendations | Jan 30 |

---

## What Each Epic Needs From Others

### Epic 1 (Steve) Needs From Epic 2-3 (Victor)

**For Milestone 2 (Dec 26):**
- Preference dimensions requirements (what dimensions Epic 2 needs)
- Youth profile field requirements (what Epic 2 writes, what Epic 3 reads)

**For Milestone 5 (Jan 30):**
- Availability for integration testing
- Bug reports if youth profile APIs don't work as expected

**Can Complete Without:** Epic 1 can complete Milestones 1-4 independently if needed

---

### Epic 2 (Victor) Needs From Epic 1 (Steve)

**For Implementation:**
- Preference dimensions schema (from DB5)
- Youth profile write API (save_preference_vector)
- Occupation taxonomy (to contextualize preferences)

**Timeline Dependency:**
- Epic 1b must complete by Dec 26 for Epic 2 to proceed

---

### Epic 3 (Victor) Needs From Epic 1 (Steve)

**For Implementation:**
- Youth profile read API (get_youth_profile)
- Taxonomy query APIs (get occupations, get skills, get jobs, get training)
- Database performance (queries must be fast)

**Timeline Dependency:**
- Epic 1a must complete by Dec 12 for Epic 3 foundation
- Epic 1b must complete by Dec 26 for Epic 3 to query youth profiles

---

### Epic 4 (Steve) Needs From Epic 2-3 (Victor)

**For Implementation:**
- None - Epic 4 is independent

**For Integration (Optional):**
- Conversation pattern feedback (if Victor wants to review)
- Swahili model decision (if Victor wants same model)

**Timeline:** Epic 4 can complete independently

---

## Communication & Reporting

**Weekly Status Updates:**
- Every Friday by 5 PM EAT
- Format: Completed tasks, next week plan, blockers
- Sent to: Nyambura Kariuki, Jasmin

**Milestone Completion:**
- Submit deliverables guide showing what was completed
- Request review meeting
- Address feedback within 48 hours

**Dependency Requests:**
- Send to Epic 2-3 contractor with 3 days notice minimum
- Include: what I need, why, by when
- Follow up if no response in 24 hours

**Escalation:**
- If blocker from Epic 2-3 contractor: escalate to Nyambura within 24 hours
- If timeline risk: notify immediately with mitigation plan
- If scope clarification needed: ask Nyambura/Jasmin within 24 hours

---

## Post-Implementation Support

**Complimentary Period:** 14 days (Jan 31 - Feb 13, 2026)

**Scope:**
- Bug fixes for my delivered functionality
- Critical issues preventing system use
- Documentation clarifications
- Quick configuration adjustments

**Response Times:**
- Critical bugs: 4 hours
- High-priority: 24 hours
- Medium-priority: 48 hours
- Low-priority: 72 hours

**Extended Support:**
- Available after complimentary period
- Terms to be agreed separately

---

## Timeline Summary

```
Week 1-2 (Nov 24 - Dec 12): Milestone 1 - Epic 1a complete
Week 3-4 (Dec 13 - Dec 26): Milestone 2 - Epic 1b complete
Week 5-6 (Dec 27 - Jan 9): Milestone 3 - Epic 4a complete
Week 7-8 (Jan 10 - Jan 23): Milestone 4 - Epic 4b complete
Week 9 (Jan 24 - Jan 30): Milestone 5 - Integration & handover
```

**Milestone 1 has zero dependencies on other contractors.**

**Milestone 2 may benefit from schema input but can proceed independently.**

**Milestone 3-4 are fully independent.**

**Milestone 5 requires integration testing coordination (if other contractors available).**

---

**Prepared by:** Steve Alila  
**Version:** 2.0