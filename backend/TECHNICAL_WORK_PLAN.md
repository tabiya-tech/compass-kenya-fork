# Technical Work Plan: Milestone 1 - Component Dependencies

**Project:** Compass - Kenya Job Market Taxonomy System  
**Milestone:** 1 (Weeks 1-4)   
**Purpose:** Planning document with dependency mapping and implementation strategy

---

## Overview

This document outlines the technical approach for building a contextualized Kenya-focused job taxonomy by mapping KeSCO occupations to ESCO taxonomy using hierarchical semantic matching.

---

## Planned System Architecture

```
                    Application Layer (Future)
                    (FastAPI + React)
                           │
                ┌──────────┼──────────┐
                │          │          │
                ▼          ▼          ▼
         Taxonomy DB  Labor Demand  Jobs DB
         
    Target: 9K+ occupations    Schema ready    5 scrapers
            14K+ skills        (Milestone 2)   operational
            130K+ relations                     
```

---

## Planned Database Schema

### **Collections to Implement:**

```
occupations
    ├─ isco_group_code ────────┐  # CRITICAL: Enables hierarchical matching
    ├─ mapped_to_esco_id ──────┼─→ Will link KeSCO to ESCO
    ├─ mapping_confidence ─────┤   Will store match quality
    ├─ mapping_method ─────────┤   Will track how match was made
    │                          │
skills                         │
    ├─ preferred_label         │
    ├─ skill_type              │
    │                          │
occupation_skill_relations     │
    ├─ occupation_id ──────────┘
    ├─ skill_id ───────────────→ FK to skills
    ├─ relation_type ──────────→ Essential/optional
    └─ inherited_from_esco_id ─→ PROVENANCE: tracks skill source
         │
job_listings (from scrapers)
    ├─ job_title
    ├─ mapped_occupation_id ───→ FK to occupations
    └─ mapped_skills ──────────→ FK array to skills
```

**Key Design Decision:** Store `isco_group_code` separately to enable fast hierarchical filtering before semantic search.

---

## Implementation Pipeline (Dependency Chain)

```
START (Week 1)
  │
  ├─→ [STEP 1: ESCO Occupations Import]
  │   PRIORITY: HIGH (Blocks KeSCO import)
  │   │
  │   Tasks:
  │   ├─ Parse ESCO CSV (3,000+ occupations expected)
  │   ├─ Extract ISCO codes from OCCUPATIONGROUPCODE field
  │   │   Example: "ISCO08-2411" → store "2411" in isco_group_code
  │   ├─ Parse alternative labels (newline-separated)
  │   └─ Store in occupations collection
  │   │
  │   Deliverable: ~3,000 ESCO occupations with ISCO codes
  │
  ├─→ [STEP 2: ESCO Skills Import]
  │   CAN RUN IN PARALLEL WITH STEP 1
  │   │
  │   Tasks:
  │   ├─ Parse ESCO skills CSV (13,000+ skills expected)
  │   ├─ Store skill types and reuse levels
  │   └─ Store in skills collection
  │   │
  │   Deliverable: ~14,000 ESCO skills
  │
  ├─→ [STEP 3: ESCO Occupation-Skill Relations]
  │   DEPENDS ON: Steps 1 & 2 must complete first
  │   │
  │   Tasks:
  │   ├─ Parse relations CSV 
  │   ├─ Map essential vs optional skills
  │   └─ Store in occupation_skill_relations collection
  │   │
  │   Deliverable: ~130,000 occupation-skill relations
  │
  └─→ [STEP 4: Build Lookup Structures]
      DEPENDS ON: Step 1 must complete first
      │
      Tasks:
      ├─ Build flat lookup: {lowercase_title: occupation_doc}
      │   Purpose: Fast exact matching
      │   Expected size: ~30,000 entries (including alt labels)
      │
      └─ Build group lookup: {isco_code: [occupation_docs]}
          Purpose: Hierarchical filtering for semantic search
          Expected size: ~400 ISCO groups
          
      Deliverable: In-memory lookups for matching
           │
           ▼
  ┌─→ [STEP 5: Hierarchical Semantic Matcher] (Week 2)
  │   DEPENDS ON: Step 4 must complete first
  │   │
  │   Strategy: Use ISCO codes as hierarchical filter
  │   │
  │   Tasks:
  │   ├─ Load sentence transformer model (all-MiniLM-L6-v2)
  │   ├─ Build group-based semantic indices
  │   │   For each ISCO group:
  │   │   └─ Generate embeddings for occupations in group
  │   │
  │   └─ Build full catalog index (fallback)
  │       Generate embeddings for all ~3,000 ESCO occupations
  │   │
  │   Expected Output: Matcher with 3-stage algorithm
  │        │
  │        ▼
  └─→ [STEP 6: KeSCO Import with Contextualization]
      DEPENDS ON: Steps 1 & 5 must complete first
      │
      Strategy: Match each KeSCO occupation to ESCO
      │
      For each KeSCO occupation:
      │
      ├─ Extract ISCO code from KeSCO code
      │   Example: "2411-12" → "2411"
      │   Store in: isco_group_code field
      │
      ├─ STAGE 0: Exact match (fastest)
      │   Check if title exists in flat lookup
      │   If found → 100% confidence, method="exact"
      │
      ├─ STAGE 1: Semantic search within ISCO group
      │   Filter ESCO to same ISCO code
      │   Semantic match within group only
      │   Threshold: ≥65% for group-based
      │   If found → confidence score, method="hierarchical_group"
      │
      └─ STAGE 2: Semantic search full catalog (fallback)
          Search across all ESCO occupations
          Threshold: ≥55% 
          If found → confidence score, method="hierarchical_fallback"
          │
          Decision Logic:
          ├─→ [≥55% confidence] → Auto-match
          │   • Set mapped_to_esco_id
          │   • Set is_localized = True
          │   • Copy alt_labels from ESCO
          │   Target: >80% of KeSCO occupations
          │
          ├─→ [45-54% confidence] → Manual review queue
          │   • Set suggested_esco_id
          │   • Set requires_manual_review = True
          │   Target: <10% of KeSCO occupations
          │
          └─→ [<45% confidence] → No match
              • Set requires_manual_skill_assignment = True
              Target: <10% of KeSCO occupations
              │
              ▼
      Export to CSV for review:
           │
           ▼
  ┌─→ [STEP 7: Skill Inheritance] (Week 3)
  │   DEPENDS ON: Steps 3 & 6 must complete first
  │   │
  │   Strategy: Leverage ESCO→KeSCO mapping to inherit skills
  │   │
  │   For each auto-matched KeSCO occupation:
  │   │
  │   ├─ Find ESCO occupation via mapped_to_esco_id
  │   ├─ Find all skills for that ESCO occupation
  │   ├─ Copy skill relations to KeSCO occupation
  │   │   Fields to populate:
  │   │   ├─ occupation_id: KeSCO occupation
  │   │   ├─ skill_id: from ESCO relation
  │   │   ├─ relation_type: from ESCO (essential/optional)
  │   │   ├─ inherited_from_esco_id: PROVENANCE tracking
  │   │   └─ source: "inherited_from_esco"
  │   │
  │   └─ Update KeSCO occupation:
  │       ├─ has_inherited_skills = True
  │       └─ inherited_skills_count = N
  │   │
  │   Expected Result: 
  │   • ~4,000+ KeSCO occupations with skills (80% of 5,000)
  │   • ~50,000+ new skill relations
  │   • Complete provenance tracking
  │        │
  │        ▼
  └─→ [STEP 8: Job Scraping Infrastructure] (Week 1-2)
      CAN RUN IN PARALLEL with Steps 5-7
      │
      Platforms to implement (6 total):
      ├─ BrighterMonday
      ├─ Careerjet
      ├─ Fuzu
      ├─ MyJobMag
      ├─ JobWebKenya
      └─ JobsinKenya
      │
      Technology: Selenium WebDriver
      Reason: Kenyan job sites use JavaScript rendering
      │
      For each platform:
      ├─ Implement parse_job_card() method
      ├─ Extract: title, company, location, salary, description
      ├─ Handle lazy loading (scroll-based)
      └─ Test with 20 jobs per platform
      │
      Expected Output: 5-6 operational scrapers
           │
           ▼
      [Job-Occupation Matching]
      DEPENDS ON: Steps 5 & 8 complete
      │
      Strategy: Reuse hierarchical semantic matcher
      │
      For each scraped job:
      ├─ Match job title to occupations (70% threshold)
      ├─ Extract skills from description (80% threshold)
      └─ Store in job_listings with:
          ├─ mapped_occupation_id
          ├─ mapped_skills array
          └─ mapping_confidence
           │
           ▼
      [MILESTONE 1 COMPLETE]
      Ready for API Layer (Future Milestones)
```

---

## Key Technical Decisions

### **1. Aggressive 55% Threshold**

**Rationale:** 
- Manual review of 5,000 occupations is impractical
- Better to auto-match with lower confidence than create bottleneck
- Quality can be verified and improved iteratively

**Trade-off:** Accept some lower-quality matches to maximize automation

**Mitigation:** 
- Export all matches to CSV for spot-checking
- Track match method and confidence for future refinement
- Support manual override via curation tools

---

### **2. Hierarchical Filtering First**

**Rationale:**
- ISCO codes are standardized (used by both ESCO and KeSCO)
- Occupations in same ISCO group are semantically related
- Reduces search space by ~99% (3,000 → ~7 per group)

**Expected Impact:**
- Faster matching (seconds vs minutes)
- Higher precision (fewer false positives)
- Better recall (catches variations within occupational family)

---

### **3. Skill Inheritance via Mapping**

**Rationale:**
- Manually assigning skills to 5,000 KeSCO occupations is infeasible
- ESCO already has curated occupation-skill relations
- Mapping quality determines skill quality

**Implementation:**
- For each auto-matched KeSCO, copy all ESCO skills
- Track provenance with `inherited_from_esco_id`
- Flag unmatched occupations for manual skill assignment

**Expected Result:** 
- 80% of KeSCO occupations get skills automatically
- 20% require manual work (manageable)

---

### **4. Alternative Labels Integration**

**Rationale:**
- ESCO has many alternative titles per occupation
- These synonyms improve search and matching
- KeSCO titles may use different terminology

**Implementation:**
- When KeSCO matches ESCO, copy ESCO's alt_labels
- Exclude KeSCO title itself (avoid duplication)
- Limit to 10 alt labels per occupation

**Expected Benefit:** Richer search capabilities in final application

---

### **5. CSV Export for Manual Review**

**Rationale:**
- Quality assurance requires human oversight
- Stakeholders need visibility into matching decisions
- Edge cases should be caught early

**Implementation:**
- Export all matches with columns:
  - kesco_code, kesco_title, kesco_isco_group
  - esco_code, esco_title, esco_isco_group
  - confidence, method
- Sort by confidence 

**Usage:** Team can review borderline matches before finalizing

---

## Risk Assessment & Mitigation

### **Risk 1: Low Auto-Match Rate (<80%)**

**Likelihood:** Medium  
**Impact:** High (creates manual work bottleneck)

**Mitigation:**
- Use aggressive 55% threshold instead of conservative 70%
- Implement fallback to full catalog if group match fails
- Export borderline cases for team review

**Contingency:** If auto-match rate is <70%, adjust threshold down to 50% after quality review

---

### **Risk 2: Semantic Model Performance**

**Likelihood:** Low  
**Impact:** Medium (slow matching)

**Mitigation:**
- Use lightweight model (all-MiniLM-L6-v2)
- Pre-compute all ESCO embeddings once
- Cache embeddings to avoid recomputation
- Use batch processing for KeSCO matching

**Contingency:** If too slow, switch to paraphrase-MiniLM-L3-v2 (even lighter)

---

### **Risk 3: ESCO/KeSCO Structural Differences**

**Likelihood:** Medium  
**Impact:** Medium (some occupations won't match)

**Mitigation:**
- Support custom Kenyan occupations (informal sector, entrepreneurship)
- Allow manual skill assignment for unmatched occupations
- Track all unmatched cases for future refinement

**Contingency:** Create "Kenyan supplement" taxonomy for gap areas

---

### **Risk 4: Job Scraper Instability**

**Likelihood:** High  
**Impact:** Low (not blocking for Milestone 1)

**Mitigation:**
- Implement robust error handling
- Use Selenium for JavaScript-rendered sites
- Add retry logic with exponential backoff
- Log all failures for debugging

**Contingency:** If scraper breaks, manual testing is acceptable for Milestone 1

---

## Success Criteria (Milestone 1)

**Database Infrastructure:**
- ✅ Schemas defined for 3 databases (taxonomy, labor demand, jobs)
- ✅ ~3,000 ESCO occupations imported with ISCO codes
- ✅ ~14,000 ESCO skills imported
- ✅ ~130,000 occupation-skill relations imported

**Contextualization:**
- ✅ ~5,000 KeSCO occupations imported
- ✅ **>80% auto-match rate achieved**
- ✅ <5% import errors
- ✅ Complete provenance tracking (source, added_by, timestamps)

**Job Scraping:**
- ✅ 4-6 platform scrapers operational
- ✅ Can scrape 20 jobs per platform
- ✅ Jobs mapped to taxonomy occupations

**Quality:**
- ✅ BDD tests covering core functionality
- ✅ No high-severity linting errors
- ✅ Documentation complete (READMEs)

---

## Team Ownership (Milestone 1)

**Steve Alila:**
- Epic 1a: Taxonomy & database foundation (Week 1-2)
- Epic 1b: Training opportunities DB (Week 3-4)
- Job scraping infrastructure setup
- Epic 4a: Skills elicitation improvements
- Epic 4b: Swahili implementation

**Victor Gitahi (https://github.com/wilfredeveloper):**
- Epic 2: Preference elicitation agent
- Epic 3: Recommender/advisor agent

---

## Dependencies on Other Contractors

**None for Milestone 1** - This work is self-contained and can proceed independently.

**Future Dependencies:**
- Epic 2-4 may require coordination with other contractors
- API integration will depend on backend team's progress
- Frontend integration will depend on React team's components

---