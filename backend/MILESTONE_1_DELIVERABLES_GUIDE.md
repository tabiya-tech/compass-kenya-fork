# Milestone 1 Deliverables Guide

**Project:** Compass - Kenya Job Market Taxonomy System  
**Milestone:** 1 (Weeks 1-2)  
**GitHub Branch:** `stevealila`

---

## Overview

This document provides a complete reference to all Milestone 1 deliverables, showing exactly where each requirement is implemented in the GitHub repository and how to use/test it.

**Base GitHub URL:** `https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend`

---

## Deliverable 1: Technical Work Plan & Dependency Map

### Location
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/TECHNICAL_WORK_PLAN.md
```
---

## Deliverable 2: Database Schemas

### Location
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/models.py
```

### What's Implemented

#### **DB1: Taxonomy Schema** 
**File: models.py**

Collections defined:
- `OccupationGroupModel` - Hierarchical occupation groups
- `OccupationModel` - **THE CORE MODEL**
  - Fields: code, isco_group_code, preferred_label, alt_labels
  - Contextualization: mapped_to_esco_id, mapping_confidence, mapping_method
  - Provenance: source, added_by, created_at, updated_at
- `SkillModel` - ESCO skills taxonomy
- `OccupationSkillRelationModel` - Links occupations to skills
- `SkillRelationModel` - Hierarchical skill relationships
- `CareerPathModel` - Career progression routes

**Key Features:**
- Complete field definitions with descriptions
- Primary/foreign keys defined (ObjectId references)
- Occupation-skill linking via occupation_skill_relations
- Skill inheritance tracking (inherited_from_esco_id)

#### **DB2: Labor Demand Schema** 
**File: models.py**

Collections defined:
- `LaborDemandModel`
  - Fields: occupation_id, region, demand_score, demand_category
  - Provenance: source, last_updated_at, data_collection_period
- `LaborMarketInsightModel` - Qualitative labor market analysis
  

#### **DB3: Jobs Schema** 
**File: models.py**

Collections defined:
- `JobListingModel`
  - Fields: job_title, description, employer, location, salary_text
  - Status: status (active/expired/filled/removed via JobStatus enum)
  - Taxonomy mapping: mapped_occupation_id, mapped_skills, mapping_confidence
  - Provenance: scraped_at, last_checked_at, source_platform
- `JobScrapingLogModel` - Tracks all scraping runs

---

## Deliverable 3: ESCO/KeSCO Taxonomy Builder Module

### Location
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/
```

### Components

#### **3.1 Configuration**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/config.py
```
- Data file paths
- MongoDB connection strings
- Validation functions

#### **3.2 ESCO Importers** (Importing Data)

**ESCO Occupations:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/esco_occupations_importer.py
```
- Imports 3,062 ESCO occupations
- Extracts ISCO group codes
- Parses alternative labels
- Class: `ESCOOccupationsImporter`

**ESCO Skills:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/esco_skills_importer.py
```
- Imports 13,896 skills
- Class: `ESCOSkillsImporter`

**ESCO Relations:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/esco_relations_importer.py
```
- Imports 130,822 occupation-skill relations
- Links occupations to skills
- Class: `ESCORelationsImporter`

#### **3.3 KeSCO Importer with Contextualization**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/kesco_importer.py
```
- Imports 5,915 KeSCO occupations
- Maps to ESCO using hierarchical semantic matching
- Class: `KeSCOImporter`

#### **3.4 Hierarchical Semantic Matcher** (Mapping)
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/hierarchical_semantic_matcher.py
```
- Three-stage matching algorithm
- ISCO group filtering
- Semantic search within groups
- Class: `HierarchicalSemanticMatcher`

#### **3.5 Skill Inheritance**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/inherit_skills.py
```
- Copies skills from ESCO to matched KeSCO occupations
- Tracks provenance with `inherited_from_esco_id`
- Function: `inherit_skills_for_matched_kesco()`

#### **3.6 Master Orchestrator**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/run_all_imports.py
```
- Runs all importers in correct order
- Handles dependencies automatically
- Complete logging and error handling

### How to Use

#### **Run Complete Import Pipeline:**
```bash
# Run all imports
python3 -m app.taxonomy.importers.run_all_imports
```

## Deliverable 4: Job Scraping Infrastructure

### Location
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/
```

### Components

#### **4.1 Base Scraper Infrastructure**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/base.py
```
- Abstract base class: `BaseScraper`
- Selenium WebDriver initialization
- Safe element extraction: `_safe_find()`
- Salary extraction: `_extract_salary()`
- Text cleaning: `_clean_text()`
- Lazy loading support: `_trigger_lazy_loading()`

#### **4.2 Platform Scrapers** (5 Operational)

**BrighterMonday:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/platforms/brightermonday.py
```
Class: `BrighterMondayScraper`

**Careerjet:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/platforms/careerjet.py
```
Class: `CareerjetScraper`

**Fuzu:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/platforms/fuzu.py
```
Class: `FuzuScraper`

**JobWebKenya:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/platforms/jobwebkenya.py
```
Class: `JobWebKenyaScraper`

**MyJobMag:**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/platforms/myjobmag.py
```
Class: `MyJobMagScraper`

**Each scraper extracts:**
- Job title
- Company/employer
- Location
- Description
- Salary (when available)
- Contract type
- Posting date
- Application URL

#### **4.3 Taxonomy Matcher** (Mapping/Normalization)
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/taxonomy_matcher.py
```
- Fuzzy matches job titles to occupations (70% threshold)
- Extracts skills from descriptions (80% threshold)
- Class: `TaxonomyMatcher`

#### **4.4 Storage Layer** (Normalization to Schema)
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/storage.py
```
- Converts scraped data to `JobListingModel`
- Normalizes field mappings:
  - title → job_title
  - company → employer
  - application_url → url
- Saves to MongoDB job_listings collection
- Logs all scraping runs
- Class: `JobStorage`

#### **4.5 Master Orchestrator**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/run_all_scrapers.py
```
- Runs all 5 scrapers sequentially
- Loads taxonomy for matching
- Saves jobs to database
- Complete logging
- Class: `MasterScraper`

#### **4.6 Configuration**
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/config.py
```
- Platform URLs and selectors
- Selenium configuration
- Wait times and strategies

### How to Use

#### **Run All Scrapers:**
```bash
# Run all platforms
python3 -m app.scrapers.run_all_scrapers

# Or is venv is set up
cd backend
.\.venv\Scripts\Activate.ps1
python -m app.scrapers.run_all_scrapers

# Or to run without MongoDB
cd backend
$env:USE_LOCAL_FIXTURE = '1'
$env:SCRAPER_FIXTURE_DIR = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/Tabiya South Africa v1.0.1-rc.1'
$env:SCRAPER_OUTPUT_DIR = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/Kenya Trial/data'
.\.venv\Scripts\Activate.ps1
python -m app.scrapers.run_all_scrapers
```

#### **Run Single Platform:**
```python
from app.scrapers.platforms.brightermonday import BrighterMondayScraper

scraper = BrighterMondayScraper()
jobs = scraper.scrape(max_jobs=20)
```

#### **Test Taxonomy Matching:**
```python
from app.scrapers.taxonomy_matcher import TaxonomyMatcher

# Load taxonomy
matcher = TaxonomyMatcher(occupations, skills)

# Match a job
result = matcher.match_job({
    'title': 'Software Developer',
    'description': 'Python, JavaScript, SQL...'
})
# Returns: mapped_occupation_id, mapped_skills, confidence scores
```

### Verification

**Check Scraped Jobs:**
```bash
# MongoDB
db.job_listings.count()
db.job_listings.find().limit(5)

# Verify fields are populated
db.job_listings.findOne({}, {
    job_title: 1,
    employer: 1,
    location: 1,
    mapped_occupation_id: 1,
    mapped_skills: 1
})
```

**Check Scraping Logs:**
```bash
db.job_scraping_logs.find().sort({started_at: -1}).limit(5)
```

---

## Deliverable 5: Testing & Quality

### Test Locations

#### **Taxonomy Tests** (28 tests)
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/taxonomy/importers/tests/
```

**Test Files:**
- `test_hierarchical_semantic_matcher.py` (14 tests)
- `test_kesco_importer.py` (14 tests)
- `conftest.py` (shared fixtures)

#### **Scraper Tests** (12 tests)
```
https://github.com/tabiya-tech/compass-kenya-fork/blob/stevealila/backend/app/scrapers/tests/
```

**Test Files:**
- `test_base_scraper.py` (8 tests)
- `test_taxonomy_matcher.py` (4 tests)
- `conftest.py` (shared fixtures)

### How to Run Tests

#### **Run All Tests:**
```bash
# Run everything
pytest

# Run with verbose output
pytest -v
```

#### **Run Specific Module:**
```bash
# Taxonomy tests only
pytest app/taxonomy/importers/tests/ -v

# Scraper tests only
pytest app/scrapers/tests/ -v

# Single test file
pytest app/taxonomy/importers/tests/test_kesco_importer.py -v
```