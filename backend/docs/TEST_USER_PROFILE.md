# Test User Profile: Hassan (Mombasa Persona)

This document describes the sample user profile used for testing the **Recommender/Advisor Agent** in the interactive test script (`test_recommender_agent_interactive.py`).

---

## 📦 JSON Schema

> ⚠️ **IMPORTANT**: The test script uses a **simplified stub** for `skills_vector`. The actual Skills Explorer Agent produces a richer `SkillEntity` structure. See comparison below.

---

### Test Script Stub (Simplified)

This is what `test_recommender_agent_interactive.py` currently uses:

```json
{
  "youth_id": "test_user_123",
  "country_of_user": "KENYA",
  
  "skills_vector": {
    "top_skills": [
      {"preferredLabel": "Basic Electrical Wiring", "proficiency": 0.6},
      {"preferredLabel": "Manual Handling / Physical Labor", "proficiency": 0.8},
      {"preferredLabel": "M-Pesa / Mobile Money", "proficiency": 0.85}
    ]
  },
  
  "preference_vector": {
    "financial_importance": 0.85,
    "work_environment_importance": 0.55,
    "career_advancement_importance": 0.60,
    "work_life_balance_importance": 0.80,
    "job_security_importance": 0.70,
    "task_preference_importance": 0.65,
    "social_impact_importance": 0.40,
    
    "confidence_score": 0.0,
    "n_vignettes_completed": 0,
    "per_dimension_uncertainty": {},
    
    "posterior_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "posterior_covariance_diagonal": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "fim_determinant": null,
    
    "decision_patterns": {},
    "tradeoff_willingness": {},
    "values_signals": {},
    "consistency_indicators": {},
    "extracted_constraints": {}
  }
}
```

---

### Actual SkillEntity Schema (from Skills Explorer Agent)

The actual `SkillEntity` produced by the Skills Explorer Agent (`app/vector_search/esco_entities.py`) has this structure:

```json
{
  "id": "string",
  "modelId": "string",
  "UUID": "string",
  "preferredLabel": "string",
  "altLabels": ["string"],
  "description": "string",
  "scopeNote": "string (optional)",
  "originUUID": "string (optional)",
  "UUIDHistory": ["string"],
  "score": 0.0,
  "skillType": "skill/competence | knowledge | language | attitude | ''"
}
```

### Actual ExperienceEntity.top_skills (Full Production Shape)

From `app/agent/experience/experience_entity.py`, the `ExperienceEntity` contains:

```json
{
  "uuid": "unique-experience-id",
  "experience_title": "Crew Member",
  "company": "McDonald's",
  "location": "Cape Town, South Africa",
  "timeline": {
    "start_date": "2023-01",
    "end_date": "2024-06"
  },
  "work_type": "WAGED_EMPLOYEE",
  
  "top_skills": [
    {
      "id": "skill-db-id",
      "modelId": "taxonomy-model-id",
      "UUID": "esco-skill-uuid",
      "preferredLabel": "customer service",
      "altLabels": ["client service", "customer care"],
      "description": "Maintain a high standard of customer service...",
      "scopeNote": "",
      "originUUID": "original-esco-uuid",
      "UUIDHistory": [],
      "score": 0.92,
      "skillType": "skill/competence"
    }
  ],
  
  "remaining_skills": [],
  "responsibilities": {
    "responsibilities": ["Taking orders", "Handling cash"],
    "non_responsibilities": [],
    "other_peoples_responsibilities": []
  },
  "esco_occupations": [],
  "questions_and_answers": [],
  "summary": "Experience summary text"
}
```

---

### ⚠️ Schema Mismatch Analysis

| Field | Test Stub | Actual SkillEntity | Notes |
|-------|-----------|-------------------|-------|
| `preferredLabel` | ✅ Present | ✅ Present | **Same** |
| `proficiency` | ✅ Present | ❌ Missing | Test-only field |
| `id` | ❌ Missing | ✅ Present | DB identifier |
| `UUID` | ❌ Missing | ✅ Present | ESCO UUID |
| `modelId` | ❌ Missing | ✅ Present | Taxonomy model |
| `altLabels` | ❌ Missing | ✅ Present | Alternative names |
| `description` | ❌ Missing | ✅ Present | Skill description |
| `score` | ❌ Missing | ✅ Present | Match score (0-1) |
| `skillType` | ❌ Missing | ✅ Present | skill/knowledge/etc |

### How the Recommender Agent Handles This

The `BasePhaseHandler._extract_skills_list()` method handles **both formats**:

```python
def _extract_skills_list(self, state: RecommenderAdvisorAgentState) -> list[str]:
    if isinstance(state.skills_vector, dict):
        if "skills" in state.skills_vector:
            return state.skills_vector["skills"]
        elif "top_skills" in state.skills_vector:
            skills = state.skills_vector.get("top_skills", [])
            # Extract skill names from either format
            return [s.get("preferredLabel", s.get("name", str(s))) 
                    if isinstance(s, dict) else str(s) for s in skills]
```

This means the agent works with:
- `{"top_skills": [{"preferredLabel": "..."}]}` (test stub)
- `{"top_skills": [SkillEntity(...)]}` (production)
- `{"skills": ["skill1", "skill2"]}` (simple list)

---

## �📋 User Identity

| Field          | Value                 |
|----------------|-----------------------|
| **Youth ID**   | `test_user_123`       |
| **Name**       | Hassan                |
| **Age**        | 24                    |
| **Location**   | Mombasa, Kenya        |
| **Country**    | `Country.KENYA`       |
| **Education**  | Completed Form 4, some technical college |

---

## 👤 Persona Background

Hassan is a 24-year-old from Mombasa with the following characteristics:

- **Education**: Completed Form 4 (secondary school), with some technical college coursework
- **Work Experience**:
  - Has worked casual jobs at the port
  - Helped his uncle with electrical repairs
- **Strengths**: 
  - Good with hands (manual/technical skills)
  - Basic phone and mobile money (M-Pesa) skills
- **Motivations**:
  - Wants stable income but values flexibility
  - Family expects him to contribute financially

---

## 🛠️ Skills Vector

The skills vector represents Hassan's current skill proficiencies (0.0 - 1.0 scale):

```python
skills_vector = {
    "top_skills": [
        {"preferredLabel": "Basic Electrical Wiring", "proficiency": 0.6},
        {"preferredLabel": "Manual Handling / Physical Labor", "proficiency": 0.8},
        {"preferredLabel": "M-Pesa / Mobile Money", "proficiency": 0.85},
        {"preferredLabel": "Customer Service", "proficiency": 0.65},
        {"preferredLabel": "Tool Usage (hand tools)", "proficiency": 0.7},
        {"preferredLabel": "Motorcycle Riding", "proficiency": 0.5},
        {"preferredLabel": "Basic Math / Pricing", "proficiency": 0.7}
    ]
}
```

### Skills Summary Table

| Skill                          | Proficiency | Level        |
|--------------------------------|-------------|--------------|
| M-Pesa / Mobile Money          | 0.85        | **High**     |
| Manual Handling / Physical Labor | 0.80      | **High**     |
| Tool Usage (hand tools)        | 0.70        | Moderate     |
| Basic Math / Pricing           | 0.70        | Moderate     |
| Customer Service               | 0.65        | Moderate     |
| Basic Electrical Wiring        | 0.60        | Moderate     |
| Motorcycle Riding              | 0.50        | Low-Moderate |

---

## 💡 Preference Vector

The preference vector represents Hassan's relative job/career attribute priorities, learned via Bayesian preference elicitation.

All importance scores are on a **[0, 1] scale**:
- **0.0 - 0.3**: Low importance
- **0.4 - 0.6**: Moderate importance  
- **0.7 - 1.0**: High importance

```python
from app.agent.preference_elicitation_agent.types import PreferenceVector

preference_vector = PreferenceVector(
    financial_importance=0.85,          # High - needs to support family
    work_environment_importance=0.55,   # Moderate - okay with physical work outdoors
    career_advancement_importance=0.60, # Moderate - interested but not primary focus
    work_life_balance_importance=0.80,  # High - values flexibility
    job_security_importance=0.70,       # Moderate-high - wants stable income
    task_preference_importance=0.65,    # Moderate - prefers hands-on work
    social_impact_importance=0.40       # Lower - practical/financial focus first
)
```

### Preference Dimensions Table

| Dimension               | Importance | Interpretation        | Rationale                                    |
|-------------------------|------------|----------------------|----------------------------------------------|
| Financial Compensation  | **0.85**   | **HIGH**             | Needs to support family                      |
| Work-Life Balance       | **0.80**   | **HIGH**             | Values flexibility                           |
| Job Security            | **0.70**   | **MODERATE-HIGH**    | Wants stable income                          |
| Task Preferences        | 0.65       | MODERATE             | Prefers hands-on work                        |
| Career Advancement      | 0.60       | MODERATE             | Interested but not primary focus             |
| Work Environment        | 0.55       | MODERATE             | Okay with physical work outdoors             |
| Social Impact           | 0.40       | LOW                  | Practical/financial focus takes priority     |

---

## 🎯 Hassan's Key Priorities (from Preference Vector)

1. **💰 Financial Security** (0.85) - *Highest priority*
   - Family pressure to contribute financially
   - Looking for good, consistent income
   
2. **⚖️ Work-Life Balance** (0.80) - *Very high priority*
   - Values flexibility in work schedule
   - Doesn't want to be locked into rigid hours
   
3. **🛡️ Job Stability** (0.70) - *High priority*
   - Prefers stable income over risky opportunities
   - Looking for reliable work arrangements

4. **🔧 Task Type Preferences** (0.65) - *Moderate priority*
   - Enjoys hands-on, practical work
   - Not interested in purely desk-based roles

---

## 📊 PreferenceVector Type Definition

The `PreferenceVector` class includes the following comprehensive fields:

### Core Preference Dimensions
| Field | Type | Description |
|-------|------|-------------|
| `financial_importance` | `float [0-1]` | How much user values financial compensation |
| `work_environment_importance` | `float [0-1]` | How much user values work environment (remote, commute, conditions) |
| `career_advancement_importance` | `float [0-1]` | How much user values career growth |
| `work_life_balance_importance` | `float [0-1]` | How much user values work-life balance |
| `job_security_importance` | `float [0-1]` | How much user values job security |
| `task_preference_importance` | `float [0-1]` | How much user values specific task types |
| `social_impact_importance` | `float [0-1]` | How much user values social impact |

### Quality Metadata
| Field | Type | Description |
|-------|------|-------------|
| `confidence_score` | `float [0-1]` | Overall confidence in preference estimates |
| `n_vignettes_completed` | `int` | Number of vignettes completed during elicitation |
| `per_dimension_uncertainty` | `dict[str, float]` | Uncertainty (variance) for each dimension |

### Bayesian Metadata
| Field | Type | Description |
|-------|------|-------------|
| `posterior_mean` | `list[float]` | Raw Bayesian posterior mean vector (7 dimensions) |
| `posterior_covariance_diagonal` | `list[float]` | Diagonal of posterior covariance matrix |
| `fim_determinant` | `float | None` | Fisher Information Matrix determinant |

### Qualitative Metadata (LLM-extracted)
| Field | Type | Description |
|-------|------|-------------|
| `decision_patterns` | `dict[str, Any]` | Patterns in how user makes decisions |
| `tradeoff_willingness` | `dict[str, bool]` | Explicit tradeoffs user is willing/unwilling to make |
| `values_signals` | `dict[str, bool]` | Deep values expressed in user's reasoning |
| `consistency_indicators` | `dict[str, float]` | Consistency in user's responses |
| `extracted_constraints` | `dict[str, Any]` | Hard constraints extracted from input |

---

## 🏗️ Usage in Test Script

The profile is created in the test script as follows:

```python
# Create initial state with all profile components
state = RecommenderAdvisorAgentState(
    session_id="test_session_12345",
    youth_id="test_user_123",
    country_of_user=Country.KENYA,
    conversation_phase=ConversationPhase.INTRO,
    recommendations=create_sample_recommendations(),
    skills_vector=create_sample_skills_vector(),
    preference_vector=create_sample_preference_vector()
)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `create_sample_skills_vector()` | Creates Hassan's skills proficiency dictionary |
| `create_sample_preference_vector()` | Creates Hassan's PreferenceVector with Bayesian preferences |
| `create_sample_recommendations()` | Creates Node2Vec recommendations tailored to Hassan |

---

## 📁 Source Files

- **Agent Implementation**: [`compass/backend/app/agent/recommender_advisor_agent/agent.py`](../app/agent/recommender_advisor_agent/agent.py)
- **Interactive Test Script**: [`compass/backend/scripts/test_recommender_agent_interactive.py`](../scripts/test_recommender_agent_interactive.py)
- **PreferenceVector Type**: [`compass/backend/app/agent/preference_elicitation_agent/types.py`](../app/agent/preference_elicitation_agent/types.py)

---

*Last updated: January 2026*
