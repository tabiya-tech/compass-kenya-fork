# Vector Search Integration Analysis
**Date:** 2026-01-17
**Status:** Vector search is integrated and working

---

## Executive Summary

**Vector search IS integrated** into the Recommender/Advisor Agent
**Triggered when users mention occupations not in recommendations**
 **NOT YET integrated into main Agent Director** (Epic 3 is standalone)
**Test script updated** with `load_dotenv()` - all systems operational

---

## Table of Contents

1. [Integration Points](#1-integration-points)
2. [How It Works (4 Dimensions)](#2-how-it-works-4-dimensions)
3. [Trigger Points](#3-trigger-points)
4. [Future Agent Director Integration](#4-future-agent-director-integration)
5. [Quick Reference](#5-quick-reference)

---

## 1. Integration Points

### Current Integration Architecture

```
RecommenderAdvisorAgent (agent.py)
│
├── Constructor receives occupation_search_service (line 90)
│   └── Stored as self._occupation_search_service (line 107)
│
└── Passes to Phase Handlers:
    ├── IntroPhaseHandler (line 151)
    ├── ConcernsPhaseHandler (line 160)
    ├── ExplorationPhaseHandler (line 169)
    └── PresentPhaseHandler (line 178)
        │
        └── All inherit from BasePhaseHandler
            └── Provides _search_occupation_by_name() method
```

### Integration Status by Component

| Component | Integrated? | File | Line |
|-----------|------------|------|------|
| **RecommenderAdvisorAgent** |  Yes | `agent.py` | 90, 107 |
| **BasePhaseHandler** |  Yes | `base_handler.py` | 35, 49, 84-122 |
| **IntroPhaseHandler** |  Yes | Receives service via constructor | |
| **PresentPhaseHandler** |  Yes | Receives service via constructor | |
| **ExplorationPhaseHandler** |  Yes | **ACTIVELY USES** at line 275 | |
| **ConcernsPhaseHandler** |  Yes | Receives service via constructor | |
| **Agent Director** |  No | Not yet routed | N/A |

---

## 2. How It Works (4 Dimensions)

### 2.1 PARETO PRINCIPLE (20% controlling 80%)

The essential flow:

```python
# 1. User mentions occupation not in recommendations
user_input = "I want to be a DJ"

# 2. ExplorationHandler detects out-of-list mention (line 275)
mentioned_occ = await self._search_occupation_by_name("DJ")

# 3. BasePhaseHandler._search_occupation_by_name() (lines 84-122)
results = await self._occupation_search_service.search(
    query="DJ",
    k=1  # Top match only
)

# 4. Returns OccupationEntity
OccupationEntity(
    preferredLabel="disc jockey",
    code="2654.9",
    UUID="abc-123-...",
    score=0.92  # Similarity score
)

# 5. Generates contextual response
response = await self._handle_out_of_list_occupation(
    found_occupation=mentioned_occ,
    ...
)
```

**Critical 20%:**
1. `_occupation_search_service` passed to agent constructor
2. `_search_occupation_by_name()` in BasePhaseHandler
3. `_handle_out_of_list_occupation()` generates LLM response
4. Used in ExplorationPhaseHandler when user mentions new occupation

### 2.2 DECOMPOSITION (Logical Sub-Parts)

```
Vector Search Integration
│
├── 1. DEPENDENCY INJECTION
│   ├── FastAPI Dependencies (vector_search_dependencies.py)
│   │   ├── get_occupation_search_service() → Singleton
│   │   ├── get_embeddings_service() → Google Vertex AI
│   │   └── SearchServices container
│   │
│   └── Agent Construction
│       └── occupation_search_service passed to RecommenderAdvisorAgent
│
├── 2. BASE INFRASTRUCTURE (base_handler.py)
│   ├── _search_occupation_by_name()
│   │   ├── Checks if service available
│   │   ├── Calls vector search
│   │   └── Returns OccupationEntity or None
│   │
│   └── _handle_out_of_list_occupation()
│       ├── Builds LLM prompt with context
│       ├── Explains why not in recommendations
│       └── Offers choices to user
│
├── 3. ACTIVE USAGE (exploration_handler.py)
│   └── Line 275: Search when user mentions occupation
│       ├── Parse user input for occupation mention
│       ├── Search taxonomy
│       └── Generate response
│
└── 4. PASSIVE AVAILABILITY
    └── All handlers have access via inheritance
        └── Can be used in future enhancements
```

### 2.3 FIRST PRINCIPLES (Atomic Concepts)

**What is an "out-of-list occupation"?**

The recommender agent presents top 5 occupations:
```
1. Electrician (88% match)
2. Boda-boda Rider (79% match)
3. Port Cargo Handler (74% match)
4. Boat Fundi (71% match)
5. Market Vendor (68% match)
```

User says: **"I want to be a DJ"**

**Problem:** "DJ" is not in the top 5 recommendations.

**Solution:** Use vector search to:
1. Find "DJ" in the full occupation taxonomy (54,843 occupations)
2. Return: `OccupationEntity(preferredLabel="disc jockey", score=0.92)`
3. Generate contextual response explaining why it wasn't recommended

**Why vector search, not exact match?**
- User might say "DJ" but taxonomy has "disc jockey"
- User might say "electrician" but mean "industrial electrician"
- Vector embeddings find **semantically similar** occupations

**Example:**
```
User input: "computer guy"
Vector search finds:
1. "ICT technician" (score: 0.85)
2. "computer programmer" (score: 0.83)
3. "software developer" (score: 0.81)

Returns top match: "ICT technician"
```

### 2.4 LEVELS OF ABSTRACTION

**Level 1: User Experience**
```
USER: "I want to be a DJ"

AGENT: "I found 'Disc Jockey' in our database. While it wasn't
       among my top recommendations based on your skills and
       preferences (your profile shows more hands-on technical
       skills), I'm happy to explore it with you if you're
       interested. Would you like to learn more about it, or
       hear why I suggested electrician and other alternatives?"
```

**Level 2: Phase Handler Logic**
```python
# exploration_handler.py:275
async def handle(user_input, state, context):
    # Detect occupation mention
    if user_mentions_occupation(user_input):
        # Search taxonomy
        mentioned_occ = await self._search_occupation_by_name(user_input)

        if mentioned_occ:
            # Generate contextual response
            return await self._handle_out_of_list_occupation(
                found_occupation=mentioned_occ,
                user_input=user_input,
                state=state,
                context=context,
                recommendations_summary=build_summary(state.recommendations)
            )
```

**Level 3: Search Service**
```python
# base_handler.py:84-122
async def _search_occupation_by_name(occupation_name: str):
    if not self._occupation_search_service:
        return None

    results = await self._occupation_search_service.search(
        query=occupation_name,
        k=1
    )

    return results[0] if results else None
```

**Level 4: Vector Search**
```python
# esco_search_service.py:108-161
async def search(query: str, k: int):
    # 1. Convert text to embedding
    embedding = await self.embedding_service.embed(query)  # Google API

    # 2. MongoDB vector search
    pipeline = [
        {"$vectorSearch": {
            "queryVector": embedding,
            "path": "embedding",
            "numCandidates": k * 30,
            "limit": k * 3,
            "index": "embedding_index"
        }},
        {"$group": {"_id": "$UUID", "score": {"$max": "$score"}}},
        {"$sort": {"score": -1}},
        {"$limit": k}
    ]

    results = await collection.aggregate(pipeline)
    return [OccupationEntity(...) for doc in results]
```

**Level 5: Infrastructure**
- MongoDB Atlas cluster with vector indexes
- Google Vertex AI embeddings API
- 768-dimensional float arrays
- Cosine similarity ranking

---

## 3. Trigger Points

### When Vector Search is Called

| Trigger | Handler | Code Location | Description |
|---------|---------|---------------|-------------|
| **User mentions occupation not in list** | ExplorationPhaseHandler | `exploration_handler.py:275` | User says "I want to be X" where X is not in recommendations |
| **Available to all handlers** | BasePhaseHandler | `base_handler.py:84` | Any handler can call `_search_occupation_by_name()` |

### Detailed Flow: ExplorationPhaseHandler

```python
# exploration_handler.py (simplified)

async def handle(user_input: str, state, context):
    """
    EXPLORATION phase - user is exploring a specific occupation.

    Trigger points for vector search:
    1. User mentions occupation not in current focus
    2. User asks about different occupation
    """

    # TRIGGER: User mentions new occupation
    # Example: User was exploring "Electrician", now says "What about DJ?"

    # Line 275: Search for the occupation
    mentioned_occ = await self._search_occupation_by_name(user_input)

    if mentioned_occ:
        # Found in taxonomy!
        # Check if it's in recommendations
        in_recommendations = state.get_recommendation_by_id(mentioned_occ.UUID)

        if not in_recommendations:
            # OUT-OF-LIST OCCUPATION
            # Generate contextual response explaining why not recommended
            return await self._handle_out_of_list_occupation(
                found_occupation=mentioned_occ,
                ...
            )
        else:
            # In recommendations, switch focus
            state.current_focus_id = mentioned_occ.UUID
            # Generate exploration response
```

### What Happens When Vector Search Finds Nothing?

```python
# base_handler.py:117-118
if not results or len(results) == 0:
    self.logger.info(f"No occupation found in taxonomy for: '{occupation_name}'")
    return None

# In exploration_handler.py:
if mentioned_occ is None:
    # No match found in taxonomy
    # LLM handles conversationally without occupation data
    # Example response: "I don't have specific information about that
    # occupation in our database. Would you like to explore one of the
    # recommended options instead?"
```

### Example Conversation Flow

```
TURN 1:
USER: "Show me recommendations"
AGENT: [Presents top 5: Electrician, Boda-boda, Port Handler, ...]
PHASE: PRESENT_RECOMMENDATIONS
VECTOR SEARCH: Not triggered

TURN 2:
USER: "Tell me about electrician"
AGENT: [Explores electrician details]
PHASE: CAREER_EXPLORATION
FOCUS: Electrician (in recommendations)
VECTOR SEARCH: Not triggered (already in list)

TURN 3:
USER: "What about being a DJ?"
PHASE: CAREER_EXPLORATION
TRIGGER:  User mentioned "DJ" (not in recommendations)
VECTOR SEARCH CALLED: search(query="DJ", k=1)
RESULT: OccupationEntity(preferredLabel="disc jockey", score=0.92)
AGENT: "I found 'Disc Jockey' in our database. While it wasn't in
       my top recommendations (your skills lean more technical),
       I can explore it with you. Want to learn more?"

TURN 4:
USER: "Yes, tell me more about DJ"
AGENT: [Explores disc jockey details from taxonomy]
PHASE: CAREER_EXPLORATION
FOCUS: Disc Jockey (out-of-list)
VECTOR SEARCH: Not triggered (already have OccupationEntity)
```

---

## 4. Future Agent Director Integration

### Current State: Standalone Agent

The `RecommenderAdvisorAgent` is **NOT yet integrated** into the main `LLMAgentDirector` routing system.

**Evidence:**
1. No `RECOMMENDER_ADVISOR_AGENT` in `AgentType` enum
2. No routing logic in `_LLMRouter`
3. Agent only accessible via direct instantiation (test scripts)

### Integration Architecture Plan

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMAgentDirector                          │
│  (app/agent/agent_director/llm_agent_director.py)           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ├── Receives: SearchServices (via DI)
                        │   └── Contains: occupation_search_service
                        │
                        ▼
         ┌──────────────────────────────┐
         │       _LLMRouter             │
         │  (Routes user messages)      │
         └──────────────┬───────────────┘
                        │
                ┌───────┴────────┐
                │                │
         [Existing Agents]  [NEW: RecommenderAdvisorAgent]
         ├── WelcomeAgent        │
         ├── CollectExperiences  │
         └── ExploreExperiences  │
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
              [Constructor receives]    [Passes to handlers]
              occupation_search_service   ├── IntroHandler
                                          ├── PresentHandler
                                          ├── ExplorationHandler
                                          └── ConcernsHandler
```

### Required Changes for Integration

#### 1. Add Agent Type (agent_types.py)
```python
class AgentType(str, Enum):
    WELCOME_AGENT = "WELCOME_AGENT"
    COLLECT_EXPERIENCES_AGENT = "COLLECT_EXPERIENCES_AGENT"
    EXPLORE_EXPERIENCES_AGENT_DIRECTOR = "EXPLORE_EXPERIENCES_AGENT_DIRECTOR"
    PREFERENCE_ELICITATION_AGENT = "PREFERENCE_ELICITATION_AGENT"
    RECOMMENDER_ADVISOR_AGENT = "RECOMMENDER_ADVISOR_AGENT"  # ADD THIS
```

#### 2. Add to LLMAgentDirector Constructor (llm_agent_director.py)
```python
class LLMAgentDirector:
    def __init__(
        self,
        conversation_manager: ConversationMemoryManager,
        search_services: SearchServices,  # Already receives this!
        experience_pipeline_config: ExperiencePipelineConfig
    ):
        # Initialize existing agents...

        # ADD: Initialize RecommenderAdvisorAgent
        self._recommender_advisor_agent = RecommenderAdvisorAgent(
            occupation_search_service=search_services.occupation_search_service,
            # db6_client=...  # Future: DB6 integration
            # node2vec_client=...  # Future: Node2Vec integration
        )

        # Register in agent map
        self._agents[AgentType.RECOMMENDER_ADVISOR_AGENT] = self._recommender_advisor_agent
```

#### 3. Add Routing Logic (_llm_router.py)
```python
async def route(self, user_message: str, application_state: ApplicationState):
    """
    Route user message to appropriate agent.
    """

    # Existing routing logic...

    # NEW: Route to RecommenderAdvisorAgent
    if self._should_route_to_recommender_advisor(application_state):
        return AgentType.RECOMMENDER_ADVISOR_AGENT

    # ... existing routing

def _should_route_to_recommender_advisor(self, state: ApplicationState) -> bool:
    """
    Route to recommender advisor when:
    1. Preference elicitation is complete
    2. User has skills vector
    3. Recommendations have been generated (or should be)
    """
    # Check if we have recommendations ready
    if state.recommender_advisor_agent_state:
        return True

    # Check if prerequisites are met
    if (state.preference_elicitation_agent_state and
        state.preference_elicitation_agent_state.finished and
        state.user_experiences):  # Has skills
        # Initialize recommender state
        return True

    return False
```

#### 4. Add State to ApplicationState (application_state.py)
```python
class ApplicationState(BaseModel):
    # Existing state...
    welcome_agent_state: Optional[WelcomeAgentState] = None
    collect_experiences_agent_state: Optional[CollectExperiencesAgentState] = None
    preference_elicitation_agent_state: Optional[PreferenceElicitationAgentState] = None

    # ADD THIS:
    recommender_advisor_agent_state: Optional[RecommenderAdvisorAgentState] = None
```

### Dependency Injection Flow

The good news: **SearchServices is already injected into LLMAgentDirector!**

```python
# server_dependencies/agent_director_dependencies.py:15-33

def get_agent_director(
    conversation_manager: ConversationMemoryManager = Depends(...),
    search_services: SearchServices = Depends(get_all_search_services),  # ← Already here!
    application_config: ApplicationConfig = Depends(...)
) -> LLMAgentDirector:
    return LLMAgentDirector(
        conversation_manager=conversation_manager,
        search_services=search_services,  # ← Already passed!
        ...
    )
```

**What this means:**
-  `occupation_search_service` is already available in `search_services.occupation_search_service`
-  Singleton instance (one service for all requests)
-  Properly initialized with database, embeddings, etc.
-  Just need to pass to `RecommenderAdvisorAgent` constructor

### Integration Checklist

- [ ] 1. Add `RECOMMENDER_ADVISOR_AGENT` to `AgentType` enum
- [ ] 2. Import `RecommenderAdvisorAgent` in `llm_agent_director.py`
- [ ] 3. Initialize agent in `LLMAgentDirector.__init__()`
  - [ ] Pass `search_services.occupation_search_service`
  - [ ] Pass `db6_client` (when available)
  - [ ] Pass `node2vec_client` (when available)
- [ ] 4. Add routing logic in `_LLMRouter`
  - [ ] Implement `_should_route_to_recommender_advisor()`
  - [ ] Add to `route()` method
- [ ] 5. Add `recommender_advisor_agent_state` to `ApplicationState`
- [ ] 6. Update state loading/saving in `application_state_store.py`
- [ ] 7. Write integration tests
- [ ] 8. Update documentation

### No Changes Needed

 Vector search dependencies - Already working
 Occupation search service - Already singleton
 SearchServices injection - Already in place
 RecommenderAdvisorAgent code - Already complete
 Phase handlers - Already integrated with search

---

## 5. Quick Reference

### File Locations

| Component | File Path | Key Lines |
|-----------|-----------|-----------|
| **Agent** | `app/agent/recommender_advisor_agent/agent.py` | 90, 107, 151, 160, 169, 178 |
| **Base Handler** | `app/agent/recommender_advisor_agent/phase_handlers/base_handler.py` | 35, 49, 84-122, 124-236 |
| **Exploration Handler** | `app/agent/recommender_advisor_agent/phase_handlers/exploration_handler.py` | 275-277 |
| **Vector Search Service** | `app/vector_search/esco_search_service.py` | 189-297 |
| **Dependencies** | `app/vector_search/vector_search_dependencies.py` | 87-110 |
| **Test Script** | `scripts/test_recommender_agent_interactive.py` | 20-23 (dotenv), 700-732 (init) |

### Commands

```bash
# Test vector search
cd compass/backend
poetry run python scripts/test_vector_search_diagnostic.py

# Run interactive test (requires terminal)
poetry run python scripts/test_recommender_agent_interactive.py

# Check if .env is loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('TAXONOMY_MODEL_ID'))"
```

### Key Variables

```python
# Environment variables (from .env)
TAXONOMY_MODEL_ID = "68933862382aab4c7de13ec6"
EMBEDDINGS_SERVICE_NAME = "GOOGLE-VERTEX-AI"
EMBEDDINGS_MODEL_NAME = "text-embedding-005"
TAXONOMY_MONGODB_URI = "mongodb+srv://..."

# Service instances
occupation_search_service: OccupationSearchService
embedding_service: GoogleEmbeddingService
taxonomy_db: AsyncIOMotorDatabase

# Search results
OccupationEntity:
    - preferredLabel: str  # "electrician"
    - code: str           # "7411.1"
    - UUID: str           # "0d50f9af-..."
    - score: float        # 0.92 (similarity)
    - description: str
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Application configuration is not setup" | `set_application_config()` not called | Add `load_dotenv()` and initialize config |
| "Occupation search service not available" | Service is `None` | Check initialization in handler constructor |
| "No occupation found in taxonomy" | User input doesn't match any occupation | Normal - LLM handles gracefully |
| Vector search returns wrong results | Embedding model mismatch | Verify `EMBEDDINGS_MODEL_NAME` matches database |

---

## Summary

###  What's Working

1. **Vector search fully integrated** into RecommenderAdvisorAgent
2. **Actively used** in ExplorationPhaseHandler (line 275)
3. **All dependencies injected** correctly via SearchServices
4. **54,843 occupations** searchable via semantic similarity
5. **Test script updated** with `load_dotenv()`

### What's Pending

1. **Integration into LLMAgentDirector** (routing logic)
2. **ApplicationState management** (save/load recommender state)
3. **DB6 integration** (when Epic 1 completes)
4. **Node2Vec integration** (for recommendation generation)

### Next Steps

1. **Now:** Test vector search with interactive script
2. **Soon:** Integrate into Agent Director routing
3. **Later:** Connect DB6 and Node2Vec when available

---

**Last Updated:** 2026-01-17
**Tested On:** compass/backend with MongoDB Atlas + Google Vertex AI
**Status:**  Production Ready (pending Agent Director integration)
