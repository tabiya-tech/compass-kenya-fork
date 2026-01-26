# Recommender Advisor Agent

## Overview
The **Recommender Advisor Agent** manages the core career recommendation conversation. It guides users from initial introductions through presenting recommendations, addressing concerns, discussing trade-offs, and finally planning actionable next steps.

## Core Logic
The agent operates through distinct phases managed by specialized handlers:
- **Intro**: Sets context and prepares the user.
- **Present Recommendations**: Displays and explains occupation, opportunity, or training options.
- **Exploration**: Deep dives into specific recommendations (details, day-to-day, etc.).
- **Address Concerns**: Handles user hesitation or questions about fit/requirements.
- **Discuss Trade-offs**: Helps users weigh pros and cons.
- **Action Planning**: Secures commitment and defines concrete next steps.

## Testing
We have a comprehensive suite including unit, evaluation (scripted), and end-to-end tests. Run these commands from the `compass/backend` directory.

### 1. Unit Tests
Tests internal routing, initialization, and error handling.
```bash
poetry run pytest app/agent/recommender_advisor_agent/test_agent.py
```

### 2. Evaluation Tests (Scripted)
Simulates a full conversation using a scripted user and mocked services.
```bash
poetry run pytest evaluation_tests/recommender_advisor_agent/test_recommender_advisor_agent_eval.py
```

### 3. End-to-End Tests
Simulates a full conversation using **real** search service fixtures to validate data flow.
```bash
poetry run pytest evaluation_tests/recommender_advisor_agent/test_recommender_advisor_agent_e2e.py
```

### Run All Tests
```bash
poetry run pytest \
  app/agent/recommender_advisor_agent/test_agent.py \
  evaluation_tests/recommender_advisor_agent/
```
