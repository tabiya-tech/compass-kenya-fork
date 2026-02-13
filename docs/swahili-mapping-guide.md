# Swahili Term Mapping Guide

## Overview

This document describes the Swahili term mapping system used by Compass Kenya to normalize Swahili and code-switched (mixed English/Swahili) job and skill terms into English taxonomy-aligned equivalents for ESCO pipeline processing.

The mapping data lives in a single source of truth: `backend/app/i18n/resources/swahili_terms.json`. This file is used by both:
- The **mapping service** (`SwahiliMappingService`) for term normalization before ESCO search.
- The **RAG glossary injection** to provide Swahili job term context in LLM prompts.

---

## Mapping Sources

The initial 65+ terms in `swahili_terms.json` were sourced from:

1. **ESCO Taxonomy (Swahili translations)** -- Where available, ESCO occupation labels have Swahili equivalents that were cross-referenced.
2. **Kenya National Bureau of Statistics (KNBS)** -- Occupation classifications used in Kenyan census and labor force surveys provided formal Swahili job titles.
3. **Field research and community input** -- Informal and slang terms (e.g., "watchie" for security guard, "mama mboga" for vegetable seller, "boda boda" for motorcycle taxi) were collected from Kenyan youth employment programs and field interviews.
4. **Standard Swahili dictionaries** -- Formal Swahili occupational terms verified against Kamusi ya Kiswahili Sanifu and TUKI dictionaries.

---

## How to Add or Update Terms

### Step 1: Edit `swahili_terms.json`

Open `backend/app/i18n/resources/swahili_terms.json` and add a new entry:

```json
{
  "term": "fundi bomba",
  "normalized": "plumber",
  "taxonomy_type": "occupation",
  "language": "sw",
  "variant_type": "formal"
}
```

**Fields:**

| Field | Required | Values | Description |
|---|---|---|---|
| `term` | Yes | Any string | The Swahili term as users would type it |
| `normalized` | Yes | English term | The English taxonomy-aligned equivalent |
| `taxonomy_type` | Yes | `"occupation"` or `"skill"` | Whether this maps to an occupation or a skill |
| `language` | Yes | `"sw"` | Always `"sw"` for Swahili |
| `variant_type` | Yes | `"formal"`, `"informal"`, or `"slang"` | Formality level of the term |

### Step 2: Verify the mapping

1. Check that the `normalized` English term exists in the ESCO taxonomy (or is close enough for embedding search to resolve).
2. Ensure there are no duplicate `term` entries with conflicting `normalized` values.
3. For multi-word terms (e.g., "mama mboga"), the mapping service handles phrase-level matching automatically.

### Step 3: Run tests

```bash
cd backend

# Run the mapping service unit tests
poetry run pytest app/i18n/ -k "swahili" -v

# Run the Swahili golden transcript tests to check for regressions
poetry run python evaluation_tests/golden_transcript_runner.py \
  --transcripts-dir evaluation_tests/golden_transcripts \
  --output-dir evaluation_tests/golden_transcripts/output
```

### Step 4: Verify RAG glossary

After adding terms, they will automatically appear in the RAG glossary injected into prompts when the locale is `sw-KE`. No separate glossary file needs updating.

---

## Regional Swahili Variations

Swahili in Kenya varies by region and social context. This section documents known variations relevant to job terminology.

### Coastal Swahili (Mombasa, Lamu, Malindi)

Coastal Swahili is considered closer to "standard" Swahili and uses more Arabic-influenced vocabulary:

| Coastal term | Standard term | English |
|---|---|---|
| mzee wa meli | nahodha | ship captain |
| fundistadi | fundi | master craftsman |
| mswahili | mfanyabiashara | trader (historical usage) |

**Status:** Partially covered. Most coastal formal terms overlap with standard Swahili in the dictionary.

### Upcountry Swahili (Nairobi, Central Kenya)

Upcountry Swahili tends to borrow more from English and local Bantu languages:

| Upcountry term | Standard term | English |
|---|---|---|
| watchie | askari | security guard |
| kondakta | mpokezi abiria | bus conductor |
| hustle | kazi ya vibarua | casual work |

**Status:** Well covered. Most upcountry terms are in the dictionary as `"slang"` variants.

### Sheng (Urban Youth Slang, Nairobi)

Sheng is a dynamic creole mixing Swahili, English, and local languages. Common among Kenyan youth (our target demographic):

| Sheng term | Standard Swahili | English |
|---|---|---|
| boda boda | pikipiki ya abiria | motorcycle taxi |
| mama mboga | muuzaji mboga | vegetable vendor |
| msee wa mjengo | mjenzi | construction worker |
| vibarua | kazi za muda | casual/odd jobs |
| makanga | kondakta | bus conductor (informal) |

**Status:** Well covered. Sheng terms are prioritized in the dictionary since they are most common among target users.

### Known Gaps

The following regional or niche terms are NOT yet in the dictionary and may be added in future iterations:

- Western Kenya agricultural terms (Luhya-influenced)
- Pastoral/nomadic work terms from Northern Kenya (Turkana, Samburu areas)
- Fishing-specific terms from Lake Victoria communities
- Mining and quarry terminology from Rift Valley areas

---

## Quality Review Process

When adding new terms, follow this checklist:

1. **Source verification** -- Confirm the Swahili term is actually used by Kenyan youth (not just dictionary Swahili).
2. **ESCO alignment** -- Check that the `normalized` English term returns relevant results when searched in the ESCO database.
3. **No conflicts** -- Ensure the term doesn't conflict with an existing entry (e.g., "fundi" already maps to "technician"; adding "fundi" -> "carpenter" would create ambiguity).
4. **Test coverage** -- Add a test case that exercises the new term in an E2E conversation if it represents a common occupation.
5. **Peer review** -- New terms should be reviewed by at least one person familiar with Kenyan Swahili usage.

---

## Architecture Reference

```
swahili_terms.json  (single source of truth)
       |
       ├──> SwahiliMappingService.normalize()     (term resolution before ESCO search)
       ├──> SwahiliMappingService.normalize_text() (code-switch handling)
       └──> SwahiliMappingService.get_glossary_sample() (RAG prompt injection)
```
