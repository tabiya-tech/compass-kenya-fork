#!/usr/bin/env python3
"""
Intended for cron jobs that receive data from external sources (Kobotoolbox, etc.).
When a user has recommendations in the DB, they skip
to the recommendation phase at conversation start.
"""


import argparse
import asyncio
from textwrap import dedent
import csv
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from pydantic_settings import BaseSettings

# Add backend to path for app imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.user_recommendations.repository.repository import UserRecommendationsRepository
from app.user_recommendations.types import (
    UserRecommendations,
    OccupationRecommendationDB,
    OpportunityRecommendationDB,
    SkillGapRecommendationDB,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ScriptSettings(BaseSettings):
    mongodb_uri: str = ""
    db_name: str = ""

    class Config:
        env_prefix = "USER_RECOMMENDATIONS_"
        extra = "ignore"


def _parse_json_list(raw: str, item_model: type) -> list:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")
    return [item_model.model_validate(item) if isinstance(item, dict) else item_model() for item in data]


def _row_to_user_recommendations(row: dict) -> UserRecommendations:
    user_id = (row.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("user_id is required and cannot be empty")

    occ_raw = row.get("occupation_recommendations", "")
    opp_raw = row.get("opportunity_recommendations", "")
    gap_raw = row.get("skill_gap_recommendations", "")

    occupations = _parse_json_list(occ_raw, OccupationRecommendationDB)
    opportunities = _parse_json_list(opp_raw, OpportunityRecommendationDB)
    skill_gaps = _parse_json_list(gap_raw, SkillGapRecommendationDB)

    return UserRecommendations(
        user_id=user_id,
        occupation_recommendations=occupations,
        opportunity_recommendations=opportunities,
        skill_gap_recommendations=skill_gaps,
    )


def _read_csv(path: str) -> list[UserRecommendations]:
    items = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_index, row in enumerate(reader):
            try:
                items.append(_row_to_user_recommendations(row))
            except (ValueError, KeyError) as exc:
                logger.error("Row %d: %s", row_index + 2, exc)
                raise
    return items


def _read_jsonl(path: str) -> list[UserRecommendations]:
    items = []
    with open(path, encoding="utf-8") as jsonl_file:
        for line_index, line in enumerate(jsonl_file):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Line %d: Invalid JSON: %s", line_index + 1, exc)
                raise
            items.append(UserRecommendations.model_validate(data))
    return items


async def import_recommendations(
    repository: UserRecommendationsRepository,
    items: list[UserRecommendations],
) -> int:
    for item in items:
        await repository.upsert(item.user_id, item)
    return len(items)


async def _main(
    *,
    input_file: str,
    mongo_uri: str,
    db_name: str,
    hot_run: bool,
    format_type: str,
) -> None:
    if format_type == "csv":
        items = _read_csv(input_file)
    elif format_type == "jsonl":
        items = _read_jsonl(input_file)
    else:
        logger.error("Invalid format: %s", format_type)
        sys.exit(1)

    logger.info("Loaded %d user recommendations from %s", len(items), input_file)
    if not items:
        logger.info("No records to import.")
        return

    if not hot_run:
        logger.info("Dry run. Would import %d user recommendations. Use --hot-run to apply.", len(items))
        return

    if not mongo_uri or not db_name:
        logger.error(
            "MongoDB URI and DB name are required for --hot-run. "
            "Set USER_RECOMMENDATIONS_MONGODB_URI and USER_RECOMMENDATIONS_DB_NAME."
        )
        sys.exit(1)

    client = AsyncIOMotorClient(mongo_uri, tlsAllowInvalidCertificates=True)
    await client.server_info()
    db = client.get_database(db_name)
    repository = UserRecommendationsRepository(db=db)

    try:
        count = await import_recommendations(repository, items)
        logger.info("Imported %d user recommendations.", count)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import user recommendations from CSV or JSONL into the user_recommendations collection.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent("""
        Environment (required for --hot-run):
          USER_RECOMMENDATIONS_MONGODB_URI, USER_RECOMMENDATIONS_DB_NAME
        """),
    )
    parser.add_argument("--input-file", "-i", required=True, help="Path to CSV or JSONL file")
    parser.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv", help="Input format")
    parser.add_argument("--hot-run", action="store_true", help="Apply changes to the database")
    args = parser.parse_args()

    settings = ScriptSettings()
    mongo_uri = settings.mongodb_uri
    db_name = settings.db_name

    asyncio.run(
        _main(
            input_file=args.input_file,
            mongo_uri=mongo_uri,
            db_name=db_name,
            hot_run=args.hot_run,
            format_type=args.format,
        )
    )
