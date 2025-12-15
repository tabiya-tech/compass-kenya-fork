"""
Tools for curating taxonomy:
- flagging irrelevant occupations
- adding custom Kenyan occupations
"""

from bson import ObjectId  # type: ignore
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient

from app.taxonomy.models import (
    OccupationModel,
    DataSource,
    OccupationType,
)

from app.taxonomy.importers.config import (
    TAXONOMY_MODEL_ID,
    MONGODB_URI,
    TAXONOMY_DB_NAME,
)


async def flag_occupation_irrelevant(db, occupation_id: str, reason: str, curator: str):
    """
    Mark an occupation as not relevant for Kenya
    """
    await db.occupations.update_one(
        {"_id": ObjectId(occupation_id)},
        {
            "$set": {
                "is_relevant_for_kenya": False,
                "kenya_specific_notes": reason,
                "updated_at": datetime.now(timezone.utc),
                "added_by": curator,
            }
        },
    )


async def add_custom_occupation(db, occupation_data: dict, curator: str):
    """
    Add a new custom Kenyan occupation
    """
    occupation = OccupationModel(
        # Primary identification
        code=f"KE-CUSTOM-{ObjectId()}",
        preferred_label=occupation_data["title"],
        alt_labels=occupation_data.get("alt_labels", []),

        # REQUIRED classification
        occupation_type=OccupationType.LOCAL_OCCUPATION,

        # Optional hierarchy (if known)
        isco_group_code=occupation_data.get("isco_group_code"),
        occupation_group_code=occupation_data.get("occupation_group_code"),

        # Kenya context
        is_relevant_for_kenya=True,
        is_informal_sector=occupation_data.get("is_informal", False),
        is_entrepreneurship=occupation_data.get("is_entrepreneurship", False),
        kenya_specific_notes=occupation_data.get("notes"),

        # Provenance
        source=DataSource.CUSTOM,
        taxonomy_model_id=TAXONOMY_MODEL_ID,
        added_by=curator,
        is_localized=True,

        # Search optimization
        search_terms=occupation_data.get("search_terms", []),
    )

    result = await db.occupations.insert_one(
        occupation.model_dump(by_alias=True, exclude_none=True)
    )

    return result.inserted_id


async def main():
    """
    Demo: flag an occupation and add a custom Kenyan occupation
    """
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[TAXONOMY_DB_NAME]

    # ------------------------------------------------------------------
    # Example 1: Flag irrelevant occupation
    # ------------------------------------------------------------------
    await flag_occupation_irrelevant(
        db=db,
        occupation_id="507f1f77bcf86cd799439011",
        reason="Not applicable in Kenyan labor market",
        curator="stevealila_contractor",
    )

    # ------------------------------------------------------------------
    # Example 2: Add custom informal occupation (Boda Boda Rider)
    # ------------------------------------------------------------------
    boda_boda_id = await add_custom_occupation(
        db=db,
        occupation_data={
            "title": "Boda Boda Rider (Motorcycle Taxi Driver)",
            "alt_labels": [
                "Bodaboda Rider",
                "Motorcycle Taxi Driver",
                "Motorbike Taxi Operator",
            ],
            "is_informal": True,
            "is_entrepreneurship": True,
            "search_terms": [
                "boda boda",
                "bodaboda",
                "motorcycle taxi",
                "motorbike rider",
            ],
            # Optional ISCO reference if you want it
            # "isco_group_code": "8321",
        },
        curator="stevealila_contractor",
    )

    print(f"Added custom occupation with ID: {boda_boda_id}")

    client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
