#!/usr/bin/env python3
"""
Diagnostic script to test vector search service setup.

This script checks each component needed for vector search to work:
1. Application configuration
2. Database connection
3. Embedding service
4. Occupation search service

Run with: poetry run python scripts/test_vector_search_diagnostic.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.app_config import get_application_config, set_application_config, ApplicationConfig
from app.server_dependencies.db_dependencies import CompassDBProvider
from app.vector_search.esco_search_service import OccupationSearchService, VectorSearchConfig
from app.vector_search.embeddings_model import GoogleEmbeddingService
from common_libs.environment_settings.constants import EmbeddingConfig
from app.version.utils import load_version_info
from app.countries import Country, get_country_from_string
from app.i18n.language_config import LanguageConfig
from app.users.cv.constants import DEFAULT_MAX_UPLOADS_PER_USER, DEFAULT_RATE_LIMIT_PER_MINUTE


def initialize_app_config():
    """Initialize application config from environment (mimics server.py initialization)."""

    # Load language config from environment (with default fallback)
    _language_config_str = os.getenv("BACKEND_LANGUAGE_CONFIG", '{"default_locale":"en-US","available_locales":[{"locale":"en-US","date_format":"MM/DD/YYYY"}]}')

    try:
        language_config = LanguageConfig(**json.loads(_language_config_str))
    except Exception as e:
        raise ValueError(f"BACKEND_LANGUAGE_CONFIG environment variable is invalid! {e}") from e

    # Get metrics enabled
    _metrics_enabled_str = os.getenv("BACKEND_ENABLE_METRICS", "false")

    # Get default country
    _default_country_of_user_str = os.getenv("BACKEND_DEFAULT_COUNTRY", "KENYA")

    config = ApplicationConfig(
        environment_name=os.getenv("TARGET_ENVIRONMENT_NAME", "development"),
        version_info=load_version_info(),
        enable_metrics=_metrics_enabled_str.lower() == "true",
        default_country_of_user=get_country_from_string(_default_country_of_user_str),
        taxonomy_model_id=os.getenv('TAXONOMY_MODEL_ID'),
        embeddings_service_name=os.getenv('EMBEDDINGS_SERVICE_NAME'),
        embeddings_model_name=os.getenv('EMBEDDINGS_MODEL_NAME'),
        features={},
        experience_pipeline_config={},
        cv_storage_bucket=os.getenv("BACKEND_CV_STORAGE_BUCKET"),
        cv_max_uploads_per_user=os.getenv("BACKEND_CV_MAX_UPLOADS_PER_USER") or DEFAULT_MAX_UPLOADS_PER_USER,
        cv_rate_limit_per_minute=os.getenv("BACKEND_CV_RATE_LIMIT_PER_MINUTE") or DEFAULT_RATE_LIMIT_PER_MINUTE,
        language_config=language_config
    )

    set_application_config(config)
    return config


async def test_setup():
    print("=" * 80)
    print("VECTOR SEARCH SERVICE DIAGNOSTIC")
    print("=" * 80)

    try:
        # 0. Initialize app config first
        print("\n[0/4] Initializing application config...")
        try:
            app_config = initialize_app_config()
            print(f"  ✓ Application config initialized")
            print(f"    - Environment: {app_config.environment_name}")
            print(f"    - Taxonomy Model ID: {app_config.taxonomy_model_id}")
            print(f"    - Embeddings Service: {app_config.embeddings_service_name}")
            print(f"    - Embeddings Model: {app_config.embeddings_model_name}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            print("\n  SOLUTION: Check .env file for required environment variables:")
            print("    - TAXONOMY_MODEL_ID")
            print("    - EMBEDDINGS_SERVICE_NAME")
            print("    - EMBEDDINGS_MODEL_NAME")
            print("    - TAXONOMY_MONGODB_URI")
            import traceback
            traceback.print_exc()
            return

        # 1. Check database
        print("\n[1/4] Checking database connection...")
        try:
            taxonomy_db = await CompassDBProvider.get_taxonomy_db()
            print(f"  ✓ Connected to database: {taxonomy_db.name}")

            # Test connection
            await taxonomy_db.command('ping')
            print(f"  ✓ Database ping successful")

            # Check collections
            embedding_config = EmbeddingConfig()
            collections = await taxonomy_db.list_collection_names()
            occupation_collection = embedding_config.occupation_collection_name

            if occupation_collection in collections:
                print(f"  ✓ Occupation collection '{occupation_collection}' exists")

                # Count documents
                count = await taxonomy_db[occupation_collection].count_documents({})
                print(f"    - Document count: {count:,}")
            else:
                print(f"  ✗ Occupation collection '{occupation_collection}' NOT FOUND")
                print(f"    Available collections: {', '.join(collections)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            print("\n  SOLUTION: Check TAXONOMY_MONGODB_URI in .env file")
            import traceback
            traceback.print_exc()
            return

        # 2. Check embedding service
        print("\n[2/4] Checking embedding service...")
        try:
            embedding_service = GoogleEmbeddingService(
                model_name=app_config.embeddings_model_name
            )
            print(f"  ✓ Embedding service initialized")

            # Test embedding generation
            print(f"  ⏳ Generating test embedding...")
            test_embedding = await embedding_service.embed("test")
            print(f"  ✓ Embedding generated successfully")
            print(f"    - Embedding dimension: {len(test_embedding)}")
            print(f"    - Sample values: [{test_embedding[0]:.4f}, {test_embedding[1]:.4f}, {test_embedding[2]:.4f}, ...]")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            print("\n  SOLUTION: Ensure Google Cloud credentials are set up:")
            print("    - Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("    - Or use 'gcloud auth application-default login'")
            import traceback
            traceback.print_exc()
            return

        # 3. Test search
        print("\n[3/4] Testing occupation search service...")
        try:
            embedding_config = EmbeddingConfig()
            search_config = VectorSearchConfig(
                collection_name=embedding_config.occupation_collection_name,
                index_name=embedding_config.embedding_index,
                embedding_key=embedding_config.embedding_key,
            )
            print(f"  ✓ Search config created:")
            print(f"    - Collection: {search_config.collection_name}")
            print(f"    - Index: {search_config.index_name}")
            print(f"    - Embedding key: {search_config.embedding_key}")

            search_service = OccupationSearchService(
                taxonomy_db,
                embedding_service,
                search_config,
                app_config.taxonomy_model_id
            )
            print(f"  ✓ Search service initialized")

            # Test search
            print(f"\n  ⏳ Searching for 'electrician'...")
            results = await search_service.search(query="electrician", k=3)
            print(f"  ✓ Search completed successfully")
            print(f"    - Found {len(results)} occupations:")

            for i, r in enumerate(results, 1):
                print(f"      {i}. {r.preferredLabel}")
                print(f"         Code: {r.code}")
                print(f"         Score: {r.score:.3f}")
                print(f"         UUID: {r.UUID}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            print("\n  POSSIBLE CAUSES:")
            print("    1. Vector search index not created in MongoDB Atlas")
            print("    2. Index name mismatch in configuration")
            print("    3. Embedding field not properly indexed")
            import traceback
            traceback.print_exc()
            return

        print("\n" + "=" * 80)
        print("✅ ALL CHECKS PASSED! Vector search service is working correctly.")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


async def main():
    await test_setup()


if __name__ == "__main__":
    asyncio.run(main())
