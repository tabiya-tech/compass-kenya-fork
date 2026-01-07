from typing import List, Dict
from datetime import datetime, timezone
import logging
import os
import json
from pathlib import Path
import asyncio

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from app.taxonomy.models import JobListingModel, JobScrapingLogModel, JobPlatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JobStorage")


class FileJobStorage:
    """Fallback storage that writes jobs/logs to local files.

    This is used when `USE_LOCAL_FIXTURE` is truthy or when a `file://` mongo_uri
    is provided. Files are written to `data/scrapes/` by default.
    """

    def __init__(self, out_dir: str = None):
        self.out_dir = Path(out_dir or os.getenv("SCRAPER_OUTPUT_DIR", "data/scrapes"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs_file = self.out_dir / "logs.jsonl"

    async def save_jobs(self, jobs: List[Dict], platform: str) -> Dict:
        if not jobs:
            logger.warning(f"No jobs to save for {platform}")
            return {'inserted': 0, 'failed': 0}

        inserted = 0
        failed = 0

        def _write():
            nonlocal inserted, failed
            file_path = self.out_dir / f"{platform}.jsonl"
            with file_path.open("a", encoding="utf-8") as fh:
                for job in jobs:
                    try:
                        # Convert via model for consistent field names
                        # normalize mapping_confidence to 0-1 if provided as 0-100
                        raw_conf = job.get('occupation_match_score') or job.get('mapping_confidence')
                        mapping_confidence = None
                        try:
                            if raw_conf is not None:
                                mc = float(raw_conf)
                                if mc > 1:
                                    mapping_confidence = mc / 100.0
                                else:
                                    mapping_confidence = mc
                        except Exception:
                            mapping_confidence = None

                        job_model = JobListingModel(**{
                            'source_platform': JobPlatform(platform),
                            'url': job.get('application_url', ''),
                            'job_title': job.get('title', ''),
                            'description': job.get('description') or '',
                            'employer': job.get('company'),
                            'location': job.get('location') or 'Kenya',
                            'employment_type': job.get('employment_type'),
                            'salary_text': job.get('salary'),
                            'closing_date': job.get('closing_date'),
                            'application_url': job.get('application_url'),
                            'mapped_occupation_id': job.get('mapped_occupation_id'),
                            'mapped_skills': job.get('mapped_skills'),
                            'mapping_confidence': mapping_confidence,
                            'scraped_at': job.get('scraped_at', datetime.now(timezone.utc)),
                            'last_checked_at': datetime.now(timezone.utc),
                            'scraper_version': '1.0.0'
                        })

                        # dump model to dict and write as JSON line
                        fh.write(json.dumps(job_model.model_dump(by_alias=True, exclude_none=True, mode='python'), default=str) + "\n")
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Failed to persist job locally: {e}")
                        failed += 1

        await asyncio.to_thread(_write)

        logger.info(f"Wrote {inserted} jobs to {self.out_dir} for {platform}")
        return {'inserted': inserted, 'failed': failed, 'total': len(jobs)}

    async def log_scrape(self, platform: str, stats: Dict, success: bool = True, error_message: str = None):
        def _write_log():
            import uuid
            log = {
                'run_id': str(uuid.uuid4()),
                'platform': platform,
                'scraper_version': '1.0.0',
                'jobs_found': stats.get('total_jobs', 0),
                'jobs_added': stats.get('inserted', 0),
                'errors_count': stats.get('failed', 0),
                'started_at': datetime.now(timezone.utc).isoformat(),
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'status': 'completed' if success else 'failed',
                'error_message': error_message
            }
            with self.logs_file.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(log, default=str) + "\n")

        await asyncio.to_thread(_write_log)

    async def get_jobs_count(self, platform: str = None) -> int:
        if platform:
            file_path = self.out_dir / f"{platform}.jsonl"
            if not file_path.exists():
                return 0
            return sum(1 for _ in file_path.open("r", encoding="utf-8"))
        # count all files
        total = 0
        for p in self.out_dir.glob("*.jsonl"):
            total += sum(1 for _ in p.open("r", encoding="utf-8"))
        return total

    async def close(self):
        return


class JobStorage:
    """Handles storage of scraped jobs to MongoDB or local files based on configuration."""

    def __init__(self, mongo_uri: str = None):
        """Initialize storage backend.

        If `USE_LOCAL_FIXTURE` env var is truthy or `mongo_uri` starts with `file://`,
        use `FileJobStorage`.
        """
        use_file = False
        if mongo_uri and mongo_uri.startswith("file://"):
            use_file = True
        if os.getenv("USE_LOCAL_FIXTURE", "").lower() in ("1", "true", "yes"):
            use_file = True

        if use_file:
            out_dir = None
            if mongo_uri and mongo_uri.startswith("file://"):
                out_dir = mongo_uri[len("file://"):]
            self._impl = FileJobStorage(out_dir)
            self._is_file = True
            return

        if mongo_uri is None:
            mongo_uri = os.getenv("APPLICATION_MONGODB_URI") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")

        db_name = os.getenv("APPLICATION_DATABASE_NAME", "compass-kenya-application-local")

        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.jobs_collection = self.db["job_listings"]
        self.logs_collection = self.db["job_scraping_logs"]
        self._is_file = False

    async def save_jobs(self, jobs: List[Dict], platform: str) -> Dict:
        if getattr(self, '_is_file', False):
            return await self._impl.save_jobs(jobs, platform)

        if not jobs:
            logger.warning(f"No jobs to save for {platform}")
            return {'inserted': 0, 'failed': 0}

        inserted_count = 0
        failed_count = 0

        for job in jobs:
            try:
                job_model = self._job_dict_to_model(job, platform)
                await self.jobs_collection.insert_one(
                    job_model.model_dump(by_alias=True, exclude_none=True, mode='python')
                )
                inserted_count += 1
            except Exception as e:
                logger.error(f"Failed to save job '{job.get('title', 'Unknown')}': {str(e)}")
                failed_count += 1

        logger.info(f"Saved {inserted_count} jobs from {platform} ({failed_count} failed)")
        return {'inserted': inserted_count, 'failed': failed_count, 'total': len(jobs)}

    async def log_scrape(self, platform: str, stats: Dict, success: bool = True, error_message: str = None):
        if getattr(self, '_is_file', False):
            return await self._impl.log_scrape(platform, stats, success, error_message)

        try:
            import uuid
            log = JobScrapingLogModel(
                run_id=str(uuid.uuid4()),
                platform=JobPlatform(platform),
                scraper_version="1.0.0",
                jobs_found=stats.get('total_jobs', 0),
                jobs_added=stats.get('inserted', 0),
                jobs_updated=0,
                jobs_marked_expired=0,
                errors_count=stats.get('failed', 0),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                status='completed' if success else 'failed',
                error_message=error_message
            )
            await self.logs_collection.insert_one(
                log.model_dump(by_alias=True, exclude_none=True, mode='python')
            )
            logger.info(f"Logged scrape for {platform}")
        except Exception as e:
            logger.error(f"Failed to log scrape: {str(e)}")

    def _job_dict_to_model(self, job: Dict, platform: str) -> JobListingModel:
        """Convert job dictionary to JobListingModel."""
        mapped_occupation_id = None
        if job.get('mapped_occupation_id'):
            mapped_occupation_id = job['mapped_occupation_id']
            if not isinstance(mapped_occupation_id, ObjectId):
                try:
                    mapped_occupation_id = ObjectId(mapped_occupation_id)
                except Exception:
                    mapped_occupation_id = mapped_occupation_id

        mapped_skills = []
        if job.get('mapped_skills'):
            mapped_skills = [
                (ObjectId(skill_id) if not isinstance(skill_id, ObjectId) else skill_id)
                for skill_id in job['mapped_skills']
            ]

        mapping_confidence = job.get('occupation_match_score')
        if mapping_confidence and mapping_confidence > 1:
            mapping_confidence = mapping_confidence / 100.0

        return JobListingModel(
            source_platform=JobPlatform(platform),
            url=job.get('application_url', ''),
            job_title=job.get('title', ''),
            description=job.get('description') or '',
            employer=job.get('company'),
            location=job.get('location') or 'Kenya',
            employment_type=job.get('employment_type'),
            salary_text=job.get('salary'),
            closing_date=job.get('closing_date'),
            application_url=job.get('application_url'),
            mapped_occupation_id=mapped_occupation_id,
            mapped_skills=mapped_skills,
            mapping_confidence=mapping_confidence,
            scraped_at=job.get('scraped_at', datetime.now(timezone.utc)),
            last_checked_at=datetime.now(timezone.utc),
            scraper_version="1.0.0"
        )

    async def get_jobs_count(self, platform: str = None) -> int:
        if getattr(self, '_is_file', False):
            return await self._impl.get_jobs_count(platform)

        query = {}
        if platform:
            query['sourcePlatform'] = platform
        return await self.jobs_collection.count_documents(query)

    async def close(self):
        if getattr(self, '_is_file', False):
            return
        self.client.close()