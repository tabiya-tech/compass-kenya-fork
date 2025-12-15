# app/scrapers/platforms/__init__.py
from .brightermonday import BrighterMondayScraper
from .careerjet import CareerjetScraper
from .fuzu import FuzuScraper
from .jobwebkenya import JobWebKenyaScraper
from .myjobmag import MyJobMagScraper

__all__ = [
    'BrighterMondayScraper',
    'CareerjetScraper',
    'FuzuScraper',
    'JobWebKenyaScraper',
    'MyJobMagScraper',
]