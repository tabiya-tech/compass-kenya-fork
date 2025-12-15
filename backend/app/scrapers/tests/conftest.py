"""Shared test fixtures for scraper tests"""

import pytest
from unittest.mock import MagicMock
from selenium import webdriver

@pytest.fixture
def mock_driver():
    """Mocked Selenium WebDriver"""
    givenMockDriver = MagicMock(spec=webdriver.Chrome)
    return givenMockDriver

@pytest.fixture
def sample_job_data():
    """Sample scraped job data"""
    return {
        'title': 'Software Developer',
        'company': 'Tech Company',
        'location': 'Nairobi',
        'description': 'Looking for a skilled developer...',
        'salary': 'KSh 100,000 - 150,000',
        'posted_date': '2 days ago',
        'application_url': 'https://example.com/job/123'
    }