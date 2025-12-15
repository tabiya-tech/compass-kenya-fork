"""
BDD tests for BaseScraper

Tests the base scraper functionality including WebDriver initialization,
element extraction, and lazy loading triggers.
"""

import pytest
from unittest.mock import Mock
from selenium.common.exceptions import NoSuchElementException

from app.scrapers.base import BaseScraper


describe = pytest.mark.describe


@describe("test BaseScraper initialization")
class TestBaseScraperInitialization:
    """Test that BaseScraper initializes correctly with platform config"""
    
    def test_should_initialize_with_platform_config(self):
        """should load platform configuration correctly"""
        # GIVEN a platform key
        givenPlatformKey = 'brightermonday'
        
        # WHEN BaseScraper is initialized with a concrete implementation
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        actualScraper = TestScraper(givenPlatformKey)
        
        # THEN expect scraper to be created
        assert actualScraper is not None
        
        # AND expect platform configuration to be loaded
        assert actualScraper.platform_key == givenPlatformKey
        assert actualScraper.platform_name == 'BrighterMonday'
        assert actualScraper.url is not None
        assert actualScraper.selectors is not None


@describe("test _safe_find() method")
class TestSafeFind:
    """Test safe element extraction with error handling"""
    
    def test_should_extract_text_from_element(self):
        """should safely extract text from element"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # AND a mock element with text
        givenElement = Mock()
        givenSubElement = Mock()
        givenSubElement.text = "Software Developer"
        givenElement.find_element.return_value = givenSubElement
        
        # AND a CSS selector
        givenSelector = 'h2.job-title'
        
        # WHEN text is extracted
        actualText = givenScraper._safe_find(givenElement, givenSelector, 'text')
        
        # THEN expect correct text to be returned
        assert actualText == "Software Developer"
    
    def test_should_return_none_when_element_not_found(self):
        """should return None when element does not exist"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # AND a mock element that raises NoSuchElementException
        givenElement = Mock()
        givenElement.find_element.side_effect = NoSuchElementException()
        
        # AND a CSS selector
        givenSelector = 'span.nonexistent'
        
        # WHEN text extraction is attempted
        actualText = givenScraper._safe_find(givenElement, givenSelector, 'text')
        
        # THEN expect None to be returned
        assert actualText is None
    
    def test_should_extract_attribute_from_element(self):
        """should extract href or other attribute from element"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # AND a mock element with href attribute
        givenElement = Mock()
        givenSubElement = Mock()
        givenSubElement.get_attribute.return_value = "https://example.com/job/123"
        givenElement.find_element.return_value = givenSubElement
        
        # AND a CSS selector
        givenSelector = 'a.job-link'
        
        # WHEN href attribute is extracted
        actualHref = givenScraper._safe_find(givenElement, givenSelector, 'href')
        
        # THEN expect correct href to be returned
        assert actualHref == "https://example.com/job/123"


@describe("test _extract_salary() method")
class TestExtractSalary:
    """Test salary extraction from text"""
    
    def test_should_extract_ksh_salary_from_text(self):
        """should extract KSh salary range from text"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # AND text containing KSh salary
        givenSalaryText = "KSh 100,000 - 150,000 per month"
        
        # WHEN salary is extracted
        actualSalary = givenScraper._extract_salary(givenSalaryText)
        
        # THEN expect KSh portion to be extracted
        assert actualSalary is not None
        assert "KSh" in actualSalary
        assert "100,000" in actualSalary
    
    def test_should_return_none_when_no_salary_in_text(self):
        """should return None when text contains no salary information"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # AND text without salary information
        givenNonSalaryText = "Nairobi, Kenya"
        
        # WHEN salary extraction is attempted
        actualSalary = givenScraper._extract_salary(givenNonSalaryText)
        
        # THEN expect None to be returned
        assert actualSalary is None


@describe("test _clean_text() method")
class TestCleanText:
    """Test text cleaning and normalization"""
    
    def test_should_remove_extra_whitespace(self):
        """should remove extra whitespace from text"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # AND text with extra whitespace
        givenMessyText = "  Software   Developer\n  Position  "
        
        # WHEN text is cleaned
        actualCleanText = givenScraper._clean_text(givenMessyText)
        
        # THEN expect whitespace to be normalized
        expectedCleanText = "Software Developer Position"
        assert actualCleanText == expectedCleanText
    
    def test_should_return_none_for_empty_text(self):
        """should return None when input text is None or empty"""
        # GIVEN a scraper instance
        class TestScraper(BaseScraper):
            def parse_job_card(self, card_element):
                return {}
        
        givenScraper = TestScraper('brightermonday')
        
        # WHEN None is cleaned
        actualResult = givenScraper._clean_text(None)
        
        # THEN expect None to be returned
        assert actualResult is None