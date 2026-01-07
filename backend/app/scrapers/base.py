from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timezone
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import logging
import os
from pathlib import Path

from .config import PLATFORMS, SELENIUM_CONFIG


class BaseScraper(ABC):
    """Base class for job scrapers with Selenium support."""
    
    def __init__(self, platform_key: str):
        """
        Initialize scraper for a specific platform.
        
        Args:
            platform_key: Key from PLATFORMS config (e.g., 'brightermonday')
        """
        self.platform_key = platform_key
        self.config = PLATFORMS[platform_key]
        self.platform_name = self.config['name']
        self.url = self.config['url']
        self.selectors = self.config['selectors']
        self.wait_time = self.config.get('wait_time', 10)
        self.wait_selector = self.config.get('wait_selector')
        
        self.driver = None
        self.jobs = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Scraper.{self.platform_name}")
    
    def _init_driver(self):
        """Initialize Selenium WebDriver with Chrome, with simple retry logic."""
        chrome_options = Options()

        if SELENIUM_CONFIG['headless']:
            chrome_options.add_argument('--headless')
        if SELENIUM_CONFIG['disable_gpu']:
            chrome_options.add_argument('--disable-gpu')
        if SELENIUM_CONFIG['no_sandbox']:
            chrome_options.add_argument('--no-sandbox')
        if SELENIUM_CONFIG['disable_dev_shm']:
            chrome_options.add_argument('--disable-dev-shm-usage')

        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        last_err = None
        for attempt in range(3):
            try:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                self.driver.set_page_load_timeout(SELENIUM_CONFIG['page_load_timeout'])
                self.driver.implicitly_wait(SELENIUM_CONFIG['implicit_wait'])
                return
            except Exception as e:
                last_err = e
                self.logger.warning(f"WebDriver init attempt {attempt+1} failed: {e}")
                # small backoff
                import time
                time.sleep(1 + attempt)

        # if we reach here, raise the last exception
        raise last_err
    
    def _wait_for_jobs(self):
        """Wait for job listings to load on the page."""
        if not self.wait_selector:
            return
        
        try:
            WebDriverWait(self.driver, self.wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.wait_selector))
            )
            self.logger.info(f"Job listings loaded for {self.platform_name}")
            
            # Scroll to trigger lazy loading
            self._trigger_lazy_loading()
            
        except TimeoutException:
            self.logger.warning(f"Timeout waiting for jobs on {self.platform_name}")
    
    def _trigger_lazy_loading(self):
        """Scroll through page to trigger lazy-loaded content."""
        import time
        
        try:
            # Get page height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Scroll in increments
            scroll_pause = 1.0
            for i in range(3):  # Scroll 3 times
                # Scroll down
                self.driver.execute_script(f"window.scrollTo(0, {(i+1) * 500});")
                time.sleep(scroll_pause)
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            self.logger.info("Triggered lazy loading by scrolling")
            
        except Exception as e:
            self.logger.warning(f"Error triggering lazy loading: {str(e)}")

    def _try_load_more(self) -> bool:
        """Attempt to click a site 'load more' or 'next' control. Returns True if clicked."""
        try:
            # First try any explicit selector in config
            sel = self.config.get('load_more') or self.config.get('next_page')
            if sel:
                try:
                    elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                    for el in elems:
                        try:
                            self.driver.execute_script("arguments[0].click();", el)
                            return True
                        except Exception:
                            continue
                except Exception:
                    pass

            # Fallback: search for buttons/links with common load-more text
            candidates = []
            try:
                candidates.extend(self.driver.find_elements(By.TAG_NAME, 'button'))
                candidates.extend(self.driver.find_elements(By.TAG_NAME, 'a'))
            except Exception:
                return False

            for el in candidates:
                text = (el.text or '').strip().lower()
                if not text:
                    continue
                if any(k in text for k in ('load more', 'show more', 'more jobs', 'next', 'next page', 'view more')):
                    try:
                        self.driver.execute_script("arguments[0].click();", el)
                        return True
                    except Exception:
                        continue
            return False
        except Exception:
            return False
    
    def _safe_find(self, element, selector: str, attribute: str = 'text') -> Optional[str]:
        """
        Safely find an element and extract text or attribute.
        
        Args:
            element: Parent element to search within
            selector: CSS selector
            attribute: 'text' or specific attribute name (e.g., 'href', 'alt')
        
        Returns:
            Extracted string or None
        """
        try:
            found = element.find_element(By.CSS_SELECTOR, selector)
            if attribute == 'text':
                return found.text.strip() if found.text else None
            else:
                return found.get_attribute(attribute)
        except NoSuchElementException:
            return None
    
    def _safe_find_all(self, element, selector: str) -> List:
        """
        Safely find all elements matching selector.
        
        Args:
            element: Parent element to search within
            selector: CSS selector
        
        Returns:
            List of elements (empty list if none found)
        """
        try:
            return element.find_elements(By.CSS_SELECTOR, selector)
        except NoSuchElementException:
            return []
    
    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean extracted text by removing extra whitespace."""
        if not text:
            return None
        return ' '.join(text.split())
    
    def _extract_salary(self, text: Optional[str]) -> Optional[str]:
        """Extract salary information from text containing 'KSh' or currency."""
        if not text:
            return None
        
        # Remove "Location:" prefix if present
        text = text.replace('Location:', '').strip()
        
        # Look for KSh pattern
        if 'KSh' in text or 'Ksh' in text:
            # Extract just the salary portion
            parts = text.split()
            salary_parts = []
            capture = False
            for part in parts:
                if 'KSh' in part or 'Ksh' in part:
                    capture = True
                if capture:
                    salary_parts.append(part)
                    # Stop after we get the range or single amount
                    if len(salary_parts) >= 3:  # e.g., "KSh 90,000 - 105,000"
                        break
            return ' '.join(salary_parts) if salary_parts else None
        
        return None
    
    @abstractmethod
    def parse_job_card(self, card_element) -> Optional[Dict]:
        """
        Parse a single job card element into structured data.
        Must be implemented by each platform scraper.
        
        Args:
            card_element: Selenium WebElement representing a job card
        
        Returns:
            Dictionary with job data or None if parsing fails
        """
        pass
    
    def scrape(self, max_jobs: int = 20) -> List[Dict]:
        """
        Main scraping method - fetches and parses jobs.
        
        Args:
            max_jobs: Maximum number of jobs to scrape
        
        Returns:
            List of job dictionaries
        """
        self.logger.info(f"Starting scrape of {self.platform_name}...")
        
        try:
            # Initialize driver
            self._init_driver()
            
            # Load page (with retries to handle transient failures)
            self.logger.info(f"Loading {self.url}")
            last_err = None
            for attempt in range(3):
                try:
                    self.driver.get(self.url)
                    break
                except Exception as e:
                    last_err = e
                    self.logger.warning(f"Driver.get attempt {attempt+1} failed: {e}")
                    try:
                        # try re-initializing the driver and retry
                        if self.driver:
                            try:
                                self.driver.quit()
                            except Exception:
                                pass
                        self._init_driver()
                    except Exception as ie:
                        self.logger.warning(f"Re-init driver failed: {ie}")
            else:
                # all attempts failed
                raise last_err
            
            # Wait for jobs to load
            self._wait_for_jobs()
            
            # Iterate pages / load-more until we reach max_jobs or no more pages
            seen = set()
            page_loads = 0
            max_page_loads = int(self.config.get('max_page_loads', 10))

            while len(self.jobs) < max_jobs:
                # Find current job cards
                job_cards = self.driver.find_elements(By.CSS_SELECTOR, self.selectors['job_cards'])
                self.logger.info(f"Found {len(job_cards)} job cards (page load {page_loads})")

                for i, card in enumerate(job_cards):
                    if len(self.jobs) >= max_jobs:
                        break
                    try:
                        # Sometimes content loads with delay - try twice with a small wait
                        job_data = None
                        for attempt in range(2):
                            job_data = self.parse_job_card(card)
                            if job_data and job_data.get('title'):
                                break
                            elif attempt == 0:
                                # Wait a moment and scroll to the card
                                import time
                                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", card)
                                time.sleep(0.5)

                        if not job_data or not job_data.get('title'):
                            self.logger.warning(f"✗ Failed to parse a job card - no title found")
                            continue

                        # deduplicate by URL or title
                        unique_key = job_data.get('application_url') or job_data.get('title')
                        if unique_key in seen:
                            continue
                        seen.add(unique_key)

                        job_data['source_platform'] = self.platform_key
                        job_data['scraped_at'] = datetime.now(timezone.utc)
                        self.jobs.append(job_data)
                        self.logger.info(f"✓ Scraped job {len(self.jobs)}: {job_data.get('title', 'Unknown')}")
                    except Exception as e:
                        self.logger.error(f"Error parsing job card: {str(e)}")
                        continue

                # If we've reached the requested number, stop
                if len(self.jobs) >= max_jobs:
                    break

                # Attempt to load more or go to next page
                if page_loads >= max_page_loads:
                    self.logger.info("Reached max page loads, stopping pagination")
                    break

                clicked = self._try_load_more()
                if clicked:
                    page_loads += 1
                    # wait a short while for new content to load
                    import time
                    time.sleep(1 + page_loads * 0.5)
                    # trigger extra lazy loading
                    self._trigger_lazy_loading()
                    continue
                else:
                    # no load-more/next found, stop
                    break

            self.logger.info(f"Successfully scraped {len(self.jobs)} jobs from {self.platform_name}")

            # If no jobs were parsed, save an HTML snapshot for debugging
            if not self.jobs and self.driver:
                try:
                    out_dir = Path(os.getenv('SCRAPER_OUTPUT_DIR', 'data/scrapes/debug'))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
                    filename = f"{self.platform_key}_page_{timestamp}.html"
                    file_path = out_dir / filename
                    with file_path.open('w', encoding='utf-8') as fh:
                        fh.write(self.driver.page_source)
                    self.logger.info(f"Saved debug HTML snapshot to {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save debug HTML snapshot: {e}")

            return self.jobs
            
        except Exception as e:
            self.logger.error(f"Error scraping {self.platform_name}: {str(e)}")
            return []
        
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("Browser closed")
    
    def get_stats(self) -> Dict:
        """Get scraping statistics."""
        return {
            'platform': self.platform_name,
            'total_jobs': len(self.jobs),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }