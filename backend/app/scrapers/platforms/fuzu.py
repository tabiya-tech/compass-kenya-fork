# app/scrapers/platforms/fuzu.py
from typing import Dict, Optional
from selenium.webdriver.common.by import By
from pathlib import Path
import os
import time
from datetime import datetime
from ..base import BaseScraper


class FuzuScraper(BaseScraper):
    """Scraper for Fuzu job listings with more robust waits and debug snapshots."""

    def __init__(self):
        super().__init__('fuzu')

    def _robust_scroll(self, attempts: int = 5, pause: float = 1.0):
        """Scroll the page several times to trigger lazy-loading."""
        try:
            for i in range(attempts):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(pause)
            # scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
        except Exception:
            pass

    def parse_job_card(self, card_element) -> Optional[Dict]:
        """
        Parse a Fuzu job card and attempt to open the detail page to extract
        the full description. On failures, save the detail page HTML for debugging.
        """
        try:
            # Title
            title = self._safe_find(card_element, self.selectors['title'])
            if not title:
                return None

            # Company
            company = self._safe_find(card_element, self.selectors['company'])

            # Location
            location = self._safe_find(card_element, self.selectors['location'])
            location = self._clean_text(location)

            # Short description (from card)
            description = self._safe_find(card_element, self.selectors.get('description'))
            description = self._clean_text(description)

            # Posted and closing dates
            date_elements = self._safe_find_all(card_element, self.selectors.get('posted_date', 'p.published'))
            posted_date = None
            closing_date = None
            for elem in date_elements:
                text = elem.text
                if 'Published:' in text:
                    posted_date = text.replace('Published:', '').strip()
                elif 'Closing:' in text:
                    closing_date = text.replace('Closing:', '').strip()

            # Link
            link = self._safe_find(card_element, self.selectors.get('link'), 'href')

            job = {
                'title': title,
                'company': company,
                'location': location,
                'description': description,
                'posted_date': posted_date,
                'closing_date': closing_date,
                'application_url': link,
            }

            # If detail link exists, open it and try to retrieve extended text.
            if link and self.driver:
                last_err = None
                for attempt in range(3):
                    try:
                        self.driver.get(link)
                        # perform robust scrolls to ensure content loads
                        self._robust_scroll(attempts=4, pause=0.8)

                        # try common selectors for full description
                        sel_candidates = [
                            'div.job-description', 'div.job-details', 'div.view-summary-content',
                            'div.description', 'section.description', 'article.job', 'div.job'
                        ]
                        full_text = None
                        for sel in sel_candidates:
                            elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                            if elems:
                                # join first element's paragraph children
                                text = elems[0].text.strip()
                                if text:
                                    full_text = self._clean_text(text)
                                    break

                        if full_text:
                            job['full_description'] = full_text
                        else:
                            # fallback: grab body text
                            body = self.driver.find_elements(By.TAG_NAME, 'body')
                            if body:
                                job['full_description'] = self._clean_text(body[0].text)

                        break
                    except Exception as e:
                        last_err = e
                        self.logger.warning(f"Detail page attempt {attempt+1} failed for Fuzu link: {e}")
                        time.sleep(0.8 + attempt)
                        continue

                if last_err:
                    # save debug snapshot of the detail page if possible
                    try:
                        out_dir = Path(os.getenv('SCRAPER_OUTPUT_DIR', 'data/scrapes/debug'))
                        out_dir.mkdir(parents=True, exist_ok=True)
                        safe_title = ''.join(c if c.isalnum() else '_' for c in (title or 'fuzu'))[:80]
                        ts = datetime.now().strftime('%Y%m%dT%H%M%S')
                        fname = f"fuzu_detail_{safe_title}_{ts}.html"
                        fp = out_dir / fname
                        with fp.open('w', encoding='utf-8') as fh:
                            fh.write(self.driver.page_source if self.driver else '')
                        self.logger.info(f"Saved Fuzu detail snapshot to {fp}")
                    except Exception as se:
                        self.logger.warning(f"Failed to save Fuzu detail snapshot: {se}")
                # navigate back to listing if possible
                try:
                    self.driver.back()
                    time.sleep(0.3)
                except Exception:
                    pass

            return job

        except Exception as e:
            self.logger.error(f"Error parsing Fuzu job card: {str(e)}")
            # save page snapshot to help debugging
            try:
                out_dir = Path(os.getenv('SCRAPER_OUTPUT_DIR', 'data/scrapes/debug'))
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%dT%H%M%S')
                fname = f"fuzu_card_error_{ts}.html"
                fp = out_dir / fname
                if self.driver:
                    with fp.open('w', encoding='utf-8') as fh:
                        fh.write(self.driver.page_source)
                    self.logger.info(f"Saved Fuzu error snapshot to {fp}")
            except Exception:
                pass
            return None