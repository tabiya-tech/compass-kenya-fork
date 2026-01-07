# app/scrapers/platforms/brightermonday.py
from typing import Dict, Optional
from selenium.webdriver.common.by import By
from ..base import BaseScraper


class BrighterMondayScraper(BaseScraper):
    """Scraper for BrighterMonday job listings."""

    def __init__(self):
        super().__init__('brightermonday')

    def _extract_section_by_heading(self, heading_texts):
        """
        Search the current page for a heading containing any of the strings
        in `heading_texts` (case-insensitive) and return the text of the
        sibling/content element that follows the heading.
        """
        try:
            # build xpath that looks for elements with heading text
            # search for h1-h6, strong, b, p elements containing heading
            for ht in heading_texts:
                xpath = (
                    f"//h1[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //h2[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //h3[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //h4[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //h5[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //h6[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //strong[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                    f" | //b[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ht}')]"
                )

                elems = self.driver.find_elements(By.XPATH, xpath)
                if not elems:
                    continue

                # For each matched heading, try to get the next sibling block text
                for el in elems:
                    try:
                        # Try following-sibling::* first
                        sib = el.find_element(By.XPATH, 'following-sibling::*[1]')
                        text = sib.text.strip()
                        if text:
                            return self._clean_text(text)
                    except Exception:
                        # fallback: look inside parent for paragraphs after heading
                        try:
                            parent = el.find_element(By.XPATH, '..')
                            paras = parent.find_elements(By.TAG_NAME, 'p')
                            if paras:
                                acc = ' '.join([p.text for p in paras if p.text])
                                if acc.strip():
                                    return self._clean_text(acc)
                        except Exception:
                            continue

            return None
        except Exception:
            return None

    def parse_job_card(self, card_element) -> Optional[Dict]:
        """
        Parse a BrighterMonday job card. If a detail link is present, open
        the job detail page and extract `job_summary` and
        `job_description_and_requirements` where available.
        """
        try:
            # Title
            title = self._safe_find(card_element, self.selectors['title'])
            if not title:
                return None

            # Company
            company = self._safe_find(card_element, self.selectors['company'])

            # Location (first span)
            location = self._safe_find(card_element, self.selectors['location'])
            location = self._clean_text(location)

            # Employment type (second span)
            employment_type = self._safe_find(card_element, self.selectors['employment_type'])
            employment_type = self._clean_text(employment_type)

            # Salary (third span - contains KSh)
            salary_text = self._safe_find(card_element, self.selectors['salary'])
            salary = self._extract_salary(salary_text)

            # Category
            category = self._safe_find(card_element, self.selectors['category'])

            # Short description (from listing card)
            short_description = self._safe_find(card_element, self.selectors.get('description'))
            short_description = self._clean_text(short_description)

            # Posted date
            posted_date = self._safe_find(card_element, self.selectors['posted_date'])

            # Link
            link = self._safe_find(card_element, self.selectors.get('link'), 'href')

            job = {
                'title': title,
                'company': company,
                'location': location,
                'employment_type': employment_type,
                'salary': salary,
                'category': category,
                'short_description': short_description,
                'posted_date': posted_date,
                'application_url': link,
            }

            # If a detail link exists, open it and attempt to extract structured
            # sections like 'Job summary' and 'Job description & requirements'.
            if link and self.driver:
                try:
                    # Open detail page in same tab (faster than new window)
                    self.driver.get(link)
                    # Allow page to load a bit
                    import time
                    time.sleep(0.8)

                    # Extract sections by heading heuristics
                    job_summary = self._extract_section_by_heading(['job summary', 'summary'])
                    job_description_and_requirements = self._extract_section_by_heading([
                        'job description', 'description & requirements', 'description and requirements', 'requirements'
                    ])

                    # Fallback: try a generic article/container selector if present
                    if not job_summary:
                        gen = self.driver.find_elements(By.CSS_SELECTOR, 'div.job-summary, div.summary, section.summary')
                        if gen:
                            job_summary = self._clean_text(gen[0].text)

                    if not job_description_and_requirements:
                        gen2 = self.driver.find_elements(By.CSS_SELECTOR, 'div.job-description, div.description, section.description')
                        if gen2:
                            job_description_and_requirements = self._clean_text(gen2[0].text)

                    job['job_summary'] = job_summary
                    job['job_description_and_requirements'] = job_description_and_requirements

                except Exception as e:
                    # Don't fail scraping the card if detail extraction fails
                    self.logger.debug(f"Detail extraction failed for {link}: {e}")
                finally:
                    # Navigate back to listing page so subsequent cards work
                    try:
                        self.driver.back()
                        time.sleep(0.3)
                    except Exception:
                        pass

            return job

        except Exception as e:
            self.logger.error(f"Error parsing BrighterMonday job card: {str(e)}")
            return None