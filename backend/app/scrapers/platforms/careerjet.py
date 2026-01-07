# app/scrapers/platforms/careerjet.py
from typing import Dict, Optional
from selenium.webdriver.common.by import By
from ..base import BaseScraper


class CareerjetScraper(BaseScraper):
    """Scraper for Careerjet job listings."""

    def __init__(self):
        super().__init__('careerjet')

    def _extract_section_by_heading(self, heading_texts):
        """
        Look for headings (h1-h6, strong, b) containing any of the
        `heading_texts` (case-insensitive) and return the following block's text.
        """
        try:
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

                for el in elems:
                    try:
                        sib = el.find_element(By.XPATH, 'following-sibling::*[1]')
                        text = sib.text.strip()
                        if text:
                            return self._clean_text(text)
                    except Exception:
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
        Parse a Careerjet job card and extract structured sections from the
        job detail page: `key_responsibilities` and `minimum_requirements`.
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

            # Salary
            salary = self._safe_find(card_element, self.selectors['salary'])
            salary = self._clean_text(salary)

            # Short description from listing
            short_description = self._safe_find(card_element, self.selectors.get('description'))
            short_description = self._clean_text(short_description)

            # Posted date
            posted_date = self._safe_find(card_element, self.selectors['posted_date'])

            # Link - special handling for data-url attribute
            link_path = card_element.get_attribute('data-url')
            link = f"https://www.careerjet.co.ke{link_path}" if link_path else None

            job = {
                'title': title,
                'company': company,
                'location': location,
                'salary': salary,
                'short_description': short_description,
                'posted_date': posted_date,
                'application_url': link,
            }

            if link and self.driver:
                try:
                    self.driver.get(link)
                    import time
                    time.sleep(0.6)

                    # Extract Key Responsibilities and Minimum Requirements
                    key_resp = self._extract_section_by_heading(['key responsibilities', 'responsibilities', 'key responsibilities:'])
                    min_reqs = self._extract_section_by_heading(['minimum requirements', 'minimum qualifications', 'requirements'])

                    # Fallbacks: generic selectors
                    if not key_resp:
                        gen = self.driver.find_elements(By.CSS_SELECTOR, 'div.key-responsibilities, section.key-responsibilities, div.responsibilities')
                        if gen:
                            key_resp = self._clean_text(gen[0].text)

                    if not min_reqs:
                        gen2 = self.driver.find_elements(By.CSS_SELECTOR, 'div.minimum-requirements, section.minimum-requirements, div.requirements')
                        if gen2:
                            min_reqs = self._clean_text(gen2[0].text)

                    job['key_responsibilities'] = key_resp
                    job['minimum_requirements'] = min_reqs

                except Exception as e:
                    self.logger.debug(f"Careerjet detail extraction failed for {link}: {e}")
                finally:
                    try:
                        self.driver.back()
                        time.sleep(0.2)
                    except Exception:
                        pass

            return job

        except Exception as e:
            self.logger.error(f"Error parsing Careerjet job card: {str(e)}")
            return None