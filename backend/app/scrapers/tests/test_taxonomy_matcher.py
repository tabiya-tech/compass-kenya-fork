"""
BDD tests for TaxonomyMatcher

Tests fuzzy matching of job titles to taxonomy occupations and skill extraction.
"""

import pytest
from app.scrapers.taxonomy_matcher import TaxonomyMatcher

describe = pytest.mark.describe


@describe("test TaxonomyMatcher initialization")
class TestTaxonomyMatcherInitialization:
    """Test that matcher initializes with taxonomy data"""
    
    def test_should_build_lookup_dictionaries(self):
        """should build occupation and skill lookup dictionaries"""
        # GIVEN taxonomy data
        givenOccupations = [
            {'id': 'occ_001', 'preferred_label': 'Software Developer'},
            {'id': 'occ_002', 'preferred_label': 'Accountant'}
        ]
        
        givenSkills = [
            {'id': 'skill_001', 'preferred_label': 'Python'},
            {'id': 'skill_002', 'preferred_label': 'JavaScript'}
        ]
        
        # WHEN matcher is initialized
        actualMatcher = TaxonomyMatcher(givenOccupations, givenSkills)
        
        # THEN expect lookup dictionaries to be created
        assert len(actualMatcher.occupation_labels) == 2
        assert 'Software Developer' in actualMatcher.occupation_labels
        assert len(actualMatcher.skill_labels) == 2


@describe("test match_occupation() method")
class TestMatchOccupation:
    """Test fuzzy matching of job titles to occupations"""
    
    def test_should_match_exact_job_title(self):
        """should match job title with 100% confidence when exact"""
        # GIVEN a matcher with test data
        givenOccupations = [
            {'id': 'occ_001', 'preferred_label': 'Software Developer'}
        ]
        givenSkills = []
        givenMatcher = TaxonomyMatcher(givenOccupations, givenSkills)
        
        # AND an exact job title
        givenJobTitle = "Software Developer"
        
        # WHEN occupation matching is performed
        actualMatch = givenMatcher.match_occupation(givenJobTitle, threshold=70)
        
        # THEN expect exact match with high confidence
        assert actualMatch is not None
        actualOccId, actualLabel, actualScore = actualMatch
        assert actualScore >= 90
        assert actualLabel == "Software Developer"
    
    def test_should_match_similar_job_title(self):
        """should match similar job title above threshold"""
        # GIVEN a matcher with test data
        givenOccupations = [
            {'id': 'occ_001', 'preferred_label': 'Software Developer'}
        ]
        givenSkills = []
        givenMatcher = TaxonomyMatcher(givenOccupations, givenSkills)
        
        # AND a similar job title
        givenJobTitle = "Software Engineer"
        givenThreshold = 70
        
        # WHEN occupation matching is performed
        actualMatch = givenMatcher.match_occupation(givenJobTitle, givenThreshold)
        
        # THEN expect match with good confidence
        assert actualMatch is not None
        actualOccId, actualLabel, actualScore = actualMatch
        assert actualScore >= givenThreshold
    
    def test_should_return_none_when_no_good_match(self):
        """should return None when no match above threshold"""
        # GIVEN a matcher with test data
        givenOccupations = [
            {'id': 'occ_001', 'preferred_label': 'Software Developer'}
        ]
        givenSkills = []
        givenMatcher = TaxonomyMatcher(givenOccupations, givenSkills)
        
        # AND a very different job title
        givenJobTitle = "Chef"
        givenThreshold = 70
        
        # WHEN occupation matching is performed
        actualMatch = givenMatcher.match_occupation(givenJobTitle, givenThreshold)
        
        # THEN expect no match
        assert actualMatch is None