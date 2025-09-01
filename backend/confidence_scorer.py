"""
Confidence Scoring Algorithm
Calculates reliability scores for spelling and grammar suggestions
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from models import TypoResult, GrammarResult, ResumeContext
from domain_vocabulary import DomainVocabulary

@dataclass
class ScoringFactors:
    """Factors that contribute to confidence scoring"""
    base_confidence: float
    context_boost: float
    domain_validation: float
    cross_validation: float
    length_penalty: float
    frequency_boost: float

class ConfidenceScorer:
    """Calculates confidence scores for analysis suggestions"""
    
    def __init__(self):
        self.domain_vocab = DomainVocabulary()
        self.confidence_threshold = 80.0
        self.scoring_weights = {
            'base_confidence': 0.4,
            'context_boost': 0.25,
            'domain_validation': 0.2,
            'cross_validation': 0.1,
            'frequency_boost': 0.05
        }
    
    def score_spelling_suggestion(self, original: str, suggestion: str, context: str, 
                                resume_context: Optional[ResumeContext] = None) -> float:
        """Calculate confidence score for a spelling suggestion"""
        
        factors = ScoringFactors(
            base_confidence=self._calculate_base_spelling_confidence(original, suggestion),
            context_boost=self._calculate_context_boost(original, suggestion, context),
            domain_validation=self._calculate_domain_validation(original, suggestion, resume_context),
            cross_validation=self._calculate_cross_validation_score(original, suggestion),
            length_penalty=self._calculate_length_penalty(original, suggestion),
            frequency_boost=self._calculate_frequency_boost(suggestion)
        )
        
        # Calculate weighted score
        final_score = (
            factors.base_confidence * self.scoring_weights['base_confidence'] +
            factors.context_boost * self.scoring_weights['context_boost'] +
            factors.domain_validation * self.scoring_weights['domain_validation'] +
            factors.cross_validation * self.scoring_weights['cross_validation'] +
            factors.frequency_boost * self.scoring_weights['frequency_boost']
        )
        
        # Apply length penalty
        final_score *= (1.0 - factors.length_penalty)
        
        return min(100.0, max(0.0, final_score))
    
    def score_grammar_suggestion(self, sentence: str, suggestion: str, issue_type: str,
                               context: str, resume_context: Optional[ResumeContext] = None) -> float:
        """Calculate confidence score for a grammar suggestion"""
        
        # Base confidence based on issue type
        base_confidence = self._get_grammar_base_confidence(issue_type)
        
        # Context appropriateness for resume writing
        context_score = self._calculate_grammar_context_score(sentence, suggestion, resume_context)
        
        # Professional language validation
        professional_score = self._calculate_professional_language_score(suggestion, resume_context)
        
        # Rule certainty (how certain we are about the grammar rule)
        rule_certainty = self._calculate_rule_certainty(issue_type)
        
        # Calculate final score
        final_score = (
            base_confidence * 0.3 +
            context_score * 0.3 +
            professional_score * 0.25 +
            rule_certainty * 0.15
        )
        
        return min(100.0, max(0.0, final_score))
    
    def _calculate_base_spelling_confidence(self, original: str, suggestion: str) -> float:
        """Calculate base confidence for spelling correction"""
        
        # Edit distance factor
        edit_distance = self._calculate_edit_distance(original.lower(), suggestion.lower())
        max_length = max(len(original), len(suggestion))
        
        if max_length == 0:
            return 0.0
        
        # Lower edit distance = higher confidence
        edit_score = max(0, 100 - (edit_distance / max_length * 100))
        
        # Character overlap factor
        overlap_score = self._calculate_character_overlap(original.lower(), suggestion.lower())
        
        # Length similarity factor
        length_diff = abs(len(original) - len(suggestion))
        length_score = max(0, 100 - (length_diff * 10))
        
        # Combine factors
        base_score = (edit_score * 0.5 + overlap_score * 0.3 + length_score * 0.2)
        
        return base_score
    
    def _calculate_context_boost(self, original: str, suggestion: str, context: str) -> float:
        """Calculate context-based confidence boost"""
        
        context_lower = context.lower()
        suggestion_lower = suggestion.lower()
        
        # Technical context boost
        tech_indicators = [
            'programming', 'development', 'software', 'web', 'mobile', 'app',
            'framework', 'library', 'tool', 'platform', 'api', 'database'
        ]
        
        tech_context = any(indicator in context_lower for indicator in tech_indicators)
        
        if tech_context and self.domain_vocab.is_valid_technical_term(suggestion, context):
            return 90.0  # High boost for valid technical terms in technical context
        
        # Professional context boost
        professional_indicators = [
            'experience', 'responsible', 'managed', 'led', 'developed', 'implemented',
            'collaborated', 'achieved', 'improved', 'optimized'
        ]
        
        professional_context = any(indicator in context_lower for indicator in professional_indicators)
        
        if professional_context:
            # Check if suggestion fits professional language
            if self._is_professional_term(suggestion_lower):
                return 75.0
        
        # General context appropriateness
        context_words = set(context_lower.split())
        suggestion_chars = set(suggestion_lower)
        
        # Check if suggestion characters appear in context
        char_overlap = len(suggestion_chars & set(''.join(context_words))) / len(suggestion_chars)
        
        return char_overlap * 50.0
    
    def _calculate_domain_validation(self, original: str, suggestion: str, 
                                   resume_context: Optional[ResumeContext]) -> float:
        """Calculate domain-specific validation score"""
        
        # Check against domain vocabulary
        if self.domain_vocab.is_valid_technical_term(suggestion):
            return 95.0
        
        # Check if it's a known company name
        if suggestion.lower() in self.domain_vocab.company_names:
            return 90.0
        
        # Check if it's a certification term
        if suggestion.lower() in self.domain_vocab.certifications:
            return 85.0
        
        # Industry-specific validation
        if resume_context and resume_context.industry_indicators:
            for industry in resume_context.industry_indicators:
                if industry in ['software_development', 'data_science', 'devops']:
                    # More lenient for technical industries
                    if self._looks_like_technical_term(suggestion):
                        return 70.0
        
        # Check for common professional terms
        if self._is_professional_term(suggestion.lower()):
            return 60.0
        
        return 30.0  # Default low score for unknown terms
    
    def _calculate_cross_validation_score(self, original: str, suggestion: str) -> float:
        """Calculate cross-validation score using multiple methods"""
        
        # Phonetic similarity (simplified)
        phonetic_score = self._calculate_phonetic_similarity(original, suggestion)
        
        # Keyboard distance (for typo detection)
        keyboard_score = self._calculate_keyboard_distance_score(original, suggestion)
        
        # Common typo patterns
        typo_pattern_score = self._calculate_typo_pattern_score(original, suggestion)
        
        # Combine scores
        cross_val_score = (phonetic_score * 0.4 + keyboard_score * 0.3 + typo_pattern_score * 0.3)
        
        return cross_val_score
    
    def _calculate_length_penalty(self, original: str, suggestion: str) -> float:
        """Calculate penalty for significant length differences"""
        
        length_diff = abs(len(original) - len(suggestion))
        max_length = max(len(original), len(suggestion))
        
        if max_length == 0:
            return 0.0
        
        # Penalty increases with length difference
        penalty_ratio = length_diff / max_length
        
        # Apply penalty curve (more penalty for larger differences)
        penalty = penalty_ratio ** 2 * 0.3
        
        return min(0.3, penalty)  # Cap penalty at 30%
    
    def _calculate_frequency_boost(self, suggestion: str) -> float:
        """Calculate boost based on word frequency/commonality"""
        
        # Common professional words get a boost
        common_professional_words = {
            'experience', 'development', 'management', 'analysis', 'implementation',
            'collaboration', 'leadership', 'optimization', 'integration', 'architecture'
        }
        
        if suggestion.lower() in common_professional_words:
            return 80.0
        
        # Technical terms get moderate boost
        if self.domain_vocab.is_valid_technical_term(suggestion):
            return 60.0
        
        # Length-based frequency estimation (shorter words often more common)
        if len(suggestion) <= 4:
            return 40.0
        elif len(suggestion) <= 7:
            return 30.0
        else:
            return 20.0
    
    def _get_grammar_base_confidence(self, issue_type: str) -> float:
        """Get base confidence for different grammar issue types"""
        
        confidence_map = {
            'Subject-verb disagreement': 95.0,
            'Plural form needed': 90.0,
            'Pronoun case error': 85.0,
            'Article error': 80.0,
            'Preposition error': 75.0,
            'Tense consistency': 70.0,
            'Punctuation': 65.0,
            'Style': 50.0,
            'Grammar/Style': 60.0
        }
        
        return confidence_map.get(issue_type, 50.0)
    
    def _calculate_grammar_context_score(self, sentence: str, suggestion: str,
                                       resume_context: Optional[ResumeContext]) -> float:
        """Calculate context appropriateness for grammar suggestions"""
        
        # Resume-specific grammar patterns
        resume_patterns = [
            r'\b\d+\+?\s+years?\s+(?:of\s+)?experience\b',  # "5+ years experience"
            r'\bresponsible\s+for\b',  # "responsible for"
            r'\bled\s+(?:a\s+)?team\b',  # "led team"
            r'\bmanaged\s+\w+\b',  # "managed projects"
        ]
        
        # Check if suggestion maintains resume-appropriate language
        for pattern in resume_patterns:
            if re.search(pattern, suggestion, re.IGNORECASE):
                return 85.0
        
        # Check section context
        if resume_context and resume_context.sections:
            # Different standards for different sections
            if 'experience' in resume_context.sections:
                # Experience section should use past tense
                if self._uses_appropriate_tense(suggestion, 'past'):
                    return 80.0
            
            if 'summary' in resume_context.sections:
                # Summary can use present tense
                if self._uses_appropriate_tense(suggestion, 'present'):
                    return 75.0
        
        return 60.0  # Default moderate score
    
    def _calculate_professional_language_score(self, suggestion: str,
                                             resume_context: Optional[ResumeContext]) -> float:
        """Calculate score for professional language appropriateness"""
        
        # Professional action verbs
        professional_verbs = {
            'achieved', 'administered', 'analyzed', 'collaborated', 'coordinated',
            'developed', 'enhanced', 'established', 'executed', 'facilitated',
            'implemented', 'improved', 'increased', 'led', 'managed', 'optimized',
            'organized', 'planned', 'reduced', 'streamlined', 'supervised'
        }
        
        suggestion_words = suggestion.lower().split()
        
        # Check for professional verbs
        if any(word in professional_verbs for word in suggestion_words):
            return 90.0
        
        # Check for quantified achievements
        if re.search(r'\d+%|\$\d+|\d+\s+(?:years?|months?)', suggestion):
            return 85.0
        
        # Check for professional phrases
        professional_phrases = [
            'cross-functional', 'stakeholder', 'best practices', 'deliverables',
            'requirements', 'specifications', 'methodology', 'framework'
        ]
        
        if any(phrase in suggestion.lower() for phrase in professional_phrases):
            return 80.0
        
        return 50.0  # Default score
    
    def _calculate_rule_certainty(self, issue_type: str) -> float:
        """Calculate certainty about the grammar rule being applied"""
        
        # High certainty rules
        high_certainty = {
            'Subject-verb disagreement', 'Plural form needed', 'Article error'
        }
        
        # Medium certainty rules
        medium_certainty = {
            'Pronoun case error', 'Preposition error', 'Tense consistency'
        }
        
        # Low certainty rules (more subjective)
        low_certainty = {
            'Style', 'Punctuation', 'Grammar/Style'
        }
        
        if issue_type in high_certainty:
            return 90.0
        elif issue_type in medium_certainty:
            return 70.0
        elif issue_type in low_certainty:
            return 40.0
        else:
            return 50.0
    
    def calculate_composite_score(self, scores: List[float]) -> float:
        """Calculate composite confidence score from multiple sources"""
        if not scores:
            return 0.0
        
        # Use weighted average with higher weight for higher scores
        weights = [score / 100.0 for score in scores]
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return sum(scores) / len(scores)
        
        return weighted_sum / weight_sum
    
    def apply_confidence_threshold(self, suggestions: List[Dict], threshold: float = None) -> List[Dict]:
        """Filter suggestions based on confidence threshold"""
        if threshold is None:
            threshold = self.confidence_threshold
        
        return [s for s in suggestions if s.get('confidence_score', 0) >= threshold]
    
    # Helper methods
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_character_overlap(self, s1: str, s2: str) -> float:
        """Calculate character overlap percentage"""
        if not s1 or not s2:
            return 0.0
        
        chars1 = set(s1)
        chars2 = set(s2)
        overlap = len(chars1 & chars2)
        total_unique = len(chars1 | chars2)
        
        return (overlap / total_unique * 100) if total_unique > 0 else 0.0
    
    def _calculate_phonetic_similarity(self, s1: str, s2: str) -> float:
        """Calculate phonetic similarity (simplified)"""
        # Simple phonetic patterns
        phonetic_replacements = {
            'ph': 'f', 'ck': 'k', 'qu': 'kw', 'x': 'ks',
            'c': 'k', 'z': 's', 'j': 'g'
        }
        
        def phonetic_normalize(s):
            s = s.lower()
            for old, new in phonetic_replacements.items():
                s = s.replace(old, new)
            return s
        
        p1 = phonetic_normalize(s1)
        p2 = phonetic_normalize(s2)
        
        if p1 == p2:
            return 90.0
        
        # Calculate similarity after phonetic normalization
        edit_dist = self._calculate_edit_distance(p1, p2)
        max_len = max(len(p1), len(p2))
        
        if max_len == 0:
            return 0.0
        
        similarity = max(0, 100 - (edit_dist / max_len * 100))
        return similarity * 0.7  # Reduce weight since it's approximate
    
    def _calculate_keyboard_distance_score(self, s1: str, s2: str) -> float:
        """Calculate score based on keyboard key proximity"""
        # Simplified keyboard layout (QWERTY)
        keyboard_layout = {
            'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
            'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
            'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4),
            'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
            'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4),
            'n': (2, 5), 'm': (2, 6)
        }
        
        if len(s1) != len(s2):
            return 30.0  # Different lengths, moderate score
        
        total_distance = 0
        valid_chars = 0
        
        for c1, c2 in zip(s1.lower(), s2.lower()):
            if c1 in keyboard_layout and c2 in keyboard_layout:
                pos1 = keyboard_layout[c1]
                pos2 = keyboard_layout[c2]
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                total_distance += distance
                valid_chars += 1
        
        if valid_chars == 0:
            return 30.0
        
        avg_distance = total_distance / valid_chars
        # Convert to score (closer keys = higher score)
        score = max(0, 100 - (avg_distance * 30))
        
        return score
    
    def _calculate_typo_pattern_score(self, original: str, suggestion: str) -> float:
        """Calculate score based on common typo patterns"""
        
        # Common typo patterns
        patterns = [
            # Letter swaps
            (r'(.)(.)(.)', r'\1\3\2'),  # Adjacent letter swap
            # Double letters
            (r'(.)\1', r'\1'),  # Remove double letter
            (r'(.)', r'\1\1'),  # Add double letter
            # Common substitutions
            ('ie', 'ei'), ('ei', 'ie'),
            ('tion', 'sion'), ('sion', 'tion'),
        ]
        
        original_lower = original.lower()
        suggestion_lower = suggestion.lower()
        
        # Check if the change matches common typo patterns
        for pattern in patterns:
            if isinstance(pattern, tuple) and len(pattern) == 2:
                if pattern[0] in original_lower and pattern[1] in suggestion_lower:
                    return 80.0
                if pattern[1] in original_lower and pattern[0] in suggestion_lower:
                    return 80.0
        
        return 40.0  # Default score
    
    def _looks_like_technical_term(self, word: str) -> bool:
        """Check if word looks like a technical term"""
        # Technical term patterns
        patterns = [
            r'^[a-z]+js$',      # ends with 'js'
            r'^[a-z]+sql$',     # ends with 'sql'
            r'^[a-z]+db$',      # ends with 'db'
            r'^[a-z]+api$',     # ends with 'api'
            r'^\w+\.\w+$',      # contains dot (like node.js)
            r'^[A-Z]{2,}$',     # all caps acronym
        ]
        
        for pattern in patterns:
            if re.match(pattern, word):
                return True
        
        return False
    
    def _is_professional_term(self, word: str) -> bool:
        """Check if word is a professional term"""
        professional_terms = {
            'experience', 'development', 'management', 'analysis', 'implementation',
            'collaboration', 'leadership', 'optimization', 'integration', 'architecture',
            'strategy', 'methodology', 'framework', 'solution', 'innovation',
            'efficiency', 'performance', 'quality', 'delivery', 'stakeholder'
        }
        
        return word in professional_terms
    
    def _uses_appropriate_tense(self, text: str, expected_tense: str) -> bool:
        """Check if text uses appropriate tense"""
        words = text.lower().split()
        
        if expected_tense == 'past':
            # Look for past tense indicators
            past_indicators = ['ed', 'led', 'managed', 'developed', 'created', 'implemented']
            return any(word.endswith('ed') or word in past_indicators for word in words)
        
        elif expected_tense == 'present':
            # Look for present tense indicators
            present_indicators = ['manage', 'develop', 'create', 'implement', 'lead']
            return any(word in present_indicators or word.endswith('ing') for word in words)
        
        return True  # Default to true if unsure