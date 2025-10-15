"""
Advanced confidence scoring system for enhanced typo detection.
Implements multi-factor confidence calculation with machine learning-inspired approaches.
"""

import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta

from enhanced_models import DetectionLayer, ConfidenceLevel
from core_interfaces import IConfidenceScorer

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceFactors:
    """Factors that contribute to confidence scoring"""
    edit_distance_score: float = 0.0
    frequency_score: float = 0.0
    context_score: float = 0.0
    domain_score: float = 0.0
    length_similarity_score: float = 0.0
    phonetic_similarity_score: float = 0.0
    layer_reliability_score: float = 0.0
    historical_accuracy_score: float = 0.0
    
    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted confidence score"""
        total_score = 0.0
        total_weight = 0.0
        
        factors = {
            'edit_distance': self.edit_distance_score,
            'frequency': self.frequency_score,
            'context': self.context_score,
            'domain': self.domain_score,
            'length_similarity': self.length_similarity_score,
            'phonetic_similarity': self.phonetic_similarity_score,
            'layer_reliability': self.layer_reliability_score,
            'historical_accuracy': self.historical_accuracy_score
        }
        
        for factor_name, score in factors.items():
            weight = weights.get(factor_name, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 1.0)

class LayerReliabilityTracker:
    """Tracks reliability of different detection layers"""
    
    def __init__(self):
        self.layer_stats: Dict[DetectionLayer, Dict[str, Any]] = {}
        self.accuracy_history: Dict[DetectionLayer, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Initialize layer stats
        for layer in DetectionLayer:
            self.layer_stats[layer] = {
                'total_suggestions': 0,
                'accepted_suggestions': 0,
                'accuracy_rate': 0.8,  # Default starting accuracy
                'confidence_calibration': 1.0,
                'last_updated': datetime.now()
            }
    
    def record_suggestion_outcome(self, layer: DetectionLayer, was_correct: bool, 
                                confidence: float) -> None:
        """Record the outcome of a suggestion for calibration"""
        stats = self.layer_stats[layer]
        stats['total_suggestions'] += 1
        
        if was_correct:
            stats['accepted_suggestions'] += 1
        
        # Update accuracy rate with exponential moving average
        new_accuracy = stats['accepted_suggestions'] / stats['total_suggestions']
        stats['accuracy_rate'] = 0.9 * stats['accuracy_rate'] + 0.1 * new_accuracy
        
        # Record in history
        self.accuracy_history[layer].append((datetime.now(), new_accuracy))
        
        # Keep only recent history (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.accuracy_history[layer] = [
            (date, acc) for date, acc in self.accuracy_history[layer] 
            if date >= cutoff_date
        ]
        
        stats['last_updated'] = datetime.now()
    
    def get_layer_reliability(self, layer: DetectionLayer) -> float:
        """Get reliability score for a layer (0-1)"""
        stats = self.layer_stats.get(layer, {})
        return stats.get('accuracy_rate', 0.8)
    
    def get_confidence_calibration(self, layer: DetectionLayer) -> float:
        """Get confidence calibration factor for a layer"""
        stats = self.layer_stats.get(layer, {})
        return stats.get('confidence_calibration', 1.0)
    
    def update_calibration(self, layer: DetectionLayer) -> None:
        """Update confidence calibration based on historical performance"""
        if layer not in self.accuracy_history:
            return
        
        history = self.accuracy_history[layer]
        if len(history) < 10:  # Need minimum data points
            return
        
        # Calculate calibration factor based on recent performance
        recent_accuracies = [acc for _, acc in history[-20:]]  # Last 20 data points
        avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
        
        # Adjust calibration factor
        if avg_accuracy > 0.9:
            calibration = 1.1  # Boost confidence for highly accurate layers
        elif avg_accuracy > 0.8:
            calibration = 1.0  # Normal confidence
        elif avg_accuracy > 0.6:
            calibration = 0.9  # Reduce confidence slightly
        else:
            calibration = 0.8  # Significantly reduce confidence
        
        self.layer_stats[layer]['confidence_calibration'] = calibration

class FrequencyAnalyzer:
    """Analyzes word frequency for confidence scoring"""
    
    def __init__(self):
        # Common English word frequencies (simplified)
        self.common_words = {
            'the': 1000000, 'be': 500000, 'to': 400000, 'of': 350000,
            'and': 300000, 'a': 250000, 'in': 200000, 'that': 150000,
            'have': 140000, 'i': 130000, 'it': 120000, 'for': 110000,
            'not': 100000, 'on': 95000, 'with': 90000, 'he': 85000,
            'as': 80000, 'you': 75000, 'do': 70000, 'at': 65000
        }
        
        # Technical term frequencies (domain-specific)
        self.technical_frequencies = {
            'python': 50000, 'javascript': 45000, 'java': 40000,
            'react': 35000, 'aws': 30000, 'docker': 25000,
            'kubernetes': 20000, 'tensorflow': 15000, 'api': 60000,
            'database': 40000, 'frontend': 30000, 'backend': 30000
        }
    
    def get_frequency_score(self, word: str, is_technical_context: bool = False) -> float:
        """Get frequency-based confidence score (0-1)"""
        word_lower = word.lower()
        
        # Check technical frequencies first if in technical context
        if is_technical_context and word_lower in self.technical_frequencies:
            freq = self.technical_frequencies[word_lower]
            # Normalize technical frequency (max 60000)
            return min(1.0, freq / 60000)
        
        # Check common word frequencies
        if word_lower in self.common_words:
            freq = self.common_words[word_lower]
            # Normalize common word frequency (max 1000000)
            return min(1.0, freq / 1000000)
        
        # Unknown word - assign low frequency score
        return 0.1

class PhoneticSimilarityCalculator:
    """Calculates phonetic similarity for confidence scoring"""
    
    def __init__(self):
        # Simplified phonetic mapping
        self.phonetic_map = {
            'c': 'k', 'ph': 'f', 'gh': 'f', 'ck': 'k',
            'x': 'ks', 'qu': 'kw', 'th': 't'
        }
    
    def get_phonetic_representation(self, word: str) -> str:
        """Get simplified phonetic representation of a word"""
        word_lower = word.lower()
        
        # Apply phonetic mappings
        for pattern, replacement in self.phonetic_map.items():
            word_lower = word_lower.replace(pattern, replacement)
        
        # Remove silent letters (simplified)
        word_lower = re.sub(r'[aeiou]$', '', word_lower)  # Remove trailing vowels
        word_lower = re.sub(r'h$', '', word_lower)  # Remove trailing h
        
        return word_lower
    
    def calculate_phonetic_similarity(self, word1: str, word2: str) -> float:
        """Calculate phonetic similarity between two words (0-1)"""
        phonetic1 = self.get_phonetic_representation(word1)
        phonetic2 = self.get_phonetic_representation(word2)
        
        if not phonetic1 or not phonetic2:
            return 0.0
        
        # Calculate edit distance on phonetic representations
        distance = self._levenshtein_distance(phonetic1, phonetic2)
        max_len = max(len(phonetic1), len(phonetic2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1 - (distance / max_len)
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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

class ContextAnalyzer:
    """Analyzes context for confidence scoring"""
    
    def __init__(self):
        self.technical_indicators = [
            'programming', 'development', 'software', 'technology',
            'framework', 'library', 'api', 'database', 'server',
            'frontend', 'backend', 'fullstack', 'devops'
        ]
        
        self.skill_indicators = [
            'skills', 'technologies', 'proficient', 'experience',
            'knowledge', 'familiar', 'expertise', 'competency'
        ]
        
        self.formal_indicators = [
            'resume', 'cv', 'professional', 'career', 'employment',
            'position', 'role', 'responsibility', 'achievement'
        ]
    
    def analyze_context(self, context: str, word: str, suggestion: str) -> Dict[str, float]:
        """Analyze context and return confidence factors"""
        context_lower = context.lower()
        word_lower = word.lower()
        suggestion_lower = suggestion.lower()
        
        scores = {
            'technical_context': 0.0,
            'skill_context': 0.0,
            'formal_context': 0.0,
            'word_position': 0.0,
            'surrounding_quality': 0.0
        }
        
        # Technical context score
        technical_matches = sum(1 for indicator in self.technical_indicators 
                              if indicator in context_lower)
        scores['technical_context'] = min(1.0, technical_matches / 3)
        
        # Skill context score
        skill_matches = sum(1 for indicator in self.skill_indicators 
                           if indicator in context_lower)
        scores['skill_context'] = min(1.0, skill_matches / 2)
        
        # Formal context score
        formal_matches = sum(1 for indicator in self.formal_indicators 
                            if indicator in context_lower)
        scores['formal_context'] = min(1.0, formal_matches / 2)
        
        # Word position score (middle of sentence is better)
        word_position = context_lower.find(word_lower)
        if word_position != -1:
            relative_position = word_position / len(context_lower)
            # Score higher for words in middle of context
            scores['word_position'] = 1 - abs(0.5 - relative_position)
        
        # Surrounding quality score (check for other errors nearby)
        surrounding_text = self._get_surrounding_text(context, word, 50)
        error_indicators = ['???', '***', 'xxx', '...']
        error_count = sum(1 for indicator in error_indicators 
                         if indicator in surrounding_text.lower())
        scores['surrounding_quality'] = max(0.0, 1.0 - (error_count * 0.3))
        
        return scores
    
    def _get_surrounding_text(self, context: str, word: str, window: int) -> str:
        """Get text surrounding the word"""
        word_pos = context.lower().find(word.lower())
        if word_pos == -1:
            return context
        
        start = max(0, word_pos - window)
        end = min(len(context), word_pos + len(word) + window)
        return context[start:end]

class AdvancedConfidenceScorer(IConfidenceScorer):
    """
    Advanced confidence scorer implementing sophisticated scoring algorithms.
    """
    
    def __init__(self):
        self.layer_tracker = LayerReliabilityTracker()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.phonetic_calculator = PhoneticSimilarityCalculator()
        self.context_analyzer = ContextAnalyzer()
        
        # Confidence scoring weights
        self.spelling_weights = {
            'edit_distance': 0.25,
            'frequency': 0.15,
            'context': 0.20,
            'domain': 0.15,
            'length_similarity': 0.10,
            'phonetic_similarity': 0.10,
            'layer_reliability': 0.05
        }
        
        self.grammar_weights = {
            'edit_distance': 0.15,
            'frequency': 0.10,
            'context': 0.30,
            'domain': 0.20,
            'length_similarity': 0.05,
            'phonetic_similarity': 0.05,
            'layer_reliability': 0.15
        }
        
        logger.info("Advanced confidence scorer initialized")
    
    def score_spelling_suggestion(self, original: str, suggestion: str, 
                                context: str, metadata: Dict[str, Any]) -> float:
        """
        Score a spelling suggestion with comprehensive analysis.
        
        Args:
            original: Original word
            suggestion: Suggested correction
            context: Surrounding text context
            metadata: Additional metadata (resume context, layer info, etc.)
            
        Returns:
            Confidence score (0-100)
        """
        factors = self._calculate_spelling_factors(original, suggestion, context, metadata)
        
        # Calculate weighted score
        base_score = factors.get_weighted_score(self.spelling_weights)
        
        # Apply layer-specific calibration
        layer = metadata.get('layer', DetectionLayer.TRADITIONAL_NLP)
        calibration = self.layer_tracker.get_confidence_calibration(layer)
        
        # Apply domain-specific adjustments
        domain_adjustment = self._get_domain_adjustment(original, suggestion, metadata)
        
        # Final score calculation
        final_score = base_score * calibration * (1 + domain_adjustment)
        
        # Ensure score is in valid range
        final_score = max(0.0, min(100.0, final_score * 100))
        
        # Log scoring details for debugging
        logger.debug(
            f"Spelling confidence: {original} -> {suggestion} = {final_score:.1f}",
            extra={
                'original': original,
                'suggestion': suggestion,
                'base_score': base_score,
                'calibration': calibration,
                'domain_adjustment': domain_adjustment,
                'layer': layer
            }
        )
        
        return final_score
    
    def _calculate_spelling_factors(self, original: str, suggestion: str,
                                  context: str, metadata: Dict[str, Any]) -> ConfidenceFactors:
        """Calculate all confidence factors for spelling suggestions"""
        factors = ConfidenceFactors()
        
        # Edit distance score
        factors.edit_distance_score = self._calculate_edit_distance_score(original, suggestion)
        
        # Frequency score
        is_technical = metadata.get('is_technical_context', False)
        factors.frequency_score = self.frequency_analyzer.get_frequency_score(suggestion, is_technical)
        
        # Context score
        context_scores = self.context_analyzer.analyze_context(context, original, suggestion)
        factors.context_score = sum(context_scores.values()) / len(context_scores)
        
        # Domain score (from domain validator if available)
        factors.domain_score = metadata.get('domain_validation_score', 0.5)
        
        # Length similarity score
        factors.length_similarity_score = self._calculate_length_similarity(original, suggestion)
        
        # Phonetic similarity score
        factors.phonetic_similarity_score = self.phonetic_calculator.calculate_phonetic_similarity(
            original, suggestion
        )
        
        # Layer reliability score
        layer = metadata.get('layer', DetectionLayer.TRADITIONAL_NLP)
        factors.layer_reliability_score = self.layer_tracker.get_layer_reliability(layer)
        
        return factors
    
    def _calculate_edit_distance_score(self, original: str, suggestion: str) -> float:
        """Calculate edit distance-based confidence score"""
        if not original or not suggestion:
            return 0.0
        
        distance = self._levenshtein_distance(original.lower(), suggestion.lower())
        max_len = max(len(original), len(suggestion))
        
        if max_len == 0:
            return 1.0
        
        # Convert distance to similarity score
        similarity = 1 - (distance / max_len)
        
        # Apply non-linear transformation to emphasize high similarity
        return math.pow(similarity, 0.5)
    
    def _calculate_length_similarity(self, original: str, suggestion: str) -> float:
        """Calculate length similarity score"""
        if not original or not suggestion:
            return 0.0
        
        len_diff = abs(len(original) - len(suggestion))
        max_len = max(len(original), len(suggestion))
        
        if max_len == 0:
            return 1.0
        
        # Penalize large length differences
        similarity = 1 - (len_diff / max_len)
        return max(0.0, similarity)
    
    def _get_domain_adjustment(self, original: str, suggestion: str, 
                             metadata: Dict[str, Any]) -> float:
        """Get domain-specific confidence adjustment"""
        adjustment = 0.0
        
        # Technical term bonus
        if metadata.get('is_technical_term', False):
            adjustment += 0.1
        
        # Resume context bonus
        if metadata.get('is_resume_context', False):
            adjustment += 0.05
        
        # Company name bonus
        if metadata.get('is_company_name', False):
            adjustment += 0.15
        
        # Certification bonus
        if metadata.get('is_certification', False):
            adjustment += 0.2
        
        return adjustment
    
    def score_grammar_suggestion(self, sentence: str, suggestion: str,
                               issue_type: str, metadata: Dict[str, Any]) -> float:
        """
        Score a grammar suggestion.
        
        Args:
            sentence: Original sentence
            suggestion: Suggested correction
            issue_type: Type of grammar issue
            metadata: Additional metadata
            
        Returns:
            Confidence score (0-100)
        """
        factors = self._calculate_grammar_factors(sentence, suggestion, issue_type, metadata)
        
        # Calculate weighted score
        base_score = factors.get_weighted_score(self.grammar_weights)
        
        # Apply layer-specific calibration
        layer = metadata.get('layer', DetectionLayer.TRADITIONAL_NLP)
        calibration = self.layer_tracker.get_confidence_calibration(layer)
        
        # Apply grammar-specific adjustments
        grammar_adjustment = self._get_grammar_adjustment(issue_type, metadata)
        
        # Final score calculation
        final_score = base_score * calibration * (1 + grammar_adjustment)
        
        # Ensure score is in valid range
        final_score = max(0.0, min(100.0, final_score * 100))
        
        logger.debug(
            f"Grammar confidence: {issue_type} = {final_score:.1f}",
            extra={
                'sentence': sentence[:50] + "..." if len(sentence) > 50 else sentence,
                'suggestion': suggestion[:50] + "..." if len(suggestion) > 50 else suggestion,
                'issue_type': issue_type,
                'base_score': base_score,
                'calibration': calibration,
                'grammar_adjustment': grammar_adjustment
            }
        )
        
        return final_score
    
    def _calculate_grammar_factors(self, sentence: str, suggestion: str,
                                 issue_type: str, metadata: Dict[str, Any]) -> ConfidenceFactors:
        """Calculate confidence factors for grammar suggestions"""
        factors = ConfidenceFactors()
        
        # Context score (more important for grammar)
        context_scores = self.context_analyzer.analyze_context(sentence, "", suggestion)
        factors.context_score = sum(context_scores.values()) / len(context_scores)
        
        # Domain score
        factors.domain_score = metadata.get('domain_validation_score', 0.5)
        
        # Layer reliability score
        layer = metadata.get('layer', DetectionLayer.TRADITIONAL_NLP)
        factors.layer_reliability_score = self.layer_tracker.get_layer_reliability(layer)
        
        # Issue type confidence (some grammar rules are more reliable)
        factors.edit_distance_score = self._get_issue_type_confidence(issue_type)
        
        # Sentence quality score
        factors.frequency_score = self._assess_sentence_quality(sentence)
        
        return factors
    
    def _get_issue_type_confidence(self, issue_type: str) -> float:
        """Get confidence based on grammar issue type"""
        issue_confidences = {
            'MORFOLOGIK_RULE_EN_US': 0.9,  # Spelling errors
            'EN_A_VS_AN': 0.95,            # Article errors
            'UPPERCASE_SENTENCE_START': 0.9, # Capitalization
            'COMMA_PARENTHESIS_WHITESPACE': 0.8, # Punctuation
            'DOUBLE_PUNCTUATION': 0.95,    # Double punctuation
            'WHITESPACE_RULE': 0.85,       # Whitespace issues
            'EN_COMPOUNDS': 0.7,           # Compound words
            'SENTENCE_FRAGMENT': 0.6,      # Sentence fragments
            'EN_UNPAIRED_BRACKETS': 0.9,   # Bracket matching
        }
        
        return issue_confidences.get(issue_type, 0.7)  # Default confidence
    
    def _assess_sentence_quality(self, sentence: str) -> float:
        """Assess overall quality of the sentence"""
        if not sentence:
            return 0.0
        
        quality_score = 1.0
        
        # Penalize very short sentences
        if len(sentence) < 10:
            quality_score *= 0.7
        
        # Penalize sentences with many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', sentence)) / len(sentence)
        if special_char_ratio > 0.1:
            quality_score *= (1 - special_char_ratio)
        
        # Bonus for proper sentence structure
        if sentence[0].isupper() and sentence.endswith(('.', '!', '?')):
            quality_score *= 1.1
        
        return min(1.0, quality_score)
    
    def _get_grammar_adjustment(self, issue_type: str, metadata: Dict[str, Any]) -> float:
        """Get grammar-specific confidence adjustment"""
        adjustment = 0.0
        
        # High-confidence grammar rules
        high_confidence_rules = [
            'EN_A_VS_AN', 'DOUBLE_PUNCTUATION', 'UPPERCASE_SENTENCE_START'
        ]
        
        if issue_type in high_confidence_rules:
            adjustment += 0.1
        
        # Resume context adjustment
        if metadata.get('is_resume_context', False):
            adjustment += 0.05
        
        return adjustment
    
    def calibrate_confidence(self, raw_score: float, layer: DetectionLayer,
                           historical_accuracy: float) -> float:
        """
        Calibrate confidence score based on historical performance.
        
        Args:
            raw_score: Raw confidence score (0-100)
            layer: Detection layer that generated the score
            historical_accuracy: Historical accuracy of the layer (0-1)
            
        Returns:
            Calibrated confidence score (0-100)
        """
        # Update layer reliability
        self.layer_tracker.record_suggestion_outcome(layer, True, raw_score)
        
        # Get calibration factor
        calibration = self.layer_tracker.get_confidence_calibration(layer)
        
        # Apply calibration
        calibrated_score = raw_score * calibration
        
        # Apply historical accuracy adjustment
        if historical_accuracy < 0.8:
            # Reduce confidence for historically inaccurate layers
            accuracy_penalty = (0.8 - historical_accuracy) * 0.5
            calibrated_score *= (1 - accuracy_penalty)
        elif historical_accuracy > 0.9:
            # Boost confidence for highly accurate layers
            accuracy_bonus = (historical_accuracy - 0.9) * 0.2
            calibrated_score *= (1 + accuracy_bonus)
        
        return max(0.0, min(100.0, calibrated_score))
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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
    
    def get_confidence_explanation(self, score: float, factors: ConfidenceFactors) -> str:
        """Generate human-readable explanation for confidence score"""
        level = self._get_confidence_level(score)
        
        explanations = []
        
        if factors.edit_distance_score > 0.8:
            explanations.append("very similar to original word")
        elif factors.edit_distance_score > 0.6:
            explanations.append("moderately similar to original word")
        else:
            explanations.append("significantly different from original word")
        
        if factors.frequency_score > 0.7:
            explanations.append("commonly used word")
        elif factors.frequency_score > 0.4:
            explanations.append("moderately common word")
        else:
            explanations.append("uncommon word")
        
        if factors.context_score > 0.7:
            explanations.append("fits well in context")
        elif factors.context_score > 0.4:
            explanations.append("reasonably fits context")
        else:
            explanations.append("may not fit context well")
        
        if factors.domain_score > 0.7:
            explanations.append("recognized domain term")
        
        explanation = f"{level} confidence ({score:.0f}%): " + ", ".join(explanations)
        return explanation
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level description"""
        if score >= 90:
            return "Very high"
        elif score >= 80:
            return "High"
        elif score >= 70:
            return "Medium"
        elif score >= 60:
            return "Low"
        else:
            return "Very low"
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scoring statistics"""
        stats = {}
        
        # Layer reliability statistics
        for layer in DetectionLayer:
            layer_stats = self.layer_tracker.layer_stats.get(layer, {})
            stats[f"{layer.value}_reliability"] = layer_stats.get('accuracy_rate', 0.0)
            stats[f"{layer.value}_calibration"] = layer_stats.get('confidence_calibration', 1.0)
            stats[f"{layer.value}_suggestions"] = layer_stats.get('total_suggestions', 0)
        
        return stats