"""
Multi-Layered Validation Pipeline
Cross-checks suggestions through multiple analysis methods
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from models import (
    TypoResult, GrammarResult, ValidatedTypoResult, ValidatedGrammarResult,
    ValidationStatus, ResumeContext
)
from domain_vocabulary import DomainVocabulary
from confidence_scorer import ConfidenceScorer

class ValidationMethod(Enum):
    """Different validation methods"""
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    CONTEXT_BASED = "context_based"
    DOMAIN_SPECIFIC = "domain_specific"

@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    confidence: float
    method: ValidationMethod
    explanation: str
    supporting_evidence: List[str]

@dataclass
class CrossValidationResult:
    """Result of cross-validation across multiple methods"""
    suggestion: Dict[str, Any]
    validation_results: List[ValidationResult]
    final_confidence: float
    consensus_score: float
    status: ValidationStatus

class ValidationPipeline:
    """Multi-layered validation system for text analysis suggestions"""
    
    def __init__(self):
        self.domain_vocab = DomainVocabulary()
        self.confidence_scorer = ConfidenceScorer()
        self.validation_threshold = 70.0
        self.consensus_threshold = 0.6  # 60% of methods must agree
        
    def validate_spelling_suggestion(self, typo: TypoResult, context: str,
                                   resume_context: Optional[ResumeContext] = None) -> CrossValidationResult:
        """Validate a spelling suggestion through multiple methods"""
        
        suggestion_dict = {
            'type': 'spelling',
            'original': typo.word,
            'suggestion': typo.suggestion,
            'position': typo.position,
            'context': context
        }
        
        # Run multiple validation methods
        validation_results = []
        
        # Method 1: Rule-based validation
        rule_result = self._validate_spelling_rule_based(typo, context)
        validation_results.append(rule_result)
        
        # Method 2: Statistical validation
        stat_result = self._validate_spelling_statistical(typo, context)
        validation_results.append(stat_result)
        
        # Method 3: Context-based validation
        context_result = self._validate_spelling_context_based(typo, context, resume_context)
        validation_results.append(context_result)
        
        # Method 4: Domain-specific validation
        domain_result = self._validate_spelling_domain_specific(typo, context, resume_context)
        validation_results.append(domain_result)
        
        # Calculate consensus and final confidence
        consensus_score = self._calculate_consensus(validation_results)
        final_confidence = self._calculate_final_confidence(validation_results, consensus_score)
        
        # Determine validation status
        status = self._determine_validation_status(validation_results, consensus_score, final_confidence)
        
        return CrossValidationResult(
            suggestion=suggestion_dict,
            validation_results=validation_results,
            final_confidence=final_confidence,
            consensus_score=consensus_score,
            status=status
        )
    
    def validate_grammar_suggestion(self, grammar: GrammarResult, context: str,
                                  resume_context: Optional[ResumeContext] = None) -> CrossValidationResult:
        """Validate a grammar suggestion through multiple methods"""
        
        suggestion_dict = {
            'type': 'grammar',
            'sentence': grammar.sentence,
            'suggestion': grammar.suggestion,
            'issue_type': grammar.issue_type,
            'position': grammar.position,
            'context': context
        }
        
        # Run multiple validation methods
        validation_results = []
        
        # Method 1: Rule-based validation
        rule_result = self._validate_grammar_rule_based(grammar, context)
        validation_results.append(rule_result)
        
        # Method 2: Context-based validation
        context_result = self._validate_grammar_context_based(grammar, context, resume_context)
        validation_results.append(context_result)
        
        # Method 3: Professional language validation
        professional_result = self._validate_grammar_professional(grammar, resume_context)
        validation_results.append(professional_result)
        
        # Method 4: Consistency validation
        consistency_result = self._validate_grammar_consistency(grammar, context, resume_context)
        validation_results.append(consistency_result)
        
        # Calculate consensus and final confidence
        consensus_score = self._calculate_consensus(validation_results)
        final_confidence = self._calculate_final_confidence(validation_results, consensus_score)
        
        # Determine validation status
        status = self._determine_validation_status(validation_results, consensus_score, final_confidence)
        
        return CrossValidationResult(
            suggestion=suggestion_dict,
            validation_results=validation_results,
            final_confidence=final_confidence,
            consensus_score=consensus_score,
            status=status
        )
    
    def cross_validate_suggestions(self, suggestions: List[Dict]) -> List[CrossValidationResult]:
        """Cross-validate multiple suggestions"""
        results = []
        
        for suggestion in suggestions:
            if suggestion.get('type') == 'spelling':
                # Convert to TypoResult for validation
                typo = TypoResult(
                    word=suggestion['original'],
                    suggestion=suggestion['suggestion'],
                    position=suggestion.get('position')
                )
                result = self.validate_spelling_suggestion(
                    typo, 
                    suggestion.get('context', ''),
                    suggestion.get('resume_context')
                )
                results.append(result)
                
            elif suggestion.get('type') == 'grammar':
                # Convert to GrammarResult for validation
                grammar = GrammarResult(
                    sentence=suggestion['sentence'],
                    suggestion=suggestion['suggestion'],
                    issue_type=suggestion['issue_type'],
                    position=suggestion.get('position')
                )
                result = self.validate_grammar_suggestion(
                    grammar,
                    suggestion.get('context', ''),
                    suggestion.get('resume_context')
                )
                results.append(result)
        
        return results
    
    def apply_conservative_filtering(self, suggestions: List[CrossValidationResult]) -> List[CrossValidationResult]:
        """Apply conservative filtering to minimize false positives"""
        
        filtered_suggestions = []
        
        for suggestion in suggestions:
            # Conservative criteria
            meets_threshold = suggestion.final_confidence >= self.validation_threshold
            has_consensus = suggestion.consensus_score >= self.consensus_threshold
            not_filtered = suggestion.status != ValidationStatus.FILTERED
            
            # Additional conservative checks
            has_strong_evidence = len([r for r in suggestion.validation_results if r.is_valid]) >= 2
            no_conflicting_evidence = not self._has_conflicting_evidence(suggestion.validation_results)
            
            if meets_threshold and has_consensus and not_filtered and has_strong_evidence and no_conflicting_evidence:
                suggestion.status = ValidationStatus.CROSS_VALIDATED
                filtered_suggestions.append(suggestion)
            else:
                suggestion.status = ValidationStatus.FILTERED
        
        return filtered_suggestions
    
    # Spelling validation methods
    
    def _validate_spelling_rule_based(self, typo: TypoResult, context: str) -> ValidationResult:
        """Rule-based spelling validation"""
        
        original = typo.word.lower()
        suggestion = typo.suggestion.lower()
        
        # Check edit distance
        edit_distance = self._calculate_edit_distance(original, suggestion)
        max_length = max(len(original), len(suggestion))
        
        # Rule: Edit distance should be reasonable
        if edit_distance > max_length * 0.5:
            return ValidationResult(
                is_valid=False,
                confidence=20.0,
                method=ValidationMethod.RULE_BASED,
                explanation="Edit distance too large for likely typo",
                supporting_evidence=[f"Edit distance: {edit_distance}, Max length: {max_length}"]
            )
        
        # Rule: Length difference should be reasonable
        length_diff = abs(len(original) - len(suggestion))
        if length_diff > 3:
            return ValidationResult(
                is_valid=False,
                confidence=30.0,
                method=ValidationMethod.RULE_BASED,
                explanation="Length difference too large",
                supporting_evidence=[f"Length difference: {length_diff}"]
            )
        
        # Rule: Should share common characters
        common_chars = set(original) & set(suggestion)
        if len(common_chars) < min(len(original), len(suggestion)) * 0.5:
            return ValidationResult(
                is_valid=False,
                confidence=25.0,
                method=ValidationMethod.RULE_BASED,
                explanation="Insufficient character overlap",
                supporting_evidence=[f"Common characters: {len(common_chars)}"]
            )
        
        # Passed all rules
        confidence = 85.0 - (edit_distance * 10) - (length_diff * 5)
        return ValidationResult(
            is_valid=True,
            confidence=max(50.0, confidence),
            method=ValidationMethod.RULE_BASED,
            explanation="Passes basic spelling correction rules",
            supporting_evidence=[
                f"Edit distance: {edit_distance}",
                f"Length difference: {length_diff}",
                f"Character overlap: {len(common_chars)}"
            ]
        )
    
    def _validate_spelling_statistical(self, typo: TypoResult, context: str) -> ValidationResult:
        """Statistical spelling validation"""
        
        original = typo.word
        suggestion = typo.suggestion
        
        # Check if suggestion is a common word
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use',
            # Professional words
            'experience', 'development', 'management', 'analysis', 'project', 'team', 'work', 'business', 'company', 'service', 'system', 'process', 'technology', 'software', 'application', 'solution', 'customer', 'client', 'support', 'quality'
        }
        
        suggestion_lower = suggestion.lower()
        
        # Higher confidence for common words
        if suggestion_lower in common_words:
            confidence = 80.0
            explanation = "Suggestion is a common word"
            evidence = ["Common word validation"]
        else:
            # Check word patterns
            if self._has_valid_word_pattern(suggestion):
                confidence = 65.0
                explanation = "Suggestion follows valid word patterns"
                evidence = ["Valid word pattern"]
            else:
                confidence = 40.0
                explanation = "Suggestion is uncommon and doesn't follow standard patterns"
                evidence = ["Uncommon word"]
        
        return ValidationResult(
            is_valid=confidence > 50.0,
            confidence=confidence,
            method=ValidationMethod.STATISTICAL,
            explanation=explanation,
            supporting_evidence=evidence
        )
    
    def _validate_spelling_context_based(self, typo: TypoResult, context: str,
                                       resume_context: Optional[ResumeContext]) -> ValidationResult:
        """Context-based spelling validation"""
        
        suggestion = typo.suggestion.lower()
        context_lower = context.lower()
        
        # Check if suggestion fits the context
        context_words = set(context_lower.split())
        
        # Look for related words in context
        related_score = 0
        evidence = []
        
        # Check for semantic relationships
        if 'develop' in context_lower and 'development' in suggestion:
            related_score += 30
            evidence.append("Semantic relationship with 'develop'")
        
        if 'manage' in context_lower and 'management' in suggestion:
            related_score += 30
            evidence.append("Semantic relationship with 'manage'")
        
        # Check for technical context
        tech_indicators = ['software', 'programming', 'code', 'system', 'application']
        if any(indicator in context_lower for indicator in tech_indicators):
            if self.domain_vocab.is_valid_technical_term(suggestion, context):
                related_score += 40
                evidence.append("Valid technical term in technical context")
        
        # Check resume section context
        if resume_context and resume_context.sections:
            section_boost = self._get_section_context_boost(suggestion, resume_context)
            related_score += section_boost
            if section_boost > 0:
                evidence.append(f"Appropriate for resume section context (+{section_boost})")
        
        confidence = min(90.0, 40.0 + related_score)
        
        return ValidationResult(
            is_valid=confidence > 60.0,
            confidence=confidence,
            method=ValidationMethod.CONTEXT_BASED,
            explanation="Context-based validation of spelling suggestion",
            supporting_evidence=evidence if evidence else ["No strong contextual evidence"]
        )
    
    def _validate_spelling_domain_specific(self, typo: TypoResult, context: str,
                                         resume_context: Optional[ResumeContext]) -> ValidationResult:
        """Domain-specific spelling validation"""
        
        suggestion = typo.suggestion
        
        # Check against domain vocabulary
        if self.domain_vocab.is_valid_technical_term(suggestion, context):
            return ValidationResult(
                is_valid=True,
                confidence=95.0,
                method=ValidationMethod.DOMAIN_SPECIFIC,
                explanation="Valid technical term in domain vocabulary",
                supporting_evidence=["Domain vocabulary match"]
            )
        
        # Check company names
        if suggestion.lower() in self.domain_vocab.company_names:
            return ValidationResult(
                is_valid=True,
                confidence=90.0,
                method=ValidationMethod.DOMAIN_SPECIFIC,
                explanation="Valid company name",
                supporting_evidence=["Company name match"]
            )
        
        # Check certifications
        if suggestion.lower() in self.domain_vocab.certifications:
            return ValidationResult(
                is_valid=True,
                confidence=85.0,
                method=ValidationMethod.DOMAIN_SPECIFIC,
                explanation="Valid certification term",
                supporting_evidence=["Certification match"]
            )
        
        # Check professional phrases
        professional_phrases = self.domain_vocab.professional_phrases
        for category, phrases in professional_phrases.items():
            if any(phrase in suggestion.lower() for phrase in phrases):
                return ValidationResult(
                    is_valid=True,
                    confidence=75.0,
                    method=ValidationMethod.DOMAIN_SPECIFIC,
                    explanation=f"Valid professional phrase ({category})",
                    supporting_evidence=[f"Professional phrase match: {category}"]
                )
        
        # Not found in domain vocabulary
        return ValidationResult(
            is_valid=False,
            confidence=30.0,
            method=ValidationMethod.DOMAIN_SPECIFIC,
            explanation="Not found in domain-specific vocabulary",
            supporting_evidence=["No domain match"]
        )
    
    # Grammar validation methods
    
    def _validate_grammar_rule_based(self, grammar: GrammarResult, context: str) -> ValidationResult:
        """Rule-based grammar validation"""
        
        issue_type = grammar.issue_type
        sentence = grammar.sentence.lower()
        suggestion = grammar.suggestion.lower()
        
        # High-confidence grammar rules
        high_confidence_rules = {
            'Subject-verb disagreement': 90.0,
            'Plural form needed': 85.0,
            'Article error': 80.0
        }
        
        if issue_type in high_confidence_rules:
            confidence = high_confidence_rules[issue_type]
            
            # Additional validation for specific rules
            if issue_type == 'Subject-verb disagreement':
                if 'i has' in sentence or 'i was' in sentence:
                    confidence = 95.0
                    evidence = ["Clear subject-verb disagreement detected"]
                else:
                    evidence = ["Subject-verb rule applied"]
            
            elif issue_type == 'Plural form needed':
                if re.search(r'\d+\s+year\s+', sentence):
                    confidence = 90.0
                    evidence = ["Number-noun agreement error detected"]
                else:
                    evidence = ["Plural form rule applied"]
            
            else:
                evidence = [f"High-confidence rule: {issue_type}"]
            
            return ValidationResult(
                is_valid=True,
                confidence=confidence,
                method=ValidationMethod.RULE_BASED,
                explanation=f"High-confidence grammar rule: {issue_type}",
                supporting_evidence=evidence
            )
        
        # Medium-confidence rules
        medium_confidence_rules = {
            'Pronoun case error': 70.0,
            'Preposition error': 65.0,
            'Tense consistency': 60.0
        }
        
        if issue_type in medium_confidence_rules:
            return ValidationResult(
                is_valid=True,
                confidence=medium_confidence_rules[issue_type],
                method=ValidationMethod.RULE_BASED,
                explanation=f"Medium-confidence grammar rule: {issue_type}",
                supporting_evidence=[f"Grammar rule: {issue_type}"]
            )
        
        # Low-confidence or unknown rules
        return ValidationResult(
            is_valid=False,
            confidence=40.0,
            method=ValidationMethod.RULE_BASED,
            explanation=f"Low-confidence or unknown grammar rule: {issue_type}",
            supporting_evidence=["Uncertain grammar rule"]
        )
    
    def _validate_grammar_context_based(self, grammar: GrammarResult, context: str,
                                      resume_context: Optional[ResumeContext]) -> ValidationResult:
        """Context-based grammar validation"""
        
        sentence = grammar.sentence
        suggestion = grammar.suggestion
        
        # Check if suggestion maintains professional tone
        professional_indicators = [
            'responsible for', 'managed', 'led', 'developed', 'implemented',
            'collaborated', 'achieved', 'improved', 'optimized'
        ]
        
        maintains_professional_tone = any(
            indicator in suggestion.lower() for indicator in professional_indicators
        )
        
        # Check section-appropriate language
        section_appropriate = True
        if resume_context and resume_context.sections:
            # Experience section should use past tense
            if 'experience' in resume_context.sections:
                if not self._uses_past_tense(suggestion):
                    section_appropriate = False
        
        confidence = 60.0
        evidence = []
        
        if maintains_professional_tone:
            confidence += 20
            evidence.append("Maintains professional tone")
        
        if section_appropriate:
            confidence += 15
            evidence.append("Section-appropriate language")
        else:
            confidence -= 20
            evidence.append("May not be section-appropriate")
        
        return ValidationResult(
            is_valid=confidence > 60.0,
            confidence=confidence,
            method=ValidationMethod.CONTEXT_BASED,
            explanation="Context-based grammar validation",
            supporting_evidence=evidence if evidence else ["No strong contextual indicators"]
        )
    
    def _validate_grammar_professional(self, grammar: GrammarResult,
                                     resume_context: Optional[ResumeContext]) -> ValidationResult:
        """Professional language validation"""
        
        suggestion = grammar.suggestion.lower()
        
        # Professional language patterns
        professional_patterns = [
            r'\b(?:achieved|accomplished|administered|analyzed|collaborated|coordinated|developed|enhanced|established|executed|facilitated|implemented|improved|increased|led|managed|optimized|organized|planned|reduced|streamlined|supervised)\b',
            r'\b\d+%\s+(?:increase|improvement|reduction)\b',
            r'\b(?:cross-functional|stakeholder|deliverable|methodology|framework)\b'
        ]
        
        professional_score = 0
        evidence = []
        
        for pattern in professional_patterns:
            if re.search(pattern, suggestion):
                professional_score += 25
                evidence.append(f"Professional pattern: {pattern}")
        
        # Check for quantified achievements
        if re.search(r'\d+(?:%|\$|k)', suggestion):
            professional_score += 20
            evidence.append("Contains quantified achievement")
        
        # Check industry appropriateness
        if resume_context and resume_context.industry_indicators:
            for industry in resume_context.industry_indicators:
                if industry in ['software_development', 'data_science']:
                    # Technical industries - check for technical language
                    tech_terms = ['system', 'application', 'platform', 'solution', 'technology']
                    if any(term in suggestion for term in tech_terms):
                        professional_score += 15
                        evidence.append("Industry-appropriate technical language")
        
        confidence = min(90.0, 40.0 + professional_score)
        
        return ValidationResult(
            is_valid=confidence > 60.0,
            confidence=confidence,
            method=ValidationMethod.DOMAIN_SPECIFIC,
            explanation="Professional language validation",
            supporting_evidence=evidence if evidence else ["No strong professional indicators"]
        )
    
    def _validate_grammar_consistency(self, grammar: GrammarResult, context: str,
                                    resume_context: Optional[ResumeContext]) -> ValidationResult:
        """Consistency validation within resume context"""
        
        suggestion = grammar.suggestion
        
        # Check tense consistency
        tense_consistent = True
        evidence = []
        
        if resume_context and resume_context.sections:
            if 'experience' in resume_context.sections:
                experience_content = resume_context.sections['experience'].content
                
                # Check if suggestion maintains tense consistency with experience section
                if self._uses_past_tense(experience_content) and not self._uses_past_tense(suggestion):
                    tense_consistent = False
                    evidence.append("Tense inconsistent with experience section")
                elif self._uses_past_tense(suggestion):
                    evidence.append("Tense consistent with experience section")
        
        # Check style consistency
        style_consistent = True
        # This could be expanded to check bullet point styles, formatting, etc.
        
        confidence = 70.0
        if tense_consistent:
            confidence += 15
        else:
            confidence -= 25
        
        if style_consistent:
            confidence += 10
        
        return ValidationResult(
            is_valid=confidence > 60.0,
            confidence=confidence,
            method=ValidationMethod.CONTEXT_BASED,
            explanation="Consistency validation within resume",
            supporting_evidence=evidence if evidence else ["No consistency issues detected"]
        )
    
    # Helper methods
    
    def _calculate_consensus(self, validation_results: List[ValidationResult]) -> float:
        """Calculate consensus score across validation methods"""
        if not validation_results:
            return 0.0
        
        valid_count = sum(1 for result in validation_results if result.is_valid)
        return valid_count / len(validation_results)
    
    def _calculate_final_confidence(self, validation_results: List[ValidationResult],
                                  consensus_score: float) -> float:
        """Calculate final confidence score"""
        if not validation_results:
            return 0.0
        
        # Weight by validation method reliability
        method_weights = {
            ValidationMethod.RULE_BASED: 0.3,
            ValidationMethod.STATISTICAL: 0.2,
            ValidationMethod.CONTEXT_BASED: 0.25,
            ValidationMethod.DOMAIN_SPECIFIC: 0.25
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            weight = method_weights.get(result.method, 0.2)
            if result.is_valid:
                weighted_sum += result.confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_sum / total_weight
        
        # Apply consensus boost/penalty
        consensus_factor = 0.5 + (consensus_score * 0.5)  # 0.5 to 1.0
        
        return base_confidence * consensus_factor
    
    def _determine_validation_status(self, validation_results: List[ValidationResult],
                                   consensus_score: float, final_confidence: float) -> ValidationStatus:
        """Determine final validation status"""
        
        if final_confidence < 50.0:
            return ValidationStatus.FILTERED
        
        if consensus_score >= self.consensus_threshold and final_confidence >= self.validation_threshold:
            return ValidationStatus.CROSS_VALIDATED
        
        if final_confidence >= 60.0:
            return ValidationStatus.VALIDATED
        
        return ValidationStatus.PENDING
    
    def _has_conflicting_evidence(self, validation_results: List[ValidationResult]) -> bool:
        """Check if validation results have conflicting evidence"""
        
        valid_results = [r for r in validation_results if r.is_valid]
        invalid_results = [r for r in validation_results if not r.is_valid]
        
        # If we have both high-confidence valid and invalid results, there's conflict
        high_conf_valid = any(r.confidence > 80.0 for r in valid_results)
        high_conf_invalid = any(r.confidence > 80.0 for r in invalid_results)
        
        return high_conf_valid and high_conf_invalid
    
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
    
    def _has_valid_word_pattern(self, word: str) -> bool:
        """Check if word follows valid English word patterns"""
        
        # Basic English word patterns
        patterns = [
            r'^[a-z]+$',  # All lowercase letters
            r'^[A-Z][a-z]+$',  # Title case
            r'^[a-z]+-[a-z]+$',  # Hyphenated words
            r'^[a-z]+ing$',  # -ing ending
            r'^[a-z]+ed$',  # -ed ending
            r'^[a-z]+tion$',  # -tion ending
            r'^[a-z]+ment$',  # -ment ending
        ]
        
        for pattern in patterns:
            if re.match(pattern, word):
                return True
        
        return False
    
    def _get_section_context_boost(self, suggestion: str, resume_context: ResumeContext) -> int:
        """Get context boost based on resume section"""
        
        suggestion_lower = suggestion.lower()
        boost = 0
        
        # Skills section context
        if 'skills' in resume_context.sections:
            if self.domain_vocab.is_valid_technical_term(suggestion):
                boost += 20
        
        # Experience section context
        if 'experience' in resume_context.sections:
            professional_terms = ['management', 'development', 'analysis', 'implementation']
            if any(term in suggestion_lower for term in professional_terms):
                boost += 15
        
        # Education section context
        if 'education' in resume_context.sections:
            education_terms = ['degree', 'university', 'college', 'bachelor', 'master']
            if any(term in suggestion_lower for term in education_terms):
                boost += 10
        
        return boost
    
    def _uses_past_tense(self, text: str) -> bool:
        """Check if text primarily uses past tense"""
        
        past_tense_indicators = [
            'managed', 'led', 'developed', 'created', 'implemented', 'designed',
            'built', 'established', 'coordinated', 'supervised', 'achieved'
        ]
        
        past_tense_patterns = [
            r'\b\w+ed\b',  # Regular past tense
            r'\b(?:was|were|had|did)\b'  # Past tense auxiliaries
        ]
        
        text_lower = text.lower()
        
        # Count past tense indicators
        past_count = sum(1 for indicator in past_tense_indicators if indicator in text_lower)
        
        # Count past tense patterns
        for pattern in past_tense_patterns:
            past_count += len(re.findall(pattern, text_lower))
        
        # Count total verbs (simplified)
        total_verbs = len(re.findall(r'\b\w+(?:ed|ing|s)?\b', text_lower))
        
        if total_verbs == 0:
            return False
        
        return (past_count / total_verbs) > 0.3  # 30% threshold for past tense