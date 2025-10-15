"""
Enhanced traditional NLP detection layer with optimizations and robust fallbacks.
Implements intelligent spell checking and grammar analysis with domain awareness.
"""

import time
import logging
import re
import subprocess
import sys
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from enhanced_models import (
    LayeredDetectionResult, DetectionLayer, AnalysisConfig, 
    LayerPerformanceMetrics, ErrorType
)
from core_interfaces import IDetectionLayer
from confidence_scoring import AdvancedConfidenceScorer
from domain_vocabulary import EnhancedDomainValidator

logger = logging.getLogger(__name__)

@dataclass
class TraditionalNLPConfig:
    """Configuration for traditional NLP layer"""
    enable_spell_checking: bool = True
    enable_grammar_checking: bool = True
    java_timeout: float = 10.0
    spell_check_timeout: float = 5.0
    confidence_threshold: float = 70.0
    max_suggestions: int = 3
    use_custom_dictionary: bool = True
    enable_context_filtering: bool = True
    parallel_processing: bool = True

class JavaDependencyManager:
    """Manages Java dependency for language-tool-python"""
    
    def __init__(self):
        self.java_available = None
        self.java_version = None
        self.language_tool = None
        self.initialization_lock = threading.Lock()
    
    def check_java_availability(self) -> bool:
        """Check if Java is available"""
        if self.java_available is not None:
            return self.java_available
        
        try:
            result = subprocess.run(
                ['java', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=5.0
            )
            
            if result.returncode == 0:
                self.java_available = True
                # Extract Java version
                version_output = result.stderr or result.stdout
                version_match = re.search(r'version "([^"]+)"', version_output)
                if version_match:
                    self.java_version = version_match.group(1)
                
                logger.info(f"Java detected: version {self.java_version}")
                return True
            else:
                self.java_available = False
                logger.warning("Java not available: command failed")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            self.java_available = False
            logger.warning(f"Java not available: {e}")
            return False
    
    def get_language_tool(self, timeout: float = 10.0) -> Optional[Any]:
        """Get language tool instance with timeout"""
        if not self.check_java_availability():
            return None
        
        with self.initialization_lock:
            if self.language_tool is not None:
                return self.language_tool
            
            try:
                import language_tool_python
                
                # Initialize with timeout
                def init_tool():
                    return language_tool_python.LanguageTool('en-US')
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(init_tool)
                    self.language_tool = future.result(timeout=timeout)
                
                logger.info("LanguageTool initialized successfully")
                return self.language_tool
                
            except TimeoutError:
                logger.error("LanguageTool initialization timed out")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize LanguageTool: {e}")
                return None
    
    def cleanup(self):
        """Clean up language tool resources"""
        if self.language_tool:
            try:
                self.language_tool.close()
            except:
                pass
            self.language_tool = None

class EnhancedSpellChecker:
    """Enhanced spell checker with domain awareness and optimization"""
    
    def __init__(self, domain_validator: EnhancedDomainValidator):
        self.domain_validator = domain_validator
        self.spell_checker = None
        self.custom_words = set()
        self.ignore_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w*\d+\w*\b',  # Words with numbers
            r'\b[a-z]+\.[a-z]+\b',  # File extensions
            r'\b\w+@\w+\.\w+\b',  # Email addresses
            r'\bhttps?://\S+\b',  # URLs
        ]
        self._initialize_spell_checker()
    
    def _initialize_spell_checker(self):
        """Initialize spell checker with optimizations"""
        try:
            from spellchecker import SpellChecker
            
            self.spell_checker = SpellChecker()
            
            # Add technical vocabulary
            technical_terms = self._get_technical_vocabulary()
            self.spell_checker.word_frequency.load_words(technical_terms)
            self.custom_words.update(technical_terms)
            
            logger.info(f"Spell checker initialized with {len(technical_terms)} technical terms")
            
        except ImportError:
            logger.error("pyspellchecker not available")
            self.spell_checker = None
        except Exception as e:
            logger.error(f"Failed to initialize spell checker: {e}")
            self.spell_checker = None
    
    def _get_technical_vocabulary(self) -> Set[str]:
        """Get comprehensive technical vocabulary"""
        technical_terms = set()
        
        # Get terms from domain validator
        if hasattr(self.domain_validator, 'vocabulary'):
            vocab = self.domain_validator.vocabulary
            
            # Add technical terms
            for term_obj in vocab.technical_terms.values():
                technical_terms.add(term_obj.term.lower())
                technical_terms.update(v.lower() for v in term_obj.variations)
            
            # Add frameworks and tools
            for tool_obj in vocab.frameworks_and_tools.values():
                technical_terms.add(tool_obj.term.lower())
                technical_terms.update(v.lower() for v in tool_obj.variations)
            
            # Add programming languages
            for lang_obj in vocab.programming_languages.values():
                technical_terms.add(lang_obj.term.lower())
                technical_terms.update(v.lower() for v in lang_obj.variations)
            
            # Add company names
            technical_terms.update(name.lower() for name in vocab.company_names)
        
        # Add common resume terms
        resume_terms = {
            'javascript', 'typescript', 'python', 'java', 'csharp', 'cplusplus',
            'html', 'css', 'scss', 'react', 'angular', 'vue', 'nodejs',
            'express', 'django', 'flask', 'fastapi', 'spring', 'laravel',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'jenkins', 'gitlab', 'github', 'jira', 'confluence',
            'agile', 'scrum', 'kanban', 'devops', 'cicd', 'api', 'rest',
            'graphql', 'microservices', 'serverless', 'frontend', 'backend',
            'fullstack', 'ui', 'ux', 'responsive', 'mobile', 'ios', 'android'
        }
        
        technical_terms.update(resume_terms)
        return technical_terms
    
    def check_spelling(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """Check spelling with context awareness"""
        if not self.spell_checker:
            return []
        
        spelling_errors = []
        
        # Extract words with positions
        word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        
        for match in word_pattern.finditer(text):
            word = match.group()
            word_lower = word.lower()
            position = match.start()
            
            # Skip if word should be ignored
            if self._should_ignore_word(word, text, position):
                continue
            
            # Check if word is misspelled
            if word_lower not in self.spell_checker:
                # Get suggestions
                suggestions = list(self.spell_checker.candidates(word_lower))
                
                if suggestions:
                    # Filter and rank suggestions
                    filtered_suggestions = self._filter_suggestions(
                        word, suggestions, context
                    )
                    
                    if filtered_suggestions:
                        spelling_errors.append({
                            'word': word,
                            'position': position,
                            'suggestions': filtered_suggestions[:3],  # Top 3
                            'context': self._extract_context(text, position, 50)
                        })
        
        return spelling_errors
    
    def _should_ignore_word(self, word: str, text: str, position: int) -> bool:
        """Determine if a word should be ignored"""
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if re.match(pattern, word):
                return True
        
        # Skip very short words
        if len(word) <= 2:
            return True
        
        # Check if it's a technical term
        if self.domain_validator.is_valid_technical_term(word, text):
            return True
        
        # Skip if it's in custom words
        if word.lower() in self.custom_words:
            return True
        
        # Skip proper nouns in certain contexts
        if word[0].isupper() and self._is_proper_noun_context(text, position):
            return True
        
        return False
    
    def _is_proper_noun_context(self, text: str, position: int) -> bool:
        """Check if position is in a proper noun context"""
        # Simple heuristic: check if preceded by common proper noun indicators
        context_before = text[max(0, position-20):position].lower()
        
        proper_noun_indicators = [
            'at ', 'from ', 'with ', 'by ', 'company', 'university',
            'college', 'inc', 'corp', 'ltd', 'llc'
        ]
        
        return any(indicator in context_before for indicator in proper_noun_indicators)
    
    def _filter_suggestions(self, original: str, suggestions: List[str], 
                          context: str) -> List[str]:
        """Filter and rank suggestions based on context"""
        if not suggestions:
            return []
        
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = 0.0
            
            # Edit distance score
            edit_distance = self._levenshtein_distance(original.lower(), suggestion.lower())
            max_len = max(len(original), len(suggestion))
            if max_len > 0:
                similarity = 1 - (edit_distance / max_len)
                score += similarity * 40
            
            # Length similarity
            length_diff = abs(len(original) - len(suggestion))
            length_score = max(0, 20 - length_diff * 2)
            score += length_score
            
            # Context relevance
            if self.domain_validator.is_valid_technical_term(suggestion, context):
                score += 30
            
            # Frequency in spell checker
            if hasattr(self.spell_checker, 'word_frequency'):
                try:
                    freq = self.spell_checker.word_frequency.get(suggestion.lower(), 0)
                    score += min(10, freq / 1000)  # Normalize frequency
                except AttributeError:
                    # Fallback if get method not available
                    score += 5  # Default frequency score
            
            scored_suggestions.append((suggestion, score))
        
        # Sort by score and return top suggestions
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return [suggestion for suggestion, _ in scored_suggestions]
    
    def _extract_context(self, text: str, position: int, window: int) -> str:
        """Extract context around position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end]
    
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

class EnhancedGrammarChecker:
    """Enhanced grammar checker with fallback strategies"""
    
    def __init__(self, java_manager: JavaDependencyManager):
        self.java_manager = java_manager
        self.fallback_patterns = self._initialize_fallback_patterns()
    
    def _initialize_fallback_patterns(self) -> List[Dict[str, Any]]:
        """Initialize fallback grammar patterns for when Java is not available"""
        return [
            {
                'pattern': r'\bI has\b',
                'replacement': 'I have',
                'explanation': 'Subject-verb agreement: Use "I have" instead of "I has"',
                'confidence': 95.0,
                'category': 'subject_verb_agreement'
            },
            {
                'pattern': r'\b(\d+)\s+year\s+(?:of\s+)?experience\b',
                'replacement': r'\1 years of experience',
                'explanation': 'Use plural "years" when the number is greater than 1',
                'confidence': 90.0,
                'category': 'plural_form'
            },
            {
                'pattern': r'\bMe and\s+\w+\s+(?:have|has|will|can|are)\b',
                'replacement': 'My colleague and I',
                'explanation': 'Use "My colleague and I" as the subject of a sentence',
                'confidence': 85.0,
                'category': 'pronoun_case'
            },
            {
                'pattern': r'\bThis are\b',
                'replacement': 'These are',
                'explanation': 'Use "These are" for plural subjects',
                'confidence': 90.0,
                'category': 'subject_verb_agreement'
            },
            {
                'pattern': r'\bThat are\b',
                'replacement': 'Those are',
                'explanation': 'Use "Those are" for plural subjects',
                'confidence': 90.0,
                'category': 'subject_verb_agreement'
            },
            {
                'pattern': r'\ban\s+(?=[bcdfghjklmnpqrstvwxyz])',
                'replacement': 'a',
                'explanation': 'Use "a" before consonant sounds',
                'confidence': 85.0,
                'category': 'article_usage'
            },
            {
                'pattern': r'\ba\s+(?=[aeiou])',
                'replacement': 'an',
                'explanation': 'Use "an" before vowel sounds',
                'confidence': 85.0,
                'category': 'article_usage'
            },
            {
                'pattern': r'\bresponsible to\b',
                'replacement': 'responsible for',
                'explanation': 'Use "responsible for" in professional contexts',
                'confidence': 80.0,
                'category': 'preposition_usage'
            },
            {
                'pattern': r'\bexperience on\b',
                'replacement': 'experience in',
                'explanation': 'Use "experience in" when describing expertise areas',
                'confidence': 75.0,
                'category': 'preposition_usage'
            },
            {
                'pattern': r'\b(\w+)\s+\1\b',
                'replacement': r'\1',
                'explanation': 'Remove duplicate words',
                'confidence': 95.0,
                'category': 'duplicate_words'
            }
        ]
    
    def check_grammar(self, text: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """Check grammar with fallback to pattern-based checking"""
        # Try LanguageTool first if Java is available
        language_tool = self.java_manager.get_language_tool(timeout)
        
        if language_tool:
            return self._check_with_language_tool(text, language_tool)
        else:
            logger.info("Using fallback grammar checking (Java not available)")
            return self._check_with_patterns(text)
    
    def _check_with_language_tool(self, text: str, language_tool: Any) -> List[Dict[str, Any]]:
        """Check grammar using LanguageTool"""
        try:
            matches = language_tool.check(text)
            grammar_issues = []
            
            for match in matches:
                # Filter out low-confidence matches
                if hasattr(match, 'category') and match.category in ['TYPOS', 'CASING']:
                    continue  # Skip spelling issues (handled by spell checker)
                
                # Extract context
                start = max(0, match.offset - 30)
                end = min(len(text), match.offset + match.errorLength + 30)
                context = text[start:end].strip()
                
                # Get suggestion
                suggestion = match.replacements[0] if match.replacements else "No suggestion"
                
                grammar_issues.append({
                    'context': context,
                    'position': match.offset,
                    'length': match.errorLength,
                    'suggestion': suggestion,
                    'explanation': match.message,
                    'confidence': self._calculate_language_tool_confidence(match),
                    'category': getattr(match, 'category', 'grammar'),
                    'rule_id': getattr(match, 'ruleId', 'unknown')
                })
            
            return grammar_issues
            
        except Exception as e:
            logger.error(f"LanguageTool grammar check failed: {e}")
            return self._check_with_patterns(text)
    
    def _check_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Check grammar using fallback patterns"""
        grammar_issues = []
        
        for pattern_info in self.fallback_patterns:
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Extract context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                # Create suggestion
                if r'\1' in pattern_info['replacement']:
                    # Handle regex group replacements
                    suggestion = re.sub(pattern, pattern_info['replacement'], 
                                      match.group(), flags=re.IGNORECASE)
                else:
                    suggestion = pattern_info['replacement']
                
                grammar_issues.append({
                    'context': context,
                    'position': match.start(),
                    'length': match.end() - match.start(),
                    'suggestion': suggestion,
                    'explanation': pattern_info['explanation'],
                    'confidence': pattern_info['confidence'],
                    'category': pattern_info['category'],
                    'rule_id': f"fallback_{pattern_info['category']}"
                })
        
        return grammar_issues
    
    def _calculate_language_tool_confidence(self, match: Any) -> float:
        """Calculate confidence for LanguageTool matches"""
        # Base confidence based on rule category
        category_confidence = {
            'GRAMMAR': 85.0,
            'STYLE': 70.0,
            'PUNCTUATION': 80.0,
            'TYPOGRAPHY': 75.0,
            'REDUNDANCY': 80.0,
            'LOGIC': 90.0,
            'MISC': 60.0
        }
        
        base_confidence = category_confidence.get(
            getattr(match, 'category', 'MISC'), 70.0
        )
        
        # Adjust based on rule specificity
        if hasattr(match, 'ruleId'):
            rule_id = match.ruleId
            
            # High confidence rules
            if any(high_conf in rule_id for high_conf in [
                'EN_A_VS_AN', 'DOUBLE_PUNCTUATION', 'UPPERCASE_SENTENCE_START'
            ]):
                base_confidence += 10
            
            # Lower confidence rules
            elif any(low_conf in rule_id for low_conf in [
                'SENTENCE_FRAGMENT', 'COMMA_PARENTHESIS'
            ]):
                base_confidence -= 10
        
        return max(50.0, min(95.0, base_confidence))

class EnhancedTraditionalNLPLayer(IDetectionLayer):
    """
    Enhanced traditional NLP layer with optimizations and robust fallbacks.
    """
    
    def __init__(self, config: Optional[TraditionalNLPConfig] = None):
        self.config = config or TraditionalNLPConfig()
        
        # Initialize components
        self.domain_validator = EnhancedDomainValidator()
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.java_manager = JavaDependencyManager()
        
        self.spell_checker = EnhancedSpellChecker(self.domain_validator)
        self.grammar_checker = EnhancedGrammarChecker(self.java_manager)
        
        # Performance tracking
        self.performance_metrics = LayerPerformanceMetrics(
            layer_name=DetectionLayer.TRADITIONAL_NLP
        )
        
        logger.info("Enhanced traditional NLP layer initialized")
    
    def detect(self, text: str, config: AnalysisConfig) -> List[LayeredDetectionResult]:
        """
        Detect errors using enhanced traditional NLP methods.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            List of detection results
        """
        start_time = time.time()
        results = []
        
        try:
            # Parallel processing if enabled
            if self.config.parallel_processing:
                results = self._detect_parallel(text, config)
            else:
                results = self._detect_sequential(text, config)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics.total_processed += 1
            self.performance_metrics.total_processing_time += processing_time
            
            # Calculate success rate
            self.performance_metrics.success_rate = (
                self.performance_metrics.total_processed - self.performance_metrics.error_count
            ) / self.performance_metrics.total_processed
            
            # Update processing time for results
            for result in results:
                result.processing_time = processing_time / len(results) if results else processing_time
            
            logger.debug(
                f"Traditional NLP detection completed: {len(results)} results in {processing_time:.3f}s",
                extra={
                    'text_length': len(text),
                    'results_count': len(results),
                    'processing_time': processing_time
                }
            )
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_metrics.error_count += 1
            
            logger.error(
                f"Traditional NLP detection failed: {e}",
                extra={
                    'text_length': len(text),
                    'processing_time': processing_time,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            
            return []
    
    def _detect_parallel(self, text: str, config: AnalysisConfig) -> List[LayeredDetectionResult]:
        """Detect errors using parallel processing"""
        results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            # Submit spell checking
            if self.config.enable_spell_checking:
                futures.append(
                    executor.submit(self._check_spelling_with_timeout, text)
                )
            
            # Submit grammar checking
            if self.config.enable_grammar_checking:
                futures.append(
                    executor.submit(self._check_grammar_with_timeout, text)
                )
            
            # Collect results
            for future in futures:
                try:
                    future_results = future.result(timeout=max(
                        self.config.spell_check_timeout,
                        self.config.java_timeout
                    ))
                    results.extend(future_results)
                except TimeoutError:
                    logger.warning("NLP detection task timed out")
                except Exception as e:
                    logger.error(f"NLP detection task failed: {e}")
        
        return results
    
    def _detect_sequential(self, text: str, config: AnalysisConfig) -> List[LayeredDetectionResult]:
        """Detect errors using sequential processing"""
        results = []
        
        # Spell checking
        if self.config.enable_spell_checking:
            try:
                spelling_results = self._check_spelling_with_timeout(text)
                results.extend(spelling_results)
            except Exception as e:
                logger.error(f"Spell checking failed: {e}")
        
        # Grammar checking
        if self.config.enable_grammar_checking:
            try:
                grammar_results = self._check_grammar_with_timeout(text)
                results.extend(grammar_results)
            except Exception as e:
                logger.error(f"Grammar checking failed: {e}")
        
        return results
    
    def _check_spelling_with_timeout(self, text: str) -> List[LayeredDetectionResult]:
        """Check spelling with timeout protection"""
        def spell_check():
            return self.spell_checker.check_spelling(text, text)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(spell_check)
            try:
                spelling_errors = future.result(timeout=self.config.spell_check_timeout)
                return self._convert_spelling_results(spelling_errors, text)
            except TimeoutError:
                logger.warning("Spell checking timed out")
                return []
    
    def _check_grammar_with_timeout(self, text: str) -> List[LayeredDetectionResult]:
        """Check grammar with timeout protection"""
        grammar_issues = self.grammar_checker.check_grammar(text, self.config.java_timeout)
        return self._convert_grammar_results(grammar_issues, text)
    
    def _convert_spelling_results(self, spelling_errors: List[Dict[str, Any]], 
                                text: str) -> List[LayeredDetectionResult]:
        """Convert spelling errors to LayeredDetectionResult"""
        results = []
        
        for error in spelling_errors:
            # Calculate confidence scores for suggestions
            confidence_scores = []
            for suggestion in error['suggestions']:
                confidence = self.confidence_scorer.score_spelling_suggestion(
                    error['word'], suggestion, error['context'], {
                        'layer': DetectionLayer.TRADITIONAL_NLP,
                        'is_technical_context': self._is_technical_context(text),
                        'domain_validation_score': 0.8
                    }
                )
                confidence_scores.append(confidence)
            
            # Filter by confidence threshold
            filtered_suggestions = []
            filtered_confidences = []
            
            for suggestion, confidence in zip(error['suggestions'], confidence_scores):
                if confidence >= self.config.confidence_threshold:
                    filtered_suggestions.append(suggestion)
                    filtered_confidences.append(confidence)
            
            if filtered_suggestions:
                result = LayeredDetectionResult(
                    layer_name=DetectionLayer.TRADITIONAL_NLP,
                    original_word=error['word'],
                    suggestions=filtered_suggestions,
                    confidence_scores=filtered_confidences,
                    detection_method="enhanced_spell_checker",
                    processing_time=0.0,  # Will be set by caller
                    error_type=ErrorType.SPELLING,
                    context=error['context'],
                    position=error['position']
                )
                results.append(result)
        
        return results
    
    def _convert_grammar_results(self, grammar_issues: List[Dict[str, Any]], 
                               text: str) -> List[LayeredDetectionResult]:
        """Convert grammar issues to LayeredDetectionResult"""
        results = []
        
        for issue in grammar_issues:
            # Use provided confidence or calculate it
            confidence = issue.get('confidence', 75.0)
            
            # Validate confidence threshold
            if confidence >= self.config.confidence_threshold:
                result = LayeredDetectionResult(
                    layer_name=DetectionLayer.TRADITIONAL_NLP,
                    original_word=issue['context'],
                    suggestions=[issue['suggestion']],
                    confidence_scores=[confidence],
                    detection_method=f"grammar_checker_{issue.get('rule_id', 'pattern')}",
                    processing_time=0.0,  # Will be set by caller
                    error_type=ErrorType.GRAMMAR,
                    context=issue['context'],
                    position=issue['position']
                )
                results.append(result)
        
        return results
    
    def _is_technical_context(self, text: str) -> bool:
        """Determine if text is in technical context"""
        technical_indicators = [
            'programming', 'software', 'development', 'technology',
            'framework', 'api', 'database', 'server', 'frontend',
            'backend', 'devops', 'cloud', 'aws', 'docker'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in technical_indicators)
    
    def get_layer_name(self) -> DetectionLayer:
        """Get the layer identifier"""
        return DetectionLayer.TRADITIONAL_NLP
    
    def is_available(self) -> bool:
        """Check if the layer is available"""
        # Traditional NLP is always available (has fallbacks)
        return True
    
    def get_performance_metrics(self) -> LayerPerformanceMetrics:
        """Get current performance metrics"""
        return self.performance_metrics
    
    def validate_config(self, config: AnalysisConfig) -> bool:
        """Validate if configuration is compatible"""
        return getattr(config, 'enable_traditional_nlp', True)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.java_manager.cleanup()
        logger.info("Traditional NLP layer cleanup completed")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about system dependencies"""
        return {
            'java_available': self.java_manager.check_java_availability(),
            'java_version': self.java_manager.java_version,
            'language_tool_available': self.java_manager.language_tool is not None,
            'spell_checker_available': self.spell_checker.spell_checker is not None,
            'custom_dictionary_size': len(self.spell_checker.custom_words),
            'fallback_patterns_count': len(self.grammar_checker.fallback_patterns)
        }
    
    def update_config(self, new_config: TraditionalNLPConfig) -> None:
        """Update layer configuration"""
        self.config = new_config
        logger.info("Traditional NLP layer configuration updated")
    
    def benchmark_performance(self, test_texts: List[str]) -> Dict[str, float]:
        """Benchmark layer performance"""
        if not test_texts:
            return {}
        
        total_time = 0.0
        total_chars = 0
        successful_runs = 0
        
        for text in test_texts:
            start_time = time.time()
            try:
                results = self.detect(text, AnalysisConfig())
                processing_time = time.time() - start_time
                
                total_time += processing_time
                total_chars += len(text)
                successful_runs += 1
                
            except Exception as e:
                logger.warning(f"Benchmark test failed: {e}")
        
        if successful_runs == 0:
            return {'error': 'All benchmark tests failed'}
        
        return {
            'average_processing_time': total_time / successful_runs,
            'characters_per_second': total_chars / total_time if total_time > 0 else 0,
            'success_rate': successful_runs / len(test_texts),
            'total_tests': len(test_texts),
            'successful_tests': successful_runs
        }