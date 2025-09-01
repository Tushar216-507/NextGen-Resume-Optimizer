import time
import re
from typing import List, Set, Optional
from spellchecker import SpellChecker
import language_tool_python
from models import (
    TypoResult, GrammarResult, AnalysisResult, AnalysisSummary,
    ValidatedTypoResult, ValidatedGrammarResult, EnhancedAnalysisResult,
    EnhancedAnalysisSummary, ConfidenceMetrics, ResumeContext, ValidationStatus
)

# Import enhanced components
try:
    from domain_vocabulary import DomainVocabulary
    from context_analyzer import ContextAnalyzer
    from confidence_scorer import ConfidenceScorer
    from validation_pipeline import ValidationPipeline
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False
    print("Enhanced components not available, falling back to basic analysis")

class TextAnalysisService:
    """Service for analyzing text quality including spelling and grammar with enhanced accuracy"""
    
    def __init__(self, enable_enhanced_analysis: bool = True):
        self.spell_checker = SpellChecker()
        self.grammar_tool = None  # Initialize lazily for better performance
        
        # Enhanced analysis configuration
        self.enable_enhanced_analysis = enable_enhanced_analysis and ENHANCED_COMPONENTS_AVAILABLE
        self.confidence_threshold = 80.0
        
        # Initialize enhanced components if available
        if self.enable_enhanced_analysis:
            try:
                self.domain_vocab = DomainVocabulary()
                self.context_analyzer = ContextAnalyzer()
                self.confidence_scorer = ConfidenceScorer()
                self.validation_pipeline = ValidationPipeline()
                print("Enhanced analysis components initialized successfully")
            except Exception as e:
                print(f"Failed to initialize enhanced components: {e}")
                self.enable_enhanced_analysis = False
        
        # Add professional/technical terms to custom dictionary
        self._add_professional_terms()
        
        # Common resume words that shouldn't be flagged (fallback vocabulary)
        self.resume_vocabulary = {
            'javascript', 'python', 'html', 'css', 'sql', 'api', 'ui', 'ux',
            'frontend', 'backend', 'fullstack', 'devops', 'agile', 'scrum',
            'github', 'git', 'aws', 'azure', 'docker', 'kubernetes', 'react',
            'angular', 'vue', 'nodejs', 'mongodb', 'postgresql', 'mysql',
            'tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy', 'django',
            'flask', 'fastapi', 'restful', 'graphql', 'microservices',
            'ci', 'cd', 'jenkins', 'terraform', 'ansible', 'linux', 'ubuntu',
            'centos', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'nginx'
        }
    
    def _add_professional_terms(self):
        """Add professional and technical terms to spell checker"""
        professional_terms = [
            'javascript', 'typescript', 'python', 'java', 'csharp', 'cplusplus',
            'html', 'css', 'scss', 'sass', 'bootstrap', 'tailwind',
            'react', 'angular', 'vue', 'svelte', 'nextjs', 'nuxtjs',
            'nodejs', 'express', 'fastapi', 'django', 'flask', 'spring',
            'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'jenkins', 'gitlab', 'github', 'bitbucket', 'jira', 'confluence',
            'agile', 'scrum', 'kanban', 'devops', 'cicd', 'microservices',
            'api', 'rest', 'graphql', 'websocket', 'oauth', 'jwt',
            'frontend', 'backend', 'fullstack', 'ui', 'ux', 'responsive',
            'tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy', 'matplotlib',
            'analytics', 'machinelearning', 'artificialintelligence', 'datascience'
        ]
        
        # Add enhanced vocabulary if available
        if self.enable_enhanced_analysis and hasattr(self, 'domain_vocab'):
            # Get all technical terms from domain vocabulary
            all_terms = []
            
            # Add technical terms
            for term_data in self.domain_vocab.technical_terms.values():
                all_terms.append(term_data.term)
                all_terms.extend(term_data.variations)
            
            # Add frameworks and tools
            for tool_data in self.domain_vocab.frameworks_and_tools.values():
                all_terms.append(tool_data.term)
                all_terms.extend(tool_data.variations)
            
            # Add company names and certifications
            all_terms.extend(self.domain_vocab.company_names)
            all_terms.extend(self.domain_vocab.certifications)
            
            professional_terms.extend(all_terms)
        
        # Add terms to spell checker's known words
        self.spell_checker.word_frequency.load_words(professional_terms)
    
    def _get_grammar_tool(self):
        """Lazy initialization of grammar tool"""
        if self.grammar_tool is None:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        return self.grammar_tool
    
    def analyze_spelling(self, text: str) -> List[TypoResult]:
        """Analyze text for spelling mistakes and typos with improved accuracy"""
        
        # Use enhanced analysis if available
        if self.enable_enhanced_analysis:
            enhanced_typos = self._analyze_spelling_enhanced(text, None)
            # Convert to basic format
            return [TypoResult(word=t.word, suggestion=t.suggestion, position=t.position) 
                   for t in enhanced_typos]
        
        # Fallback to basic analysis
        typos = []
        
        # Split text into words and preserve original case
        word_pattern = re.finditer(r'\b[a-zA-Z]+\b', text)
        
        for match in word_pattern:
            original_word = match.group()
            word_lower = original_word.lower()
            
            # Skip if word is in our professional vocabulary
            if word_lower in self.resume_vocabulary:
                continue
                
            # Skip very short words (likely abbreviations)
            if len(word_lower) <= 2:
                continue
                
            # Skip if word contains numbers or special patterns
            if re.search(r'\d', word_lower):
                continue
            
            # Check if word is misspelled
            if word_lower not in self.spell_checker:
                suggestions = self.spell_checker.candidates(word_lower)
                
                if suggestions:
                    # Get the best suggestion using edit distance and frequency
                    best_suggestion = self._get_best_suggestion(word_lower, suggestions)
                    
                    # Only suggest if the suggestion is significantly different and better
                    if best_suggestion and self._is_valid_suggestion(word_lower, best_suggestion):
                        typos.append(TypoResult(
                            word=original_word,
                            suggestion=best_suggestion.title() if original_word.istitle() else best_suggestion,
                            position=match.start()
                        ))
        
        return typos
    
    def _get_best_suggestion(self, word: str, suggestions: Set[str]) -> str:
        """Get the best spelling suggestion based on edit distance and frequency"""
        if not suggestions:
            return ""
            
        # Convert to list
        suggestion_list = list(suggestions)
        
        # If only one suggestion, return it
        if len(suggestion_list) == 1:
            return suggestion_list[0]
        
        # Prefer suggestions with higher frequency in the spell checker's corpus
        try:
            best_suggestion = max(suggestion_list, 
                                key=lambda s: self.spell_checker.word_frequency[s] if s in self.spell_checker.word_frequency else 0)
        except:
            # Fallback: return the first suggestion
            best_suggestion = suggestion_list[0]
        
        return best_suggestion
    
    def _is_valid_suggestion(self, original: str, suggestion: str) -> bool:
        """Check if a suggestion is valid and worth showing"""
        # Don't suggest if words are too similar (might be intentional variation)
        if len(original) == len(suggestion):
            differences = sum(1 for a, b in zip(original, suggestion) if a != b)
            if differences <= 1:  # Only 1 character difference
                return False
        
        # Don't suggest if suggestion is much shorter (likely wrong)
        if len(suggestion) < len(original) - 2:
            return False
            
        # Don't suggest if suggestion contains the original as substring
        if original in suggestion or suggestion in original:
            return False
            
        return True
    
    def analyze_grammar(self, text: str) -> List[GrammarResult]:
        """Analyze text for grammar and style issues"""
        
        # Use enhanced analysis if available
        if self.enable_enhanced_analysis:
            enhanced_grammar = self._analyze_grammar_enhanced(text, None)
            # Convert to basic format
            return [GrammarResult(sentence=g.sentence, suggestion=g.suggestion, 
                                issue_type=g.issue_type, position=g.position)
                   for g in enhanced_grammar]
        
        # Fallback to basic analysis
        grammar_issues = []
        
        try:
            tool = self._get_grammar_tool()
            matches = tool.check(text)
            
            for match in matches:
                # Extract the sentence containing the error
                start = max(0, match.offset - 50)
                end = min(len(text), match.offset + match.errorLength + 50)
                context = text[start:end].strip()
                
                # Get the suggested replacement
                suggestion = match.replacements[0] if match.replacements else "No suggestion available"
                
                grammar_issues.append(GrammarResult(
                    sentence=context,
                    suggestion=f"Replace '{match.context}' with '{suggestion}'",
                    issue_type=match.category,
                    position=match.offset
                ))
        
        except Exception as e:
            print(f"Grammar analysis error: {e}")
            # Add basic grammar checks as fallback
            grammar_issues.extend(self._basic_grammar_check(text))
        
        return grammar_issues
    
    def _basic_grammar_check(self, text: str) -> List[GrammarResult]:
        """Basic grammar checking without external dependencies"""
        issues = []
        
        # Improved pattern-based grammar checks for resumes
        patterns = [
            # Subject-verb agreement
            (r'\bI has\b', 'I have', 'Subject-verb disagreement'),
            (r'\bI was\s+(\w+ing)\b', r'I \1', 'Tense consistency'),
            
            # Plural forms
            (r'\b(\d+)\s+year\s+(?:of\s+)?experience\b', r'\1 years of experience', 'Plural form needed'),
            (r'\b(\d+)\s+year\s+(?:in|with)\b', r'\1 years in', 'Plural form needed'),
            
            # Common pronoun errors
            (r'\bMe and\s+\w+\s+(?:have|has|will|can|are)\b', 'My colleague and I', 'Pronoun case error'),
            (r'\bThis are\b', 'These are', 'Subject-verb disagreement'),
            (r'\bThat are\b', 'Those are', 'Subject-verb disagreement'),
            
            # Articles
            (r'\ban\s+(?=[bcdfghjklmnpqrstvwxyz])', 'a', 'Article error'),
            (r'\ba\s+(?=[aeiou])', 'an', 'Article error'),
            
            # Common resume mistakes
            (r'\bresponsible to\b', 'responsible for', 'Preposition error'),
            (r'\bexperience on\b', 'experience in', 'Preposition error'),
        ]
        
        for pattern, suggestion, issue_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                # Create a more specific suggestion
                matched_text = match.group()
                if '\\1' in suggestion or '\\2' in suggestion:
                    # Handle regex group replacements
                    suggestion_text = f"Replace '{matched_text}' with the correct plural/form"
                else:
                    suggestion_text = f"Replace '{matched_text}' with '{suggestion}'"
                
                issues.append(GrammarResult(
                    sentence=context,
                    suggestion=suggestion_text,
                    issue_type=issue_type,
                    position=match.start()
                ))
        
        return issues
    
    def analyze_full_text(self, text: str, check_spelling: bool = True, check_grammar: bool = True) -> AnalysisResult:
        """Perform complete text analysis with enhanced accuracy when available"""
        
        # Use enhanced analysis if available
        if self.enable_enhanced_analysis:
            enhanced_result = self.analyze_with_confidence(text, check_spelling, check_grammar)
            
            # Convert enhanced results back to basic format for backward compatibility
            basic_typos = [TypoResult(word=t.word, suggestion=t.suggestion, position=t.position) 
                          for t in enhanced_result.typos]
            basic_grammar = [GrammarResult(sentence=g.sentence, suggestion=g.suggestion, 
                                         issue_type=g.issue_type, position=g.position)
                            for g in enhanced_result.grammar_issues]
            
            basic_summary = AnalysisSummary(
                total_typos=enhanced_result.summary.total_typos,
                total_grammar_issues=enhanced_result.summary.total_grammar_issues,
                word_count=enhanced_result.summary.word_count
            )
            
            return AnalysisResult(
                typos=basic_typos,
                grammar_issues=basic_grammar,
                summary=basic_summary,
                processing_time=enhanced_result.processing_time
            )
        
        # Fallback to basic analysis
        start_time = time.time()
        
        typos = []
        grammar_issues = []
        
        if check_spelling:
            typos = self.analyze_spelling(text)
        
        if check_grammar:
            grammar_issues = self.analyze_grammar(text)
        
        # Calculate summary statistics
        word_count = len(text.split())
        processing_time = time.time() - start_time
        
        summary = AnalysisSummary(
            total_typos=len(typos),
            total_grammar_issues=len(grammar_issues),
            word_count=word_count
        )
        
        return AnalysisResult(
            typos=typos,
            grammar_issues=grammar_issues,
            summary=summary,
            processing_time=processing_time
        )
    
    def analyze_with_confidence(self, text: str, check_spelling: bool = True, 
                              check_grammar: bool = True) -> EnhancedAnalysisResult:
        """Perform enhanced text analysis with confidence scoring and validation"""
        
        if not self.enable_enhanced_analysis:
            raise RuntimeError("Enhanced analysis components not available")
        
        start_time = time.time()
        
        # Analyze resume context
        resume_context = None
        try:
            resume_context = self.context_analyzer.analyze_resume_context(text)
        except Exception as e:
            print(f"Context analysis failed: {e}")
            # Create fallback context
            resume_context = ResumeContext(
                sections={}, 
                formatting_style="traditional", 
                professional_level="mid_level",
                industry_indicators=[],
                detected_technologies=[]
            )
        
        # Perform enhanced analysis
        typos = []
        grammar_issues = []
        
        if check_spelling:
            typos = self._analyze_spelling_enhanced(text, resume_context)
        
        if check_grammar:
            grammar_issues = self._analyze_grammar_enhanced(text, resume_context)
        
        # Validate suggestions if validation pipeline is available
        try:
            typos = self._validate_spelling_suggestions(typos, text, resume_context)
            grammar_issues = self._validate_grammar_suggestions(grammar_issues, text, resume_context)
        except Exception as e:
            print(f"Validation failed: {e}")
        
        # Calculate metrics
        processing_time = time.time() - start_time
        word_count = len(text.split())
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(typos, grammar_issues)
        
        # Create enhanced summary
        summary = EnhancedAnalysisSummary(
            total_typos=len(typos),
            total_grammar_issues=len(grammar_issues),
            word_count=word_count,
            confidence_metrics=confidence_metrics,
            context_analysis=resume_context
        )
        
        return EnhancedAnalysisResult(
            typos=typos,
            grammar_issues=grammar_issues,
            summary=summary,
            processing_time=processing_time
        )
    
    def _analyze_spelling_enhanced(self, text: str, resume_context: Optional[ResumeContext]) -> List[ValidatedTypoResult]:
        """Enhanced spelling analysis with context awareness"""
        validated_typos = []
        
        # Split text into words and preserve original case
        word_pattern = re.finditer(r'\b[a-zA-Z]+\b', text)
        
        for match in word_pattern:
            original_word = match.group()
            word_lower = original_word.lower()
            position = match.start()
            
            # Skip very short words (likely abbreviations)
            if len(word_lower) <= 2:
                continue
                
            # Skip if word contains numbers
            if re.search(r'\d', word_lower):
                continue
            
            # Enhanced domain vocabulary check
            if self.domain_vocab.is_valid_technical_term(original_word, text):
                continue
            
            # Check if word is misspelled
            if word_lower not in self.spell_checker:
                # Get context around the word
                context_start = max(0, position - 100)
                context_end = min(len(text), position + len(original_word) + 100)
                context = text[context_start:context_end]
                
                # Get suggestions
                suggestions = self.spell_checker.candidates(word_lower)
                
                if suggestions:
                    # Get context-appropriate suggestions
                    context_suggestions = self.domain_vocab.get_context_appropriate_suggestions(
                        original_word, context
                    )
                    
                    # Combine and rank suggestions
                    all_suggestions = list(suggestions) + context_suggestions
                    
                    if all_suggestions:
                        best_suggestion = self._get_best_suggestion_enhanced(
                            original_word, all_suggestions, context, resume_context
                        )
                        
                        if best_suggestion:
                            # Calculate confidence score
                            confidence = self.confidence_scorer.score_spelling_suggestion(
                                original_word, best_suggestion, context, resume_context
                            )
                            
                            # Only include if meets confidence threshold
                            if confidence >= self.confidence_threshold:
                                # Generate explanation
                                explanation = self._generate_spelling_explanation(
                                    original_word, best_suggestion, context, confidence
                                )
                                
                                validated_typos.append(ValidatedTypoResult(
                                    word=original_word,
                                    suggestion=best_suggestion.title() if original_word.istitle() else best_suggestion,
                                    confidence_score=confidence,
                                    explanation=explanation,
                                    context=context,
                                    validation_status=ValidationStatus.VALIDATED,
                                    position=position
                                ))
        
        return validated_typos
    
    def _analyze_grammar_enhanced(self, text: str, resume_context: Optional[ResumeContext]) -> List[ValidatedGrammarResult]:
        """Enhanced grammar analysis with context awareness"""
        validated_grammar = []
        
        try:
            # Use language-tool-python for grammar checking
            tool = self._get_grammar_tool()
            matches = tool.check(text)
            
            for match in matches:
                # Extract context
                start = max(0, match.offset - 50)
                end = min(len(text), match.offset + match.errorLength + 50)
                context = text[start:end].strip()
                
                # Get suggested replacement
                suggestion = match.replacements[0] if match.replacements else "No suggestion available"
                
                # Create full suggestion text
                full_suggestion = f"Replace '{match.context}' with '{suggestion}'"
                
                # Calculate confidence score
                confidence = self.confidence_scorer.score_grammar_suggestion(
                    context, full_suggestion, match.category, text, resume_context
                )
                
                # Only include if meets confidence threshold
                if confidence >= self.confidence_threshold:
                    # Generate explanation
                    explanation = self._generate_grammar_explanation(
                        match.category, match.context, suggestion, confidence
                    )
                    
                    validated_grammar.append(ValidatedGrammarResult(
                        sentence=context,
                        suggestion=full_suggestion,
                        confidence_score=confidence,
                        explanation=explanation,
                        issue_type=match.category,
                        rule_category=match.category,
                        validation_status=ValidationStatus.VALIDATED,
                        position=match.offset
                    ))
        
        except Exception as e:
            print(f"Grammar analysis error: {e}")
            # Fallback to basic grammar checks
            basic_issues = self._basic_grammar_check_enhanced(text, resume_context)
            validated_grammar.extend(basic_issues)
        
        return validated_grammar
    
    def _get_best_suggestion_enhanced(self, original: str, suggestions: List[str], 
                                    context: str, resume_context: Optional[ResumeContext]) -> Optional[str]:
        """Get the best spelling suggestion using enhanced scoring"""
        
        if not suggestions:
            return None
        
        # Score each suggestion
        scored_suggestions = []
        for suggestion in suggestions:
            confidence = self.confidence_scorer.score_spelling_suggestion(
                original, suggestion, context, resume_context
            )
            scored_suggestions.append((suggestion, confidence))
        
        # Sort by confidence and return the best
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        # Return best suggestion if it meets minimum threshold
        best_suggestion, best_confidence = scored_suggestions[0]
        if best_confidence >= 60.0:  # Lower threshold for suggestion selection
            return best_suggestion
        
        return None
    
    def _basic_grammar_check_enhanced(self, text: str, resume_context: Optional[ResumeContext]) -> List[ValidatedGrammarResult]:
        """Enhanced basic grammar checking with context awareness"""
        issues = []
        
        # Resume-specific grammar patterns with confidence scores
        patterns = [
            {
                'pattern': r'\bI has\b',
                'suggestion': 'I have',
                'issue_type': 'Subject-verb disagreement',
                'confidence': 95.0,
                'explanation': 'Use "I have" instead of "I has" for correct subject-verb agreement'
            },
            {
                'pattern': r'\b(\d+)\s+year\s+(?:of\s+)?experience\b',
                'suggestion': r'\1 years of experience',
                'issue_type': 'Plural form needed',
                'confidence': 90.0,
                'explanation': 'Use plural "years" when the number is greater than 1'
            },
            {
                'pattern': r'\bMe and\s+\w+\s+(?:have|has|will|can|are)\b',
                'suggestion': 'My colleague and I',
                'issue_type': 'Pronoun case error',
                'confidence': 85.0,
                'explanation': 'Use "My colleague and I" as the subject of a sentence'
            },
            {
                'pattern': r'\bThis are\b',
                'suggestion': 'These are',
                'issue_type': 'Subject-verb disagreement',
                'confidence': 90.0,
                'explanation': 'Use "These are" for plural subjects'
            },
            {
                'pattern': r'\bresponsible to\b',
                'suggestion': 'responsible for',
                'issue_type': 'Preposition error',
                'confidence': 80.0,
                'explanation': 'Use "responsible for" in professional contexts'
            },
            {
                'pattern': r'\bexperience on\b',
                'suggestion': 'experience in',
                'issue_type': 'Preposition error',
                'confidence': 75.0,
                'explanation': 'Use "experience in" when describing expertise areas'
            }
        ]
        
        for pattern_info in patterns:
            matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                # Create suggestion
                matched_text = match.group()
                if r'\1' in pattern_info['suggestion']:
                    # Handle regex group replacements
                    corrected = re.sub(pattern_info['pattern'], pattern_info['suggestion'], 
                                     matched_text, flags=re.IGNORECASE)
                    suggestion_text = f'Replace "{matched_text}" with "{corrected}"'
                else:
                    suggestion_text = f'Replace "{matched_text}" with "{pattern_info["suggestion"]}"'
                
                # Apply context-based confidence adjustment
                confidence = pattern_info['confidence']
                if resume_context:
                    confidence = self._adjust_confidence_for_context(
                        confidence, pattern_info['issue_type'], resume_context
                    )
                
                # Only include if meets threshold
                if confidence >= self.confidence_threshold:
                    issues.append(ValidatedGrammarResult(
                        sentence=context,
                        suggestion=suggestion_text,
                        confidence_score=confidence,
                        explanation=pattern_info['explanation'],
                        issue_type=pattern_info['issue_type'],
                        rule_category=pattern_info['issue_type'],
                        validation_status=ValidationStatus.VALIDATED,
                        position=match.start()
                    ))
        
        return issues
    
    def _validate_spelling_suggestions(self, typos: List[ValidatedTypoResult], text: str,
                                     resume_context: Optional[ResumeContext]) -> List[ValidatedTypoResult]:
        """Validate spelling suggestions through the validation pipeline"""
        
        if not hasattr(self, 'validation_pipeline'):
            return typos
        
        validated_typos = []
        
        for typo in typos:
            # Convert to basic TypoResult for validation
            basic_typo = TypoResult(
                word=typo.word,
                suggestion=typo.suggestion,
                position=typo.position
            )
            
            try:
                # Run through validation pipeline
                validation_result = self.validation_pipeline.validate_spelling_suggestion(
                    basic_typo, typo.context, resume_context
                )
                
                # Update validation status and confidence based on validation
                if validation_result.status == ValidationStatus.CROSS_VALIDATED:
                    typo.validation_status = ValidationStatus.CROSS_VALIDATED
                    typo.confidence_score = validation_result.final_confidence
                    validated_typos.append(typo)
                elif validation_result.status == ValidationStatus.VALIDATED and validation_result.final_confidence >= self.confidence_threshold:
                    typo.validation_status = ValidationStatus.VALIDATED
                    typo.confidence_score = validation_result.final_confidence
                    validated_typos.append(typo)
                # Filtered suggestions are not included
            except Exception as e:
                print(f"Validation error for typo '{typo.word}': {e}")
                # Include original suggestion if validation fails
                validated_typos.append(typo)
        
        return validated_typos
    
    def _validate_grammar_suggestions(self, grammar_issues: List[ValidatedGrammarResult], text: str,
                                    resume_context: Optional[ResumeContext]) -> List[ValidatedGrammarResult]:
        """Validate grammar suggestions through the validation pipeline"""
        
        if not hasattr(self, 'validation_pipeline'):
            return grammar_issues
        
        validated_grammar = []
        
        for issue in grammar_issues:
            # Convert to basic GrammarResult for validation
            basic_grammar = GrammarResult(
                sentence=issue.sentence,
                suggestion=issue.suggestion,
                issue_type=issue.issue_type,
                position=issue.position
            )
            
            try:
                # Run through validation pipeline
                validation_result = self.validation_pipeline.validate_grammar_suggestion(
                    basic_grammar, text, resume_context
                )
                
                # Update validation status and confidence based on validation
                if validation_result.status == ValidationStatus.CROSS_VALIDATED:
                    issue.validation_status = ValidationStatus.CROSS_VALIDATED
                    issue.confidence_score = validation_result.final_confidence
                    validated_grammar.append(issue)
                elif validation_result.status == ValidationStatus.VALIDATED and validation_result.final_confidence >= self.confidence_threshold:
                    issue.validation_status = ValidationStatus.VALIDATED
                    issue.confidence_score = validation_result.final_confidence
                    validated_grammar.append(issue)
                # Filtered suggestions are not included
            except Exception as e:
                print(f"Validation error for grammar issue: {e}")
                # Include original suggestion if validation fails
                validated_grammar.append(issue)
        
        return validated_grammar
    
    def _calculate_confidence_metrics(self, typos: List[ValidatedTypoResult], 
                                    grammar_issues: List[ValidatedGrammarResult]) -> ConfidenceMetrics:
        """Calculate confidence metrics for the analysis"""
        
        all_suggestions = typos + grammar_issues
        
        if not all_suggestions:
            return ConfidenceMetrics(
                average_confidence=0.0,
                high_confidence_suggestions=0,
                filtered_low_confidence=0,
                validation_pass_rate=0.0
            )
        
        # Calculate metrics
        confidences = [s.confidence_score for s in all_suggestions]
        average_confidence = sum(confidences) / len(confidences)
        
        high_confidence_suggestions = sum(1 for c in confidences if c >= 90.0)
        
        cross_validated = sum(1 for s in all_suggestions 
                            if s.validation_status == ValidationStatus.CROSS_VALIDATED)
        validation_pass_rate = cross_validated / len(all_suggestions)
        
        # Estimate filtered suggestions (this would be tracked in a real implementation)
        filtered_low_confidence = 0  # Placeholder
        
        return ConfidenceMetrics(
            average_confidence=average_confidence,
            high_confidence_suggestions=high_confidence_suggestions,
            filtered_low_confidence=filtered_low_confidence,
            validation_pass_rate=validation_pass_rate
        )
    
    def _generate_spelling_explanation(self, original: str, suggestion: str, 
                                     context: str, confidence: float) -> str:
        """Generate explanation for spelling correction"""
        
        # Check if it's a technical term
        if hasattr(self, 'domain_vocab') and self.domain_vocab.is_valid_technical_term(suggestion, context):
            return f"'{suggestion}' is a recognized technical term (confidence: {confidence:.0f}%)"
        
        # Check if it's a common professional word
        professional_words = ['experience', 'development', 'management', 'analysis']
        if suggestion.lower() in professional_words:
            return f"'{suggestion}' is a common professional term (confidence: {confidence:.0f}%)"
        
        # Check edit distance
        edit_distance = self._calculate_edit_distance(original.lower(), suggestion.lower())
        if edit_distance == 1:
            return f"Single character correction from '{original}' to '{suggestion}' (confidence: {confidence:.0f}%)"
        elif edit_distance == 2:
            return f"Minor spelling correction from '{original}' to '{suggestion}' (confidence: {confidence:.0f}%)"
        else:
            return f"Spelling correction from '{original}' to '{suggestion}' (confidence: {confidence:.0f}%)"
    
    def _generate_grammar_explanation(self, issue_type: str, original: str, 
                                    suggestion: str, confidence: float) -> str:
        """Generate explanation for grammar correction"""
        
        explanations = {
            'Subject-verb disagreement': f"Subject and verb must agree in number. Replace '{original}' with '{suggestion}' (confidence: {confidence:.0f}%)",
            'Plural form needed': f"Use plural form when referring to multiple items. '{suggestion}' is correct (confidence: {confidence:.0f}%)",
            'Pronoun case error': f"Use correct pronoun case. '{suggestion}' is appropriate as the subject (confidence: {confidence:.0f}%)",
            'Article error': f"Use correct article. '{suggestion}' is the appropriate article here (confidence: {confidence:.0f}%)",
            'Preposition error': f"Use correct preposition. '{suggestion}' is more appropriate in professional contexts (confidence: {confidence:.0f}%)",
            'Tense consistency': f"Maintain consistent tense throughout the section. '{suggestion}' maintains proper tense (confidence: {confidence:.0f}%)"
        }
        
        return explanations.get(issue_type, 
                              f"Grammar improvement suggested: replace '{original}' with '{suggestion}' (confidence: {confidence:.0f}%)")
    
    def _adjust_confidence_for_context(self, base_confidence: float, issue_type: str, 
                                     resume_context: ResumeContext) -> float:
        """Adjust confidence based on resume context"""
        
        adjusted_confidence = base_confidence
        
        # Boost confidence for issues in appropriate sections
        if issue_type == 'Tense consistency':
            if 'experience' in resume_context.sections:
                adjusted_confidence += 10  # Higher confidence in experience section
        
        # Adjust based on professional level
        if resume_context.professional_level == "executive":
            adjusted_confidence += 5  # Higher standards for executive resumes
        elif resume_context.professional_level == "entry_level":
            adjusted_confidence -= 5  # More lenient for entry level
        
        # Adjust based on industry
        if 'software_development' in resume_context.industry_indicators:
            if issue_type == 'Preposition error':
                adjusted_confidence += 5  # Technical writing standards
        
        return min(100.0, max(0.0, adjusted_confidence))
    
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
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for suggestions"""
        self.confidence_threshold = max(0.0, min(100.0, threshold))
    
    def get_analysis_capabilities(self) -> dict:
        """Get information about available analysis capabilities"""
        return {
            'enhanced_analysis_available': self.enable_enhanced_analysis,
            'confidence_scoring': self.enable_enhanced_analysis,
            'context_analysis': self.enable_enhanced_analysis,
            'domain_vocabulary': self.enable_enhanced_analysis,
            'validation_pipeline': self.enable_enhanced_analysis,
            'confidence_threshold': self.confidence_threshold
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.grammar_tool:
            self.grammar_tool.close()