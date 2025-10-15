"""
GECToR Analysis Service for enhanced grammatical error correction.

This module implements the GECToR (Grammatical Error Correction Transformer) model
for more accurate spelling and grammar correction in resume analysis.
"""

import re
import time
from typing import List, Dict, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models import (
    TypoResult, 
    GrammarResult, 
    ValidatedTypoResult,
    ValidatedGrammarResult,
    ValidationStatus,
    ResumeContext
)

class GECToRAnalysisService:
    """
    GECToR-based analysis service for enhanced grammatical error correction.
    
    This service uses a transformer-based approach for more accurate
    spelling and grammar correction compared to traditional NLP methods.
    """
    
    def __init__(self, 
                 model_name: str = "grammarly/gector", 
                 confidence_threshold: float = 0.8,
                 device: str = None):
        """
        Initialize the GECToR analysis service.
        
        Args:
            model_name: The name of the pretrained model to use
            confidence_threshold: Minimum confidence score for suggestions
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy loading of model and tokenizer
        self._model = None
        self._tokenizer = None
        
        # Performance metrics
        self.total_processing_time = 0
        self.total_texts_processed = 0
        
    def _load_model(self):
        """Lazy initialization of the model and tokenizer"""
        if self._model is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()  # Set to evaluation mode
            except Exception as e:
                print(f"Error loading GECToR model: {e}")
                # Fallback to a simpler model if available
                try:
                    fallback_model = "vennify/t5-base-grammar-correction"
                    self._tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
                    self._model.to(self.device)
                    self._model.eval()
                except Exception as e2:
                    print(f"Error loading fallback model: {e2}")
                    raise RuntimeError("Failed to load GECToR model and fallback model")
    
    def analyze_text(self, text: str, resume_context: Optional[ResumeContext] = None) -> Tuple[List[ValidatedTypoResult], List[ValidatedGrammarResult]]:
        """
        Analyze text using GECToR for both spelling and grammar errors.
        
        Args:
            text: The text to analyze
            resume_context: Optional context about the resume
            
        Returns:
            Tuple of (typo_results, grammar_results)
        """
        start_time = time.time()
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
        
        # Split text into manageable chunks (GECToR works best with sentences/paragraphs)
        chunks = self._split_into_chunks(text)
        
        all_typos = []
        all_grammar = []
        
        # Process each chunk
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Get corrected text from GECToR
            corrected_chunk, corrections = self._get_corrections(chunk)
            
            # Process corrections into typos and grammar issues
            typos, grammar = self._process_corrections(chunk, corrected_chunk, corrections, resume_context)
            
            all_typos.extend(typos)
            all_grammar.extend(grammar)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.total_texts_processed += 1
        
        return all_typos, all_grammar
    
    def _split_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into manageable chunks for processing.
        
        Args:
            text: The text to split
            max_length: Maximum token length for the model
            
        Returns:
            List of text chunks
        """
        # First try to split by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph is too long, split by sentences
            if len(para) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                if len(current_chunk) + len(para) <= max_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _get_corrections(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get corrections from the GECToR model.
        
        Args:
            text: The text to correct
            
        Returns:
            Tuple of (corrected_text, corrections)
        """
        try:
            # Tokenize the input text
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate corrected text
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_length=512)
            
            # Decode the generated text
            corrected_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract corrections by comparing original and corrected text
            corrections = self._extract_corrections(text, corrected_text)
            
            return corrected_text, corrections
            
        except Exception as e:
            print(f"Error in GECToR correction: {e}")
            return text, []
    
    def _extract_corrections(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """
        Extract corrections by comparing original and corrected text.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            List of correction dictionaries
        """
        corrections = []
        
        # If texts are identical, no corrections needed
        if original == corrected:
            return corrections
        
        # Use difflib to find differences
        import difflib
        
        # Split into words for more accurate comparison
        original_words = re.findall(r'\S+|\s+', original)
        corrected_words = re.findall(r'\S+|\s+', corrected)
        
        # Get sequence matcher
        matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
        
        # Extract operations
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'replace':
                # This is a replacement (could be spelling or grammar)
                orig_text = ''.join(original_words[i1:i2])
                corr_text = ''.join(corrected_words[j1:j2])
                
                # Calculate position in original text
                position = sum(len(w) for w in original_words[:i1])
                
                # Determine if this is likely a spelling or grammar issue
                is_spelling = len(orig_text.split()) == 1 and len(corr_text.split()) == 1
                
                corrections.append({
                    'type': 'spelling' if is_spelling else 'grammar',
                    'original': orig_text,
                    'correction': corr_text,
                    'position': position,
                    'confidence': 0.9  # GECToR typically has high confidence
                })
                
            elif op == 'insert':
                # This is an insertion (likely grammar)
                corr_text = ''.join(corrected_words[j1:j2])
                
                # Calculate position in original text
                position = sum(len(w) for w in original_words[:i1])
                
                corrections.append({
                    'type': 'grammar',
                    'original': '',
                    'correction': corr_text,
                    'position': position,
                    'confidence': 0.85
                })
                
            elif op == 'delete':
                # This is a deletion (likely grammar)
                orig_text = ''.join(original_words[i1:i2])
                
                # Calculate position in original text
                position = sum(len(w) for w in original_words[:i1])
                
                corrections.append({
                    'type': 'grammar',
                    'original': orig_text,
                    'correction': '',
                    'position': position,
                    'confidence': 0.85
                })
        
        return corrections
    
    def _process_corrections(self, 
                           original: str, 
                           corrected: str, 
                           corrections: List[Dict[str, Any]],
                           resume_context: Optional[ResumeContext]) -> Tuple[List[ValidatedTypoResult], List[ValidatedGrammarResult]]:
        """
        Process corrections into typo and grammar results.
        
        Args:
            original: Original text
            corrected: Corrected text
            corrections: List of correction dictionaries
            resume_context: Optional resume context
            
        Returns:
            Tuple of (typo_results, grammar_results)
        """
        typos = []
        grammar = []
        
        for correction in corrections:
            # Skip corrections below confidence threshold
            if correction['confidence'] < self.confidence_threshold:
                continue
                
            if correction['type'] == 'spelling':
                # Create typo result
                typo = ValidatedTypoResult(
                    word=correction['original'],
                    suggestion=correction['correction'],
                    position=correction['position'],
                    confidence_score=correction['confidence'] * 100,  # Convert to percentage
                    explanation=self._generate_spelling_explanation(
                        correction['original'], 
                        correction['correction'],
                        original,
                        correction['confidence'] * 100
                    ),
                    validation_status=ValidationStatus.VALIDATED,
                    context=original  # <-- Added context field
                )
                typos.append(typo)
                
            else:
                # Create grammar result
                # Get context around the correction
                start = max(0, correction['position'] - 50)
                end = min(len(original), correction['position'] + len(correction['original']) + 50)
                context = original[start:end].strip()
                
                grammar_issue = ValidatedGrammarResult(
                    sentence=context,
                    suggestion=f"Replace '{correction['original']}' with '{correction['correction']}'",
                    confidence_score=correction['confidence'] * 100,  # Convert to percentage
                    explanation=self._generate_grammar_explanation(
                        'Grammar error',  # GECToR doesn't provide specific error types
                        correction['original'],
                        correction['correction'],
                        correction['confidence'] * 100
                    ),
                    issue_type='Grammar error',
                    rule_category='GECToR correction',
                    validation_status=ValidationStatus.VALIDATED,
                    position=correction['position']
                )
                grammar.append(grammar_issue)
        
        return typos, grammar
    
    def _generate_spelling_explanation(self, original: str, suggestion: str, 
                                     context: str, confidence: float) -> str:
        """Generate explanation for spelling correction"""
        
        # Check edit distance
        edit_distance = self._calculate_edit_distance(original.lower(), suggestion.lower())
        if edit_distance == 1:
            return f"GECToR detected single character correction from '{original}' to '{suggestion}' (confidence: {confidence:.0f}%)"
        elif edit_distance == 2:
            return f"GECToR detected minor spelling correction from '{original}' to '{suggestion}' (confidence: {confidence:.0f}%)"
        else:
            return f"GECToR spelling correction from '{original}' to '{suggestion}' (confidence: {confidence:.0f}%)"
    
    def _generate_grammar_explanation(self, issue_type: str, original: str, 
                                    suggestion: str, confidence: float) -> str:
        """Generate explanation for grammar correction"""
        
        if not original and suggestion:
            return f"GECToR suggests adding '{suggestion}' for grammatical completeness (confidence: {confidence:.0f}%)"
        elif original and not suggestion:
            return f"GECToR suggests removing '{original}' for grammatical correctness (confidence: {confidence:.0f}%)"
        else:
            return f"GECToR grammar improvement: replace '{original}' with '{suggestion}' (confidence: {confidence:.0f}%)"
    
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
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the GECToR service.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.total_texts_processed == 0:
            return {
                'average_processing_time': 0,
                'texts_processed': 0
            }
            
        return {
            'average_processing_time': self.total_processing_time / self.total_texts_processed,
            'texts_processed': self.total_texts_processed
        }