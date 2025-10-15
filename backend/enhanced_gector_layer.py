"""
Enhanced GECToR detection layer with robust fallbacks and optimization.
Implements the IDetectionLayer interface with advanced error handling and performance optimization.
"""

import time
import logging
import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import re
from pathlib import Path

from enhanced_models import (
    LayeredDetectionResult, DetectionLayer, AnalysisConfig, 
    LayerPerformanceMetrics, ErrorType
)
from core_interfaces import IDetectionLayer
from confidence_scoring import AdvancedConfidenceScorer
from domain_vocabulary import EnhancedDomainValidator

logger = logging.getLogger(__name__)

@dataclass
class GECToRConfig:
    """Configuration for GECToR layer"""
    model_name: str = "vennify/t5-base-grammar-correction"
    fallback_model: str = "grammarly/coedit-large"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 4
    confidence_threshold: float = 0.7
    enable_chunking: bool = True
    chunk_overlap: int = 50
    cache_model: bool = True
    use_local_models: bool = False
    local_model_path: Optional[str] = None

class ModelManager:
    """Manages GECToR model loading with fallback strategies"""
    
    def __init__(self, config: GECToRConfig):
        self.config = config
        self.primary_model = None
        self.primary_tokenizer = None
        self.fallback_model = None
        self.fallback_tokenizer = None
        self.device = self._determine_device()
        self.model_loaded = False
        self.fallback_loaded = False
        
    def _determine_device(self) -> str:
        """Determine the best device to use"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def load_primary_model(self) -> bool:
        """Load the primary GECToR model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logger.info(f"Loading primary model: {self.config.model_name}")
            
            # Try local model first if configured
            if self.config.use_local_models and self.config.local_model_path:
                model_path = Path(self.config.local_model_path)
                if model_path.exists():
                    self.primary_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    self.primary_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
                    logger.info("Loaded local model successfully")
                else:
                    logger.warning(f"Local model path not found: {model_path}")
                    return False
            else:
                # Load from HuggingFace
                self.primary_tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.primary_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            
            # Move to device
            self.primary_model.to(self.device)
            self.primary_model.eval()
            
            self.model_loaded = True
            logger.info(f"Primary model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            return False
    
    def load_fallback_model(self) -> bool:
        """Load fallback model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logger.info(f"Loading fallback model: {self.config.fallback_model}")
            
            self.fallback_tokenizer = AutoTokenizer.from_pretrained(self.config.fallback_model)
            self.fallback_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.fallback_model)
            
            self.fallback_model.to(self.device)
            self.fallback_model.eval()
            
            self.fallback_loaded = True
            logger.info("Fallback model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False
    
    def get_model_and_tokenizer(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Get available model and tokenizer"""
        if self.model_loaded:
            return self.primary_model, self.primary_tokenizer
        elif self.fallback_loaded:
            return self.fallback_model, self.fallback_tokenizer
        else:
            return None, None
    
    def is_available(self) -> bool:
        """Check if any model is available"""
        return self.model_loaded or self.fallback_loaded
    
    def cleanup(self):
        """Clean up model resources"""
        if self.primary_model:
            del self.primary_model
            self.primary_model = None
        if self.fallback_model:
            del self.fallback_model
            self.fallback_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class TextChunker:
    """Intelligent text chunking for GECToR processing"""
    
    def __init__(self, max_length: int = 512, overlap: int = 50):
        self.max_length = max_length
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if len(text) <= self.max_length:
            return [{'text': text, 'start': 0, 'end': len(text)}]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.max_length, len(text))
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end
            
            chunk_text = text[start:end]
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end
            })
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap)
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within range"""
        # Look for sentence endings
        sentence_endings = ['.', '!', '?', '\n']
        
        best_pos = end
        for i in range(end - 1, start - 1, -1):
            if text[i] in sentence_endings:
                # Make sure it's not an abbreviation
                if text[i] == '.' and i > 0 and text[i-1].isupper():
                    continue
                best_pos = i + 1
                break
        
        return best_pos
    
    def merge_chunk_results(self, chunk_results: List[List[LayeredDetectionResult]], 
                          chunks: List[Dict[str, Any]]) -> List[LayeredDetectionResult]:
        """Merge results from overlapping chunks"""
        merged_results = []
        seen_positions = set()
        
        for i, (results, chunk) in enumerate(zip(chunk_results, chunks)):
            for result in results:
                # Adjust position to global coordinates
                if result.position is not None:
                    global_position = result.position + chunk['start']
                    
                    # Skip if we've already seen this position (from overlap)
                    if global_position in seen_positions:
                        continue
                    
                    seen_positions.add(global_position)
                    
                    # Create adjusted result
                    adjusted_result = LayeredDetectionResult(
                        layer_name=result.layer_name,
                        original_word=result.original_word,
                        suggestions=result.suggestions,
                        confidence_scores=result.confidence_scores,
                        detection_method=result.detection_method,
                        processing_time=result.processing_time,
                        error_type=result.error_type,
                        context=result.context,
                        position=global_position
                    )
                    
                    merged_results.append(adjusted_result)
        
        return merged_results

class CorrectionProcessor:
    """Processes GECToR corrections and extracts structured results"""
    
    def __init__(self, confidence_scorer: AdvancedConfidenceScorer, 
                 domain_validator: EnhancedDomainValidator):
        self.confidence_scorer = confidence_scorer
        self.domain_validator = domain_validator
    
    def process_correction(self, original: str, corrected: str, 
                         chunk_start: int = 0) -> List[LayeredDetectionResult]:
        """Process a single correction and extract results"""
        if original == corrected:
            return []
        
        # Find differences between original and corrected text
        corrections = self._extract_corrections(original, corrected)
        
        results = []
        for correction in corrections:
            # Determine error type
            error_type = self._classify_error_type(correction)
            
            # Calculate confidence
            confidence = self._calculate_confidence(correction, original)
            
            # Validate with domain knowledge
            domain_validation = self.domain_validator.validate_suggestion(
                correction['original'], 
                correction['corrected'], 
                original
            )
            
            # Adjust confidence based on domain validation
            if domain_validation.is_valid:
                confidence += domain_validation.confidence_adjustment
            
            confidence = max(0.0, min(100.0, confidence))
            
            # Create result
            result = LayeredDetectionResult(
                layer_name=DetectionLayer.GECTOR_TRANSFORMER,
                original_word=correction['original'],
                suggestions=[correction['corrected']] if correction['corrected'] else [],
                confidence_scores=[confidence],
                detection_method="gector_transformer",
                processing_time=0.0,  # Will be set by caller
                error_type=error_type,
                context=self._extract_context(original, correction['position'], 50),
                position=correction['position'] + chunk_start
            )
            
            results.append(result)
        
        return results
    
    def _extract_corrections(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """Extract individual corrections from text comparison"""
        import difflib
        
        corrections = []
        
        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, original, corrected)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                corrections.append({
                    'type': 'replace',
                    'original': original[i1:i2],
                    'corrected': corrected[j1:j2],
                    'position': i1
                })
            elif tag == 'delete':
                corrections.append({
                    'type': 'delete',
                    'original': original[i1:i2],
                    'corrected': '',
                    'position': i1
                })
            elif tag == 'insert':
                corrections.append({
                    'type': 'insert',
                    'original': '',
                    'corrected': corrected[j1:j2],
                    'position': i1
                })
        
        return corrections
    
    def _classify_error_type(self, correction: Dict[str, Any]) -> ErrorType:
        """Classify the type of error based on correction"""
        original = correction['original'].strip()
        corrected = correction['corrected'].strip()
        
        # Single word changes are likely spelling errors
        if ' ' not in original and ' ' not in corrected and original and corrected:
            return ErrorType.SPELLING
        
        # Punctuation changes
        if re.match(r'^[^\w\s]+$', original) or re.match(r'^[^\w\s]+$', corrected):
            return ErrorType.PUNCTUATION
        
        # Grammar changes (multiple words, insertions, deletions)
        return ErrorType.GRAMMAR
    
    def _calculate_confidence(self, correction: Dict[str, Any], context: str) -> float:
        """Calculate confidence for a correction"""
        original = correction['original']
        corrected = correction['corrected']
        
        # Base confidence from correction type
        if correction['type'] == 'replace':
            if len(original.split()) == 1 and len(corrected.split()) == 1:
                # Single word replacement - likely spelling
                base_confidence = 85.0
            else:
                # Multi-word replacement - grammar
                base_confidence = 75.0
        elif correction['type'] == 'insert':
            # Insertion - grammar
            base_confidence = 70.0
        elif correction['type'] == 'delete':
            # Deletion - grammar
            base_confidence = 70.0
        else:
            base_confidence = 60.0
        
        # Adjust based on edit distance for spelling corrections
        if original and corrected and correction['type'] == 'replace':
            edit_distance = self._levenshtein_distance(original.lower(), corrected.lower())
            max_len = max(len(original), len(corrected))
            
            if max_len > 0:
                similarity = 1 - (edit_distance / max_len)
                # Higher similarity = higher confidence for spelling
                base_confidence *= (0.7 + 0.3 * similarity)
        
        return base_confidence
    
    def _extract_context(self, text: str, position: int, window: int) -> str:
        """Extract context around a position"""
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

class EnhancedGECToRLayer(IDetectionLayer):
    """
    Enhanced GECToR detection layer with robust fallbacks and optimization.
    """
    
    def __init__(self, config: Optional[GECToRConfig] = None):
        self.config = config or GECToRConfig()
        self.model_manager = ModelManager(self.config)
        self.chunker = TextChunker(self.config.max_length, self.config.chunk_overlap)
        
        # Initialize supporting components
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.domain_validator = EnhancedDomainValidator()
        self.processor = CorrectionProcessor(self.confidence_scorer, self.domain_validator)
        
        # Performance tracking
        self.performance_metrics = LayerPerformanceMetrics(
            layer_name=DetectionLayer.GECTOR_TRANSFORMER
        )
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Enhanced GECToR layer initialized")
    
    def _initialize_models(self):
        """Initialize GECToR models with fallback strategy"""
        # Try to load primary model
        if not self.model_manager.load_primary_model():
            logger.warning("Primary model failed to load, trying fallback")
            
            # Try fallback model
            if not self.model_manager.load_fallback_model():
                logger.error("Both primary and fallback models failed to load")
            else:
                logger.info("Fallback model loaded successfully")
        else:
            logger.info("Primary model loaded successfully")
    
    def detect(self, text: str, config: AnalysisConfig) -> List[LayeredDetectionResult]:
        """
        Detect errors using GECToR with intelligent chunking and fallbacks.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            List of detection results
        """
        start_time = time.time()
        
        try:
            # Check if model is available
            if not self.is_available():
                logger.warning("GECToR model not available")
                return []
            
            # Get model and tokenizer
            model, tokenizer = self.model_manager.get_model_and_tokenizer()
            if not model or not tokenizer:
                logger.error("No model or tokenizer available")
                return []
            
            # Chunk text if needed
            if self.config.enable_chunking and len(text) > self.config.max_length:
                chunks = self.chunker.chunk_text(text)
                chunk_results = []
                
                for chunk in chunks:
                    chunk_result = self._process_chunk(chunk['text'], model, tokenizer)
                    chunk_results.append(chunk_result)
                
                # Merge results from chunks
                results = self.chunker.merge_chunk_results(chunk_results, chunks)
            else:
                # Process as single chunk
                results = self._process_chunk(text, model, tokenizer)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics.total_processed += 1
            self.performance_metrics.total_processing_time += processing_time
            self.performance_metrics.success_rate = (
                self.performance_metrics.total_processed - self.performance_metrics.error_count
            ) / self.performance_metrics.total_processed
            
            # Update processing time for all results
            for result in results:
                result.processing_time = processing_time / len(results) if results else processing_time
            
            logger.debug(
                f"GECToR detection completed: {len(results)} results in {processing_time:.3f}s",
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
                f"GECToR detection failed: {e}",
                extra={
                    'text_length': len(text),
                    'processing_time': processing_time,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            
            return []
    
    def _process_chunk(self, text: str, model: Any, tokenizer: Any) -> List[LayeredDetectionResult]:
        """Process a single text chunk"""
        try:
            # Tokenize input
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            # Generate correction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=self.config.max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode output
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Process corrections
            results = self.processor.process_correction(text, corrected_text)
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in results 
                if result.confidence_scores and max(result.confidence_scores) >= self.config.confidence_threshold * 100
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return []
    
    def get_layer_name(self) -> DetectionLayer:
        """Get the layer identifier"""
        return DetectionLayer.GECTOR_TRANSFORMER
    
    def is_available(self) -> bool:
        """Check if the layer is available"""
        return self.model_manager.is_available()
    
    def get_performance_metrics(self) -> LayerPerformanceMetrics:
        """Get current performance metrics"""
        return self.performance_metrics
    
    def validate_config(self, config: AnalysisConfig) -> bool:
        """Validate if configuration is compatible"""
        # Check if GECToR is enabled in config
        return getattr(config, 'enable_gector', False)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.model_manager.cleanup()
        logger.info("GECToR layer cleanup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'primary_model_loaded': self.model_manager.model_loaded,
            'fallback_model_loaded': self.model_manager.fallback_loaded,
            'device': self.model_manager.device,
            'model_name': self.config.model_name,
            'fallback_model': self.config.fallback_model,
            'max_length': self.config.max_length,
            'confidence_threshold': self.config.confidence_threshold
        }
    
    def update_config(self, new_config: GECToRConfig) -> bool:
        """Update configuration and reload models if necessary"""
        old_model_name = self.config.model_name
        self.config = new_config
        
        # Reload models if model name changed
        if old_model_name != new_config.model_name:
            self.model_manager.cleanup()
            self.model_manager = ModelManager(new_config)
            self._initialize_models()
            return self.is_available()
        
        return True
    
    def benchmark_performance(self, test_texts: List[str]) -> Dict[str, float]:
        """Benchmark layer performance on test texts"""
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