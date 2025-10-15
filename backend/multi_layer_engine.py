"""
Multi-layer detection engine for enhanced typo detection.
Orchestrates multiple detection layers with intelligent coordination and fallback strategies.
"""

import asyncio
import threading
import time
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
from datetime import datetime

from enhanced_models import (
    AnalysisConfig, DetectionLayer, LayeredDetectionResult, MultiLayerAnalysisResult,
    EnhancedTypoResult, EnhancedGrammarResult, SystemPerformanceMetrics,
    ProcessingStatus, ErrorType, AnalysisError, LayerError
)
from core_interfaces import (
    IMultiLayerEngine, IDetectionLayer, ICacheManager, IErrorHandler,
    IPerformanceMonitor, ITextPreprocessor
)
from error_handling import EnhancedErrorHandler
from intelligent_cache import IntelligentCacheManager
from logging_config import log_performance_metrics

logger = logging.getLogger(__name__)

@dataclass
class LayerExecutionResult:
    """Result of executing a detection layer"""
    layer: DetectionLayer
    results: List[LayeredDetectionResult]
    processing_time: float
    success: bool
    error: Optional[Exception] = None

class TextChunk:
    """Represents a chunk of text for processing"""
    def __init__(self, text: str, start_position: int, chunk_id: int):
        self.text = text
        self.start_position = start_position
        self.chunk_id = chunk_id
        self.end_position = start_position + len(text)
    
    def adjust_positions(self, results: List[LayeredDetectionResult]) -> List[LayeredDetectionResult]:
        """Adjust result positions to account for chunk offset"""
        adjusted_results = []
        for result in results:
            adjusted_result = LayeredDetectionResult(
                layer_name=result.layer_name,
                original_word=result.original_word,
                suggestions=result.suggestions,
                confidence_scores=result.confidence_scores,
                detection_method=result.detection_method,
                processing_time=result.processing_time,
                error_type=result.error_type,
                context=result.context,
                position=(result.position + self.start_position) if result.position else None
            )
            adjusted_results.append(adjusted_result)
        return adjusted_results

class IntelligentTextPreprocessor(ITextPreprocessor):
    """Advanced text preprocessor with intelligent chunking"""
    
    def __init__(self):
        self.sentence_boundaries = r'[.!?]+\s+'
        self.paragraph_boundaries = r'\n\s*\n'
        
    def preprocess(self, text: str) -> str:
        """Preprocess text for optimal analysis"""
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 512) -> List[TextChunk]:
        """Split text into optimal chunks for processing"""
        if len(text) <= chunk_size:
            return [TextChunk(text, 0, 0)]
        
        chunks = []
        chunk_id = 0
        
        # First, try to split by paragraphs
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(TextChunk(current_chunk.strip(), current_position, chunk_id))
                    chunk_id += 1
                    current_position += len(current_chunk) + 1
                
                # If paragraph itself is too large, split by sentences
                if len(paragraph) > chunk_size:
                    sentence_chunks = self._split_by_sentences(paragraph, chunk_size)
                    for sentence_chunk in sentence_chunks:
                        chunks.append(TextChunk(sentence_chunk, current_position, chunk_id))
                        chunk_id += 1
                        current_position += len(sentence_chunk) + 1
                    current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(TextChunk(current_chunk.strip(), current_position, chunk_id))
        
        return chunks
    
    def _split_by_sentences(self, text: str, max_size: int) -> List[str]:
        """Split text by sentences when paragraphs are too large"""
        import re
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single sentence is too large, split by words
                if len(sentence) > max_size:
                    word_chunks = self._split_by_words(sentence, max_size)
                    chunks.extend(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_words(self, text: str, max_size: int) -> List[str]:
        """Split text by words as last resort"""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_context(self, text: str, position: int, window_size: int = 100) -> str:
        """Extract context around a specific position"""
        start = max(0, position - window_size)
        end = min(len(text), position + window_size)
        return text[start:end]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing"""
        # Convert to lowercase for comparison
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove punctuation for word matching
        import string
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        
        return normalized

class LayerCoordinator:
    """Coordinates execution of multiple detection layers"""
    
    def __init__(self, error_handler: IErrorHandler, cache_manager: ICacheManager):
        self.error_handler = error_handler
        self.cache_manager = cache_manager
        self.registered_layers: Dict[DetectionLayer, IDetectionLayer] = {}
        self.layer_priorities: Dict[DetectionLayer, int] = {
            DetectionLayer.DOMAIN_VALIDATION: 3,  # Fastest, run first
            DetectionLayer.TRADITIONAL_NLP: 2,    # Medium speed
            DetectionLayer.GECTOR_TRANSFORMER: 1  # Slowest, run last
        }
    
    def register_layer(self, layer: IDetectionLayer) -> None:
        """Register a detection layer"""
        layer_name = layer.get_layer_name()
        self.registered_layers[layer_name] = layer
        logger.info(f"Registered detection layer: {layer_name}")
    
    def get_available_layers(self, config: AnalysisConfig) -> List[DetectionLayer]:
        """Get list of available and enabled layers"""
        available_layers = []
        
        for layer_name, layer_impl in self.registered_layers.items():
            # Check if layer is available
            if not layer_impl.is_available():
                continue
            
            # Check if layer is enabled in config
            if layer_name == DetectionLayer.TRADITIONAL_NLP and not config.enable_traditional_nlp:
                continue
            elif layer_name == DetectionLayer.GECTOR_TRANSFORMER and not config.enable_gector:
                continue
            elif layer_name == DetectionLayer.DOMAIN_VALIDATION and not config.enable_domain_validation:
                continue
            
            # Check if layer is healthy
            healthy_layers = self.error_handler.get_healthy_layers()
            if layer_name not in healthy_layers:
                continue
            
            available_layers.append(layer_name)
        
        # Sort by priority
        available_layers.sort(key=lambda x: self.layer_priorities.get(x, 0), reverse=True)
        return available_layers
    
    def execute_layers_parallel(self, text_chunks: List[TextChunk], 
                              config: AnalysisConfig) -> List[LayerExecutionResult]:
        """Execute multiple layers in parallel"""
        available_layers = self.get_available_layers(config)
        
        if not available_layers:
            logger.warning("No available detection layers")
            return []
        
        execution_results = []
        
        with ThreadPoolExecutor(max_workers=min(len(available_layers), 4)) as executor:
            # Submit tasks for each layer
            future_to_layer = {}
            
            for layer_name in available_layers:
                layer_impl = self.registered_layers[layer_name]
                
                # Submit task with timeout
                future = executor.submit(
                    self._execute_layer_with_timeout,
                    layer_impl, text_chunks, config
                )
                future_to_layer[future] = layer_name
            
            # Collect results as they complete
            for future in as_completed(future_to_layer, timeout=config.max_processing_time):
                layer_name = future_to_layer[future]
                
                try:
                    result = future.result(timeout=2.0)  # Individual layer timeout
                    execution_results.append(result)
                    
                    # Record success
                    self.error_handler.record_layer_success(layer_name)
                    
                except TimeoutError:
                    error = TimeoutError(f"Layer {layer_name} timed out")
                    layer_error = self.error_handler.handle_layer_error(layer_name, error)
                    
                    execution_results.append(LayerExecutionResult(
                        layer=layer_name,
                        results=[],
                        processing_time=config.max_processing_time,
                        success=False,
                        error=error
                    ))
                    
                except Exception as e:
                    layer_error = self.error_handler.handle_layer_error(layer_name, e)
                    
                    execution_results.append(LayerExecutionResult(
                        layer=layer_name,
                        results=[],
                        processing_time=0.0,
                        success=False,
                        error=e
                    ))
        
        return execution_results
    
    def _execute_layer_with_timeout(self, layer: IDetectionLayer, 
                                  text_chunks: List[TextChunk], 
                                  config: AnalysisConfig) -> LayerExecutionResult:
        """Execute a single layer with timeout protection"""
        start_time = time.time()
        layer_name = layer.get_layer_name()
        
        try:
            all_results = []
            
            # Process each chunk
            for chunk in text_chunks:
                # Check cache first
                cache_key = self.cache_manager.create_cache_key(chunk.text, config, layer_name)
                cached_result = self.cache_manager.get(cache_key)
                
                if cached_result:
                    # Adjust positions for cached results
                    adjusted_results = chunk.adjust_positions(cached_result)
                    all_results.extend(adjusted_results)
                else:
                    # Execute layer detection
                    chunk_results = layer.detect(chunk.text, config)
                    
                    # Adjust positions
                    adjusted_results = chunk.adjust_positions(chunk_results)
                    all_results.extend(adjusted_results)
                    
                    # Cache results
                    self.cache_manager.set(cache_key, chunk_results)
            
            processing_time = time.time() - start_time
            
            return LayerExecutionResult(
                layer=layer_name,
                results=all_results,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return LayerExecutionResult(
                layer=layer_name,
                results=[],
                processing_time=processing_time,
                success=False,
                error=e
            )

class ResultAggregator:
    """Aggregates and deduplicates results from multiple layers"""
    
    def __init__(self):
        self.position_tolerance = 5  # Characters
    
    def aggregate_results(self, layer_results: List[LayerExecutionResult]) -> Tuple[List[EnhancedTypoResult], List[EnhancedGrammarResult]]:
        """Aggregate results from multiple layers"""
        # Separate successful results
        successful_results = [r for r in layer_results if r.success]
        
        if not successful_results:
            return [], []
        
        # Collect all detection results
        all_detections = []
        for layer_result in successful_results:
            for detection in layer_result.results:
                all_detections.append(detection)
        
        # Group by position and type
        typo_groups = self._group_detections(
            [d for d in all_detections if d.error_type == ErrorType.SPELLING]
        )
        grammar_groups = self._group_detections(
            [d for d in all_detections if d.error_type == ErrorType.GRAMMAR]
        )
        
        # Create enhanced results
        enhanced_typos = [self._create_enhanced_typo(group) for group in typo_groups]
        enhanced_grammar = [self._create_enhanced_grammar(group) for group in grammar_groups]
        
        return enhanced_typos, enhanced_grammar
    
    def _group_detections(self, detections: List[LayeredDetectionResult]) -> List[List[LayeredDetectionResult]]:
        """Group detections by position and similarity"""
        if not detections:
            return []
        
        # Sort by position
        detections.sort(key=lambda x: x.position or 0)
        
        groups = []
        current_group = [detections[0]]
        
        for detection in detections[1:]:
            # Check if this detection should be grouped with current group
            if self._should_group_detections(current_group[0], detection):
                current_group.append(detection)
            else:
                groups.append(current_group)
                current_group = [detection]
        
        groups.append(current_group)
        return groups
    
    def _should_group_detections(self, detection1: LayeredDetectionResult, 
                               detection2: LayeredDetectionResult) -> bool:
        """Determine if two detections should be grouped together"""
        # Check position proximity
        pos1 = detection1.position or 0
        pos2 = detection2.position or 0
        
        if abs(pos1 - pos2) > self.position_tolerance:
            return False
        
        # Check word similarity
        word1 = detection1.original_word.lower()
        word2 = detection2.original_word.lower()
        
        if word1 == word2:
            return True
        
        # Check if words overlap significantly
        overlap = len(set(word1) & set(word2)) / max(len(word1), len(word2))
        return overlap > 0.7
    
    def _create_enhanced_typo(self, detections: List[LayeredDetectionResult]) -> EnhancedTypoResult:
        """Create enhanced typo result from grouped detections"""
        # Use the detection with highest confidence as primary
        primary = max(detections, key=lambda x: max(x.confidence_scores) if x.confidence_scores else 0)
        
        # Calculate ensemble confidence
        all_confidences = []
        for detection in detections:
            all_confidences.extend(detection.confidence_scores)
        
        ensemble_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Get best suggestion (most common among high-confidence suggestions)
        suggestion_votes = {}
        for detection in detections:
            for i, suggestion in enumerate(detection.suggestions):
                confidence = detection.confidence_scores[i] if i < len(detection.confidence_scores) else 0
                if confidence > 70:  # Only consider high-confidence suggestions
                    suggestion_votes[suggestion] = suggestion_votes.get(suggestion, 0) + confidence
        
        best_suggestion = max(suggestion_votes.keys(), key=suggestion_votes.get) if suggestion_votes else primary.suggestions[0] if primary.suggestions else primary.original_word
        
        # Generate explanation
        layer_names = [d.layer_name.value for d in detections]
        explanation = f"Detected by {', '.join(set(layer_names))} with {ensemble_confidence:.0f}% confidence"
        
        return EnhancedTypoResult(
            word=primary.original_word,
            suggestion=best_suggestion,
            confidence_score=ensemble_confidence,
            explanation=explanation,
            context=primary.context,
            validation_status="validated",
            position=primary.position,
            error_type=ErrorType.SPELLING,
            layer_results=detections
        )
    
    def _create_enhanced_grammar(self, detections: List[LayeredDetectionResult]) -> EnhancedGrammarResult:
        """Create enhanced grammar result from grouped detections"""
        # Use the detection with highest confidence as primary
        primary = max(detections, key=lambda x: max(x.confidence_scores) if x.confidence_scores else 0)
        
        # Calculate ensemble confidence
        all_confidences = []
        for detection in detections:
            all_confidences.extend(detection.confidence_scores)
        
        ensemble_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Get best suggestion
        best_suggestion = primary.suggestions[0] if primary.suggestions else "No suggestion available"
        
        # Generate explanation
        layer_names = [d.layer_name.value for d in detections]
        explanation = f"Grammar issue detected by {', '.join(set(layer_names))} with {ensemble_confidence:.0f}% confidence"
        
        return EnhancedGrammarResult(
            sentence=primary.context,
            suggestion=best_suggestion,
            confidence_score=ensemble_confidence,
            explanation=explanation,
            issue_type=primary.detection_method,
            rule_category="multi_layer",
            validation_status="validated",
            position=primary.position,
            error_type=ErrorType.GRAMMAR,
            layer_results=detections
        )

class MultiLayerDetectionEngine(IMultiLayerEngine):
    """
    Advanced multi-layer detection engine with intelligent coordination.
    """
    
    def __init__(self, cache_manager: Optional[ICacheManager] = None,
                 error_handler: Optional[IErrorHandler] = None):
        self.cache_manager = cache_manager or IntelligentCacheManager()
        self.error_handler = error_handler or EnhancedErrorHandler()
        self.preprocessor = IntelligentTextPreprocessor()
        self.coordinator = LayerCoordinator(self.error_handler, self.cache_manager)
        self.aggregator = ResultAggregator()
        
        # Performance tracking
        self.performance_metrics = SystemPerformanceMetrics()
        self.analysis_count = 0
        
        logger.info("Multi-layer detection engine initialized")
    
    def analyze(self, text: str, config: AnalysisConfig) -> MultiLayerAnalysisResult:
        """
        Perform comprehensive multi-layer analysis.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            Complete multi-layer analysis result
            
        Raises:
            AnalysisError: If analysis fails completely
        """
        start_time = time.time()
        self.analysis_count += 1
        
        with log_performance_metrics(logger, f"multi_layer_analysis_{self.analysis_count}"):
            try:
                # Preprocess text
                processed_text = self.preprocessor.preprocess(text)
                
                # Check cache for complete analysis
                cache_key = self.cache_manager.create_cache_key(processed_text, config)
                cached_result = self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info("Returning cached analysis result")
                    cached_result.cache_hits = 1
                    return cached_result
                
                # Split into chunks for parallel processing
                text_chunks = self.preprocessor.chunk_text(processed_text, config.chunk_size)
                
                # Execute layers in parallel
                layer_results = self.coordinator.execute_layers_parallel(text_chunks, config)
                
                # Check if any layers succeeded
                successful_layers = [r.layer for r in layer_results if r.success]
                failed_layers = [r.layer for r in layer_results if not r.success]
                
                if not successful_layers:
                    # All layers failed - create error
                    layer_errors = [
                        LayerError(
                            layer_name=r.layer,
                            error_message=str(r.error) if r.error else "Unknown error",
                            error_type="execution_failure",
                            recoverable=True
                        ) for r in layer_results if not r.success
                    ]
                    
                    analysis_error = self.error_handler.create_analysis_error(
                        "All detection layers failed",
                        layer_errors,
                        fallback_used=False
                    )
                    
                    raise AnalysisError(
                        error_message=analysis_error.error_message,
                        error_code=analysis_error.error_code,
                        layer_errors=analysis_error.layer_errors
                    )
                
                # Aggregate results
                enhanced_typos, enhanced_grammar = self.aggregator.aggregate_results(layer_results)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Update performance metrics
                self._update_performance_metrics(processing_time, len(enhanced_typos) + len(enhanced_grammar))
                
                # Determine processing status
                if failed_layers:
                    processing_status = ProcessingStatus.PARTIAL_SUCCESS
                else:
                    processing_status = ProcessingStatus.SUCCESS
                
                # Create result
                result = MultiLayerAnalysisResult(
                    typos=enhanced_typos,
                    grammar_issues=enhanced_grammar,
                    processing_time=processing_time,
                    layers_used=successful_layers,
                    performance_metrics=self.performance_metrics,
                    config_used=config,
                    processing_status=processing_status,
                    cache_hits=0,
                    total_suggestions=len(enhanced_typos) + len(enhanced_grammar)
                )
                
                # Cache result
                self.cache_manager.set(cache_key, result)
                
                logger.info(
                    f"Analysis completed: {len(enhanced_typos)} typos, {len(enhanced_grammar)} grammar issues",
                    extra={
                        'processing_time': processing_time,
                        'layers_used': [layer.value for layer in successful_layers],
                        'failed_layers': [layer.value for layer in failed_layers],
                        'total_suggestions': result.total_suggestions
                    }
                )
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                logger.error(
                    f"Multi-layer analysis failed: {str(e)}",
                    extra={
                        'processing_time': processing_time,
                        'text_length': len(text),
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
                
                # Create analysis error
                analysis_error = self.error_handler.create_analysis_error(
                    f"Analysis failed: {str(e)}",
                    [],
                    fallback_used=False
                )
                
                raise AnalysisError(
                    error_message=analysis_error.error_message,
                    error_code=analysis_error.error_code,
                    layer_errors=analysis_error.layer_errors
                )
    
    def register_layer(self, layer: IDetectionLayer) -> None:
        """Register a new detection layer"""
        self.coordinator.register_layer(layer)
    
    def unregister_layer(self, layer_name: DetectionLayer) -> None:
        """Unregister a detection layer"""
        if layer_name in self.coordinator.registered_layers:
            del self.coordinator.registered_layers[layer_name]
            logger.info(f"Unregistered detection layer: {layer_name}")
    
    def get_available_layers(self) -> List[DetectionLayer]:
        """Get list of available detection layers"""
        return list(self.coordinator.registered_layers.keys())
    
    def validate_system_health(self) -> Dict[str, bool]:
        """Validate the health of all system components"""
        health_status = {}
        
        # Check registered layers
        for layer_name, layer_impl in self.coordinator.registered_layers.items():
            try:
                health_status[f"layer_{layer_name.value}"] = layer_impl.is_available()
            except Exception as e:
                health_status[f"layer_{layer_name.value}"] = False
                logger.warning(f"Layer {layer_name} health check failed: {e}")
        
        # Check cache manager
        try:
            cache_stats = self.cache_manager.get_stats()
            health_status["cache_manager"] = True
        except Exception as e:
            health_status["cache_manager"] = False
            logger.warning(f"Cache manager health check failed: {e}")
        
        # Check error handler
        try:
            error_summary = self.error_handler.get_error_summary()
            health_status["error_handler"] = True
        except Exception as e:
            health_status["error_handler"] = False
            logger.warning(f"Error handler health check failed: {e}")
        
        # Overall system health
        healthy_components = sum(1 for status in health_status.values() if status)
        total_components = len(health_status)
        health_status["overall_system"] = healthy_components / total_components > 0.7
        
        return health_status
    
    def optimize_performance(self) -> None:
        """Optimize system performance based on metrics"""
        logger.info("Starting performance optimization")
        
        # Optimize cache
        self.cache_manager.optimize()
        
        # Update layer reliability tracking
        for layer_name in self.coordinator.registered_layers:
            if hasattr(self.error_handler, 'layer_tracker'):
                self.error_handler.layer_tracker.update_calibration(layer_name)
        
        # Log optimization results
        health_status = self.validate_system_health()
        healthy_layers = sum(1 for k, v in health_status.items() if k.startswith('layer_') and v)
        
        logger.info(
            f"Performance optimization completed: {healthy_layers} healthy layers",
            extra={
                'healthy_layers': healthy_layers,
                'cache_optimized': health_status.get('cache_manager', False),
                'overall_health': health_status.get('overall_system', False)
            }
        )
    
    def _update_performance_metrics(self, processing_time: float, suggestion_count: int) -> None:
        """Update system performance metrics"""
        self.performance_metrics.total_texts_analyzed += 1
        self.performance_metrics.total_processing_time += processing_time
        
        # Update cache hit rate
        cache_stats = self.cache_manager.get_stats()
        if 'global_stats' in cache_stats:
            self.performance_metrics.cache_hit_rate = cache_stats['global_stats'].get('hit_rate', 0.0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        health_status = self.validate_system_health()
        cache_stats = self.cache_manager.get_stats()
        error_summary = self.error_handler.get_error_summary()
        
        return {
            'system_health': health_status,
            'performance_metrics': {
                'total_analyses': self.performance_metrics.total_texts_analyzed,
                'average_processing_time': self.performance_metrics.average_processing_time,
                'cache_hit_rate': self.performance_metrics.cache_hit_rate
            },
            'cache_performance': cache_stats,
            'error_statistics': error_summary,
            'layer_status': {
                layer_name.value: {
                    'registered': True,
                    'available': layer_impl.is_available(),
                    'healthy': layer_name in self.error_handler.get_healthy_layers()
                } for layer_name, layer_impl in self.coordinator.registered_layers.items()
            }
        }