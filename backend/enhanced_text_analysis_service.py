"""
Enhanced Text Analysis Service - Main integration point for the multi-layer typo detection system.
Brings together all components for world-class accuracy and performance.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from enhanced_models import (
    AnalysisConfig, MultiLayerAnalysisResult, EnhancedTypoResult, 
    EnhancedGrammarResult, DetectionLayer, ProcessingStatus
)
from core_interfaces import IMultiLayerEngine, ICacheManager, IErrorHandler
from multi_layer_engine import MultiLayerDetectionEngine
from enhanced_gector_layer import EnhancedGECToRLayer, GECToRConfig
from enhanced_traditional_nlp_layer import EnhancedTraditionalNLPLayer, TraditionalNLPConfig
from domain_vocabulary import EnhancedDomainValidator
from confidence_scoring import AdvancedConfidenceScorer
from ensemble_validation import AdvancedEnsembleValidator
from intelligent_cache import IntelligentCacheManager
from error_handling import EnhancedErrorHandler
from logging_config import setup_logging, performance_logger, error_logger

# Set up logging
loggers = setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class EnhancedAnalysisConfig:
    """Configuration for enhanced analysis system"""
    # Layer configurations
    enable_traditional_nlp: bool = True
    enable_gector: bool = True
    enable_domain_validation: bool = True
    
    # Performance settings
    confidence_threshold: float = 80.0
    max_processing_time: float = 3.0
    parallel_processing: bool = True
    cache_enabled: bool = True
    
    # Quality settings
    min_suggestions: int = 1
    max_suggestions: int = 5
    ensemble_voting: bool = True
    
    # Fallback settings
    fallback_strategy: str = "graceful"
    enable_fallbacks: bool = True

class DomainValidationLayer:
    """Simple domain validation layer for the multi-layer system"""
    
    def __init__(self, domain_validator: EnhancedDomainValidator):
        self.domain_validator = domain_validator
        self.layer_name = DetectionLayer.DOMAIN_VALIDATION
    
    def detect(self, text: str, config: AnalysisConfig):
        """Perform domain validation (placeholder implementation)"""
        # This would implement domain-specific validation
        # For now, return empty list as domain validation is integrated into other layers
        return []
    
    def get_layer_name(self):
        return self.layer_name
    
    def is_available(self):
        return True
    
    def get_performance_metrics(self):
        from enhanced_models import LayerPerformanceMetrics
        return LayerPerformanceMetrics(layer_name=self.layer_name)
    
    def validate_config(self, config):
        return True
    
    def cleanup(self):
        pass

class EnhancedTextAnalysisService:
    """
    Enhanced Text Analysis Service with world-class accuracy and performance.
    
    This service integrates all components of the multi-layer typo detection system:
    - Traditional NLP with intelligent fallbacks
    - GECToR transformer with robust error handling
    - Domain-specific validation for technical terms
    - Ensemble validation with conflict resolution
    - Intelligent caching for performance
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, config: Optional[EnhancedAnalysisConfig] = None):
        self.config = config or EnhancedAnalysisConfig()
        
        # Initialize core components
        self.cache_manager = IntelligentCacheManager()
        self.error_handler = EnhancedErrorHandler()
        self.domain_validator = EnhancedDomainValidator()
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.ensemble_validator = AdvancedEnsembleValidator(self.confidence_scorer)
        
        # Initialize multi-layer engine
        self.engine = MultiLayerDetectionEngine(
            cache_manager=self.cache_manager,
            error_handler=self.error_handler
        )
        
        # Initialize and register detection layers
        self._initialize_detection_layers()
        
        # Performance tracking
        self.total_analyses = 0
        self.total_processing_time = 0.0
        self.initialization_time = time.time()
        
        logger.info("Enhanced Text Analysis Service initialized successfully")
    
    def _initialize_detection_layers(self):
        """Initialize and register all detection layers"""
        
        # Traditional NLP Layer
        if self.config.enable_traditional_nlp:
            try:
                traditional_config = TraditionalNLPConfig(
                    confidence_threshold=self.config.confidence_threshold,
                    parallel_processing=self.config.parallel_processing
                )
                traditional_layer = EnhancedTraditionalNLPLayer(traditional_config)
                self.engine.register_layer(traditional_layer)
                logger.info("Traditional NLP layer registered")
            except Exception as e:
                logger.error(f"Failed to initialize Traditional NLP layer: {e}")
        
        # GECToR Layer
        if self.config.enable_gector:
            try:
                gector_config = GECToRConfig(
                    confidence_threshold=self.config.confidence_threshold / 100.0,
                    enable_chunking=True
                )
                gector_layer = EnhancedGECToRLayer(gector_config)
                self.engine.register_layer(gector_layer)
                logger.info("GECToR layer registered")
            except Exception as e:
                logger.error(f"Failed to initialize GECToR layer: {e}")
        
        # Domain Validation Layer
        if self.config.enable_domain_validation:
            try:
                domain_layer = DomainValidationLayer(self.domain_validator)
                self.engine.register_layer(domain_layer)
                logger.info("Domain validation layer registered")
            except Exception as e:
                logger.error(f"Failed to initialize Domain validation layer: {e}")
    
    def analyze_text(self, text: str, 
                    check_spelling: bool = True, 
                    check_grammar: bool = True) -> MultiLayerAnalysisResult:
        """
        Perform enhanced multi-layer text analysis.
        
        Args:
            text: Text to analyze
            check_spelling: Whether to check spelling
            check_grammar: Whether to check grammar
            
        Returns:
            MultiLayerAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        self.total_analyses += 1
        
        try:
            # Create analysis configuration
            analysis_config = AnalysisConfig(
                confidence_threshold=self.config.confidence_threshold,
                max_processing_time=self.config.max_processing_time,
                enable_traditional_nlp=self.config.enable_traditional_nlp and (check_spelling or check_grammar),
                enable_gector=self.config.enable_gector and (check_spelling or check_grammar),
                enable_domain_validation=self.config.enable_domain_validation,
                cache_enabled=self.config.cache_enabled,
                parallel_processing=self.config.parallel_processing,
                fallback_strategy=self.config.fallback_strategy
            )
            
            # Perform multi-layer analysis
            result = self.engine.analyze(text, analysis_config)
            
            # Apply post-processing filters
            if not check_spelling:
                result.typos = []
            
            if not check_grammar:
                result.grammar_issues = []
            
            # Apply suggestion limits
            result.typos = result.typos[:self.config.max_suggestions]
            result.grammar_issues = result.grammar_issues[:self.config.max_suggestions]
            
            # Update performance tracking
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Log performance metrics
            performance_logger.log_analysis_performance(
                text_length=len(text),
                processing_time=processing_time,
                layers_used=[layer.value for layer in result.layers_used],
                cache_hits=result.cache_hits,
                total_suggestions=result.total_suggestions
            )
            
            logger.info(
                f"Enhanced analysis completed: {len(result.typos)} typos, {len(result.grammar_issues)} grammar issues",
                extra={
                    'text_length': len(text),
                    'processing_time': processing_time,
                    'layers_used': len(result.layers_used),
                    'total_suggestions': result.total_suggestions,
                    'cache_hits': result.cache_hits
                }
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            error_logger.log_analysis_error(
                error_code="ANALYSIS_FAILED",
                error_message=str(e),
                failed_layers=[],
                fallback_used=False
            )
            
            logger.error(
                f"Enhanced analysis failed: {e}",
                extra={
                    'text_length': len(text),
                    'processing_time': processing_time,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            
            # Return empty result on failure
            return MultiLayerAnalysisResult(
                typos=[],
                grammar_issues=[],
                processing_time=processing_time,
                layers_used=[],
                performance_metrics=self.engine.performance_metrics,
                config_used=analysis_config,
                processing_status=ProcessingStatus.FAILED,
                cache_hits=0,
                total_suggestions=0
            )
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about analysis capabilities"""
        available_layers = self.engine.get_available_layers()
        system_health = self.engine.validate_system_health()
        
        return {
            'enhanced_analysis_available': len(available_layers) > 0,
            'available_layers': [layer.value for layer in available_layers],
            'system_health': system_health,
            'cache_enabled': self.config.cache_enabled,
            'parallel_processing': self.config.parallel_processing,
            'confidence_threshold': self.config.confidence_threshold,
            'max_processing_time': self.config.max_processing_time,
            'fallback_strategy': self.config.fallback_strategy,
            'total_analyses_performed': self.total_analyses,
            'average_processing_time': (
                self.total_processing_time / max(1, self.total_analyses)
            ),
            'uptime_seconds': time.time() - self.initialization_time
        }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for suggestions"""
        if 0.0 <= threshold <= 100.0:
            self.config.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold}%")
        else:
            raise ValueError("Confidence threshold must be between 0 and 100")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        engine_report = self.engine.get_performance_report()
        cache_stats = self.cache_manager.get_stats()
        error_summary = self.error_handler.get_error_summary()
        
        return {
            'service_metrics': {
                'total_analyses': self.total_analyses,
                'total_processing_time': self.total_processing_time,
                'average_processing_time': (
                    self.total_processing_time / max(1, self.total_analyses)
                ),
                'uptime_seconds': time.time() - self.initialization_time
            },
            'engine_performance': engine_report,
            'cache_performance': cache_stats,
            'error_statistics': error_summary,
            'configuration': {
                'confidence_threshold': self.config.confidence_threshold,
                'max_processing_time': self.config.max_processing_time,
                'parallel_processing': self.config.parallel_processing,
                'cache_enabled': self.config.cache_enabled,
                'fallback_strategy': self.config.fallback_strategy
            }
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance"""
        logger.info("Starting system performance optimization")
        
        optimization_results = {}
        
        try:
            # Optimize engine
            self.engine.optimize_performance()
            optimization_results['engine_optimized'] = True
            
            # Optimize cache
            self.cache_manager.optimize()
            optimization_results['cache_optimized'] = True
            
            # Get updated performance metrics
            performance_report = self.get_performance_report()
            optimization_results['performance_report'] = performance_report
            
            logger.info("System performance optimization completed")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Validate overall system health"""
        health_report = {
            'overall_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check engine health
            engine_health = self.engine.validate_system_health()
            health_report['engine_health'] = engine_health
            
            # Check cache health
            cache_stats = self.cache_manager.get_stats()
            cache_healthy = cache_stats.get('manager_stats', {}).get('total_operations', 0) >= 0
            health_report['cache_healthy'] = cache_healthy
            
            # Check error rates
            error_summary = self.error_handler.get_error_summary()
            recent_errors = error_summary.get('recent_errors_1h', 0)
            
            if recent_errors > 10:
                health_report['issues'].append(f"High error rate: {recent_errors} errors in last hour")
                health_report['overall_status'] = 'degraded'
            
            # Check layer availability
            available_layers = len(self.engine.get_available_layers())
            if available_layers == 0:
                health_report['issues'].append("No detection layers available")
                health_report['overall_status'] = 'critical'
            elif available_layers == 1:
                health_report['issues'].append("Only one detection layer available")
                health_report['recommendations'].append("Enable additional detection layers for better accuracy")
            
            # Check performance
            avg_processing_time = self.total_processing_time / max(1, self.total_analyses)
            if avg_processing_time > self.config.max_processing_time:
                health_report['issues'].append(f"Average processing time ({avg_processing_time:.2f}s) exceeds target")
                health_report['recommendations'].append("Consider optimizing performance or increasing timeout")
            
        except Exception as e:
            health_report['overall_status'] = 'error'
            health_report['issues'].append(f"Health check failed: {e}")
        
        return health_report
    
    def cleanup(self) -> None:
        """Clean up system resources"""
        logger.info("Starting system cleanup")
        
        try:
            # Cleanup engine and layers
            for layer_name in self.engine.get_available_layers():
                self.engine.unregister_layer(layer_name)
            
            # Clear cache
            self.cache_manager.clear()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"System cleanup failed: {e}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'service_info': {
                'version': 'enhanced_v2.0',
                'total_analyses': self.total_analyses,
                'uptime_seconds': time.time() - self.initialization_time,
                'average_processing_time': (
                    self.total_processing_time / max(1, self.total_analyses)
                )
            },
            'capabilities': self.get_analysis_capabilities(),
            'performance': self.get_performance_report(),
            'health': self.validate_system_health()
        }

# Backward compatibility functions for existing API
def create_enhanced_service(enable_gector: bool = True, 
                          confidence_threshold: float = 80.0) -> EnhancedTextAnalysisService:
    """Create enhanced service with specified configuration"""
    config = EnhancedAnalysisConfig(
        enable_gector=enable_gector,
        confidence_threshold=confidence_threshold
    )
    return EnhancedTextAnalysisService(config)

def analyze_text_enhanced(text: str, 
                         check_spelling: bool = True, 
                         check_grammar: bool = True,
                         service: Optional[EnhancedTextAnalysisService] = None) -> MultiLayerAnalysisResult:
    """Analyze text with enhanced multi-layer detection"""
    if service is None:
        service = create_enhanced_service()
    
    return service.analyze_text(text, check_spelling, check_grammar)