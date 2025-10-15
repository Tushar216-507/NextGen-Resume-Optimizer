"""
Core interfaces and abstract base classes for the enhanced typo detection system.
Designed with SOLID principles for maximum extensibility and maintainability.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from enhanced_models import (
    LayeredDetectionResult, EnsembleResult, MultiLayerAnalysisResult,
    AnalysisConfig, LayerConfig, DetectionLayer, CacheKey, CacheEntry,
    LayerPerformanceMetrics, SystemPerformanceMetrics, ValidationResult,
    ValidationRule, AnalysisError, LayerError
)

class IDetectionLayer(ABC):
    """Interface for all detection layers"""
    
    @abstractmethod
    def detect(self, text: str, config: AnalysisConfig) -> List[LayeredDetectionResult]:
        """
        Detect errors in the given text.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            List of detection results
            
        Raises:
            LayerError: If detection fails
        """
        pass
    
    @abstractmethod
    def get_layer_name(self) -> DetectionLayer:
        """Get the unique identifier for this layer"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this layer is available and functional"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> LayerPerformanceMetrics:
        """Get current performance metrics for this layer"""
        pass
    
    @abstractmethod
    def validate_config(self, config: AnalysisConfig) -> bool:
        """Validate if the configuration is compatible with this layer"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by this layer"""
        pass

class IConfidenceScorer(ABC):
    """Interface for confidence scoring systems"""
    
    @abstractmethod
    def score_spelling_suggestion(self, original: str, suggestion: str, 
                                context: str, metadata: Dict[str, Any]) -> float:
        """
        Score a spelling suggestion.
        
        Args:
            original: Original word
            suggestion: Suggested correction
            context: Surrounding text context
            metadata: Additional metadata (resume context, etc.)
            
        Returns:
            Confidence score (0-100)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def calibrate_confidence(self, raw_score: float, layer: DetectionLayer,
                           historical_accuracy: float) -> float:
        """
        Calibrate confidence score based on historical performance.
        
        Args:
            raw_score: Raw confidence score
            layer: Detection layer that generated the score
            historical_accuracy: Historical accuracy of the layer
            
        Returns:
            Calibrated confidence score
        """
        pass

class IDomainValidator(ABC):
    """Interface for domain-specific validation"""
    
    @abstractmethod
    def is_valid_technical_term(self, word: str, context: str) -> bool:
        """Check if a word is a valid technical term in context"""
        pass
    
    @abstractmethod
    def get_context_appropriate_suggestions(self, word: str, context: str) -> List[str]:
        """Get context-appropriate suggestions for a word"""
        pass
    
    @abstractmethod
    def validate_suggestion(self, original: str, suggestion: str, 
                          context: str) -> ValidationResult:
        """Validate a suggestion against domain rules"""
        pass
    
    @abstractmethod
    def get_industry_terms(self, industry: str) -> List[str]:
        """Get terms specific to an industry"""
        pass
    
    @abstractmethod
    def update_vocabulary(self, new_terms: List[str]) -> None:
        """Update the domain vocabulary with new terms"""
        pass

class IEnsembleValidator(ABC):
    """Interface for ensemble validation systems"""
    
    @abstractmethod
    def validate_ensemble(self, layer_results: List[LayeredDetectionResult],
                         config: AnalysisConfig) -> EnsembleResult:
        """
        Validate and combine results from multiple layers.
        
        Args:
            layer_results: Results from different detection layers
            config: Analysis configuration
            
        Returns:
            Ensemble validation result
        """
        pass
    
    @abstractmethod
    def resolve_conflicts(self, conflicting_results: List[LayeredDetectionResult]) -> str:
        """Resolve conflicts between different layer suggestions"""
        pass
    
    @abstractmethod
    def calculate_ensemble_confidence(self, layer_results: List[LayeredDetectionResult],
                                    weights: Dict[DetectionLayer, float]) -> float:
        """Calculate ensemble confidence from layer results"""
        pass

class ICacheManager(ABC):
    """Interface for cache management"""
    
    @abstractmethod
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get cached result by key"""
        pass
    
    @abstractmethod
    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached result with optional TTL"""
        pass
    
    @abstractmethod
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass
    
    @abstractmethod
    def optimize(self) -> None:
        """Optimize cache performance"""
        pass

class IPerformanceMonitor(ABC):
    """Interface for performance monitoring"""
    
    @abstractmethod
    def record_analysis(self, processing_time: float, accuracy_score: float,
                       layers_used: List[DetectionLayer]) -> None:
        """Record analysis performance metrics"""
        pass
    
    @abstractmethod
    def record_layer_performance(self, layer: DetectionLayer, processing_time: float,
                               success: bool, confidence: float) -> None:
        """Record individual layer performance"""
        pass
    
    @abstractmethod
    def get_system_metrics(self) -> SystemPerformanceMetrics:
        """Get overall system performance metrics"""
        pass
    
    @abstractmethod
    def get_layer_metrics(self, layer: DetectionLayer) -> LayerPerformanceMetrics:
        """Get metrics for a specific layer"""
        pass
    
    @abstractmethod
    def detect_performance_degradation(self) -> List[str]:
        """Detect performance issues and return recommendations"""
        pass

class IErrorHandler(ABC):
    """Interface for error handling and recovery"""
    
    @abstractmethod
    def handle_layer_error(self, layer: DetectionLayer, error: Exception) -> LayerError:
        """Handle and log layer-specific errors"""
        pass
    
    @abstractmethod
    def determine_fallback_strategy(self, failed_layers: List[DetectionLayer],
                                  config: AnalysisConfig) -> List[DetectionLayer]:
        """Determine which layers to use as fallbacks"""
        pass
    
    @abstractmethod
    def is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable"""
        pass
    
    @abstractmethod
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        pass

class ITextPreprocessor(ABC):
    """Interface for text preprocessing"""
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text for analysis"""
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into optimal chunks for processing"""
        pass
    
    @abstractmethod
    def extract_context(self, text: str, position: int, window_size: int) -> str:
        """Extract context around a specific position"""
        pass
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing"""
        pass

class IMultiLayerEngine(ABC):
    """Interface for the main multi-layer detection engine"""
    
    @abstractmethod
    def analyze(self, text: str, config: AnalysisConfig) -> MultiLayerAnalysisResult:
        """
        Perform multi-layer analysis on text.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            Complete analysis result
            
        Raises:
            AnalysisError: If analysis fails completely
        """
        pass
    
    @abstractmethod
    def register_layer(self, layer: IDetectionLayer) -> None:
        """Register a new detection layer"""
        pass
    
    @abstractmethod
    def unregister_layer(self, layer_name: DetectionLayer) -> None:
        """Unregister a detection layer"""
        pass
    
    @abstractmethod
    def get_available_layers(self) -> List[DetectionLayer]:
        """Get list of available detection layers"""
        pass
    
    @abstractmethod
    def validate_system_health(self) -> Dict[str, bool]:
        """Validate the health of all system components"""
        pass
    
    @abstractmethod
    def optimize_performance(self) -> None:
        """Optimize system performance based on metrics"""
        pass

class IExplanationGenerator(ABC):
    """Interface for generating explanations"""
    
    @abstractmethod
    def generate_spelling_explanation(self, original: str, suggestion: str,
                                    confidence: float, context: str) -> str:
        """Generate explanation for spelling corrections"""
        pass
    
    @abstractmethod
    def generate_grammar_explanation(self, issue_type: str, original: str,
                                   suggestion: str, confidence: float) -> str:
        """Generate explanation for grammar corrections"""
        pass
    
    @abstractmethod
    def generate_ensemble_explanation(self, ensemble_result: EnsembleResult,
                                    layer_results: List[LayeredDetectionResult]) -> str:
        """Generate explanation for ensemble decisions"""
        pass

# Factory Interfaces
class ILayerFactory(ABC):
    """Factory interface for creating detection layers"""
    
    @abstractmethod
    def create_layer(self, layer_type: DetectionLayer, config: LayerConfig) -> IDetectionLayer:
        """Create a detection layer instance"""
        pass
    
    @abstractmethod
    def get_supported_layers(self) -> List[DetectionLayer]:
        """Get list of supported layer types"""
        pass

class IValidatorFactory(ABC):
    """Factory interface for creating validators"""
    
    @abstractmethod
    def create_domain_validator(self, domain: str) -> IDomainValidator:
        """Create a domain-specific validator"""
        pass
    
    @abstractmethod
    def create_ensemble_validator(self, strategy: str) -> IEnsembleValidator:
        """Create an ensemble validator with specific strategy"""
        pass

# Configuration and Registry Interfaces
class IConfigurationManager(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def load_config(self, config_path: str) -> AnalysisConfig:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def save_config(self, config: AnalysisConfig, config_path: str) -> None:
        """Save configuration to file"""
        pass
    
    @abstractmethod
    def validate_config(self, config: AnalysisConfig) -> List[str]:
        """Validate configuration and return any errors"""
        pass
    
    @abstractmethod
    def get_default_config(self) -> AnalysisConfig:
        """Get default configuration"""
        pass

class IComponentRegistry(ABC):
    """Interface for component registry"""
    
    @abstractmethod
    def register_component(self, name: str, component: Any) -> None:
        """Register a system component"""
        pass
    
    @abstractmethod
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component"""
        pass
    
    @abstractmethod
    def unregister_component(self, name: str) -> None:
        """Unregister a component"""
        pass
    
    @abstractmethod
    def list_components(self) -> List[str]:
        """List all registered components"""
        pass