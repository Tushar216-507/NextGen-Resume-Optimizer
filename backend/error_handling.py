"""
Comprehensive error handling and recovery system for enhanced typo detection.
Implements graceful degradation and intelligent fallback strategies.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enhanced_models import (
    DetectionLayer, AnalysisError, LayerError, AnalysisConfig,
    ProcessingStatus
)
from core_interfaces import IErrorHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity:
    """Error severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ErrorCategory:
    """Error categories for classification"""
    DEPENDENCY_MISSING = "dependency_missing"
    MODEL_LOADING = "model_loading"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_ERROR = "data_error"
    UNKNOWN = "unknown"

class RecoveryStrategy:
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"

class ErrorPattern:
    """Pattern for error detection and handling"""
    def __init__(self, error_type: str, pattern: str, severity: str, 
                 recovery_strategy: str, max_occurrences: int = 5):
        self.error_type = error_type
        self.pattern = pattern
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.max_occurrences = max_occurrences
        self.occurrence_count = 0
        self.last_occurrence = None

class EnhancedErrorHandler(IErrorHandler):
    """
    Advanced error handler with intelligent recovery strategies.
    Implements circuit breaker pattern and adaptive fallback mechanisms.
    """
    
    def __init__(self, max_error_history: int = 1000):
        self.error_history: deque = deque(maxlen=max_error_history)
        self.layer_health: Dict[DetectionLayer, Dict[str, Any]] = {}
        self.error_patterns: List[ErrorPattern] = []
        self.circuit_breakers: Dict[DetectionLayer, Dict[str, Any]] = {}
        self.recovery_statistics: Dict[str, int] = defaultdict(int)
        
        # Initialize layer health tracking
        for layer in DetectionLayer:
            self.layer_health[layer] = {
                'status': 'healthy',
                'error_count': 0,
                'last_error': None,
                'consecutive_failures': 0,
                'last_success': datetime.now(),
                'circuit_breaker_open': False,
                'circuit_breaker_open_time': None
            }
            
            self.circuit_breakers[layer] = {
                'failure_threshold': 5,
                'timeout_seconds': 60,
                'half_open_max_calls': 3,
                'half_open_calls': 0
            }
        
        self._initialize_error_patterns()
    
    def _initialize_error_patterns(self):
        """Initialize common error patterns and their handling strategies"""
        patterns = [
            # Java dependency errors
            ErrorPattern(
                error_type=ErrorCategory.DEPENDENCY_MISSING,
                pattern="No java install detected",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                max_occurrences=3
            ),
            # Model loading errors
            ErrorPattern(
                error_type=ErrorCategory.MODEL_LOADING,
                pattern="is not a local folder and is not a valid model identifier",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                max_occurrences=2
            ),
            # Network errors
            ErrorPattern(
                error_type=ErrorCategory.NETWORK_ERROR,
                pattern="Connection.*refused|Network.*unreachable|Timeout",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.RETRY,
                max_occurrences=3
            ),
            # Memory errors
            ErrorPattern(
                error_type=ErrorCategory.MEMORY_ERROR,
                pattern="Out of memory|CUDA out of memory",
                severity=ErrorSeverity.CRITICAL,
                recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_occurrences=1
            ),
            # Timeout errors
            ErrorPattern(
                error_type=ErrorCategory.TIMEOUT,
                pattern="Timeout|TimeoutError",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                max_occurrences=2
            )
        ]
        
        self.error_patterns.extend(patterns)
    
    def handle_layer_error(self, layer: DetectionLayer, error: Exception) -> LayerError:
        """
        Handle and classify layer-specific errors with intelligent recovery.
        
        Args:
            layer: The detection layer that failed
            error: The exception that occurred
            
        Returns:
            LayerError with classification and recovery information
        """
        error_message = str(error)
        error_type = self._classify_error(error_message)
        
        # Create layer error
        layer_error = LayerError(
            layer_name=layer,
            error_message=error_message,
            error_type=error_type,
            recoverable=self.is_recoverable_error(error)
        )
        
        # Update layer health
        self._update_layer_health(layer, layer_error)
        
        # Check circuit breaker
        self._check_circuit_breaker(layer)
        
        # Log error with context
        logger.error(
            f"Layer {layer} error: {error_message}",
            extra={
                'layer': layer,
                'error_type': error_type,
                'recoverable': layer_error.recoverable,
                'traceback': traceback.format_exc()
            }
        )
        
        # Add to error history
        self.error_history.append({
            'timestamp': datetime.now(),
            'layer': layer,
            'error': layer_error,
            'traceback': traceback.format_exc()
        })
        
        # Update recovery statistics
        self.recovery_statistics[f"{layer}_{error_type}"] += 1
        
        return layer_error
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error based on message patterns"""
        import re
        
        for pattern in self.error_patterns:
            if re.search(pattern.pattern, error_message, re.IGNORECASE):
                pattern.occurrence_count += 1
                pattern.last_occurrence = datetime.now()
                return pattern.error_type
        
        return ErrorCategory.UNKNOWN
    
    def _update_layer_health(self, layer: DetectionLayer, error: LayerError):
        """Update health status for a layer"""
        health = self.layer_health[layer]
        health['error_count'] += 1
        health['last_error'] = error
        health['consecutive_failures'] += 1
        
        # Determine health status
        if health['consecutive_failures'] >= 5:
            health['status'] = 'critical'
        elif health['consecutive_failures'] >= 3:
            health['status'] = 'degraded'
        else:
            health['status'] = 'unhealthy'
    
    def _check_circuit_breaker(self, layer: DetectionLayer):
        """Check and update circuit breaker status"""
        health = self.layer_health[layer]
        breaker = self.circuit_breakers[layer]
        
        # Open circuit breaker if failure threshold exceeded
        if (health['consecutive_failures'] >= breaker['failure_threshold'] and 
            not health['circuit_breaker_open']):
            health['circuit_breaker_open'] = True
            health['circuit_breaker_open_time'] = datetime.now()
            logger.warning(f"Circuit breaker opened for layer {layer}")
        
        # Check if circuit breaker should be closed
        elif health['circuit_breaker_open']:
            open_time = health['circuit_breaker_open_time']
            if open_time and (datetime.now() - open_time).seconds >= breaker['timeout_seconds']:
                # Try half-open state
                if breaker['half_open_calls'] < breaker['half_open_max_calls']:
                    logger.info(f"Circuit breaker half-open for layer {layer}")
                else:
                    # Close circuit breaker
                    health['circuit_breaker_open'] = False
                    health['circuit_breaker_open_time'] = None
                    breaker['half_open_calls'] = 0
                    health['consecutive_failures'] = 0
                    health['status'] = 'healthy'
                    logger.info(f"Circuit breaker closed for layer {layer}")
    
    def determine_fallback_strategy(self, failed_layers: List[DetectionLayer],
                                  config: AnalysisConfig) -> List[DetectionLayer]:
        """
        Determine optimal fallback strategy based on failed layers and config.
        
        Args:
            failed_layers: List of layers that have failed
            config: Current analysis configuration
            
        Returns:
            List of layers to use as fallbacks
        """
        available_layers = []
        
        # Get all possible layers
        all_layers = [
            DetectionLayer.TRADITIONAL_NLP,
            DetectionLayer.GECTOR_TRANSFORMER,
            DetectionLayer.DOMAIN_VALIDATION
        ]
        
        # Filter out failed layers and check health
        for layer in all_layers:
            if layer in failed_layers:
                continue
                
            health = self.layer_health[layer]
            
            # Skip if circuit breaker is open
            if health['circuit_breaker_open']:
                continue
                
            # Check configuration compatibility
            if layer == DetectionLayer.TRADITIONAL_NLP and config.enable_traditional_nlp:
                available_layers.append(layer)
            elif layer == DetectionLayer.GECTOR_TRANSFORMER and config.enable_gector:
                available_layers.append(layer)
            elif layer == DetectionLayer.DOMAIN_VALIDATION and config.enable_domain_validation:
                available_layers.append(layer)
        
        # Prioritize layers based on reliability and performance
        layer_priority = {
            DetectionLayer.DOMAIN_VALIDATION: 3,  # Fastest, most reliable
            DetectionLayer.TRADITIONAL_NLP: 2,    # Good balance
            DetectionLayer.GECTOR_TRANSFORMER: 1  # Slowest but potentially most accurate
        }
        
        # Sort by priority and health
        available_layers.sort(key=lambda x: (
            layer_priority.get(x, 0),
            -self.layer_health[x]['consecutive_failures']
        ), reverse=True)
        
        # Implement fallback strategies based on config
        if config.fallback_strategy == "aggressive":
            # Use all available layers
            return available_layers
        elif config.fallback_strategy == "minimal":
            # Use only the most reliable layer
            return available_layers[:1] if available_layers else []
        else:  # graceful
            # Use top 2 most reliable layers
            return available_layers[:2]
    
    def is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable.
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error is recoverable
        """
        error_message = str(error).lower()
        
        # Non-recoverable errors
        non_recoverable_patterns = [
            "out of memory",
            "cuda out of memory",
            "disk full",
            "permission denied",
            "file not found",
            "module not found"
        ]
        
        for pattern in non_recoverable_patterns:
            if pattern in error_message:
                return False
        
        # Recoverable errors
        recoverable_patterns = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "retry"
        ]
        
        for pattern in recoverable_patterns:
            if pattern in error_message:
                return True
        
        # Default to recoverable for unknown errors
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive error summary and statistics.
        
        Returns:
            Dictionary with error statistics and recommendations
        """
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Count recent errors
        recent_errors = [e for e in self.error_history if e['timestamp'] >= last_hour]
        daily_errors = [e for e in self.error_history if e['timestamp'] >= last_day]
        
        # Group errors by layer and type
        error_by_layer = defaultdict(int)
        error_by_type = defaultdict(int)
        
        for error_entry in daily_errors:
            layer = error_entry['layer']
            error_type = error_entry['error'].error_type
            error_by_layer[layer] += 1
            error_by_type[error_type] += 1
        
        # Get layer health summary
        layer_health_summary = {}
        for layer, health in self.layer_health.items():
            layer_health_summary[layer] = {
                'status': health['status'],
                'error_count': health['error_count'],
                'consecutive_failures': health['consecutive_failures'],
                'circuit_breaker_open': health['circuit_breaker_open']
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_1h': len(recent_errors),
            'recent_errors_24h': len(daily_errors),
            'errors_by_layer': dict(error_by_layer),
            'errors_by_type': dict(error_by_type),
            'layer_health': layer_health_summary,
            'recovery_statistics': dict(self.recovery_statistics),
            'recommendations': recommendations,
            'error_patterns': [
                {
                    'type': p.error_type,
                    'pattern': p.pattern,
                    'occurrences': p.occurrence_count,
                    'last_occurrence': p.last_occurrence
                } for p in self.error_patterns if p.occurrence_count > 0
            ]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        # Check for common issues
        for layer, health in self.layer_health.items():
            if health['circuit_breaker_open']:
                recommendations.append(
                    f"Layer {layer} circuit breaker is open. "
                    f"Check dependencies and configuration."
                )
            elif health['status'] == 'critical':
                recommendations.append(
                    f"Layer {layer} is in critical state with "
                    f"{health['consecutive_failures']} consecutive failures."
                )
        
        # Check error patterns
        for pattern in self.error_patterns:
            if pattern.occurrence_count >= pattern.max_occurrences:
                if pattern.error_type == ErrorCategory.DEPENDENCY_MISSING:
                    recommendations.append(
                        "Install missing dependencies (Java for language-tool-python, "
                        "proper model files for GECToR)."
                    )
                elif pattern.error_type == ErrorCategory.MODEL_LOADING:
                    recommendations.append(
                        "Check model configuration and network connectivity. "
                        "Consider using local model files."
                    )
                elif pattern.error_type == ErrorCategory.MEMORY_ERROR:
                    recommendations.append(
                        "Reduce batch size or enable memory optimization. "
                        "Consider using CPU instead of GPU."
                    )
        
        return recommendations
    
    def record_layer_success(self, layer: DetectionLayer):
        """Record successful layer operation"""
        health = self.layer_health[layer]
        health['consecutive_failures'] = 0
        health['last_success'] = datetime.now()
        health['status'] = 'healthy'
        
        # Update circuit breaker for half-open state
        if health['circuit_breaker_open']:
            breaker = self.circuit_breakers[layer]
            breaker['half_open_calls'] += 1
    
    def reset_layer_health(self, layer: DetectionLayer):
        """Reset health status for a layer"""
        health = self.layer_health[layer]
        health['status'] = 'healthy'
        health['error_count'] = 0
        health['consecutive_failures'] = 0
        health['circuit_breaker_open'] = False
        health['circuit_breaker_open_time'] = None
        
        breaker = self.circuit_breakers[layer]
        breaker['half_open_calls'] = 0
        
        logger.info(f"Reset health status for layer {layer}")
    
    def get_healthy_layers(self) -> List[DetectionLayer]:
        """Get list of currently healthy layers"""
        healthy_layers = []
        
        for layer, health in self.layer_health.items():
            if (health['status'] == 'healthy' and 
                not health['circuit_breaker_open']):
                healthy_layers.append(layer)
        
        return healthy_layers
    
    def create_analysis_error(self, message: str, layer_errors: List[LayerError],
                            fallback_used: bool = False) -> AnalysisError:
        """Create a comprehensive analysis error"""
        # Determine error code based on layer errors
        if not layer_errors:
            error_code = "UNKNOWN_ERROR"
        elif len(layer_errors) == len(DetectionLayer):
            error_code = "ALL_LAYERS_FAILED"
        else:
            error_code = "PARTIAL_LAYER_FAILURE"
        
        return AnalysisError(
            error_message=message,
            error_code=error_code,
            layer_errors=layer_errors,
            fallback_used=fallback_used
        )