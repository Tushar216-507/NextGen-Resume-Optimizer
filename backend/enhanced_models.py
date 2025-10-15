"""
Enhanced models for multi-layer typo detection system.
Designed with precision for maximum accuracy and performance.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import hashlib
from abc import ABC, abstractmethod

# Core Enums
class DetectionLayer(str, Enum):
    """Available detection layers"""
    TRADITIONAL_NLP = "traditional_nlp"
    GECTOR_TRANSFORMER = "gector_transformer"
    DOMAIN_VALIDATION = "domain_validation"
    ENSEMBLE = "ensemble"

class ConfidenceLevel(str, Enum):
    """Confidence level categories"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 80-89%
    MEDIUM = "medium"       # 70-79%
    LOW = "low"            # 60-69%
    VERY_LOW = "very_low"  # <60%

class ErrorType(str, Enum):
    """Types of errors detected"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    TECHNICAL_TERM = "technical_term"

class ProcessingStatus(str, Enum):
    """Processing status for analysis"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CACHED = "cached"

# Configuration Models
class AnalysisConfig(BaseModel):
    """Configuration for enhanced analysis system"""
    enable_traditional_nlp: bool = True
    enable_gector: bool = True
    enable_domain_validation: bool = True
    confidence_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    max_processing_time: float = Field(default=3.0, gt=0.0)
    cache_enabled: bool = True
    fallback_strategy: str = Field(default="graceful", pattern="^(graceful|aggressive|minimal)$")
    parallel_processing: bool = True
    chunk_size: int = Field(default=512, gt=0)
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError('Confidence threshold must be between 0 and 100')
        return v

class LayerConfig(BaseModel):
    """Configuration for individual detection layers"""
    layer_name: DetectionLayer
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: float = Field(default=2.0, gt=0.0)
    fallback_enabled: bool = True
    cache_results: bool = True

# Detection Results
class LayeredDetectionResult(BaseModel):
    """Result from a single detection layer"""
    layer_name: DetectionLayer
    original_word: str
    suggestions: List[str]
    confidence_scores: List[float]
    detection_method: str
    processing_time: float
    error_type: ErrorType
    context: str
    position: Optional[int] = None
    
    @validator('confidence_scores')
    def validate_confidence_scores(cls, v, values):
        suggestions = values.get('suggestions', [])
        if len(v) != len(suggestions):
            raise ValueError('Number of confidence scores must match number of suggestions')
        return v

class EnsembleResult(BaseModel):
    """Result from ensemble validation"""
    final_suggestion: str
    ensemble_confidence: float
    layer_votes: Dict[DetectionLayer, float]
    explanation: str
    validation_status: str
    conflict_resolution_method: str
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category"""
        if self.ensemble_confidence >= 90:
            return ConfidenceLevel.VERY_HIGH
        elif self.ensemble_confidence >= 80:
            return ConfidenceLevel.HIGH
        elif self.ensemble_confidence >= 70:
            return ConfidenceLevel.MEDIUM
        elif self.ensemble_confidence >= 60:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

# Performance Metrics
class LayerPerformanceMetrics(BaseModel):
    """Performance metrics for individual layers"""
    layer_name: DetectionLayer
    total_processed: int = 0
    total_processing_time: float = 0.0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    
    @property
    def average_processing_time(self) -> float:
        return self.total_processing_time / max(1, self.total_processed)

class SystemPerformanceMetrics(BaseModel):
    """Overall system performance metrics"""
    total_texts_analyzed: int = 0
    total_processing_time: float = 0.0
    accuracy_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    cache_hit_rate: float = 0.0
    layer_metrics: Dict[DetectionLayer, LayerPerformanceMetrics] = {}
    
    @property
    def average_processing_time(self) -> float:
        return self.total_processing_time / max(1, self.total_texts_analyzed)

# Cache Models
class CacheKey(BaseModel):
    """Cache key for analysis results"""
    text_hash: str
    config_hash: str
    layer_name: Optional[DetectionLayer] = None
    
    @classmethod
    def create(cls, text: str, config: AnalysisConfig, layer_name: Optional[DetectionLayer] = None):
        """Create cache key from text and config"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        config_str = f"{config.confidence_threshold}_{config.enable_traditional_nlp}_{config.enable_gector}_{config.enable_domain_validation}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        return cls(text_hash=text_hash, config_hash=config_hash, layer_name=layer_name)
    
    def __str__(self) -> str:
        layer_suffix = f"_{self.layer_name}" if self.layer_name else ""
        return f"{self.text_hash}_{self.config_hash}{layer_suffix}"

class CacheEntry(BaseModel):
    """Cache entry for storing analysis results"""
    key: CacheKey
    result: Any  # Can be LayeredDetectionResult or full analysis result
    timestamp: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    
    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > max_age_seconds

# Abstract Base Classes
class DetectionLayerInterface(ABC):
    """Abstract interface for detection layers"""
    
    @abstractmethod
    def detect(self, text: str, config: AnalysisConfig) -> List[LayeredDetectionResult]:
        """Detect errors in text"""
        pass
    
    @abstractmethod
    def get_layer_name(self) -> DetectionLayer:
        """Get the layer name"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if layer is available"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> LayerPerformanceMetrics:
        """Get performance metrics"""
        pass

class CacheInterface(ABC):
    """Abstract interface for caching system"""
    
    @abstractmethod
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get cached result"""
        pass
    
    @abstractmethod
    def set(self, key: CacheKey, value: Any) -> None:
        """Set cached result"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached results"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

# Enhanced Analysis Results
class EnhancedTypoResult(BaseModel):
    """Enhanced typo result with multi-layer validation"""
    word: str
    suggestion: str
    confidence_score: float
    explanation: str
    context: str
    validation_status: str
    position: Optional[int] = None
    error_type: ErrorType = ErrorType.SPELLING
    layer_results: List[LayeredDetectionResult] = []
    ensemble_result: Optional[EnsembleResult] = None
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category"""
        if self.confidence_score >= 90:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence_score >= 80:
            return ConfidenceLevel.HIGH
        elif self.confidence_score >= 70:
            return ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 60:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class EnhancedGrammarResult(BaseModel):
    """Enhanced grammar result with multi-layer validation"""
    sentence: str
    suggestion: str
    confidence_score: float
    explanation: str
    issue_type: str
    rule_category: str
    validation_status: str
    position: Optional[int] = None
    error_type: ErrorType = ErrorType.GRAMMAR
    layer_results: List[LayeredDetectionResult] = []
    ensemble_result: Optional[EnsembleResult] = None
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category"""
        if self.confidence_score >= 90:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence_score >= 80:
            return ConfidenceLevel.HIGH
        elif self.confidence_score >= 70:
            return ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 60:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class MultiLayerAnalysisResult(BaseModel):
    """Complete multi-layer analysis result"""
    typos: List[EnhancedTypoResult]
    grammar_issues: List[EnhancedGrammarResult]
    processing_time: float
    layers_used: List[DetectionLayer]
    performance_metrics: SystemPerformanceMetrics
    config_used: AnalysisConfig
    processing_status: ProcessingStatus
    cache_hits: int = 0
    total_suggestions: int = 0
    
    @property
    def total_issues(self) -> int:
        return len(self.typos) + len(self.grammar_issues)
    
    @property
    def high_confidence_issues(self) -> int:
        high_conf_typos = sum(1 for t in self.typos if t.confidence_score >= 80)
        high_conf_grammar = sum(1 for g in self.grammar_issues if g.confidence_score >= 80)
        return high_conf_typos + high_conf_grammar

# Error Handling Models
class LayerError(BaseModel):
    """Error information for layer failures"""
    layer_name: DetectionLayer
    error_message: str
    error_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    recoverable: bool = True

class AnalysisError(Exception):
    """Error information for analysis failures"""
    def __init__(self, error_message: str, error_code: str, layer_errors: List[LayerError] = None, fallback_used: bool = False):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_code = error_code
        self.layer_errors = layer_errors or []
        self.fallback_used = fallback_used
        self.timestamp = datetime.now()

# Validation Models
class ValidationRule(BaseModel):
    """Rule for validating suggestions"""
    rule_name: str
    rule_type: str
    pattern: Optional[str] = None
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    enabled: bool = True

class ValidationResult(BaseModel):
    """Result of suggestion validation"""
    is_valid: bool
    confidence_adjustment: float = 0.0
    explanation: str
    rules_applied: List[str] = []