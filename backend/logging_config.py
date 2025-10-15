"""
Advanced logging configuration for the enhanced typo detection system.
Provides structured logging with performance metrics and error tracking.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'layer'):
            log_entry['layer'] = record.layer
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        if hasattr(record, 'confidence_score'):
            log_entry['confidence_score'] = record.confidence_score
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'cache_hit'):
            log_entry['cache_hit'] = record.cache_hit
        if hasattr(record, 'text_length'):
            log_entry['text_length'] = record.text_length
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create performance log file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "performance.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_analysis_performance(self, text_length: int, processing_time: float,
                               layers_used: list, cache_hits: int, total_suggestions: int):
        """Log analysis performance metrics"""
        self.logger.info(
            "Analysis completed",
            extra={
                'text_length': text_length,
                'processing_time': processing_time,
                'layers_used': layers_used,
                'cache_hits': cache_hits,
                'total_suggestions': total_suggestions,
                'suggestions_per_second': total_suggestions / max(processing_time, 0.001)
            }
        )
    
    def log_layer_performance(self, layer: str, processing_time: float,
                            success: bool, confidence: float, suggestions_count: int):
        """Log individual layer performance"""
        self.logger.info(
            f"Layer {layer} performance",
            extra={
                'layer': layer,
                'processing_time': processing_time,
                'success': success,
                'confidence_score': confidence,
                'suggestions_count': suggestions_count
            }
        )
    
    def log_cache_performance(self, cache_hits: int, cache_misses: int,
                            cache_size: int, hit_rate: float):
        """Log cache performance metrics"""
        self.logger.info(
            "Cache performance",
            extra={
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_size': cache_size,
                'hit_rate': hit_rate
            }
        )

class ErrorLogger:
    """Specialized logger for error tracking"""
    
    def __init__(self, name: str = "errors"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.WARNING)
        
        # Create error log file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_layer_error(self, layer: str, error_type: str, error_message: str,
                       recoverable: bool, traceback: Optional[str] = None):
        """Log layer-specific errors"""
        extra = {
            'layer': layer,
            'error_type': error_type,
            'recoverable': recoverable
        }
        
        if traceback:
            extra['traceback'] = traceback
        
        self.logger.error(f"Layer error: {error_message}", extra=extra)
    
    def log_analysis_error(self, error_code: str, error_message: str,
                          failed_layers: list, fallback_used: bool):
        """Log analysis-level errors"""
        self.logger.error(
            f"Analysis error: {error_message}",
            extra={
                'error_code': error_code,
                'failed_layers': failed_layers,
                'fallback_used': fallback_used
            }
        )
    
    def log_configuration_error(self, config_section: str, error_message: str):
        """Log configuration errors"""
        self.logger.error(
            f"Configuration error in {config_section}: {error_message}",
            extra={
                'config_section': config_section,
                'error_type': 'configuration'
            }
        )

def setup_logging(log_level: str = "INFO", enable_console: bool = True,
                 enable_file: bool = True, log_dir: str = "logs") -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for the enhanced typo detection system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        log_dir: Directory for log files
        
    Returns:
        Dictionary of configured loggers
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    loggers = {}
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use simple format for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            log_path / "application.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5
        )
        main_handler.setLevel(getattr(logging, log_level.upper()))
        main_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(main_handler)
    
    # Create specialized loggers
    loggers['performance'] = PerformanceLogger()
    loggers['errors'] = ErrorLogger()
    
    # Create component-specific loggers
    component_names = [
        'multi_layer_engine',
        'traditional_nlp',
        'gector_analyzer',
        'domain_validator',
        'confidence_scorer',
        'ensemble_validator',
        'cache_manager'
    ]
    
    for component in component_names:
        logger = logging.getLogger(component)
        logger.setLevel(getattr(logging, log_level.upper()))
        loggers[component] = logger
    
    # Log system startup
    root_logger.info(
        "Enhanced typo detection system logging initialized",
        extra={
            'log_level': log_level,
            'console_enabled': enable_console,
            'file_enabled': enable_file,
            'log_directory': str(log_path)
        }
    )
    
    return loggers

class LoggingContextManager:
    """Context manager for adding context to log messages"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)

# Utility functions for common logging patterns
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with timing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logger.debug(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'processing_time': processing_time,
                        'success': True
                    }
                )
                
                return result
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logger.error(
                    f"Function {func.__name__} failed: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'processing_time': processing_time,
                        'success': False,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator

def log_performance_metrics(logger: logging.Logger, operation: str):
    """Context manager for logging performance metrics"""
    class PerformanceContext:
        def __init__(self):
            self.start_time = None
            self.operation = operation
        
        def __enter__(self):
            self.start_time = datetime.now()
            logger.debug(f"Starting {self.operation}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            processing_time = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                logger.info(
                    f"Completed {self.operation}",
                    extra={
                        'operation': self.operation,
                        'processing_time': processing_time,
                        'success': True
                    }
                )
            else:
                logger.error(
                    f"Failed {self.operation}: {str(exc_val)}",
                    extra={
                        'operation': self.operation,
                        'processing_time': processing_time,
                        'success': False,
                        'error_type': exc_type.__name__
                    }
                )
    
    return PerformanceContext()

# Initialize default loggers
default_loggers = setup_logging()
performance_logger = default_loggers['performance']
error_logger = default_loggers['errors']