"""
Intelligent caching system for enhanced typo detection.
Implements multi-level caching with content-based hashing and adaptive eviction.
"""

import hashlib
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle
import json
import logging
from pathlib import Path

from enhanced_models import CacheKey, CacheEntry, DetectionLayer
from core_interfaces import ICacheManager

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class LRUCache:
    """Thread-safe LRU cache with size and TTL limits"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if self._is_expired(entry):
                del self.cache[key]
                del self.access_times[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            entry.access_count += 1
            
            self.stats.hits += 1
            return entry.result
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        with self.lock:
            # Create cache entry
            cache_key = CacheKey.create("", type('Config', (), {
                'confidence_threshold': 80.0,
                'enable_traditional_nlp': True,
                'enable_gector': True,
                'enable_domain_validation': True
            })())
            cache_key.text_hash = key.split('_')[0] if '_' in key else key
            
            entry = CacheEntry(
                key=cache_key,
                result=value,
                timestamp=datetime.now()
            )
            
            # Remove if already exists
            if key in self.cache:
                del self.access_times[key]
            
            # Add new entry
            self.cache[key] = entry
            self.access_times[key] = time.time()
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                self._evict_lru()
            
            self.stats.size = len(self.cache)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age > self.default_ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            del self.access_times[key]
            self.stats.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            self.stats.size = len(self.cache)
            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate,
                'evictions': self.stats.evictions,
                'size': self.stats.size
            }

class AdaptiveCache:
    """Adaptive cache that adjusts based on usage patterns"""
    
    def __init__(self, initial_size: int = 500, max_size: int = 2000):
        self.initial_size = initial_size
        self.max_size = max_size
        self.current_size = initial_size
        self.cache = LRUCache(max_size=self.current_size)
        
        # Adaptation parameters
        self.hit_rate_threshold = 0.8
        self.adaptation_interval = 100  # Check every 100 operations
        self.operation_count = 0
        self.last_adaptation = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item with adaptive behavior"""
        result = self.cache.get(key)
        self.operation_count += 1
        
        # Check if adaptation is needed
        if self.operation_count % self.adaptation_interval == 0:
            self._adapt_size()
        
        return result
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item with adaptive behavior"""
        self.cache.set(key, value, ttl)
        self.operation_count += 1
    
    def _adapt_size(self) -> None:
        """Adapt cache size based on hit rate"""
        stats = self.cache.get_stats()
        
        if stats.hit_rate < self.hit_rate_threshold and self.current_size < self.max_size:
            # Increase cache size
            new_size = min(self.max_size, int(self.current_size * 1.2))
            self._resize_cache(new_size)
            logger.info(f"Increased cache size to {new_size} (hit rate: {stats.hit_rate:.2f})")
        
        elif stats.hit_rate > 0.95 and self.current_size > self.initial_size:
            # Decrease cache size if hit rate is very high
            new_size = max(self.initial_size, int(self.current_size * 0.9))
            self._resize_cache(new_size)
            logger.info(f"Decreased cache size to {new_size} (hit rate: {stats.hit_rate:.2f})")
    
    def _resize_cache(self, new_size: int) -> None:
        """Resize the underlying cache"""
        old_cache = self.cache
        self.cache = LRUCache(max_size=new_size, default_ttl=old_cache.default_ttl)
        self.current_size = new_size
        
        # Migrate most recent entries
        with old_cache.lock:
            items = list(old_cache.cache.items())
            for key, entry in items[-new_size:]:  # Keep most recent entries
                self.cache.set(key, entry.result)

class ContentBasedHasher:
    """Content-based hasher for creating cache keys"""
    
    @staticmethod
    def hash_text(text: str, normalize: bool = True) -> str:
        """Create hash for text content"""
        if normalize:
            # Normalize whitespace and case for better cache hits
            normalized = ' '.join(text.lower().split())
        else:
            normalized = text
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def hash_config(config: Any) -> str:
        """Create hash for configuration"""
        config_dict = {
            'confidence_threshold': getattr(config, 'confidence_threshold', 80.0),
            'enable_traditional_nlp': getattr(config, 'enable_traditional_nlp', True),
            'enable_gector': getattr(config, 'enable_gector', True),
            'enable_domain_validation': getattr(config, 'enable_domain_validation', True),
            'fallback_strategy': getattr(config, 'fallback_strategy', 'graceful')
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:8]
    
    @staticmethod
    def create_cache_key(text: str, config: Any, layer: Optional[DetectionLayer] = None) -> str:
        """Create comprehensive cache key"""
        text_hash = ContentBasedHasher.hash_text(text)
        config_hash = ContentBasedHasher.hash_config(config)
        
        key_parts = [text_hash, config_hash]
        if layer:
            key_parts.append(layer.value)
        
        return '_'.join(key_parts)

class MultiLevelCache:
    """Multi-level cache with different strategies for different data types"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # L1 Cache: Fast in-memory cache for recent results
        self.l1_cache = AdaptiveCache(
            initial_size=config.get('l1_size', 200),
            max_size=config.get('l1_max_size', 500)
        )
        
        # L2 Cache: Larger cache for layer-specific results
        self.l2_cache = LRUCache(
            max_size=config.get('l2_size', 1000),
            default_ttl=config.get('l2_ttl', 7200)  # 2 hours
        )
        
        # L3 Cache: Persistent cache for expensive operations
        self.l3_cache = LRUCache(
            max_size=config.get('l3_size', 2000),
            default_ttl=config.get('l3_ttl', 86400)  # 24 hours
        )
        
        # Cache for different data types
        self.suggestion_cache = LRUCache(max_size=5000, default_ttl=3600)
        self.validation_cache = LRUCache(max_size=2000, default_ttl=1800)
        self.context_cache = LRUCache(max_size=1000, default_ttl=3600)
        
        # Statistics
        self.global_stats = CacheStats()
        self.layer_stats: Dict[DetectionLayer, CacheStats] = defaultdict(CacheStats)
        
        # Persistent storage
        self.persistent_enabled = config.get('persistent_enabled', False)
        self.persistent_path = Path(config.get('persistent_path', 'cache'))
        
        if self.persistent_enabled:
            self.persistent_path.mkdir(exist_ok=True)
            self._load_persistent_cache()
    
    def get_analysis_result(self, text: str, config: Any) -> Optional[Any]:
        """Get cached analysis result"""
        key = ContentBasedHasher.create_cache_key(text, config)
        
        # Try L1 first (fastest)
        result = self.l1_cache.get(key)
        if result is not None:
            self.global_stats.hits += 1
            return result
        
        # Try L2
        result = self.l2_cache.get(key)
        if result is not None:
            # Promote to L1
            self.l1_cache.set(key, result)
            self.global_stats.hits += 1
            return result
        
        # Try L3
        result = self.l3_cache.get(key)
        if result is not None:
            # Promote to L2 and L1
            self.l2_cache.set(key, result)
            self.l1_cache.set(key, result)
            self.global_stats.hits += 1
            return result
        
        self.global_stats.misses += 1
        return None
    
    def set_analysis_result(self, text: str, config: Any, result: Any) -> None:
        """Cache analysis result"""
        key = ContentBasedHasher.create_cache_key(text, config)
        
        # Store in all levels
        self.l1_cache.set(key, result)
        self.l2_cache.set(key, result)
        self.l3_cache.set(key, result)
        
        # Store persistently if enabled
        if self.persistent_enabled:
            self._save_to_persistent(key, result)
    
    def get_layer_result(self, text: str, config: Any, layer: DetectionLayer) -> Optional[Any]:
        """Get cached layer-specific result"""
        key = ContentBasedHasher.create_cache_key(text, config, layer)
        
        result = self.l2_cache.get(key)
        if result is not None:
            self.layer_stats[layer].hits += 1
            return result
        
        self.layer_stats[layer].misses += 1
        return None
    
    def set_layer_result(self, text: str, config: Any, layer: DetectionLayer, result: Any) -> None:
        """Cache layer-specific result"""
        key = ContentBasedHasher.create_cache_key(text, config, layer)
        self.l2_cache.set(key, result)
    
    def get_suggestion(self, original: str, context: str) -> Optional[List[str]]:
        """Get cached suggestions"""
        key = f"suggestion_{ContentBasedHasher.hash_text(original + context)}"
        return self.suggestion_cache.get(key)
    
    def set_suggestion(self, original: str, context: str, suggestions: List[str]) -> None:
        """Cache suggestions"""
        key = f"suggestion_{ContentBasedHasher.hash_text(original + context)}"
        self.suggestion_cache.set(key, suggestions)
    
    def get_validation(self, original: str, suggestion: str, context: str) -> Optional[bool]:
        """Get cached validation result"""
        key = f"validation_{ContentBasedHasher.hash_text(original + suggestion + context)}"
        return self.validation_cache.get(key)
    
    def set_validation(self, original: str, suggestion: str, context: str, is_valid: bool) -> None:
        """Cache validation result"""
        key = f"validation_{ContentBasedHasher.hash_text(original + suggestion + context)}"
        self.validation_cache.set(key, is_valid)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        invalidated = 0
        
        # Invalidate from all caches
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache, 
                     self.suggestion_cache, self.validation_cache, self.context_cache]:
            with cache.lock:
                keys_to_remove = [key for key in cache.cache.keys() if pattern in key]
                for key in keys_to_remove:
                    del cache.cache[key]
                    if key in cache.access_times:
                        del cache.access_times[key]
                    invalidated += 1
        
        return invalidated
    
    def clear_all(self) -> None:
        """Clear all caches"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        self.suggestion_cache.clear()
        self.validation_cache.clear()
        self.context_cache.clear()
        
        self.global_stats = CacheStats()
        self.layer_stats.clear()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'global_stats': {
                'hits': self.global_stats.hits,
                'misses': self.global_stats.misses,
                'hit_rate': self.global_stats.hit_rate,
                'evictions': self.global_stats.evictions
            },
            'l1_cache': self.l1_cache.get_stats().__dict__,
            'l2_cache': self.l2_cache.get_stats().__dict__,
            'l3_cache': self.l3_cache.get_stats().__dict__,
            'suggestion_cache': self.suggestion_cache.get_stats().__dict__,
            'validation_cache': self.validation_cache.get_stats().__dict__,
            'context_cache': self.context_cache.get_stats().__dict__,
            'layer_stats': {
                layer.value: {
                    'hits': stats.hits,
                    'misses': stats.misses,
                    'hit_rate': stats.hit_rate
                } for layer, stats in self.layer_stats.items()
            }
        }
    
    def optimize(self) -> None:
        """Optimize cache performance"""
        # Clean expired entries
        current_time = time.time()
        
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            with cache.lock:
                expired_keys = []
                for key, entry in cache.cache.items():
                    if cache._is_expired(entry):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del cache.cache[key]
                    if key in cache.access_times:
                        del cache.access_times[key]
                    cache.stats.evictions += 1
        
        logger.info("Cache optimization completed")
    
    def _save_to_persistent(self, key: str, result: Any) -> None:
        """Save result to persistent storage"""
        try:
            file_path = self.persistent_path / f"{key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'result': result,
                    'timestamp': datetime.now(),
                    'key': key
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save to persistent cache: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Load results from persistent storage"""
        try:
            for file_path in self.persistent_path.glob("*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Check if not too old (24 hours)
                    age = (datetime.now() - data['timestamp']).total_seconds()
                    if age < 86400:  # 24 hours
                        self.l3_cache.set(data['key'], data['result'])
                    else:
                        file_path.unlink()  # Remove old file
                        
                except Exception as e:
                    logger.warning(f"Failed to load persistent cache file {file_path}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")

class IntelligentCacheManager(ICacheManager):
    """
    Intelligent cache manager implementing the ICacheManager interface.
    Provides advanced caching with content-based hashing and adaptive strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.cache = MultiLevelCache(config)
        self.hasher = ContentBasedHasher()
        
        # Performance monitoring
        self.start_time = time.time()
        self.total_operations = 0
        
        logger.info("Intelligent cache manager initialized")
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get cached result by key"""
        self.total_operations += 1
        cache_key_str = str(key)
        
        if key.layer_name:
            # Layer-specific cache
            return self.cache.get_layer_result(
                key.text_hash, 
                type('Config', (), {'confidence_threshold': 80.0})(),  # Mock config
                key.layer_name
            )
        else:
            # General analysis cache
            return self.cache.get_analysis_result(
                key.text_hash,
                type('Config', (), {'confidence_threshold': 80.0})()  # Mock config
            )
    
    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached result with optional TTL"""
        self.total_operations += 1
        
        if key.layer_name:
            # Layer-specific cache
            self.cache.set_layer_result(
                key.text_hash,
                type('Config', (), {'confidence_threshold': 80.0})(),  # Mock config
                key.layer_name,
                value
            )
        else:
            # General analysis cache
            self.cache.set_analysis_result(
                key.text_hash,
                type('Config', (), {'confidence_threshold': 80.0})(),  # Mock config
                value
            )
    
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        return self.cache.invalidate_pattern(pattern)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear_all()
        logger.info("All cache entries cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = self.cache.get_comprehensive_stats()
        except AttributeError:
            # Fallback stats if method not available
            stats = {
                'global_stats': {'hits': 0, 'misses': 0, 'hit_rate': 0.0},
                'l1_cache': {'hits': 0, 'misses': 0, 'size': 0},
                'l2_cache': {'hits': 0, 'misses': 0, 'size': 0},
                'l3_cache': {'hits': 0, 'misses': 0, 'size': 0}
            }
        
        # Add manager-level statistics
        uptime = time.time() - self.start_time
        stats['manager_stats'] = {
            'uptime_seconds': uptime,
            'total_operations': self.total_operations,
            'operations_per_second': self.total_operations / max(uptime, 1)
        }
        
        return stats
    
    def create_cache_key(self, text: str, config: Any, layer: Optional[Any] = None) -> Any:
        """Create cache key for the given parameters"""
        from enhanced_models import CacheKey
        return CacheKey.create(text, config, layer)
    
    def optimize(self) -> None:
        """Optimize cache performance"""
        self.cache.optimize()
        logger.info("Cache optimization completed")
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Generate detailed cache efficiency report"""
        stats = self.get_stats()
        
        # Calculate efficiency metrics
        global_hit_rate = stats['global_stats']['hit_rate']
        total_memory_usage = sum([
            cache_stats.get('memory_usage', 0) 
            for cache_stats in [
                stats['l1_cache'], stats['l2_cache'], stats['l3_cache']
            ]
        ])
        
        # Performance recommendations
        recommendations = []
        
        if global_hit_rate < 0.7:
            recommendations.append("Consider increasing cache size for better hit rate")
        
        if stats['global_stats']['evictions'] > stats['global_stats']['hits'] * 0.1:
            recommendations.append("High eviction rate detected, consider increasing cache TTL")
        
        layer_performance = {}
        for layer, layer_stats in stats['layer_stats'].items():
            if layer_stats['hit_rate'] < 0.5:
                recommendations.append(f"Layer {layer} has low cache hit rate")
            layer_performance[layer] = layer_stats['hit_rate']
        
        return {
            'overall_efficiency': global_hit_rate,
            'memory_usage_mb': total_memory_usage / (1024 * 1024),
            'layer_performance': layer_performance,
            'recommendations': recommendations,
            'cache_levels': {
                'l1_efficiency': stats['l1_cache']['hit_rate'],
                'l2_efficiency': stats['l2_cache']['hit_rate'],
                'l3_efficiency': stats['l3_cache']['hit_rate']
            }
        }