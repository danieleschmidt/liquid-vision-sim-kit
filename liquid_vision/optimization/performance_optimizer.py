"""
Advanced performance optimization system for Generation 3.
Provides caching, memory optimization, parallel processing, and auto-scaling.
"""

import time
import threading
import functools
import math
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from collections import OrderedDict
import logging


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    execution_time: float
    memory_usage: float
    cache_hits: int
    cache_misses: int
    parallel_efficiency: float
    throughput: float


class LRUCache:
    """Thread-safe LRU cache with performance metrics."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
                
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new
                self.cache[key] = value
                # Remove oldest if over capacity
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total
            }


class AdaptiveCache:
    """Adaptive cache that adjusts behavior based on usage patterns."""
    
    def __init__(self, initial_size: int = 100):
        self.cache = LRUCache(initial_size)
        self.access_patterns: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
        
    def _record_access(self, key: str) -> None:
        """Record access time for pattern analysis."""
        current_time = time.time()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] 
            if t > cutoff_time
        ]
        
    def get(self, key: str) -> Optional[Any]:
        """Get item with access pattern recording."""
        self._record_access(key)
        return self.cache.get(key)
        
    def put(self, key: str, value: Any) -> None:
        """Put item with intelligent caching decisions."""
        self._record_access(key)
        
        # Decide whether to cache based on access frequency
        if key in self.access_patterns:
            access_frequency = len(self.access_patterns[key])
            if access_frequency >= 2:  # Cache if accessed multiple times
                self.cache.put(key, value)
        else:
            # Always cache new items
            self.cache.put(key, value)
            
    def optimize_size(self) -> None:
        """Optimize cache size based on usage patterns."""
        stats = self.cache.stats()
        
        if stats["hit_rate"] > 0.8 and stats["size"] >= stats["max_size"] * 0.9:
            # High hit rate and nearly full - increase size
            new_size = min(stats["max_size"] * 2, 10000)
            self.logger.info(f"Increasing cache size from {stats['max_size']} to {new_size}")
            
            # Create new cache with larger size
            old_cache = self.cache
            self.cache = LRUCache(new_size)
            
            # Transfer most frequently accessed items
            for key, value in list(old_cache.cache.items())[-new_size:]:
                self.cache.put(key, value)
                
        elif stats["hit_rate"] < 0.3 and stats["max_size"] > 50:
            # Low hit rate - decrease size
            new_size = max(stats["max_size"] // 2, 50)
            self.logger.info(f"Decreasing cache size from {stats['max_size']} to {new_size}")
            
            # Create new cache with smaller size
            old_cache = self.cache
            self.cache = LRUCache(new_size)
            
            # Transfer most recently accessed items
            for key, value in list(old_cache.cache.items())[-new_size:]:
                self.cache.put(key, value)


class MemoryOptimizer:
    """Memory usage optimizer with automatic garbage collection."""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.logger = logging.getLogger(__name__)
        
    def check_memory_usage(self) -> float:
        """Check current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback without psutil
            return 0.5  # Assume moderate usage
            
    def optimize_memory(self) -> bool:
        """Optimize memory usage if needed."""
        memory_usage = self.check_memory_usage()
        
        if memory_usage > self.memory_threshold:
            self.logger.warning(f"High memory usage detected: {memory_usage:.1%}")
            
            # Force garbage collection
            import gc
            before_gc = len(gc.get_objects())
            gc.collect()
            after_gc = len(gc.get_objects())
            
            freed_objects = before_gc - after_gc
            self.logger.info(f"Garbage collection freed {freed_objects} objects")
            
            return True
            
        return False
        
    def memory_efficient_decorator(self, func: Callable) -> Callable:
        """Decorator to monitor and optimize memory usage."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before execution
            self.optimize_memory()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check memory after execution
            self.optimize_memory()
            
            return result
            
        return wrapper


class ParallelProcessor:
    """Parallel processing for batch operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        import multiprocessing
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.logger = logging.getLogger(__name__)
        
    def process_batch(self, func: Callable, items: List[Any], 
                     chunk_size: Optional[int] = None) -> List[Any]:
        """Process items in parallel batches."""
        if not items:
            return []
            
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))
            
        # For small batches, use sequential processing
        if len(items) <= self.max_workers:
            return [func(item) for item in items]
            
        try:
            # Use threading for I/O bound tasks (our neural network operations)
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split items into chunks
                chunks = [
                    items[i:i + chunk_size] 
                    for i in range(0, len(items), chunk_size)
                ]
                
                # Process chunks in parallel
                def process_chunk(chunk):
                    return [func(item) for item in chunk]
                
                # Submit all chunks
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                
                # Collect results
                results = []
                for future in futures:
                    results.extend(future.result())
                    
                return results
                
        except Exception as e:
            self.logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
            return [func(item) for item in items]
            
    def benchmark_parallel_efficiency(self, func: Callable, 
                                    test_items: List[Any]) -> float:
        """Benchmark parallel vs sequential efficiency."""
        if len(test_items) < 4:
            return 0.0
            
        # Sequential timing
        start_time = time.time()
        sequential_results = [func(item) for item in test_items]
        sequential_time = time.time() - start_time
        
        # Parallel timing
        start_time = time.time()
        parallel_results = self.process_batch(func, test_items)
        parallel_time = time.time() - start_time
        
        # Calculate efficiency (speedup ratio)
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            efficiency = speedup / self.max_workers
            
            self.logger.info(
                f"Parallel efficiency: {efficiency:.2f} "
                f"(speedup: {speedup:.2f}x, workers: {self.max_workers})"
            )
            
            return efficiency
        else:
            return 0.0


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self):
        self.performance_history: List[PerformanceMetrics] = []
        self.current_scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 4.0
        self.logger = logging.getLogger(__name__)
        
    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for scaling decisions."""
        self.performance_history.append(metrics)
        
        # Keep only recent history (last 100 measurements)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            
    def calculate_optimal_scale(self) -> float:
        """Calculate optimal scaling factor based on performance history."""
        if len(self.performance_history) < 5:
            return self.current_scale
            
        recent_metrics = self.performance_history[-10:]
        
        # Calculate average metrics
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # Scaling decisions based on metrics
        target_scale = self.current_scale
        
        # Scale up if high latency or low throughput
        if avg_execution_time > 0.1:  # >100ms latency
            target_scale *= 1.2
        if avg_throughput < 50:  # <50 ops/sec
            target_scale *= 1.1
            
        # Scale down if very fast and low memory usage
        if avg_execution_time < 0.01 and avg_memory < 0.3:  # <10ms, <30% memory
            target_scale *= 0.9
            
        # Apply bounds
        target_scale = max(self.min_scale, min(self.max_scale, target_scale))
        
        return target_scale
        
    def should_scale(self) -> Tuple[bool, float]:
        """Determine if scaling is needed and return new scale factor."""
        optimal_scale = self.calculate_optimal_scale()
        
        # Only scale if change is significant (>10%)
        change_ratio = abs(optimal_scale - self.current_scale) / self.current_scale
        
        if change_ratio > 0.1:
            return True, optimal_scale
        else:
            return False, self.current_scale
            
    def apply_scaling(self, new_scale: float) -> None:
        """Apply new scaling factor."""
        old_scale = self.current_scale
        self.current_scale = new_scale
        
        self.logger.info(f"Scaled from {old_scale:.2f} to {new_scale:.2f}")


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache = AdaptiveCache()
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()
        self.auto_scaler = AutoScaler()
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization settings
        self.enable_caching = True
        self.enable_parallel = True
        self.enable_memory_opt = True
        self.enable_auto_scale = True
        
    def optimize_function(self, cache_key_func: Optional[Callable] = None):
        """Decorator to optimize function performance."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Generate cache key
                if self.enable_caching and cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                
                # Apply memory optimization
                if self.enable_memory_opt:
                    self.memory_optimizer.optimize_memory()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                if self.enable_caching and cache_key_func:
                    self.cache.put(cache_key, result)
                
                # Record performance metrics
                execution_time = time.time() - start_time
                memory_usage = self.memory_optimizer.check_memory_usage()
                cache_stats = self.cache.cache.stats()
                
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cache_hits=cache_stats["hits"],
                    cache_misses=cache_stats["misses"],
                    parallel_efficiency=1.0,  # Single execution
                    throughput=1.0 / execution_time if execution_time > 0 else 0.0
                )
                
                if self.enable_auto_scale:
                    self.auto_scaler.record_performance(metrics)
                
                return result
                
            return wrapper
        return decorator
        
    def optimize_batch_processing(self, func: Callable, items: List[Any]) -> List[Any]:
        """Optimize batch processing with parallelization and caching."""
        if not items:
            return []
            
        start_time = time.time()
        
        # Check cache for individual items
        cached_results = {}
        uncached_items = []
        
        if self.enable_caching:
            for i, item in enumerate(items):
                cache_key = f"batch_item_{hash(str(item))}"
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    cached_results[i] = cached_result
                else:
                    uncached_items.append((i, item))
        else:
            uncached_items = list(enumerate(items))
            
        # Process uncached items
        if uncached_items:
            indices, uncached_data = zip(*uncached_items)
            
            if self.enable_parallel and len(uncached_data) > 4:
                # Parallel processing
                processed_results = self.parallel_processor.process_batch(func, uncached_data)
            else:
                # Sequential processing
                processed_results = [func(item) for item in uncached_data]
            
            # Cache new results
            if self.enable_caching:
                for idx, item, result in zip(indices, uncached_data, processed_results):
                    cache_key = f"batch_item_{hash(str(item))}"
                    self.cache.put(cache_key, result)
                    
            # Merge with cached results
            for idx, result in zip(indices, processed_results):
                cached_results[idx] = result
        
        # Reconstruct final results in original order
        final_results = [cached_results[i] for i in range(len(items))]
        
        # Record performance metrics
        execution_time = time.time() - start_time
        throughput = len(items) / execution_time if execution_time > 0 else 0.0
        
        cache_stats = self.cache.cache.stats()
        hit_rate = cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if cache_stats["hits"] + cache_stats["misses"] > 0 else 0.0
        
        self.logger.info(
            f"Batch processed {len(items)} items in {execution_time:.3f}s "
            f"({throughput:.1f} items/sec, cache hit rate: {hit_rate:.1%})"
        )
        
        return final_results
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.cache.stats()
        
        return {
            "caching": {
                "enabled": self.enable_caching,
                "stats": cache_stats,
            },
            "memory": {
                "enabled": self.enable_memory_opt,
                "current_usage": self.memory_optimizer.check_memory_usage(),
            },
            "parallel": {
                "enabled": self.enable_parallel,
                "max_workers": self.parallel_processor.max_workers,
            },
            "auto_scaling": {
                "enabled": self.enable_auto_scale,
                "current_scale": self.auto_scaler.current_scale,
                "performance_samples": len(self.auto_scaler.performance_history),
            },
            "optimization_features": [
                feature for feature, enabled in [
                    ("caching", self.enable_caching),
                    ("parallel_processing", self.enable_parallel),
                    ("memory_optimization", self.enable_memory_opt),
                    ("auto_scaling", self.enable_auto_scale),
                ] if enabled
            ]
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


# Convenient decorators
def cached(cache_key_func: Callable = None):
    """Decorator for caching function results."""
    if cache_key_func is None:
        cache_key_func = lambda *args, **kwargs: f"func_{hash(str(args))}_{hash(str(kwargs))}"
    
    return performance_optimizer.optimize_function(cache_key_func)


def memory_optimized(func: Callable) -> Callable:
    """Decorator for memory optimization."""
    return performance_optimizer.memory_optimizer.memory_efficient_decorator(func)


def batch_optimized(func: Callable) -> Callable:
    """Decorator for batch processing optimization."""
    @functools.wraps(func)
    def wrapper(items: List[Any]) -> List[Any]:
        return performance_optimizer.optimize_batch_processing(func, items)
    return wrapper