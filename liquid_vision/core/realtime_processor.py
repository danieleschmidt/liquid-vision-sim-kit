"""
Real-time event stream processing for liquid neural networks.
Optimized for low-latency, memory-bounded processing of neuromorphic data streams.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
import logging

from ..utils.logging import log_exceptions, log_performance
from ..utils.validation import validate_inputs
from .liquid_neurons import LiquidNet
from .event_encoding import EventEncoder


@dataclass
class RealTimeConfig:
    """Configuration for real-time processing."""
    buffer_size: int = 10000          # Maximum events in buffer
    batch_size: int = 32              # Processing batch size
    time_window_ms: float = 50.0      # Temporal window for batching
    max_latency_ms: float = 10.0      # Maximum allowed latency
    memory_limit_mb: float = 128.0    # Memory usage limit
    drop_policy: str = "oldest"       # "oldest", "newest", "random"
    enable_backpressure: bool = True   # Enable flow control
    num_worker_threads: int = 2       # Number of processing threads
    priority_mode: bool = False       # Enable priority-based processing


class EventBuffer:
    """Thread-safe circular buffer for event data."""
    
    def __init__(self, max_size: int, drop_policy: str = "oldest"):
        self.max_size = max_size
        self.drop_policy = drop_policy
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self._dropped_count = 0
        
    def put(self, event_batch: torch.Tensor, timeout: Optional[float] = None) -> bool:
        """Add events to buffer with optional timeout."""
        with self.not_full:
            if len(self.buffer) >= self.max_size:
                if self.drop_policy == "oldest":
                    self.buffer.popleft()
                    self._dropped_count += 1
                elif self.drop_policy == "newest":
                    # Drop the new event
                    self._dropped_count += 1
                    return False
                elif self.drop_policy == "random":
                    import random
                    if len(self.buffer) > 0:
                        idx = random.randint(0, len(self.buffer) - 1)
                        del self.buffer[idx]
                        self._dropped_count += 1
            
            self.buffer.append({
                'events': event_batch,
                'timestamp': time.time(),
                'size_bytes': event_batch.nelement() * event_batch.element_size()
            })
            self.not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get events from buffer with optional timeout."""
        with self.not_empty:
            if not self.buffer:
                if timeout is None:
                    self.not_empty.wait()
                else:
                    if not self.not_empty.wait(timeout):
                        return None
                        
            if self.buffer:
                item = self.buffer.popleft()
                self.not_full.notify()
                return item
            return None
    
    def size(self) -> int:
        with self.lock:
            return len(self.buffer)
    
    def memory_usage_mb(self) -> float:
        with self.lock:
            return sum(item['size_bytes'] for item in self.buffer) / (1024 * 1024)
    
    def dropped_count(self) -> int:
        return self._dropped_count
    
    def clear(self):
        with self.lock:
            self.buffer.clear()


class RealTimeEventProcessor:
    """
    Real-time event stream processor for liquid neural networks.
    
    Features:
    - Low-latency processing with configurable time windows
    - Memory-bounded operation with backpressure control
    - Multi-threaded processing pipeline
    - Adaptive batching and priority handling
    - Performance monitoring and statistics
    """
    
    def __init__(
        self,
        model: LiquidNet,
        encoder: EventEncoder,
        config: RealTimeConfig,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.encoder = encoder
        self.config = config
        self.device = device
        
        # Processing pipeline
        self.input_buffer = EventBuffer(config.buffer_size, config.drop_policy)
        self.output_queue = queue.Queue(maxsize=config.buffer_size // 2)
        
        # Worker threads
        self.workers = []
        self.running = False
        self.stats_lock = threading.Lock()
        
        # Performance statistics
        self.reset_stats()
        
        # Model state management
        self.model.eval()
        self.last_reset_time = time.time()
        
        logger = logging.getLogger(__name__)
        logger.info(f"Initialized RealTimeEventProcessor with {config.num_worker_threads} workers")
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self.stats_lock:
            self.stats = {
                'events_processed': 0,
                'batches_processed': 0,
                'total_latency': 0.0,
                'max_latency': 0.0,
                'dropped_events': 0,
                'memory_warnings': 0,
                'start_time': time.time()
            }
    
    @log_performance
    def process_event_batch(self, events: torch.Tensor) -> torch.Tensor:
        """Process a batch of events through the model."""
        start_time = time.time()
        
        try:
            # Encode events
            encoded_events = self.encoder.encode_batch(events)
            encoded_events = encoded_events.to(self.device)
            
            # Forward pass through model
            with torch.no_grad():
                output = self.model(encoded_events)
            
            # Update statistics
            latency = (time.time() - start_time) * 1000  # ms
            with self.stats_lock:
                self.stats['events_processed'] += events.shape[0]
                self.stats['batches_processed'] += 1
                self.stats['total_latency'] += latency
                self.stats['max_latency'] = max(self.stats['max_latency'], latency)
                
                # Check latency constraint
                if latency > self.config.max_latency_ms:
                    logging.warning(f"Latency constraint violated: {latency:.2f}ms > {self.config.max_latency_ms}ms")
            
            return output
            
        except Exception as e:
            logging.error(f"Error processing event batch: {e}")
            raise
    
    def _worker_loop(self, worker_id: int):
        """Main processing loop for worker threads."""
        logger = logging.getLogger(__name__)
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get batch from buffer
                batch_data = self.input_buffer.get(timeout=0.1)
                if batch_data is None:
                    continue
                
                events = batch_data['events']
                timestamp = batch_data['timestamp']
                
                # Check for stale data
                age_ms = (time.time() - timestamp) * 1000
                if age_ms > self.config.max_latency_ms * 2:
                    logger.warning(f"Dropping stale batch (age: {age_ms:.2f}ms)")
                    continue
                
                # Process batch
                try:
                    output = self.process_event_batch(events)
                    
                    # Put result in output queue
                    result = {
                        'output': output,
                        'timestamp': timestamp,
                        'processed_at': time.time(),
                        'worker_id': worker_id
                    }
                    
                    if not self.output_queue.full():
                        self.output_queue.put(result, timeout=0.1)
                    else:
                        logger.warning("Output queue full, dropping result")
                        
                except Exception as e:
                    logger.error(f"Worker {worker_id} processing error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} stopped")
    
    def start(self):
        """Start the real-time processing pipeline."""
        if self.running:
            return
            
        self.running = True
        self.reset_stats()
        
        # Start worker threads
        for i in range(self.config.num_worker_threads):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Started {len(self.workers)} worker threads")
    
    def stop(self, timeout: float = 5.0):
        """Stop the processing pipeline."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        self.workers.clear()
        logging.info("Stopped all worker threads")
    
    def put_events(self, events: torch.Tensor, timeout: Optional[float] = None) -> bool:
        """Add events to processing pipeline."""
        if not self.running:
            raise RuntimeError("Processor not started")
        
        # Memory check
        if self.input_buffer.memory_usage_mb() > self.config.memory_limit_mb:
            with self.stats_lock:
                self.stats['memory_warnings'] += 1
            if self.config.enable_backpressure:
                return False
        
        return self.input_buffer.put(events, timeout)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get processed results."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.stats_lock:
            uptime = time.time() - self.stats['start_time']
            avg_latency = (self.stats['total_latency'] / max(1, self.stats['batches_processed']))
            throughput = self.stats['events_processed'] / max(1, uptime)
            
            return {
                **self.stats,
                'uptime_seconds': uptime,
                'average_latency_ms': avg_latency,
                'throughput_events_per_sec': throughput,
                'buffer_size': self.input_buffer.size(),
                'buffer_memory_mb': self.input_buffer.memory_usage_mb(),
                'dropped_events': self.input_buffer.dropped_count(),
                'output_queue_size': self.output_queue.qsize()
            }
    
    def reset_model_state(self):
        """Reset the model's internal state."""
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()
        self.last_reset_time = time.time()
    
    def adaptive_batch_size(self) -> int:
        """Dynamically adjust batch size based on performance."""
        stats = self.get_stats()
        avg_latency = stats['average_latency_ms']
        
        if avg_latency > self.config.max_latency_ms * 0.8:
            # Reduce batch size to improve latency
            return max(1, self.config.batch_size // 2)
        elif avg_latency < self.config.max_latency_ms * 0.3:
            # Increase batch size for better throughput
            return min(self.config.batch_size * 2, 128)
        
        return self.config.batch_size
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class StreamProcessor:
    """High-level streaming interface for continuous event processing."""
    
    def __init__(
        self,
        model: LiquidNet,
        encoder: EventEncoder,
        config: Optional[RealTimeConfig] = None,
        device: str = "cpu"
    ):
        if config is None:
            config = RealTimeConfig()
        
        self.processor = RealTimeEventProcessor(model, encoder, config, device)
        self.callbacks = []
        self.running = False
    
    def add_callback(self, callback: Callable[[torch.Tensor, Dict[str, Any]], None]):
        """Add result callback function."""
        self.callbacks.append(callback)
    
    def start_stream(self):
        """Start streaming processing."""
        self.processor.start()
        self.running = True
        
        # Result handler thread
        def result_handler():
            while self.running:
                result = self.processor.get_result(timeout=0.1)
                if result is not None:
                    for callback in self.callbacks:
                        try:
                            callback(result['output'], result)
                        except Exception as e:
                            logging.error(f"Callback error: {e}")
        
        self.result_thread = threading.Thread(target=result_handler)
        self.result_thread.daemon = True
        self.result_thread.start()
    
    def stop_stream(self):
        """Stop streaming processing."""
        self.running = False
        if hasattr(self, 'result_thread'):
            self.result_thread.join(timeout=2.0)
        self.processor.stop()
    
    def process_event_stream(self, event_stream) -> None:
        """Process continuous event stream."""
        for events in event_stream:
            if not self.processor.put_events(events, timeout=0.01):
                logging.warning("Failed to enqueue events - backpressure activated")
    
    def get_performance_report(self) -> str:
        """Get formatted performance report."""
        stats = self.processor.get_stats()
        
        report = f"""
=== Real-Time Processing Performance Report ===
Uptime: {stats['uptime_seconds']:.1f}s
Events Processed: {stats['events_processed']:,}
Batches Processed: {stats['batches_processed']:,}
Throughput: {stats['throughput_events_per_sec']:.1f} events/sec
Average Latency: {stats['average_latency_ms']:.2f}ms
Max Latency: {stats['max_latency']:.2f}ms
Dropped Events: {stats['dropped_events']:,}
Memory Warnings: {stats['memory_warnings']:,}
Buffer Usage: {stats['buffer_size']:,} events ({stats['buffer_memory_mb']:.1f}MB)
Output Queue: {stats['output_queue_size']:,} items
"""
        return report


def create_realtime_processor(
    model: LiquidNet,
    encoder: EventEncoder,
    target_latency_ms: float = 10.0,
    memory_limit_mb: float = 128.0,
    device: str = "cpu"
) -> RealTimeEventProcessor:
    """Factory function to create optimized real-time processor."""
    
    config = RealTimeConfig(
        buffer_size=int(10000 * (128.0 / memory_limit_mb)),  # Scale with memory
        batch_size=32 if target_latency_ms > 5.0 else 16,    # Smaller batches for lower latency
        max_latency_ms=target_latency_ms,
        memory_limit_mb=memory_limit_mb,
        num_worker_threads=2 if target_latency_ms > 10.0 else 1,  # More workers for higher throughput
        enable_backpressure=True,
        drop_policy="oldest"
    )
    
    return RealTimeEventProcessor(model, encoder, config, device)