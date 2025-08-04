"""
Energy profiling and performance benchmarking for liquid neural networks.
Measures power consumption, latency, and energy efficiency on various hardware.
"""

import time
import psutil
import threading
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
import numpy as np

try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EnergyProfile:
    """Energy profiling results."""
    avg_power_mw: float
    peak_power_mw: float
    min_power_mw: float
    total_energy_mj: float
    duration_s: float
    inference_count: int
    avg_latency_ms: float
    throughput_fps: float
    efficiency_inferences_per_mj: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "avg_power_mw": self.avg_power_mw,
            "peak_power_mw": self.peak_power_mw,
            "min_power_mw": self.min_power_mw,
            "total_energy_mj": self.total_energy_mj,
            "duration_s": self.duration_s,
            "inference_count": self.inference_count,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_fps": self.throughput_fps,
            "efficiency_inferences_per_mj": self.efficiency_inferences_per_mj,
        }


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmarking."""
    duration_s: float = 60.0
    warmup_s: float = 5.0
    power_sampling_hz: float = 10.0
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32])
    input_shapes: List[Tuple[int, ...]] = field(default_factory=lambda: [(1, 64, 64)])
    device: str = "cpu"
    save_detailed_logs: bool = True
    output_dir: str = "benchmarks/results"


class PowerMonitor:
    """
    Monitor system power consumption during inference.
    Uses system utilities to estimate power usage.
    """
    
    def __init__(self, sampling_hz: float = 10.0):
        self.sampling_hz = sampling_hz
        self.sampling_interval = 1.0 / sampling_hz
        self.power_measurements = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start power monitoring."""
        self.power_measurements.clear()
        self.timestamps.clear()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> List[Tuple[float, float]]:
        """Stop monitoring and return measurements."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        return list(zip(self.timestamps, self.power_measurements))
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        start_time = time.time()
        
        while self.monitoring:
            timestamp = time.time() - start_time
            power_mw = self._measure_power()
            
            self.timestamps.append(timestamp)
            self.power_measurements.append(power_mw)
            
            time.sleep(self.sampling_interval)
            
    def _measure_power(self) -> float:
        """
        Measure current power consumption.
        This is a simplified implementation - real hardware would use
        dedicated power measurement tools.
        """
        # Use CPU usage as a proxy for power consumption
        cpu_percent = psutil.cpu_percent()
        
        # Estimate power based on CPU usage (rough approximation)
        # Typical values: idle ~5W, full load ~15-30W for mobile CPUs
        idle_power_w = 5.0
        max_power_w = 25.0
        
        estimated_power_w = idle_power_w + (max_power_w - idle_power_w) * (cpu_percent / 100.0)
        return estimated_power_w * 1000  # Convert to milliwatts
        
    def get_statistics(self) -> Dict[str, float]:
        """Get power consumption statistics."""
        if not self.power_measurements:
            return {}
            
        return {
            "avg_power_mw": statistics.mean(self.power_measurements),
            "peak_power_mw": max(self.power_measurements),
            "min_power_mw": min(self.power_measurements),
            "std_power_mw": statistics.stdev(self.power_measurements) if len(self.power_measurements) > 1 else 0,
            "samples": len(self.power_measurements),
        }


class LatencyProfiler:
    """
    Profile inference latency and throughput.
    """
    
    def __init__(self):
        self.latencies = []
        
    def profile_function(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Profile function execution time.
        
        Args:
            func: Function to profile
            args: Function arguments
            kwargs: Function keyword arguments
            num_iterations: Number of profiling iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Latency statistics
        """
        kwargs = kwargs or {}
        
        # Warmup
        for _ in range(warmup_iterations):
            func(*args, **kwargs)
            
        # Profile
        latencies = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "throughput_fps": 1000 / statistics.mean(latencies),
        }


class EnergyProfiler:
    """
    Comprehensive energy profiling for neural networks.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.power_monitor = PowerMonitor(config.power_sampling_hz)
        self.latency_profiler = LatencyProfiler()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def profile_model(
        self,
        model,
        input_generator: Callable,
        model_name: str = "liquid_model",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Profile model energy consumption and performance.
        
        Args:
            model: Model to profile
            input_generator: Function that generates input data
            model_name: Name for saving results
            
        Returns:
            Profiling results
        """
        print(f"Profiling {model_name}...")
        
        results = {}
        
        # Profile different batch sizes
        for batch_size in self.config.batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            batch_results = {}
            
            # Profile different input shapes
            for input_shape in self.config.input_shapes:
                shape_key = f"shape_{'x'.join(map(str, input_shape))}"
                print(f"    Input shape: {input_shape}")
                
                # Generate test input
                test_input = input_generator(batch_size, input_shape)
                
                # Profile energy and latency
                profile = self._profile_inference(
                    model, test_input, f"{model_name}_{batch_size}_{shape_key}"
                )
                
                batch_results[shape_key] = profile.to_dict()
                
            results[f"batch_{batch_size}"] = batch_results
            
        # Save results
        self._save_results(results, model_name)
        
        return results
        
    def _profile_inference(
        self,
        model,
        test_input,
        profile_name: str,
    ) -> EnergyProfile:
        """Profile single inference configuration."""
        # Set model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
            
        # Warmup
        print(f"      Warming up for {self.config.warmup_s}s...")
        warmup_end = time.time() + self.config.warmup_s
        while time.time() < warmup_end:
            if hasattr(model, 'reset_states'):
                model.reset_states()
            _ = model(test_input)
            
        # Start monitoring
        print(f"      Profiling for {self.config.duration_s}s...")
        self.power_monitor.start_monitoring()
        
        # Run inference loop
        start_time = time.time()
        inference_count = 0
        latencies = []
        
        while time.time() - start_time < self.config.duration_s:
            if hasattr(model, 'reset_states'):
                model.reset_states()
                
            inference_start = time.perf_counter()
            _ = model(test_input)
            inference_end = time.perf_counter()
            
            latency_ms = (inference_end - inference_start) * 1000
            latencies.append(latency_ms)
            inference_count += 1
            
        end_time = time.time()
        duration_s = end_time - start_time
        
        # Stop monitoring
        power_measurements = self.power_monitor.stop_monitoring()
        power_stats = self.power_monitor.get_statistics()
        
        # Calculate energy metrics
        avg_power_mw = power_stats.get("avg_power_mw", 0)
        peak_power_mw = power_stats.get("peak_power_mw", 0)
        min_power_mw = power_stats.get("min_power_mw", 0)
        total_energy_mj = avg_power_mw * duration_s  # mW * s = mJ
        
        avg_latency_ms = statistics.mean(latencies) if latencies else 0
        throughput_fps = inference_count / duration_s if duration_s > 0 else 0
        efficiency = inference_count / total_energy_mj if total_energy_mj > 0 else 0
        
        # Save detailed logs if requested
        if self.config.save_detailed_logs:
            self._save_detailed_logs(
                profile_name, power_measurements, latencies, duration_s
            )
            
        return EnergyProfile(
            avg_power_mw=avg_power_mw,
            peak_power_mw=peak_power_mw,
            min_power_mw=min_power_mw,
            total_energy_mj=total_energy_mj,
            duration_s=duration_s,
            inference_count=inference_count,
            avg_latency_ms=avg_latency_ms,
            throughput_fps=throughput_fps,
            efficiency_inferences_per_mj=efficiency,
        )
        
    def _save_results(self, results: Dict, model_name: str):
        """Save profiling results."""
        # Save JSON summary
        json_path = self.output_dir / f"{model_name}_profile.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save CSV summary
        csv_path = self.output_dir / f"{model_name}_summary.csv"
        self._save_csv_summary(results, csv_path)
        
        print(f"      Results saved to {json_path}")
        
    def _save_csv_summary(self, results: Dict, csv_path: Path):
        """Save CSV summary of results."""
        rows = []
        
        for batch_key, batch_data in results.items():
            for shape_key, metrics in batch_data.items():
                row = {
                    "batch_size": batch_key.replace("batch_", ""),
                    "input_shape": shape_key.replace("shape_", "").replace("x", "×"),
                    **metrics
                }
                rows.append(row)
                
        if rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                
    def _save_detailed_logs(
        self,
        profile_name: str,
        power_measurements: List[Tuple[float, float]],
        latencies: List[float],
        duration_s: float,
    ):
        """Save detailed profiling logs."""
        # Power measurements
        power_path = self.output_dir / f"{profile_name}_power.csv"
        with open(power_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "power_mw"])
            for timestamp, power in power_measurements:
                writer.writerow([timestamp, power])
                
        # Latency measurements
        latency_path = self.output_dir / f"{profile_name}_latency.csv"
        with open(latency_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["inference_id", "latency_ms"])
            for i, latency in enumerate(latencies):
                writer.writerow([i, latency])


def compare_models(
    models: Dict[str, Any],
    input_generator: Callable,
    config: BenchmarkConfig,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare energy efficiency of multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        input_generator: Function to generate test inputs
        config: Benchmark configuration
        output_path: Path to save comparison results
        
    Returns:
        Comparison results
    """
    profiler = EnergyProfiler(config)
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nProfiling {model_name}...")
        results = profiler.profile_model(model, input_generator, model_name)
        all_results[model_name] = results
        
    # Generate comparison summary
    comparison = _generate_comparison_summary(all_results)
    
    # Save comparison results
    if output_path:
        comparison_path = Path(output_path)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(comparison_path, 'w') as f:
            json.dump({
                "individual_results": all_results,
                "comparison": comparison
            }, f, indent=2)
            
        print(f"\nComparison results saved to {comparison_path}")
        
    return {
        "individual_results": all_results,
        "comparison": comparison
    }


def _generate_comparison_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comparison summary from multiple model results."""
    summary = {
        "models": list(results.keys()),
        "metrics": {},
        "rankings": {}
    }
    
    # Extract metrics for comparison
    metrics_data = {}
    
    for model_name, model_results in results.items():
        model_metrics = []
        
        for batch_key, batch_data in model_results.items():
            for shape_key, metrics in batch_data.items():
                model_metrics.append(metrics)
                
        # Average across all configurations
        if model_metrics:
            avg_metrics = {}
            for key in model_metrics[0].keys():
                values = [m[key] for m in model_metrics]
                avg_metrics[key] = statistics.mean(values)
                
            metrics_data[model_name] = avg_metrics
            
    summary["metrics"] = metrics_data
    
    # Generate rankings
    ranking_criteria = [
        ("efficiency_inferences_per_mj", "higher_better"),
        ("avg_power_mw", "lower_better"),
        ("avg_latency_ms", "lower_better"),
        ("throughput_fps", "higher_better"),
    ]
    
    for criterion, direction in ranking_criteria:
        if criterion in next(iter(metrics_data.values()), {}):
            values = [(name, metrics[criterion]) for name, metrics in metrics_data.items()]
            
            if direction == "higher_better":
                ranked = sorted(values, key=lambda x: x[1], reverse=True)
            else:
                ranked = sorted(values, key=lambda x: x[1])
                
            summary["rankings"][criterion] = [name for name, _ in ranked]
            
    return summary


# Example usage and testing functions
def create_dummy_input_generator():
    """Create dummy input generator for testing."""
    def generator(batch_size: int, input_shape: Tuple[int, ...]):
        # Generate random input data
        full_shape = (batch_size,) + input_shape
        return np.random.randn(*full_shape).astype(np.float32)
    
    return generator


def create_dummy_model():
    """Create dummy model for testing."""
    class DummyModel:
        def __call__(self, x):
            # Simulate some computation
            time.sleep(0.001)  # 1ms processing time
            return x.mean(axis=-1) if hasattr(x, 'mean') else sum(x) / len(x)
            
        def eval(self):
            pass
            
        def reset_states(self):
            pass
            
    return DummyModel()


if __name__ == "__main__":
    # Example usage
    config = BenchmarkConfig(
        duration_s=10.0,
        warmup_s=2.0,
        batch_sizes=[1, 4],
        input_shapes=[(64, 64), (128, 128)],
        output_dir="benchmarks/example_results"
    )
    
    # Create test models
    models = {
        "dummy_fast": create_dummy_model(),
        "dummy_slow": create_dummy_model(),  # Could be modified to be slower
    }
    
    input_gen = create_dummy_input_generator()
    
    print("Running benchmark comparison...")
    results = compare_models(
        models=models,
        input_generator=input_gen,
        config=config,
        output_path="benchmarks/example_results/comparison.json"
    )
    
    print("\nBenchmarking completed!")
    print(f"Models ranked by efficiency: {results['comparison']['rankings'].get('efficiency_inferences_per_mj', 'N/A')}")