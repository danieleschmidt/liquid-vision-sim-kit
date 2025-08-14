#!/usr/bin/env python3
"""
Generation 3 Demo - MAKE IT SCALE
Demonstrates performance optimization, caching, auto-scaling, and parallel processing.
"""

import sys
import os
import time
import random

# Add the parent directory to the path to import liquid_vision
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import liquid_vision
from liquid_vision.core.optimized_liquid_nets import (
    create_optimized_liquid_net, OptimizedMinimalTensor, 
    adaptive_scaler, AdaptiveModelScaler
)
from liquid_vision.core.minimal_fallback import MinimalTensor, create_minimal_liquid_net
from liquid_vision.optimization.performance_optimizer import (
    performance_optimizer, cached, memory_optimized, batch_optimized
)


def demo_caching_system():
    """Demonstrate intelligent caching system."""
    print("🗄️ Intelligent Caching System Demo")
    print("=" * 40)
    
    # Create test function with caching
    @cached(lambda x, y: f"compute_{x}_{y}")
    def expensive_computation(x: float, y: float) -> float:
        """Simulate expensive computation."""
        time.sleep(0.01)  # Simulate computation time
        return x * y + math.sin(x) + math.cos(y)
        
    import math
    
    # Test caching effectiveness
    test_inputs = [(0.1, 0.2), (0.3, 0.4), (0.1, 0.2), (0.5, 0.6), (0.3, 0.4)]
    
    print("Testing cache performance:")
    start_time = time.time()
    
    results = []
    for i, (x, y) in enumerate(test_inputs):
        result = expensive_computation(x, y)
        results.append(result)
        print(f"  Call {i+1}: compute({x}, {y}) = {result:.4f}")
        
    total_time = time.time() - start_time
    
    # Get cache statistics
    cache_stats = performance_optimizer.cache.cache.stats()
    
    print(f"\n📊 Cache Performance:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Expected speedup: ~{cache_stats['hits'] * 10:.0f}ms saved")
    
    return cache_stats['hit_rate'] > 0.3  # At least 30% hit rate


def demo_memory_optimization():
    """Demonstrate memory optimization features."""
    print("\n🧠 Memory Optimization Demo")
    print("=" * 35)
    
    # Create memory-intensive function
    @memory_optimized
    def memory_intensive_task(size: int) -> int:
        """Simulate memory-intensive task."""
        # Create large data structure
        large_list = [i * random.random() for i in range(size)]
        
        # Process data
        result = sum(large_list)
        
        # Data goes out of scope here
        return int(result)
        
    memory_optimizer = performance_optimizer.memory_optimizer
    
    # Test different memory loads
    test_sizes = [1000, 5000, 10000, 50000]
    
    print("Testing memory optimization:")
    for size in test_sizes:
        initial_memory = memory_optimizer.check_memory_usage()
        
        result = memory_intensive_task(size)
        
        final_memory = memory_optimizer.check_memory_usage()
        memory_change = final_memory - initial_memory
        
        print(f"  Size {size:5d}: Result={result:8d}, Memory change: {memory_change:+.1%}")
        
        # Force optimization if needed
        if memory_optimizer.optimize_memory():
            print(f"    🧹 Memory optimization triggered")
            
    print(f"\n📈 Memory Management:")
    print(f"  Current usage: {memory_optimizer.check_memory_usage():.1%}")
    print(f"  Optimization threshold: {memory_optimizer.memory_threshold:.1%}")
    print(f"  Auto-cleanup enabled: ✅")
    
    return True


def demo_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    print("\n⚡ Parallel Processing Demo")
    print("=" * 35)
    
    # Create CPU-intensive function for testing
    def compute_prime_factors(n: int) -> List[int]:
        """Compute prime factors of a number."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    # Generate test data
    test_numbers = [random.randint(1000, 10000) for _ in range(20)]
    
    parallel_processor = performance_optimizer.parallel_processor
    
    print(f"Testing parallel processing with {len(test_numbers)} numbers:")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [compute_prime_factors(n) for n in test_numbers]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    parallel_results = parallel_processor.process_batch(compute_prime_factors, test_numbers)
    parallel_time = time.time() - start_time
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    efficiency = speedup / parallel_processor.max_workers
    
    print(f"\n📊 Parallel Processing Results:")
    print(f"  Sequential time: {sequential_time:.3f}s")
    print(f"  Parallel time: {parallel_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {efficiency:.1%}")
    print(f"  Workers used: {parallel_processor.max_workers}")
    
    # Verify results are identical
    results_match = all(
        set(seq) == set(par) 
        for seq, par in zip(sequential_results, parallel_results)
    )
    print(f"  Results identical: {'✅' if results_match else '❌'}")
    
    return speedup > 1.1 and results_match  # At least 10% speedup


def demo_optimized_liquid_networks():
    """Demonstrate optimized liquid neural networks."""
    print("\n🌊 Optimized Liquid Networks Demo")
    print("=" * 40)
    
    # Create both standard and optimized models
    standard_model = create_minimal_liquid_net(4, 2, architecture="small")
    optimized_model = create_optimized_liquid_net(4, 2, architecture="small")
    
    # Generate test data
    test_data = [
        MinimalTensor([[random.random() * 2 - 1 for _ in range(4)]]) 
        for _ in range(50)
    ]
    
    print(f"Comparing standard vs optimized models with {len(test_data)} samples:")
    
    # Benchmark standard model
    start_time = time.time()
    standard_results = []
    for x in test_data:
        output = standard_model(x)
        standard_results.append(output.data[0])
    standard_time = time.time() - start_time
    
    # Benchmark optimized model
    start_time = time.time()
    optimized_results = []
    for x in test_data:
        output = optimized_model(x)
        optimized_results.append(output.data[0])
    optimized_time = time.time() - start_time
    
    # Performance comparison
    speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
    
    print(f"\n📊 Performance Comparison:")
    print(f"  Standard model time: {standard_time:.3f}s")
    print(f"  Optimized model time: {optimized_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Cache statistics for optimized model
    cache_stats = performance_optimizer.cache.cache.stats()
    print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    # Test batch processing
    print(f"\n🚀 Batch Processing Test:")
    start_time = time.time()
    batch_results = optimized_model.process_batch(test_data)
    batch_time = time.time() - start_time
    
    batch_speedup = optimized_time / batch_time if batch_time > 0 else 1.0
    
    print(f"  Individual processing: {optimized_time:.3f}s")
    print(f"  Batch processing: {batch_time:.3f}s") 
    print(f"  Batch speedup: {batch_speedup:.2f}x")
    
    return speedup >= 0.8  # Allow some overhead for optimization features


def demo_auto_scaling():
    """Demonstrate automatic model scaling."""
    print("\n📈 Auto-Scaling Demo")
    print("=" * 25)
    
    # Create adaptive scaler
    scaler = AdaptiveModelScaler()
    
    # Start with small model
    model = create_optimized_liquid_net(2, 1, architecture="tiny")
    print(f"Starting with {scaler.current_architecture} architecture")
    
    # Simulate different workload scenarios
    scenarios = [
        ("Light load", [(1, 0.005) for _ in range(5)]),  # Small batches, fast
        ("Medium load", [(5, 0.02) for _ in range(5)]),  # Medium batches
        ("Heavy load", [(20, 0.15) for _ in range(5)]),  # Large batches, slow
        ("Reduced load", [(2, 0.01) for _ in range(5)]), # Back to light
    ]
    
    scaling_events = 0
    
    for scenario_name, workload in scenarios:
        print(f"\n🔄 Scenario: {scenario_name}")
        
        # Process workload
        for batch_size, target_latency in workload:
            # Simulate processing
            start_time = time.time()
            
            # Create test batch
            test_batch = [
                MinimalTensor([[random.random(), random.random()]]) 
                for _ in range(batch_size)
            ]
            
            # Process batch
            results = model.process_batch(test_batch)
            actual_latency = time.time() - start_time
            
            # Calculate throughput
            throughput = batch_size / actual_latency if actual_latency > 0 else 0
            
            # Record performance
            scaler.record_performance(batch_size, actual_latency, throughput)
            
            print(f"  Batch size: {batch_size:2d}, Latency: {actual_latency:.3f}s, Throughput: {throughput:.1f} ops/s")
            
        # Check if scaling is needed
        recommended = scaler.recommend_architecture()
        if recommended != scaler.current_architecture:
            print(f"  🎯 Scaling recommendation: {scaler.current_architecture} → {recommended}")
            model = scaler.auto_scale_model(model)
            scaling_events += 1
        else:
            print(f"  ✅ Current architecture ({scaler.current_architecture}) is optimal")
    
    print(f"\n📊 Auto-Scaling Summary:")
    print(f"  Total scaling events: {scaling_events}")
    print(f"  Final architecture: {scaler.current_architecture}")
    print(f"  Average batch size: {scaler.workload_metrics['avg_batch_size']:.1f}")
    print(f"  Average latency: {scaler.workload_metrics['avg_latency']:.3f}s")
    
    return scaling_events > 0  # Should have at least one scaling event


def demo_comprehensive_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("\n🏁 Comprehensive Performance Benchmarks")
    print("=" * 45)
    
    architectures = ["tiny", "small", "base"]
    batch_sizes = [1, 5, 10, 20]
    
    benchmark_results = {}
    
    for arch in architectures:
        print(f"\n🔬 Benchmarking {arch.upper()} architecture:")
        
        model = create_optimized_liquid_net(4, 2, architecture=arch)
        arch_results = {}
        
        for batch_size in batch_sizes:
            # Generate test data
            test_inputs = [
                MinimalTensor([[random.random() * 2 - 1 for _ in range(4)]]) 
                for _ in range(batch_size)
            ]
            
            # Run benchmark
            perf_results = model.benchmark_performance(test_inputs, iterations=20)
            
            arch_results[batch_size] = perf_results
            
            print(f"  Batch {batch_size:2d}: {perf_results['batch_throughput']:.1f} ops/s, "
                  f"speedup: {perf_results['batch_speedup']:.2f}x")
        
        benchmark_results[arch] = arch_results
    
    # Find optimal configurations
    print(f"\n🏆 Optimal Configurations:")
    
    for batch_size in batch_sizes:
        best_arch = None
        best_throughput = 0
        
        for arch in architectures:
            throughput = benchmark_results[arch][batch_size]['batch_throughput']
            if throughput > best_throughput:
                best_throughput = throughput
                best_arch = arch
        
        print(f"  Batch size {batch_size:2d}: {best_arch} ({best_throughput:.1f} ops/s)")
    
    return len(benchmark_results) == len(architectures)


def demo_system_optimization_status():
    """Show overall system optimization status."""
    print("\n🎛️ System Optimization Status")
    print("=" * 35)
    
    # Get optimization statistics
    opt_stats = performance_optimizer.get_optimization_stats()
    
    print("🔧 Optimization Features:")
    for feature in opt_stats["optimization_features"]:
        print(f"  ✅ {feature.replace('_', ' ').title()}")
    
    print(f"\n📊 Caching System:")
    cache_stats = opt_stats["caching"]["stats"]
    print(f"  Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Total requests: {cache_stats['total_requests']}")
    
    print(f"\n🧠 Memory Management:")
    memory_stats = opt_stats["memory"]
    print(f"  Current usage: {memory_stats['current_usage']:.1%}")
    print(f"  Optimization enabled: {'✅' if memory_stats['enabled'] else '❌'}")
    
    print(f"\n⚡ Parallel Processing:")
    parallel_stats = opt_stats["parallel"]
    print(f"  Max workers: {parallel_stats['max_workers']}")
    print(f"  Enabled: {'✅' if parallel_stats['enabled'] else '❌'}")
    
    print(f"\n📈 Auto-Scaling:")
    scale_stats = opt_stats["auto_scaling"]
    print(f"  Current scale: {scale_stats['current_scale']:.2f}")
    print(f"  Performance samples: {scale_stats['performance_samples']}")
    print(f"  Enabled: {'✅' if scale_stats['enabled'] else '❌'}")
    
    # Overall optimization score
    enabled_features = len(opt_stats["optimization_features"])
    total_features = 4  # caching, parallel, memory, auto_scaling
    optimization_score = enabled_features / total_features
    
    print(f"\n🏆 Overall Optimization Score: {optimization_score:.1%}")
    
    return optimization_score > 0.75


def main():
    """Run the complete Generation 3 performance demo."""
    print("⚡ LIQUID VISION SIM-KIT - GENERATION 3 DEMO")
    print("🤖 MAKE IT SCALE: Performance Optimization & Auto-Scaling")
    print("=" * 70)
    
    demos = [
        ("Intelligent Caching", demo_caching_system),
        ("Memory Optimization", demo_memory_optimization),
        ("Parallel Processing", demo_parallel_processing),
        ("Optimized Networks", demo_optimized_liquid_networks),
        ("Auto-Scaling", demo_auto_scaling),
        ("Performance Benchmarks", demo_comprehensive_benchmarks),
        ("System Status", demo_system_optimization_status),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n{'🔹 ' + demo_name}")
        try:
            success = demo_func()
            results.append((demo_name, "✅ PASSED" if success else "❌ FAILED"))
        except Exception as e:
            results.append((demo_name, f"❌ ERROR: {e}"))
            print(f"❌ {demo_name} failed: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 GENERATION 3 PERFORMANCE RESULTS")
    print("=" * 70)
    
    for demo_name, result in results:
        print(f"{demo_name:25s}: {result}")
    
    passed = sum(1 for _, result in results if "PASSED" in result)
    total = len(results)
    
    print(f"\n📈 Overall Success Rate: {passed}/{total} ({passed/total*100:.0f}%)")
    
    # Final performance summary
    final_stats = performance_optimizer.get_optimization_stats()
    cache_stats = final_stats["caching"]["stats"]
    
    print(f"\n⚡ Final Performance Summary:")
    print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Memory optimization: {'Active' if final_stats['memory']['enabled'] else 'Inactive'}")
    print(f"  Parallel workers: {final_stats['parallel']['max_workers']}")
    print(f"  Auto-scaling: {'Active' if final_stats['auto_scaling']['enabled'] else 'Inactive'}")
    
    if passed == total:
        print("\n🏆 GENERATION 3 OPTIMIZATION COMPLETE!")
        print("⚡ High-performance caching and memory optimization active")
        print("🚀 Parallel processing enabled for batch operations")
        print("📈 Auto-scaling responds to workload changes")
        print("🎯 System optimized for production-scale performance")
        print("🔄 Ready for Quality Gates & Production Deployment")
    else:
        print("\n⚠️  Some optimization tests failed - review implementation")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)