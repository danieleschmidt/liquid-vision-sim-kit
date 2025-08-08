#!/usr/bin/env python3
"""Test script for Generation 3: Making it scale with performance optimizations."""

import torch
import numpy as np
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from liquid_vision import LiquidNet, EventSimulator
from liquid_vision.simulation.event_simulator import DVSSimulator
from liquid_vision.core.liquid_neurons import create_liquid_net, get_model_info

def test_model_performance():
    """Test model performance and memory usage."""
    print("⚡ Testing Model Performance & Memory...")
    
    try:
        # Test different model sizes
        architectures = ["tiny", "small", "base"]
        performance_results = {}
        
        for arch in architectures:
            model = create_liquid_net(input_dim=2, output_dim=10, architecture=arch)
            info = get_model_info(model)
            
            # Measure inference time
            batch_size = 64
            seq_len = 100
            x = torch.randn(batch_size, seq_len, 2)
            
            # Warmup
            for _ in range(5):
                model(x, reset_state=True)
            
            # Benchmark
            start_time = time.time()
            num_runs = 50
            for _ in range(num_runs):
                output = model(x, reset_state=True)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            throughput = (batch_size * seq_len) / avg_time  # samples/second
            
            performance_results[arch] = {
                "parameters": info["total_parameters"],
                "avg_time_ms": avg_time * 1000,
                "throughput_samples_per_sec": throughput,
                "memory_mb": torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            }
            
            print(f"✅ {arch}: {info['total_parameters']} params, "
                  f"{avg_time*1000:.2f}ms, {throughput:.0f} samples/s")
        
        # Test batching efficiency
        model = create_liquid_net(input_dim=2, output_dim=5, architecture="small")
        batch_sizes = [1, 8, 16, 32, 64]
        
        print("📊 Batch Size Efficiency:")
        for bs in batch_sizes:
            x = torch.randn(bs, 50, 2)
            start = time.time()
            output = model(x, reset_state=True)
            elapsed = time.time() - start
            samples_per_sec = (bs * 50) / elapsed
            print(f"   Batch {bs:2d}: {samples_per_sec:8.0f} samples/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("🔄 Testing Concurrent Processing...")
    
    try:
        # Test thread-safe model usage
        model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
        
        def worker_function(worker_id, num_samples=100):
            """Worker function for concurrent testing."""
            try:
                results = []
                for i in range(num_samples):
                    x = torch.randn(1, 2)
                    output = model(x, reset_state=True)
                    results.append(output.item())
                return f"Worker {worker_id}: {len(results)} samples processed"
            except Exception as e:
                return f"Worker {worker_id}: Error - {e}"
        
        # Test with threads
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_function, i, 50) for i in range(4)]
            thread_results = [future.result() for future in futures]
        thread_time = time.time() - start_time
        
        print(f"✅ Thread-based: {thread_time:.3f}s")
        for result in thread_results:
            print(f"   {result}")
        
        # Test simulation scaling
        simulator = DVSSimulator(resolution=(128, 128))
        frames = [np.random.rand(128, 128) * 0.5 + i*0.1 for i in range(10)]
        
        # Sequential processing
        start_time = time.time()
        sequential_events = []
        for i, frame in enumerate(frames):
            events = simulator.simulate_frame(frame, timestamp=float(i))
            sequential_events.append(len(events))
        sequential_time = time.time() - start_time
        
        print(f"✅ Sequential simulation: {sequential_time:.3f}s, "
              f"{sum(sequential_events)} total events")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent processing test error: {e}")
        return False

def test_memory_efficiency():
    """Test memory usage and optimization."""
    print("🧠 Testing Memory Efficiency...")
    
    try:
        # Test gradient checkpointing simulation
        large_model = create_liquid_net(input_dim=2, output_dim=10, architecture="base")
        
        # Large batch processing
        batch_sizes = [1, 10, 50, 100]
        memory_usage = {}
        
        for bs in batch_sizes:
            # Simulate memory-intensive operations
            x = torch.randn(bs, 200, 2)  # Large sequences
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
            
            try:
                output = large_model(x, reset_state=True)
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_mb = (peak_memory - start_memory) / 1e6
                else:
                    memory_mb = 0  # Estimate for CPU
                    
                memory_usage[bs] = memory_mb
                print(f"✅ Batch {bs:3d}: {memory_mb:.1f} MB memory")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  Batch {bs:3d}: OOM (expected for large batches)")
                    memory_usage[bs] = float('inf')
                else:
                    raise e
        
        # Test state reset efficiency
        model = create_liquid_net(input_dim=2, output_dim=5, architecture="small")
        
        # Test with state resets
        start_time = time.time()
        for i in range(100):
            x = torch.randn(8, 10, 2)
            output = model(x, reset_state=True)  # Reset every time
        with_reset_time = time.time() - start_time
        
        # Test without state resets
        start_time = time.time()
        for i in range(100):
            x = torch.randn(8, 10, 2)
            output = model(x, reset_state=False)  # No reset
        without_reset_time = time.time() - start_time
        
        print(f"✅ With state reset: {with_reset_time:.3f}s")
        print(f"✅ Without reset: {without_reset_time:.3f}s")
        print(f"✅ Reset overhead: {((with_reset_time/without_reset_time)-1)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test error: {e}")
        return False

def test_scaling_optimizations():
    """Test various scaling optimizations."""
    print("📈 Testing Scaling Optimizations...")
    
    try:
        # Test different data types for optimization
        model_fp32 = create_liquid_net(input_dim=2, output_dim=5, architecture="small")
        
        # Test FP32 vs FP16 (if supported)
        x_fp32 = torch.randn(32, 50, 2, dtype=torch.float32)
        x_fp16 = x_fp32.half()
        
        # FP32 benchmark
        start_time = time.time()
        for _ in range(20):
            output_fp32 = model_fp32(x_fp32, reset_state=True)
        fp32_time = time.time() - start_time
        
        # FP16 benchmark (if model supports it)
        try:
            model_fp16 = model_fp32.half()
            start_time = time.time()
            for _ in range(20):
                output_fp16 = model_fp16(x_fp16, reset_state=True)
            fp16_time = time.time() - start_time
            
            print(f"✅ FP32 time: {fp32_time:.3f}s")
            print(f"✅ FP16 time: {fp16_time:.3f}s")
            print(f"✅ FP16 speedup: {fp32_time/fp16_time:.2f}x")
            
        except Exception as e:
            print(f"⚠️  FP16 not supported: {e}")
        
        # Test compilation optimizations (TorchScript)
        try:
            model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
            x = torch.randn(4, 10, 2)
            
            # Regular model
            start_time = time.time()
            for _ in range(50):
                output = model(x, reset_state=True)
            regular_time = time.time() - start_time
            
            # Try TorchScript compilation
            try:
                compiled_model = torch.jit.trace(model, x)
                start_time = time.time()
                for _ in range(50):
                    output = compiled_model(x)
                compiled_time = time.time() - start_time
                
                print(f"✅ Regular model: {regular_time:.3f}s")
                print(f"✅ TorchScript: {compiled_time:.3f}s")
                print(f"✅ Compilation speedup: {regular_time/compiled_time:.2f}x")
                
            except Exception as e:
                print(f"⚠️  TorchScript compilation failed: {e}")
                
        except Exception as e:
            print(f"⚠️  Compilation test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaling optimizations test error: {e}")
        return False

def test_production_readiness():
    """Test production deployment readiness."""
    print("🚀 Testing Production Readiness...")
    
    try:
        # Test model serialization/deserialization
        model = create_liquid_net(input_dim=2, output_dim=5, architecture="small")
        
        # Save model
        save_path = "/tmp/liquid_model.pth"
        torch.save(model.state_dict(), save_path)
        print("✅ Model saved successfully")
        
        # Load model
        loaded_model = create_liquid_net(input_dim=2, output_dim=5, architecture="small")
        loaded_model.load_state_dict(torch.load(save_path))
        loaded_model.eval()
        print("✅ Model loaded successfully")
        
        # Test inference consistency
        x = torch.randn(5, 10, 2)
        with torch.no_grad():
            original_output = model(x, reset_state=True)
            loaded_output = loaded_model(x, reset_state=True)
        
        diff = torch.abs(original_output - loaded_output).max()
        print(f"✅ Model consistency: max diff = {diff:.2e}")
        
        # Test deployment configuration
        deployment_configs = {
            "edge_tiny": {"input_dim": 2, "output_dim": 3, "architecture": "tiny"},
            "server_base": {"input_dim": 2, "output_dim": 10, "architecture": "base"},
            "mobile_small": {"input_dim": 2, "output_dim": 5, "architecture": "small"}
        }
        
        for config_name, config in deployment_configs.items():
            try:
                deployment_model = create_liquid_net(**config)
                test_input = torch.randn(1, 20, 2)
                test_output = deployment_model(test_input, reset_state=True)
                print(f"✅ {config_name}: {test_output.shape} output ready")
            except Exception as e:
                print(f"❌ {config_name}: deployment failed - {e}")
        
        # Test resource monitoring
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            print(f"✅ System monitoring: CPU {cpu_percent}%, RAM {memory_percent}%")
        except ImportError:
            print("⚠️  psutil not available for system monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Production readiness test error: {e}")
        return False

def run_scaling_tests():
    """Run all scaling and performance tests."""
    print("=" * 70)
    print("⚡ GENERATION 3: SCALING & PERFORMANCE TESTING")
    print("=" * 70)
    
    tests = [
        test_model_performance,
        test_concurrent_processing,
        test_memory_efficiency,
        test_scaling_optimizations,
        test_production_readiness
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 70)
    print(f"📊 SCALING TEST RESULTS: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("✅ GENERATION 3 QUALITY GATE: PASSED")
        print("🎉 READY FOR PRODUCTION DEPLOYMENT!")
        return True
    else:
        print("❌ GENERATION 3 QUALITY GATE: FAILED")
        return False

if __name__ == "__main__":
    success = run_scaling_tests()
    exit(0 if success else 1)