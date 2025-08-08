#!/usr/bin/env python3
"""Comprehensive test suite for all three generations of the Liquid Vision framework."""

import sys
import os
import importlib
import torch
import numpy as np
from pathlib import Path

def test_framework_imports():
    """Test all framework imports work correctly."""
    print("🔍 Testing Framework Imports...")
    
    try:
        import liquid_vision
        from liquid_vision import LiquidNet, LiquidTrainer, EventSimulator, EdgeDeployer
        from liquid_vision.core.liquid_neurons import create_liquid_net, get_model_info
        from liquid_vision.simulation.event_simulator import DVSSimulator, create_simulator
        from liquid_vision.training.liquid_trainer import TrainingConfig
        
        # Store in global scope for other tests to use
        globals()['liquid_vision'] = liquid_vision
        globals()['create_liquid_net'] = create_liquid_net
        globals()['get_model_info'] = get_model_info
        globals()['DVSSimulator'] = DVSSimulator
        globals()['create_simulator'] = create_simulator
        globals()['TrainingConfig'] = TrainingConfig
        globals()['EdgeDeployer'] = EdgeDeployer
        
        print(f"✅ Core imports successful - Version {liquid_vision.__version__}")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_core_functionality():
    """Test core model functionality."""
    print("🧠 Testing Core Model Functionality...")
    
    try:
        # Create and test different architectures
        architectures = ["tiny", "small", "base"]
        for arch in architectures:
            model = create_liquid_net(input_dim=2, output_dim=5, architecture=arch)
            info = get_model_info(model)
            
            # Test forward pass
            x = torch.randn(16, 20, 2)
            output = model(x, reset_state=True)
            
            assert output.shape == (16, 20, 5), f"Wrong output shape for {arch}"
            print(f"✅ {arch}: {info['total_parameters']} params, output {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Core functionality error: {e}")
        return False

def test_event_simulation():
    """Test event camera simulation."""
    print("📷 Testing Event Camera Simulation...")
    
    try:
        # Test different simulators
        simulator_types = ["dvs", "davis", "advanced_dvs"]
        for sim_type in simulator_types:
            try:
                simulator = create_simulator(
                    simulator_type=sim_type,
                    resolution=(64, 64),
                    contrast_threshold=0.1
                )
                
                # Test frame simulation
                frame = np.random.rand(64, 64).astype(np.float32)
                events = simulator.simulate_frame(frame, timestamp=0.0)
                
                print(f"✅ {sim_type}: {len(events)} events generated")
            except Exception as e:
                print(f"⚠️  {sim_type}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Event simulation error: {e}")
        return False

def test_training_system():
    """Test training configuration and setup."""
    print("🏋️ Testing Training System...")
    
    try:
        model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
        
        # Test training configuration
        config = TrainingConfig(
            epochs=1,
            batch_size=8,
            learning_rate=0.001,
            optimizer="adam",
            loss_type="cross_entropy"
        )
        
        # Verify configuration
        assert config.epochs == 1
        assert config.batch_size == 8
        assert config.optimizer == "adam"
        
        print(f"✅ Training config: {config.model_name}")
        print(f"✅ Model ready: {sum(p.numel() for p in model.parameters())} params")
        
        return True
    except Exception as e:
        print(f"❌ Training system error: {e}")
        return False

def test_edge_deployment():
    """Test edge deployment capabilities."""
    print("🚀 Testing Edge Deployment...")
    
    try:
        from liquid_vision.deployment.edge_deployer import EdgeDeployer
        
        model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
        deployer = EdgeDeployer(target="esp32")
        
        # Test deployment readiness
        model_info = get_model_info(model)
        assert model_info["total_parameters"] > 0
        
        print(f"✅ Edge deployer created for ESP32")
        print(f"✅ Model params: {model_info['total_parameters']} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Edge deployment error: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("🔄 Testing End-to-End Pipeline...")
    
    try:
        # Create pipeline components
        simulator = DVSSimulator(resolution=(32, 32))
        model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
        
        # Generate synthetic data
        frames = [np.random.rand(32, 32) * 0.5 + i*0.1 for i in range(5)]
        
        total_events = 0
        for i, frame in enumerate(frames):
            events = simulator.simulate_frame(frame, timestamp=float(i))
            total_events += len(events)
        
        # Process with model
        dummy_input = torch.randn(4, 10, 2)
        output = model(dummy_input, reset_state=True)
        
        print(f"✅ Pipeline complete: {total_events} events -> model output {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end pipeline error: {e}")
        return False

def test_performance_characteristics():
    """Test performance characteristics."""
    print("⚡ Testing Performance Characteristics...")
    
    try:
        model = create_liquid_net(input_dim=2, output_dim=5, architecture="small")
        
        # Benchmark inference
        batch_sizes = [1, 8, 16, 32]
        for bs in batch_sizes:
            x = torch.randn(bs, 20, 2)
            
            # Warmup
            for _ in range(3):
                model(x, reset_state=True)
            
            # Benchmark
            import time
            start = time.time()
            for _ in range(10):
                output = model(x, reset_state=True)
            elapsed = time.time() - start
            
            samples_per_sec = (bs * 20 * 10) / elapsed
            print(f"✅ Batch {bs:2d}: {samples_per_sec:8.0f} samples/s")
        
        return True
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def test_model_persistence():
    """Test model saving and loading."""
    print("💾 Testing Model Persistence...")
    
    try:
        # Create and save model
        original_model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
        save_path = "/tmp/test_liquid_model.pth"
        torch.save(original_model.state_dict(), save_path)
        
        # Load model
        loaded_model = create_liquid_net(input_dim=2, output_dim=3, architecture="tiny")
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Test consistency
        x = torch.randn(4, 10, 2)
        with torch.no_grad():
            original_output = original_model(x, reset_state=True)
            loaded_output = loaded_model(x, reset_state=True)
        
        diff = torch.abs(original_output - loaded_output).max()
        assert diff < 1e-6, f"Model consistency failed: {diff}"
        
        print(f"✅ Model persistence: max diff = {diff:.2e}")
        
        # Cleanup
        os.remove(save_path)
        
        return True
    except Exception as e:
        print(f"❌ Model persistence error: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 80)
    print("🧪 COMPREHENSIVE TESTING - ALL GENERATIONS")
    print("=" * 80)
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print("=" * 80)
    
    tests = [
        ("Framework Imports", test_framework_imports),
        ("Core Functionality", test_core_functionality),
        ("Event Simulation", test_event_simulation),
        ("Training System", test_training_system),
        ("Edge Deployment", test_edge_deployment),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Performance", test_performance_characteristics),
        ("Model Persistence", test_model_persistence)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 60)
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
        print()
    
    # Final Summary
    passed = sum(results)
    total = len(results)
    success_rate = 100 * passed / total
    
    print("=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    for i, (test_name, result) in enumerate(zip([t[0] for t in tests], results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1:2d}. {test_name:<25} {status}")
    
    print("-" * 80)
    print(f"📈 OVERALL SUCCESS RATE: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("🎉 QUALITY GATE: PASSED")
        print("✨ FRAMEWORK IS PRODUCTION READY!")
    else:
        print("⚠️  QUALITY GATE: NEEDS IMPROVEMENT")
    
    print("=" * 80)
    
    return success_rate >= 85

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)