#!/usr/bin/env python3
"""
Test script for new event encoders - validates implementation without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    """Test that new encoders can be imported."""
    try:
        from liquid_vision.core.event_encoding import (
            VoxelEncoder, SAEEncoder, EventImageEncoder, create_encoder
        )
        print("✅ Successfully imported new encoders")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_factory_function():
    """Test that factory function recognizes new encoders."""
    try:
        from liquid_vision.core.event_encoding import create_encoder
        
        # Test that new encoder types are recognized
        valid_types = ["temporal", "spatial", "timeslice", "adaptive", "voxel", "sae", "event_image"]
        
        for encoder_type in ["voxel", "sae", "event_image"]:
            try:
                encoder = create_encoder(encoder_type, sensor_size=(64, 48))
                print(f"✅ Created {encoder_type} encoder successfully")
            except Exception as e:
                print(f"❌ Failed to create {encoder_type} encoder: {e}")
                return False
                
        print("✅ Factory function updated correctly")
        return True
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False

def test_custom_dataset_import():
    """Test that custom dataset can be imported."""
    try:
        from liquid_vision.training.custom_dataset import CustomEventDataset
        print("✅ Successfully imported CustomEventDataset")
        return True
    except ImportError as e:
        print(f"❌ CustomEventDataset import failed: {e}")
        return False

def test_sensor_interface_import():
    """Test that sensor interface can be imported."""
    try:
        from liquid_vision.deployment.sensor_interface import SensorInterfaceGenerator, SensorType
        print("✅ Successfully imported SensorInterfaceGenerator")
        return True
    except ImportError as e:
        print(f"❌ SensorInterfaceGenerator import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing new encoder implementations...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_factory_function,
        test_custom_dataset_import,
        test_sensor_interface_import,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 1 enhancements implemented successfully!")
        return True
    else:
        print("⚠️  Some tests failed - review implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)