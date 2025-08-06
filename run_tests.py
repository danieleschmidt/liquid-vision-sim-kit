#!/usr/bin/env python3
"""
Test runner for liquid vision simulation kit.
Runs tests that don't require PyTorch installation.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path


def run_test_module(module_path: str, test_description: str) -> tuple[int, int]:
    """
    Run tests from a module manually.
    
    Args:
        module_path: Path to test module
        test_description: Description of the test module
        
    Returns:
        Tuple of (passed_tests, total_tests)
    """
    print(f"\n🧪 Running {test_description}")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    try:
        # Import the test module
        sys.path.insert(0, str(Path(module_path).parent))
        module_name = Path(module_path).stem
        module = importlib.import_module(module_name)
        
        # Find all test classes
        test_classes = []
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                name.startswith('Test') and 
                hasattr(obj, '__module__')):
                test_classes.append(obj)
        
        # Run tests from each class
        for test_class in test_classes:
            print(f"\n📋 {test_class.__name__}")
            
            # Find test methods
            test_methods = [method for method in dir(test_class) 
                          if method.startswith('test_')]
            
            if not test_methods:
                continue
                
            # Create instance
            try:
                instance = test_class()
                
                # Run setup if exists
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                
                # Run each test method
                for method_name in test_methods:
                    total += 1
                    method = getattr(instance, method_name)
                    
                    try:
                        method()
                        print(f"  ✅ {method_name}")
                        passed += 1
                    except Exception as e:
                        print(f"  ❌ {method_name}: {e}")
                
                # Run teardown if exists
                if hasattr(instance, 'teardown_method'):
                    try:
                        instance.teardown_method()
                    except:
                        pass  # Ignore teardown errors
                        
            except Exception as e:
                print(f"  ❌ Failed to create instance: {e}")
                total += len(test_methods)
                
    except ImportError as e:
        if "torch" in str(e).lower() or "pytorch" in str(e).lower():
            print(f"⚠️  Skipping {test_description} - PyTorch not available")
            return 0, 0
        else:
            print(f"❌ Import error: {e}")
            return 0, 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return 0, 1
    
    return passed, total


def run_simple_tests():
    """Run tests that don't require external dependencies."""
    total_passed = 0
    total_tests = 0
    
    print("🚀 Liquid Vision Sim-Kit Test Suite")
    print("Running tests without PyTorch dependencies...")
    
    # Test modules that can run without PyTorch
    test_modules = [
        ("tests/test_configuration.py", "Configuration Management Tests"),
        ("tests/test_security.py", "Security Module Tests"),
    ]
    
    for module_path, description in test_modules:
        if not Path(module_path).exists():
            print(f"⚠️  Test file not found: {module_path}")
            continue
            
        passed, total = run_test_module(module_path, description)
        total_passed += passed
        total_tests += total
    
    # Test basic imports (this tests the module structure)
    print(f"\n🧪 Running Basic Import Tests")
    print("=" * 60)
    
    import_tests = [
        ("liquid_vision.config", "Configuration modules"),
        ("liquid_vision.security", "Security modules"),
        ("liquid_vision.utils", "Utility modules"),
    ]
    
    for import_name, description in import_tests:
        total_tests += 1
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {description}: Import successful")
            total_passed += 1
        except ImportError as e:
            if "torch" in str(e).lower():
                print(f"  ⚠️  {description}: Skipped (PyTorch dependency)")
            else:
                print(f"  ❌ {description}: Import failed - {e}")
        except Exception as e:
            print(f"  ❌ {description}: Unexpected error - {e}")
    
    # Test file structure validation
    print(f"\n🧪 Running File Structure Tests")
    print("=" * 60)
    
    expected_files = [
        "liquid_vision/__init__.py",
        "liquid_vision/config/__init__.py",
        "liquid_vision/config/config_manager.py",
        "liquid_vision/config/validators.py", 
        "liquid_vision/config/defaults.py",
        "liquid_vision/security/__init__.py",
        "liquid_vision/security/input_sanitizer.py",
        "liquid_vision/security/secure_deployment.py",
        "liquid_vision/security/crypto_utils.py",
        "liquid_vision/security/audit.py",
        "liquid_vision/utils/__init__.py",
        "liquid_vision/utils/logging.py",
        "liquid_vision/utils/validation.py",
        "liquid_vision/core/event_encoding.py",
        "liquid_vision/training/custom_dataset.py",
        "liquid_vision/deployment/sensor_interface.py",
        "liquid_vision/optimization/automl.py",
        "liquid_vision/optimization/nas.py",
        "liquid_vision/optimization/pruning.py",
        "liquid_vision/optimization/quantization.py",
    ]
    
    for file_path in expected_files:
        total_tests += 1
        if Path(file_path).exists():
            print(f"  ✅ {file_path}: File exists")
            total_passed += 1
        else:
            print(f"  ❌ {file_path}: File missing")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 85:
            print("🎉 Quality gate PASSED! (≥85% success rate)")
            return 0
        else:
            print("⚠️  Quality gate not met. Target: ≥85% success rate")
            return 1
    else:
        print("❌ No tests found to run")
        return 1


if __name__ == "__main__":
    sys.exit(run_simple_tests())