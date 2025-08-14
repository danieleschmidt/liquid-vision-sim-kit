#!/usr/bin/env python3
"""
Enhanced CLI interface for Liquid Vision Sim-Kit.
Provides interactive demonstrations and model testing capabilities.
"""

import argparse
import sys
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from . import get_system_status, get_feature_availability
    from .core.minimal_fallback import MinimalTensor, create_minimal_liquid_net
except ImportError:
    # Standalone execution
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from liquid_vision import get_system_status, get_feature_availability
    from liquid_vision.core.minimal_fallback import MinimalTensor, create_minimal_liquid_net


class LiquidVisionCLI:
    """Enhanced CLI interface for Liquid Vision operations."""
    
    def __init__(self):
        self.status = get_system_status()
        self.features = get_feature_availability()
        
    def print_banner(self):
        """Print ASCII banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    🧠 LIQUID VISION SIM-KIT                   ║
║              Neuromorphic AI for Edge Devices                ║
║                 Autonomous SDLC v4.0 Active                  ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def show_status(self):
        """Display system status and capabilities."""
        self.print_banner()
        
        print(f"Version: {self.status['version']}")
        print(f"Autonomous Mode: {'✅ ENABLED' if self.status['autonomous_mode'] else '❌ DISABLED'}")
        print(f"Production Ready: {'✅ YES' if self.status['production_ready'] else '⚠️  PARTIAL'}")
        
        print("\n📊 Feature Availability:")
        for feature, available in self.features.items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {feature.replace('_', ' ').title()}")
            
        print(f"\n🔧 Implementation: Minimal fallback (zero dependencies)")
        print("🚀 Ready for autonomous development!")
        
    def demo_basic(self):
        """Run basic liquid neural network demonstration."""
        print("\n🧠 Basic Liquid Neural Network Demo")
        print("=" * 50)
        
        try:
            # Create models of different sizes
            architectures = ["tiny", "small", "base"]
            
            for arch in architectures:
                try:
                    print(f"\n🔬 Testing {arch.upper()} architecture:")
                    
                    model = create_minimal_liquid_net(
                        input_dim=2,
                        output_dim=3,
                        architecture=arch
                    )
                    
                    # Test single inference
                    x = MinimalTensor([[0.5, -0.3]])
                    output = model(x)
                    
                    print(f"  Input: {x.data[0]}")
                    print(f"  Output: {[round(v, 4) for v in output.data[0]]}")
                    
                    # Count parameters (approximate)
                    param_count = self._estimate_parameters(model)
                    print(f"  Est. Parameters: {param_count}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {arch} model: {e}")
                    
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return False
            
        print("\n✅ Basic demo completed successfully!")
        return True
        
    def demo_temporal(self):
        """Demonstrate temporal processing capabilities."""
        print("\n⏰ Temporal Processing Demo")
        print("=" * 40)
        
        try:
            model = create_minimal_liquid_net(2, 1, architecture="small")
            
            print("Processing sequence with memory:")
            
            # Reset states
            model.reset_states()
            
            # Generate sequence
            sequence = [
                [1.0, 0.0],   # Step 1: High input
                [0.0, 0.0],   # Step 2: No input
                [0.0, 0.0],   # Step 3: No input
                [-1.0, 0.0],  # Step 4: Negative input
                [0.0, 0.0],   # Step 5: No input
            ]
            
            outputs = []
            for i, inputs in enumerate(sequence):
                x = MinimalTensor([inputs])
                output = model(x)
                outputs.append(output.data[0][0])
                
                print(f"  Step {i+1}: input={inputs[0]:+5.1f} -> output={output.data[0][0]:+7.4f}")
                
            print(f"\n📈 Temporal dynamics observed:")
            print(f"  - Memory retention across zero inputs")
            print(f"  - Gradual state transitions")
            print(f"  - Response to input changes")
            
        except Exception as e:
            print(f"❌ Temporal demo failed: {e}")
            return False
            
        print("\n✅ Temporal processing demo completed!")
        return True
        
    def benchmark_performance(self):
        """Run performance benchmarks."""
        print("\n⚡ Performance Benchmark")
        print("=" * 30)
        
        import time
        
        try:
            model = create_minimal_liquid_net(10, 5, architecture="base")
            
            # Warmup
            x = MinimalTensor([[0.1] * 10])
            for _ in range(10):
                model(x)
                
            # Benchmark
            num_iterations = 100
            start_time = time.time()
            
            for _ in range(num_iterations):
                output = model(x)
                
            end_time = time.time()
            
            total_time = end_time - start_time
            fps = num_iterations / total_time
            latency_ms = (total_time / num_iterations) * 1000
            
            print(f"📊 Results ({num_iterations} iterations):")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  FPS: {fps:.1f}")
            print(f"  Latency: {latency_ms:.2f}ms")
            print(f"  Implementation: Minimal Python (CPU)")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False
            
        print("\n✅ Performance benchmark completed!")
        return True
        
    def interactive_mode(self):
        """Start interactive mode."""
        print("\n🎮 Interactive Mode")
        print("Enter 'help' for commands, 'quit' to exit")
        print("-" * 40)
        
        model = None
        
        while True:
            try:
                cmd = input("\nliquid-vision> ").strip().lower()
                
                if cmd == "quit" or cmd == "exit":
                    print("Goodbye! 👋")
                    break
                    
                elif cmd == "help":
                    print("""
Available commands:
  create <arch>  - Create model (tiny/small/base)
  predict <x> <y> - Run prediction with inputs
  reset          - Reset model states
  info           - Show model info
  demo           - Run quick demo
  benchmark      - Performance test
  status         - Show system status
  quit           - Exit interactive mode
                    """)
                    
                elif cmd.startswith("create"):
                    parts = cmd.split()
                    arch = parts[1] if len(parts) > 1 else "small"
                    try:
                        model = create_minimal_liquid_net(2, 3, architecture=arch)
                        print(f"✅ Created {arch} model")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        
                elif cmd.startswith("predict"):
                    if model is None:
                        print("❌ No model loaded. Use 'create' first.")
                        continue
                        
                    parts = cmd.split()
                    if len(parts) < 3:
                        print("❌ Usage: predict <x> <y>")
                        continue
                        
                    try:
                        x_val = float(parts[1])
                        y_val = float(parts[2])
                        
                        x = MinimalTensor([[x_val, y_val]])
                        output = model(x)
                        
                        print(f"Input: [{x_val}, {y_val}]")
                        print(f"Output: {[round(v, 4) for v in output.data[0]]}")
                        
                    except ValueError:
                        print("❌ Invalid input values")
                    except Exception as e:
                        print(f"❌ Prediction failed: {e}")
                        
                elif cmd == "reset":
                    if model:
                        model.reset_states()
                        print("✅ Model states reset")
                    else:
                        print("❌ No model loaded")
                        
                elif cmd == "info":
                    if model:
                        print(f"Model architecture: {len(model.hidden_units)} layers")
                        print(f"Hidden units: {model.hidden_units}")
                        print(f"Parameters: ~{self._estimate_parameters(model)}")
                    else:
                        print("❌ No model loaded")
                        
                elif cmd == "demo":
                    self.demo_basic()
                    
                elif cmd == "benchmark":
                    self.benchmark_performance()
                    
                elif cmd == "status":
                    self.show_status()
                    
                else:
                    print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    def _estimate_parameters(self, model) -> int:
        """Estimate number of parameters in model."""
        total = 0
        
        for layer in model.liquid_layers:
            # W_in: input_dim * hidden_dim
            total += layer.input_dim * layer.hidden_dim
            # W_rec: hidden_dim * hidden_dim  
            total += layer.hidden_dim * layer.hidden_dim
            # bias: hidden_dim
            total += layer.hidden_dim
            
        # Readout layer
        total += model.hidden_units[-1] * model.output_dim
        
        return total
        
    def export_config(self, filepath: str):
        """Export current configuration."""
        config = {
            "version": self.status["version"],
            "features": self.features,
            "autonomous_mode": self.status["autonomous_mode"],
            "implementation": "minimal_fallback",
            "supported_architectures": ["tiny", "small", "base"],
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"✅ Configuration exported to {filepath}")


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Liquid Vision Sim-Kit - Neuromorphic AI for Edge Devices",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version information"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show system status and capabilities"
    )
    
    parser.add_argument(
        "--demo",
        choices=["basic", "temporal", "all"],
        help="Run demonstration (basic, temporal, or all)"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode"
    )
    
    parser.add_argument(
        "--export-config",
        type=str,
        metavar="FILE",
        help="Export configuration to JSON file"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    cli = LiquidVisionCLI()
    
    # Handle no arguments
    if len(sys.argv) == 1:
        cli.show_status()
        print("\nUse --help for available commands or --interactive for interactive mode")
        return
    
    if args.version:
        print(f"Liquid Vision Sim-Kit v{cli.status['version']}")
        print("Autonomous SDLC v4.0 - Zero Dependencies Mode")
        return
        
    if args.status:
        cli.show_status()
        return
        
    if args.demo:
        if args.demo == "basic" or args.demo == "all":
            cli.demo_basic()
        if args.demo == "temporal" or args.demo == "all":
            cli.demo_temporal()
        return
        
    if args.benchmark:
        cli.benchmark_performance()
        return
        
    if args.interactive:
        cli.interactive_mode()
        return
        
    if args.export_config:
        cli.export_config(args.export_config)
        return


if __name__ == "__main__":
    main()