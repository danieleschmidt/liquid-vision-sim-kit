"""
Hardware abstraction layer for liquid neural networks on edge devices.
Provides unified interface for ESP32, Cortex-M, Arduino, and other platforms.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import time
import threading
import queue
import logging
from dataclasses import dataclass
import json
from pathlib import Path

from ..core.liquid_neurons import LiquidNet
from ..optimization.memory_efficient import MemoryEfficientLiquidNet


class HardwarePlatform(Enum):
    """Supported hardware platforms."""
    ESP32 = "esp32"
    ESP32_S3 = "esp32-s3"
    ESP32_C3 = "esp32-c3"
    CORTEX_M4 = "cortex-m4"
    CORTEX_M7 = "cortex-m7"
    CORTEX_M33 = "cortex-m33"
    ARDUINO_UNO = "arduino-uno"
    ARDUINO_NANO = "arduino-nano"
    RASPBERRY_PI = "raspberry-pi"
    RASPBERRY_PI_PICO = "raspberry-pi-pico"
    GENERIC_ARM = "generic-arm"
    X86_64 = "x86-64"


@dataclass
class HardwareSpecs:
    """Hardware specifications for a platform."""
    cpu_mhz: int
    ram_kb: int
    flash_kb: int
    sram_kb: Optional[int] = None
    has_fpu: bool = False
    has_dsp: bool = False
    has_simd: bool = False
    has_cache: bool = False
    cache_kb: Optional[int] = None
    max_clock_mhz: Optional[int] = None
    power_consumption_mw: Optional[int] = None
    gpio_pins: int = 0
    adc_channels: int = 0
    uart_ports: int = 1
    spi_ports: int = 1
    i2c_ports: int = 1


@dataclass
class PerformanceMetrics:
    """Performance metrics for model execution."""
    inference_time_ms: float
    memory_usage_kb: float
    power_consumption_mw: float
    cpu_utilization_percent: float
    accuracy: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    energy_per_inference_mj: Optional[float] = None


class HardwareProfiler:
    """Profile hardware performance and capabilities."""
    
    def __init__(self, platform: HardwarePlatform):
        self.platform = platform
        self.specs = self._get_hardware_specs(platform)
        self.benchmarks = {}
        
    def _get_hardware_specs(self, platform: HardwarePlatform) -> HardwareSpecs:
        """Get hardware specifications for platform."""
        specs_database = {
            HardwarePlatform.ESP32: HardwareSpecs(
                cpu_mhz=240, ram_kb=520, flash_kb=4096, has_fpu=True,
                gpio_pins=34, adc_channels=18, power_consumption_mw=500
            ),
            HardwarePlatform.ESP32_S3: HardwareSpecs(
                cpu_mhz=240, ram_kb=512, flash_kb=8192, sram_kb=384, has_fpu=True,
                gpio_pins=45, adc_channels=20, power_consumption_mw=400
            ),
            HardwarePlatform.ESP32_C3: HardwareSpecs(
                cpu_mhz=160, ram_kb=400, flash_kb=4096, has_fpu=False,
                gpio_pins=22, adc_channels=6, power_consumption_mw=300
            ),
            HardwarePlatform.CORTEX_M4: HardwareSpecs(
                cpu_mhz=180, ram_kb=256, flash_kb=1024, has_fpu=True, has_dsp=True,
                gpio_pins=100, power_consumption_mw=200
            ),
            HardwarePlatform.CORTEX_M7: HardwareSpecs(
                cpu_mhz=400, ram_kb=512, flash_kb=2048, has_fpu=True, has_dsp=True,
                has_cache=True, cache_kb=32, power_consumption_mw=400
            ),
            HardwarePlatform.ARDUINO_NANO: HardwareSpecs(
                cpu_mhz=16, ram_kb=32, flash_kb=32, has_fpu=False,
                gpio_pins=14, adc_channels=8, power_consumption_mw=20
            ),
            HardwarePlatform.RASPBERRY_PI_PICO: HardwareSpecs(
                cpu_mhz=133, ram_kb=264, flash_kb=2048, has_fpu=False,
                gpio_pins=26, adc_channels=3, power_consumption_mw=100
            )
        }
        
        return specs_database.get(platform, HardwareSpecs(
            cpu_mhz=100, ram_kb=128, flash_kb=512, has_fpu=False
        ))
    
    def benchmark_arithmetic_operations(self) -> Dict[str, float]:
        """Benchmark basic arithmetic operations."""
        # This would be implemented with platform-specific code
        # For now, return estimates based on hardware specs
        
        base_ops_per_sec = self.specs.cpu_mhz * 1000
        
        if self.specs.has_fpu:
            float_ops_per_sec = base_ops_per_sec * 0.8
        else:
            float_ops_per_sec = base_ops_per_sec * 0.1
        
        if self.specs.has_dsp:
            mac_ops_per_sec = base_ops_per_sec * 1.5
        else:
            mac_ops_per_sec = base_ops_per_sec * 0.5
        
        return {
            'int_ops_per_sec': base_ops_per_sec,
            'float_ops_per_sec': float_ops_per_sec,
            'mac_ops_per_sec': mac_ops_per_sec,
            'memory_bandwidth_mb_per_sec': self.specs.ram_kb * 0.1
        }
    
    def profile_model_inference(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> PerformanceMetrics:
        """Profile model inference performance."""
        # Estimate based on model complexity and hardware specs
        num_params = sum(p.numel() for p in model.parameters())
        num_operations = self._estimate_operations(model, input_tensor)
        
        # Performance estimates
        benchmark = self.benchmark_arithmetic_operations()
        inference_time_ms = (num_operations / benchmark['float_ops_per_sec']) * 1000
        
        # Memory usage estimate
        param_memory_kb = (num_params * 4) / 1024  # Assume float32
        activation_memory_kb = self._estimate_activation_memory(model, input_tensor) / 1024
        total_memory_kb = param_memory_kb + activation_memory_kb
        
        # Power consumption estimate
        cpu_utilization = min(100, (num_operations / benchmark['float_ops_per_sec']) * 100)
        power_mw = self.specs.power_consumption_mw * (cpu_utilization / 100)
        
        return PerformanceMetrics(
            inference_time_ms=inference_time_ms,
            memory_usage_kb=total_memory_kb,
            power_consumption_mw=power_mw,
            cpu_utilization_percent=cpu_utilization,
            throughput_ops_per_sec=1000 / inference_time_ms if inference_time_ms > 0 else 0,
            energy_per_inference_mj=power_mw * inference_time_ms / 1000
        )
    
    def _estimate_operations(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
        """Estimate number of operations for model inference."""
        operations = 0
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                operations += module.in_features * module.out_features * 2  # MAC operations
            elif isinstance(module, torch.nn.Conv2d):
                # Simplified convolution operation count
                output_size = input_tensor.numel()  # Approximation
                operations += module.in_channels * module.out_channels * np.prod(module.kernel_size) * output_size
        
        return operations
    
    def _estimate_activation_memory(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
        """Estimate activation memory requirements."""
        # Simplified estimation - would need more sophisticated analysis for real use
        return input_tensor.numel() * 4 * len(list(model.modules()))  # Rough estimate


class HardwareAdapter(ABC):
    """Abstract base class for hardware adapters."""
    
    def __init__(self, platform: HardwarePlatform):
        self.platform = platform
        self.profiler = HardwareProfiler(platform)
        self.is_connected = False
        
    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to the hardware platform."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the hardware platform."""
        pass
    
    @abstractmethod
    def upload_model(self, model_path: str) -> bool:
        """Upload model to the hardware platform."""
        pass
    
    @abstractmethod
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on the hardware platform."""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        pass
    
    @abstractmethod
    def monitor_performance(self, duration_seconds: float) -> PerformanceMetrics:
        """Monitor performance for specified duration."""
        pass


class ESP32Adapter(HardwareAdapter):
    """Hardware adapter for ESP32 microcontrollers."""
    
    def __init__(self):
        super().__init__(HardwarePlatform.ESP32)
        self.serial_port = None
        self.baudrate = 115200
        
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to ESP32 via serial port."""
        try:
            import serial
            
            port = connection_params.get('port', '/dev/ttyUSB0')
            baudrate = connection_params.get('baudrate', self.baudrate)
            
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for connection
            
            # Send test command
            self.serial_port.write(b'AT\r\n')
            response = self.serial_port.readline().decode().strip()
            
            self.is_connected = 'OK' in response
            return self.is_connected
            
        except Exception as e:
            logging.error(f"Failed to connect to ESP32: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from ESP32."""
        try:
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            self.is_connected = False
            return True
        except Exception as e:
            logging.error(f"Failed to disconnect from ESP32: {e}")
            return False
    
    def upload_model(self, model_path: str) -> bool:
        """Upload model to ESP32 flash memory."""
        if not self.is_connected:
            logging.error("Not connected to ESP32")
            return False
        
        try:
            # Read model file
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            # Send upload command
            cmd = f"UPLOAD_MODEL {len(model_data)}\r\n"
            self.serial_port.write(cmd.encode())
            
            # Send model data in chunks
            chunk_size = 1024
            for i in range(0, len(model_data), chunk_size):
                chunk = model_data[i:i+chunk_size]
                self.serial_port.write(chunk)
                
                # Wait for acknowledgment
                response = self.serial_port.readline().decode().strip()
                if 'ACK' not in response:
                    logging.error(f"Upload failed at chunk {i//chunk_size}")
                    return False
            
            # Verify upload
            self.serial_port.write(b'VERIFY_MODEL\r\n')
            response = self.serial_port.readline().decode().strip()
            
            return 'MODEL_OK' in response
            
        except Exception as e:
            logging.error(f"Failed to upload model to ESP32: {e}")
            return False
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on ESP32."""
        if not self.is_connected:
            raise RuntimeError("Not connected to ESP32")
        
        try:
            # Serialize input data
            input_bytes = input_data.astype(np.float32).tobytes()
            
            # Send inference command
            cmd = f"INFERENCE {len(input_bytes)}\r\n"
            self.serial_port.write(cmd.encode())
            
            # Send input data
            self.serial_port.write(input_bytes)
            
            # Read output size
            response = self.serial_port.readline().decode().strip()
            output_size = int(response.split()[1])
            
            # Read output data
            output_bytes = self.serial_port.read(output_size)
            output_data = np.frombuffer(output_bytes, dtype=np.float32)
            
            return output_data
            
        except Exception as e:
            logging.error(f"Failed to run inference on ESP32: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get ESP32 system status."""
        if not self.is_connected:
            return {'error': 'Not connected'}
        
        try:
            self.serial_port.write(b'STATUS\r\n')
            response = self.serial_port.readline().decode().strip()
            
            # Parse status response
            status = {}
            for part in response.split(','):
                if '=' in part:
                    key, value = part.split('=', 1)
                    status[key] = value
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def monitor_performance(self, duration_seconds: float) -> PerformanceMetrics:
        """Monitor ESP32 performance."""
        if not self.is_connected:
            raise RuntimeError("Not connected to ESP32")
        
        start_time = time.time()
        measurements = []
        
        while (time.time() - start_time) < duration_seconds:
            status = self.get_system_status()
            
            if 'error' not in status:
                measurement = {
                    'timestamp': time.time(),
                    'cpu_usage': float(status.get('cpu_usage', 0)),
                    'memory_free': float(status.get('memory_free', 0)),
                    'temperature': float(status.get('temperature', 0))
                }
                measurements.append(measurement)
            
            time.sleep(0.1)  # Sample every 100ms
        
        # Calculate average metrics
        if measurements:
            avg_cpu = np.mean([m['cpu_usage'] for m in measurements])
            avg_memory_free = np.mean([m['memory_free'] for m in measurements])
            memory_used = self.profiler.specs.ram_kb - avg_memory_free
            
            return PerformanceMetrics(
                inference_time_ms=0,  # Would need separate timing
                memory_usage_kb=memory_used,
                power_consumption_mw=self.profiler.specs.power_consumption_mw * (avg_cpu / 100),
                cpu_utilization_percent=avg_cpu
            )
        
        return PerformanceMetrics(0, 0, 0, 0)


class CortexMAdapter(HardwareAdapter):
    """Hardware adapter for ARM Cortex-M microcontrollers."""
    
    def __init__(self, variant: str = "m4"):
        if variant == "m4":
            platform = HardwarePlatform.CORTEX_M4
        elif variant == "m7":
            platform = HardwarePlatform.CORTEX_M7
        elif variant == "m33":
            platform = HardwarePlatform.CORTEX_M33
        else:
            platform = HardwarePlatform.CORTEX_M4
            
        super().__init__(platform)
        self.variant = variant
        self.debugger = None
    
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to Cortex-M via debugger interface (OpenOCD/J-Link)."""
        try:
            # This would use actual debugger interface
            # For now, simulate connection
            debugger_type = connection_params.get('debugger', 'openocd')
            target = connection_params.get('target', 'stm32f4xx')
            
            # Simulate debugger connection
            logging.info(f"Connecting to {self.variant.upper()} via {debugger_type}")
            self.is_connected = True
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to Cortex-M: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Cortex-M."""
        self.is_connected = False
        return True
    
    def upload_model(self, model_path: str) -> bool:
        """Upload model to Cortex-M flash memory."""
        if not self.is_connected:
            return False
        
        try:
            # This would use debugger to program flash
            logging.info(f"Uploading model to {self.variant.upper()} flash memory")
            return True
            
        except Exception as e:
            logging.error(f"Failed to upload model: {e}")
            return False
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on Cortex-M."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Cortex-M")
        
        # This would interface with the running firmware
        # For now, simulate inference
        time.sleep(0.01)  # Simulate processing time
        return np.random.randn(10).astype(np.float32)  # Dummy output
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get Cortex-M system status."""
        if not self.is_connected:
            return {'error': 'Not connected'}
        
        # Simulate status reading
        return {
            'cpu_frequency_mhz': self.profiler.specs.cpu_mhz,
            'memory_free_kb': self.profiler.specs.ram_kb * 0.7,
            'flash_used_kb': self.profiler.specs.flash_kb * 0.3,
            'temperature_c': 45.0
        }
    
    def monitor_performance(self, duration_seconds: float) -> PerformanceMetrics:
        """Monitor Cortex-M performance."""
        # Simulate performance monitoring
        return PerformanceMetrics(
            inference_time_ms=5.0,
            memory_usage_kb=self.profiler.specs.ram_kb * 0.3,
            power_consumption_mw=self.profiler.specs.power_consumption_mw,
            cpu_utilization_percent=60.0
        )


class HardwareManager:
    """Central manager for hardware interfaces."""
    
    def __init__(self):
        self.adapters: Dict[str, HardwareAdapter] = {}
        self.active_adapter: Optional[HardwareAdapter] = None
        
    def register_adapter(self, name: str, adapter: HardwareAdapter):
        """Register a hardware adapter."""
        self.adapters[name] = adapter
        
    def connect_to_device(self, adapter_name: str, connection_params: Dict[str, Any]) -> bool:
        """Connect to a specific device."""
        if adapter_name not in self.adapters:
            logging.error(f"Unknown adapter: {adapter_name}")
            return False
        
        adapter = self.adapters[adapter_name]
        if adapter.connect(connection_params):
            self.active_adapter = adapter
            logging.info(f"Connected to {adapter_name}")
            return True
        
        return False
    
    def disconnect_active_device(self) -> bool:
        """Disconnect from active device."""
        if self.active_adapter:
            result = self.active_adapter.disconnect()
            self.active_adapter = None
            return result
        return True
    
    def deploy_model(self, model: torch.nn.Module, model_path: Optional[str] = None) -> bool:
        """Deploy model to active device."""
        if not self.active_adapter:
            logging.error("No active device connection")
            return False
        
        # Export model if needed
        if model_path is None:
            model_path = self._export_model_for_device(model, self.active_adapter.platform)
        
        return self.active_adapter.upload_model(model_path)
    
    def run_inference_batch(self, input_batch: np.ndarray) -> List[np.ndarray]:
        """Run inference on batch of inputs."""
        if not self.active_adapter:
            raise RuntimeError("No active device connection")
        
        results = []
        for input_data in input_batch:
            output = self.active_adapter.run_inference(input_data)
            results.append(output)
        
        return results
    
    def benchmark_device(self, test_model: torch.nn.Module, test_input: torch.Tensor, duration_seconds: float = 10.0) -> Dict[str, Any]:
        """Benchmark active device performance."""
        if not self.active_adapter:
            raise RuntimeError("No active device connection")
        
        # Profile the model
        metrics = self.active_adapter.profiler.profile_model_inference(test_model, test_input)
        
        # Monitor actual performance
        actual_metrics = self.active_adapter.monitor_performance(duration_seconds)
        
        return {
            'platform': self.active_adapter.platform.value,
            'estimated_metrics': metrics,
            'actual_metrics': actual_metrics,
            'system_status': self.active_adapter.get_system_status()
        }
    
    def _export_model_for_device(self, model: torch.nn.Module, platform: HardwarePlatform) -> str:
        """Export model in device-appropriate format."""
        # This would implement platform-specific model export
        # For now, use a generic export
        
        export_path = f"/tmp/model_{platform.value}.bin"
        
        # Convert to appropriate format based on platform
        if platform in [HardwarePlatform.ESP32, HardwarePlatform.ESP32_S3]:
            # Export for ESP32 (C code generation)
            self._export_for_esp32(model, export_path)
        elif "cortex" in platform.value:
            # Export for Cortex-M (CMSIS-NN format)
            self._export_for_cortex_m(model, export_path)
        else:
            # Generic export
            torch.jit.script(model).save(export_path)
        
        return export_path
    
    def _export_for_esp32(self, model: torch.nn.Module, path: str):
        """Export model for ESP32 deployment."""
        # This would generate C code for ESP32
        with open(path, 'w') as f:
            f.write("// ESP32 model implementation\n")
            f.write("// Generated model code would go here\n")
    
    def _export_for_cortex_m(self, model: torch.nn.Module, path: str):
        """Export model for Cortex-M deployment."""
        # This would generate CMSIS-NN compatible code
        with open(path, 'w') as f:
            f.write("// Cortex-M model implementation\n")
            f.write("// Generated CMSIS-NN code would go here\n")
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available hardware adapters."""
        return list(self.adapters.keys())
    
    def get_active_device_info(self) -> Dict[str, Any]:
        """Get information about active device."""
        if not self.active_adapter:
            return {'error': 'No active device'}
        
        return {
            'platform': self.active_adapter.platform.value,
            'specs': self.active_adapter.profiler.specs.__dict__,
            'status': self.active_adapter.get_system_status(),
            'connected': self.active_adapter.is_connected
        }


def create_hardware_manager() -> HardwareManager:
    """Create hardware manager with common adapters."""
    manager = HardwareManager()
    
    # Register common adapters
    manager.register_adapter('esp32', ESP32Adapter())
    manager.register_adapter('cortex-m4', CortexMAdapter('m4'))
    manager.register_adapter('cortex-m7', CortexMAdapter('m7'))
    
    return manager


def auto_detect_platform() -> Optional[HardwarePlatform]:
    """Auto-detect connected hardware platform."""
    # This would implement actual platform detection
    # For now, return None to indicate no auto-detection
    return None


def optimize_model_for_platform(
    model: torch.nn.Module,
    platform: HardwarePlatform,
    memory_budget_kb: Optional[float] = None
) -> torch.nn.Module:
    """Optimize model for specific hardware platform."""
    
    profiler = HardwareProfiler(platform)
    
    if memory_budget_kb is None:
        memory_budget_kb = profiler.specs.ram_kb * 0.7  # Use 70% of available RAM
    
    # Platform-specific optimizations
    if platform == HardwarePlatform.ARDUINO_NANO or memory_budget_kb < 20:
        # Extremely constrained - use minimal model
        from ..optimization.memory_efficient import create_edge_optimized_model
        return create_edge_optimized_model(
            input_dim=model.input_dim if hasattr(model, 'input_dim') else 10,
            output_dim=model.output_dim if hasattr(model, 'output_dim') else 3,
            target_platform="arduino_nano",
            memory_budget_kb=memory_budget_kb
        )
    
    elif "esp32" in platform.value:
        # ESP32 optimizations
        if isinstance(model, LiquidNet):
            return MemoryEfficientLiquidNet(
                input_dim=model.input_dim,
                hidden_units=model.hidden_units,
                output_dim=model.output_dim,
                memory_efficient=True,
                gradient_checkpointing=False,  # Not needed for inference
                activation_offloading=memory_budget_kb < 100
            )
    
    elif "cortex" in platform.value:
        # Cortex-M optimizations - leverage DSP capabilities
        if isinstance(model, LiquidNet) and profiler.specs.has_dsp:
            # Enable DSP-optimized operations
            return MemoryEfficientLiquidNet(
                input_dim=model.input_dim,
                hidden_units=model.hidden_units,
                output_dim=model.output_dim,
                memory_efficient=True,
                gradient_checkpointing=False
            )
    
    # Return original model if no specific optimizations needed
    return model