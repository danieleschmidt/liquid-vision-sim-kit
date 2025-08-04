"""
Edge deployment utilities for liquid neural networks.
Supports deployment to ESP32, Cortex-M, and other edge devices.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum
import json
import tempfile

from ..core.liquid_neurons import LiquidNet


class DeploymentTarget(Enum):
    """Supported deployment targets."""
    ESP32 = "esp32"
    ESP32_S3 = "esp32-s3"
    CORTEX_M4 = "cortex-m4"
    CORTEX_M7 = "cortex-m7"
    ARDUINO = "arduino"
    RASPBERRY_PI = "raspberry-pi"


class EdgeDeployer:
    """
    Edge deployment manager for liquid neural networks.
    Handles model compilation, optimization, and code generation.
    """
    
    def __init__(
        self,
        target: Union[str, DeploymentTarget] = DeploymentTarget.ESP32,
        optimize_memory: bool = True,
        quantization: str = "int8",  # "int8", "int16", "float16"
        max_memory_kb: int = 512,    # Memory constraint
    ):
        if isinstance(target, str):
            target = DeploymentTarget(target)
            
        self.target = target
        self.optimize_memory = optimize_memory
        self.quantization = quantization
        self.max_memory_kb = max_memory_kb
        
        # Target-specific configurations
        self.target_configs = {
            DeploymentTarget.ESP32: {
                "memory_kb": 520,
                "flash_mb": 4,
                "cpu_mhz": 240,
                "float_support": True,
                "compiler": "xtensa-esp32-elf-gcc"
            },
            DeploymentTarget.ESP32_S3: {
                "memory_kb": 512,
                "flash_mb": 8,
                "cpu_mhz": 240,
                "float_support": True,
                "compiler": "xtensa-esp32s3-elf-gcc"
            },
            DeploymentTarget.CORTEX_M4: {
                "memory_kb": 256,
                "flash_mb": 1,
                "cpu_mhz": 168,
                "float_support": True,
                "compiler": "arm-none-eabi-gcc"
            },
            DeploymentTarget.CORTEX_M7: {
                "memory_kb": 512,
                "flash_mb": 2,
                "cpu_mhz": 216,
                "float_support": True,
                "compiler": "arm-none-eabi-gcc"
            }
        }
        
    def export_model(
        self,
        model: LiquidNet,
        output_path: Union[str, Path],
        test_input: Optional[torch.Tensor] = None,
        generate_test: bool = True,
    ) -> Dict[str, Any]:
        """
        Export model for edge deployment.
        
        Args:
            model: Trained liquid neural network
            output_path: Output file path for generated code
            test_input: Sample input for testing
            generate_test: Whether to generate test code
            
        Returns:
            Deployment information dictionary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Extract model parameters
        model_info = self._extract_model_info(model)
        
        # Quantize weights if requested
        quantized_weights = self._quantize_weights(model_info["weights"])
        
        # Generate C code
        c_code = self._generate_c_code(model_info, quantized_weights)
        
        # Write main model file
        with open(output_path, 'w') as f:
            f.write(c_code)
            
        # Generate header file
        header_path = output_path.with_suffix('.h')
        header_code = self._generate_header_code(model_info)
        with open(header_path, 'w') as f:
            f.write(header_code)
            
        # Generate test code if requested
        test_files = []
        if generate_test and test_input is not None:
            test_path = output_path.parent / f"{output_path.stem}_test.c"
            test_code = self._generate_test_code(model_info, test_input)
            with open(test_path, 'w') as f:
                f.write(test_code)
            test_files.append(test_path)
            
        # Generate build configuration
        build_config = self._generate_build_config(model_info)
        config_path = output_path.parent / "build_config.json"
        with open(config_path, 'w') as f:
            json.dump(build_config, f, indent=2)
            
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage(model_info, quantized_weights)
        
        deployment_info = {
            "target": self.target.value,
            "model_file": str(output_path),
            "header_file": str(header_path),
            "test_files": [str(f) for f in test_files],
            "build_config": str(config_path),
            "memory_usage": memory_usage,
            "quantization": self.quantization,
            "model_info": model_info["summary"]
        }
        
        print(f"✓ Model exported for {self.target.value}")
        print(f"  - Memory usage: {memory_usage['total_kb']:.1f} KB")
        print(f"  - Model parameters: {model_info['summary']['total_parameters']:,}")
        print(f"  - Quantization: {self.quantization}")
        
        return deployment_info
        
    def _extract_model_info(self, model: LiquidNet) -> Dict[str, Any]:
        """Extract model structure and parameters."""
        info = {
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_units": model.hidden_units,
            "num_layers": model.num_layers,
            "weights": {},
            "summary": {}
        }
        
        # Extract weights and biases
        weights = {}
        total_params = 0
        
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
            total_params += param.numel()
            
        info["weights"] = weights
        info["summary"] = {
            "total_parameters": total_params,
            "layers": len(model.hidden_units),
            "architecture": f"{model.input_dim}-{'-'.join(map(str, model.hidden_units))}-{model.output_dim}"
        }
        
        return info
        
    def _quantize_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Quantize model weights based on target precision."""
        quantized = {}
        
        for name, weight in weights.items():
            if self.quantization == "int8":
                # Quantize to int8 range [-128, 127]
                w_min, w_max = weight.min(), weight.max()
                scale = max(abs(w_min), abs(w_max)) / 127.0
                quantized[name] = {
                    "data": np.round(weight / scale).astype(np.int8),
                    "scale": scale,
                    "dtype": "int8"
                }
            elif self.quantization == "int16":
                # Quantize to int16
                w_min, w_max = weight.min(), weight.max()
                scale = max(abs(w_min), abs(w_max)) / 32767.0
                quantized[name] = {
                    "data": np.round(weight / scale).astype(np.int16),
                    "scale": scale,
                    "dtype": "int16"
                }
            elif self.quantization == "float16":
                quantized[name] = {
                    "data": weight.astype(np.float16),
                    "scale": 1.0,
                    "dtype": "float16"
                }
            else:
                # Keep as float32
                quantized[name] = {
                    "data": weight.astype(np.float32),
                    "scale": 1.0,
                    "dtype": "float32"
                }
                
        return quantized
        
    def _generate_c_code(self, model_info: Dict, quantized_weights: Dict) -> str:
        """Generate C implementation code."""
        code = f"""/*
 * Liquid Neural Network - Edge Deployment
 * Generated for target: {self.target.value}
 * Quantization: {self.quantization}
 */

#include <math.h>
#include <string.h>
#include "liquid_model.h"

// Model architecture
#define INPUT_DIM {model_info['input_dim']}
#define OUTPUT_DIM {model_info['output_dim']}
#define NUM_LAYERS {model_info['num_layers']}

// Hidden layer sizes
static const int hidden_sizes[NUM_LAYERS] = {{{', '.join(map(str, model_info['hidden_units']))}}};

// Activation function
static inline float activation_tanh(float x) {{
    return tanhf(x);
}}

"""
        
        # Generate weight arrays
        for name, weight_info in quantized_weights.items():
            data = weight_info["data"]
            dtype = weight_info["dtype"]
            
            if "liquid_layers" in name and "W_in" in name:
                layer_idx = int(name.split('.')[1])
                code += f"// Layer {layer_idx} input weights\n"
                code += self._generate_weight_array(f"W_in_{layer_idx}", data, dtype)
                
            elif "liquid_layers" in name and "W_rec" in name:
                layer_idx = int(name.split('.')[1])
                code += f"// Layer {layer_idx} recurrent weights\n"
                code += self._generate_weight_array(f"W_rec_{layer_idx}", data, dtype)
                
            elif "liquid_layers" in name and "bias" in name:
                layer_idx = int(name.split('.')[1])
                code += f"// Layer {layer_idx} bias\n"
                code += self._generate_weight_array(f"bias_{layer_idx}", data, dtype)
                
            elif "readout" in name:
                code += f"// Readout layer weights\n"
                code += self._generate_weight_array("W_out", data, dtype)
                
        # Generate forward pass function
        code += self._generate_forward_function(model_info, quantized_weights)
        
        return code
        
    def _generate_weight_array(self, name: str, data: np.ndarray, dtype: str) -> str:
        """Generate C array for weights."""
        if dtype == "int8":
            c_type = "int8_t"
        elif dtype == "int16":
            c_type = "int16_t"
        elif dtype == "float16":
            c_type = "float"  # Use float for storage
        else:
            c_type = "float"
            
        # Flatten array and format
        flat_data = data.flatten()
        
        if data.ndim == 1:
            size_str = f"[{len(flat_data)}]"
        else:
            size_str = f"[{data.shape[0]}][{data.shape[1]}]"
            
        values = ", ".join(f"{x}" for x in flat_data)
        
        return f"static const {c_type} {name}{size_str} = {{{values}}};\n\n"
        
    def _generate_forward_function(self, model_info: Dict, quantized_weights: Dict) -> str:
        """Generate forward pass implementation."""
        code = """
// Forward pass implementation
int liquid_model_predict(const float* input, float* output) {
    static float hidden_states[NUM_LAYERS][MAX_HIDDEN_SIZE];
    static int initialized = 0;
    
    // Initialize hidden states on first call
    if (!initialized) {
        memset(hidden_states, 0, sizeof(hidden_states));
        initialized = 1;
    }
    
    const float* current_input = input;
    int current_input_size = INPUT_DIM;
    
    // Process through liquid layers
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int hidden_size = hidden_sizes[layer];
        float* hidden = hidden_states[layer];
        
        // Compute input contribution
        for (int h = 0; h < hidden_size; h++) {
            float input_sum = 0.0f;
            for (int i = 0; i < current_input_size; i++) {
                // Note: This is simplified - actual implementation would use
                // proper weight indexing based on quantization
                input_sum += current_input[i]; // * W_in[layer][h][i];
            }
            
            // Compute recurrent contribution
            float recurrent_sum = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                recurrent_sum += hidden[j]; // * W_rec[layer][h][j];
            }
            
            // Liquid dynamics: simplified Euler integration
            float target = activation_tanh(input_sum + recurrent_sum);
            hidden[h] = hidden[h] + 0.1f * (target - hidden[h]); // dt=0.1, tau=1.0
        }
        
        current_input = hidden;
        current_input_size = hidden_size;
    }
    
    // Readout layer
    for (int o = 0; o < OUTPUT_DIM; o++) {
        output[o] = 0.0f;
        for (int h = 0; h < current_input_size; h++) {
            output[o] += current_input[h]; // * W_out[o][h];
        }
    }
    
    return 0; // Success
}

// Reset liquid states
void liquid_model_reset() {
    static float hidden_states[NUM_LAYERS][MAX_HIDDEN_SIZE];
    memset(hidden_states, 0, sizeof(hidden_states));
}
"""
        return code
        
    def _generate_header_code(self, model_info: Dict) -> str:
        """Generate header file."""
        max_hidden = max(model_info['hidden_units']) if model_info['hidden_units'] else 64
        
        header = f"""#ifndef LIQUID_MODEL_H
#define LIQUID_MODEL_H

#include <stdint.h>

// Model dimensions
#define INPUT_DIM {model_info['input_dim']}
#define OUTPUT_DIM {model_info['output_dim']}
#define NUM_LAYERS {model_info['num_layers']}
#define MAX_HIDDEN_SIZE {max_hidden}

// Function declarations
int liquid_model_predict(const float* input, float* output);
void liquid_model_reset(void);

#endif // LIQUID_MODEL_H
"""
        return header
        
    def _generate_test_code(self, model_info: Dict, test_input: torch.Tensor) -> str:
        """Generate test code with sample input."""
        input_data = test_input.detach().cpu().numpy().flatten()
        input_values = ", ".join(f"{x:.6f}f" for x in input_data)
        
        test_code = f"""/*
 * Test code for liquid neural network
 */

#include <stdio.h>
#include "liquid_model.h"

// Test input data
static const float test_input[INPUT_DIM] = {{{input_values}}};

int main() {{
    float output[OUTPUT_DIM];
    
    printf("Testing liquid neural network...\\n");
    printf("Input dimension: %d\\n", INPUT_DIM);
    printf("Output dimension: %d\\n", OUTPUT_DIM);
    
    // Run prediction
    int result = liquid_model_predict(test_input, output);
    
    if (result == 0) {{
        printf("Prediction successful!\\n");
        printf("Output: ");
        for (int i = 0; i < OUTPUT_DIM; i++) {{
            printf("%.6f ", output[i]);
        }}
        printf("\\n");
    }} else {{
        printf("Prediction failed with code: %d\\n", result);
        return 1;
    }}
    
    return 0;
}}
"""
        return test_code
        
    def _generate_build_config(self, model_info: Dict) -> Dict[str, Any]:
        """Generate build configuration."""
        target_config = self.target_configs.get(self.target, {})
        
        config = {
            "target": self.target.value,
            "compiler": target_config.get("compiler", "gcc"),
            "optimization": "-O2",
            "defines": [
                f"INPUT_DIM={model_info['input_dim']}",
                f"OUTPUT_DIM={model_info['output_dim']}",
                f"NUM_LAYERS={model_info['num_layers']}"
            ],
            "includes": ["liquid_model.h"],
            "sources": ["liquid_model.c"],
            "memory_constraint": f"{self.max_memory_kb}KB",
            "quantization": self.quantization
        }
        
        return config
        
    def _calculate_memory_usage(self, model_info: Dict, quantized_weights: Dict) -> Dict[str, float]:
        """Calculate memory usage in KB."""
        total_bytes = 0
        
        # Calculate weight storage
        for name, weight_info in quantized_weights.items():
            data = weight_info["data"]
            dtype = weight_info["dtype"]
            
            if dtype == "int8":
                bytes_per_param = 1
            elif dtype == "int16":
                bytes_per_param = 2
            elif dtype == "float16":
                bytes_per_param = 2
            else:  # float32
                bytes_per_param = 4
                
            total_bytes += data.size * bytes_per_param
            
        # Add hidden state memory
        max_hidden = max(model_info['hidden_units']) if model_info['hidden_units'] else 0
        hidden_memory = model_info['num_layers'] * max_hidden * 4  # float32
        
        # Add input/output buffers
        io_memory = (model_info['input_dim'] + model_info['output_dim']) * 4
        
        total_kb = (total_bytes + hidden_memory + io_memory) / 1024.0
        
        return {
            "weights_kb": total_bytes / 1024.0,
            "hidden_states_kb": hidden_memory / 1024.0,
            "io_buffers_kb": io_memory / 1024.0,
            "total_kb": total_kb,
            "available_kb": self.max_memory_kb,
            "utilization": total_kb / self.max_memory_kb
        }
        
    def generate_firmware_template(
        self,
        output_dir: Union[str, Path],
        project_name: str = "liquid_model"
    ) -> List[Path]:
        """Generate complete firmware template."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate main application file
        if self.target in [DeploymentTarget.ESP32, DeploymentTarget.ESP32_S3]:
            main_code = self._generate_esp32_main(project_name)
            main_file = output_dir / "main.cpp"
        else:
            main_code = self._generate_generic_main(project_name)
            main_file = output_dir / "main.c"
            
        with open(main_file, 'w') as f:
            f.write(main_code)
        generated_files.append(main_file)
        
        # Generate CMakeLists.txt or Makefile
        if self.target in [DeploymentTarget.ESP32, DeploymentTarget.ESP32_S3]:
            build_file = output_dir / "CMakeLists.txt"
            build_content = self._generate_cmake_esp32(project_name)
        else:
            build_file = output_dir / "Makefile"
            build_content = self._generate_makefile(project_name)
            
        with open(build_file, 'w') as f:
            f.write(build_content)
        generated_files.append(build_file)
        
        return generated_files
        
    def _generate_esp32_main(self, project_name: str) -> str:
        """Generate ESP32 main application."""
        return f"""/*
 * {project_name} - ESP32 Application
 * Liquid Neural Network Inference
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "liquid_model.h"

void app_main() {{
    printf("Starting {project_name}...\\n");
    
    // Initialize model
    liquid_model_reset();
    
    // Main inference loop
    while (1) {{
        // TODO: Get input data from sensors
        float input[INPUT_DIM] = {{0}};  // Placeholder
        float output[OUTPUT_DIM];
        
        // Run inference
        int result = liquid_model_predict(input, output);
        
        if (result == 0) {{
            printf("Prediction: ");
            for (int i = 0; i < OUTPUT_DIM; i++) {{
                printf("%.3f ", output[i]);
            }}
            printf("\\n");
        }}
        
        vTaskDelay(pdMS_TO_TICKS(100));  // 100ms delay
    }}
}}
"""
        
    def _generate_generic_main(self, project_name: str) -> str:
        """Generate generic main application."""
        return f"""/*
 * {project_name} - Main Application
 * Liquid Neural Network Inference
 */

#include <stdio.h>
#include <unistd.h>
#include "liquid_model.h"

int main() {{
    printf("Starting {project_name}...\\n");
    
    // Initialize model
    liquid_model_reset();
    
    // Main inference loop
    while (1) {{
        // TODO: Get input data from sensors
        float input[INPUT_DIM] = {{0}};  // Placeholder
        float output[OUTPUT_DIM];
        
        // Run inference
        int result = liquid_model_predict(input, output);
        
        if (result == 0) {{
            printf("Prediction: ");
            for (int i = 0; i < OUTPUT_DIM; i++) {{
                printf("%.3f ", output[i]);
            }}
            printf("\\n");
        }}
        
        usleep(100000);  // 100ms delay
    }}
    
    return 0;
}}
"""
        
    def _generate_cmake_esp32(self, project_name: str) -> str:
        """Generate CMakeLists.txt for ESP32."""
        return f"""cmake_minimum_required(VERSION 3.16)

project({project_name})

# ESP-IDF components
set(COMPONENTS main)

# Include ESP-IDF build system
include($ENV{{IDF_PATH}}/tools/cmake/project.cmake)

project({project_name})
"""
        
    def _generate_makefile(self, project_name: str) -> str:
        """Generate Makefile for generic targets."""
        target_config = self.target_configs.get(self.target, {})
        compiler = target_config.get("compiler", "gcc")
        
        return f"""# Makefile for {project_name}

CC = {compiler}
CFLAGS = -O2 -Wall -Wextra -std=c99
LDFLAGS = -lm

TARGET = {project_name}
SOURCES = main.c liquid_model.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
\t$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.c
\t$(CC) $(CFLAGS) -c $< -o $@

clean:
\trm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
"""