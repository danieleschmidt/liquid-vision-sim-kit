# Deployment Guide

## Edge Device Deployment

### ESP32 Deployment

#### Prerequisites
- ESP-IDF 4.4 or later
- ESP32 development board
- USB cable for programming

#### Step 1: Export Model
```python
from liquid_vision.deployment import EdgeDeployer, DeploymentTarget

# Train your model first
model = train_your_model()  # Your trained LiquidNet

# Create deployer
deployer = EdgeDeployer(
    target=DeploymentTarget.ESP32,
    quantization="int8",
    max_memory_kb=512
)

# Export model
deployment_info = deployer.export_model(
    model=model,
    output_path="firmware/liquid_model.c",
    test_input=sample_input,
    generate_test=True
)

print(f"Memory usage: {deployment_info['memory_usage']['total_kb']:.1f} KB")
```

#### Step 2: Generate Firmware
```python
# Generate complete firmware template
firmware_files = deployer.generate_firmware_template(
    output_dir="firmware/esp32_project",
    project_name="liquid_classifier"
)
```

#### Step 3: Build and Flash
```bash
cd firmware/esp32_project

# Set up ESP-IDF environment
. $IDF_PATH/export.sh

# Configure project
idf.py menuconfig

# Build
idf.py build

# Flash to device
idf.py -p /dev/ttyUSB0 flash monitor
```

### Cortex-M Deployment

#### STM32 Example
```python
# Deploy to Cortex-M4
deployer = EdgeDeployer(
    target=DeploymentTarget.CORTEX_M4,
    quantization="int16",  # Better precision for floating point MCUs
    max_memory_kb=256
)

deployment_info = deployer.export_model(
    model=model,
    output_path="firmware/stm32/liquid_model.c"
)
```

#### Build with ARM GCC
```bash
cd firmware/stm32

# Compile
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O2 \
    -c liquid_model.c -o liquid_model.o

# Link with your main application
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb \
    main.o liquid_model.o -o liquid_classifier.elf
```

## Performance Optimization

### Model Quantization

#### Quantization-Aware Training
```python
from liquid_vision.training.quantization import QuantizationAwareTrainer

# Create quantized model
qat_trainer = QuantizationAwareTrainer(
    model=your_model,
    quantization_config={
        "weight_scheme": "uniform",
        "activation_scheme": "uniform", 
        "weight_bits": 8,
        "activation_bits": 8
    }
)

quantized_model = qat_trainer.get_model()

# Train with quantization
trainer = LiquidTrainer(quantized_model, config, train_loader)
history = trainer.fit()
```

#### Post-Training Quantization
```python
from liquid_vision.training.quantization import ModelQuantizer

quantizer = ModelQuantizer(calibration_loader=cal_loader)

quantized_model = quantizer.quantize_model(
    model=trained_model,
    quantization_scheme="uniform",
    weight_bits=8,
    activation_bits=8,
    calibrate=True
)

# Compare models
comparison = quantizer.compare_models(
    original_model=trained_model,
    quantized_model=quantized_model,
    test_loader=test_loader
)

print(f"Accuracy drop: {comparison['accuracy_drop']:.3f}")
print(f"Compression ratio: {comparison['compression_ratio']:.1f}x")
```

### Memory Optimization

#### Memory-Efficient Training
```python
from liquid_vision.optimization import optimize_memory_usage

# Convert to memory-efficient version
efficient_model = optimize_memory_usage(
    model=your_model,
    optimization_level="aggressive"  # "minimal", "moderate", "aggressive"
)

# Use memory-efficient mode during training
with efficient_model.memory_efficient_mode():
    output = efficient_model(batch_input)
    loss = criterion(output, targets)
    loss.backward()
```

#### Gradient Checkpointing
```python
from liquid_vision.optimization import MemoryEfficientLiquidNet

model = MemoryEfficientLiquidNet(
    input_dim=input_dim,
    hidden_units=[256, 128, 64],
    output_dim=output_dim,
    gradient_checkpointing=True,
    activation_offloading=True
)

# Monitor memory usage
stats = model.get_memory_stats()
print(f"Memory utilization: {stats['utilization']:.2f}")
```

## Distributed Training

### Multi-GPU Training
```python
from liquid_vision.core.distributed import setup_distributed_training, DistributedLiquidNet

# Setup distributed environment
setup_distributed_training(
    local_rank=0,
    world_size=4,  # 4 GPUs
    backend="nccl"
)

# Wrap model for distributed training
distributed_model = DistributedLiquidNet(
    model=your_model,
    device_ids=[0],  # Local GPU
    find_unused_parameters=True
)

# Use DistributedSampler for data loading
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4
)
```

### Model Parallelism
```python
from liquid_vision.core.distributed import ModelParallelLiquidNet

# For very large models that don't fit on single GPU
model = ModelParallelLiquidNet(
    input_dim=input_dim,
    hidden_units=[512, 256, 128, 64],  # Large model
    output_dim=output_dim,
    device_map={
        0: "cuda:0",  # First layer on GPU 0
        1: "cuda:0",  # Second layer on GPU 0  
        2: "cuda:1",  # Third layer on GPU 1
        3: "cuda:1",  # Fourth layer on GPU 1
        4: "cuda:1",  # Readout on GPU 1
    }
)
```

## Benchmarking and Profiling

### Energy Profiling
```python
from liquid_vision.benchmarks import EnergyProfiler, BenchmarkConfig

config = BenchmarkConfig(
    duration_s=60.0,
    warmup_s=5.0,
    batch_sizes=[1, 8, 16, 32],
    input_shapes=[(64, 64), (128, 128)],
    device="cuda",
    output_dir="benchmarks/results"
)

profiler = EnergyProfiler(config)

def input_generator(batch_size, input_shape):
    return torch.randn(batch_size, *input_shape)

# Profile single model
results = profiler.profile_model(
    model=your_model,
    input_generator=input_generator,
    model_name="liquid_net_base"
)

print(f"Average latency: {results['batch_1']['shape_64x64']['avg_latency_ms']:.2f} ms")
print(f"Throughput: {results['batch_1']['shape_64x64']['throughput_fps']:.1f} FPS")
print(f"Energy efficiency: {results['batch_1']['shape_64x64']['efficiency_inferences_per_mj']:.1f} inf/mJ")
```

### Model Comparison
```python
from liquid_vision.benchmarks import compare_models

# Compare multiple models
models = {
    "liquid_small": create_liquid_net(input_dim, output_dim, "small"),
    "liquid_base": create_liquid_net(input_dim, output_dim, "base"),  
    "liquid_quantized": quantized_model,
    "liquid_efficient": efficient_model,
}

comparison_results = compare_models(
    models=models,
    input_generator=input_generator,
    config=config,
    output_path="benchmarks/model_comparison.json"
)

# Get efficiency rankings
rankings = comparison_results["comparison"]["rankings"]
print(f"Most efficient: {rankings['efficiency_inferences_per_mj'][0]}")
print(f"Lowest latency: {rankings['avg_latency_ms'][0]}")
```

## Production Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install liquid-vision-sim-kit
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Run training
CMD ["python", "examples/basic_usage.py"]
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liquid-vision-training
spec:
  replicas: 1
  selector:
    matchLabels:
      app: liquid-vision-training
  template:
    metadata:
      labels:
        app: liquid-vision-training
    spec:
      containers:
      - name: training
        image: liquid-vision:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: results-volume
          mountPath: /app/results
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: results-volume
        persistentVolumeClaim:
          claimName: training-results-pvc
```

### Model Serving with TorchServe
```python
# model_handler.py
import torch
from ts.torch_handler.base_handler import BaseHandler
from liquid_vision import LiquidNet

class LiquidNetHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def initialize(self, context):
        # Load model
        model_path = context.system_properties.get("model_dir") + "/model.pth"
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
    def preprocess(self, data):
        # Convert input to tensor
        input_data = data[0].get("data") or data[0].get("body")
        tensor = torch.tensor(input_data).float().to(self.device)
        return tensor
        
    def inference(self, data):
        with torch.no_grad():
            return self.model(data)
            
    def postprocess(self, data):
        return data.cpu().numpy().tolist()
```

### Monitoring and Logging
```python
import logging
from liquid_vision.training import LiquidTrainer, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Add custom metrics logging
class MetricsLogger:
    def __init__(self, log_file="metrics.json"):
        self.log_file = log_file
        self.metrics = []
        
    def log_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc):
        metric = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss, 
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "timestamp": time.time()
        }
        self.metrics.append(metric)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```python
# Solution 1: Reduce batch size
config.batch_size = 16  # Instead of 32

# Solution 2: Use gradient checkpointing
model = MemoryEfficientLiquidNet(
    input_dim=input_dim,
    hidden_units=hidden_units,
    output_dim=output_dim,
    gradient_checkpointing=True
)

# Solution 3: Clear cache frequently
if epoch % 10 == 0:
    torch.cuda.empty_cache()
```

#### Slow Training
```python
# Solution 1: Use mixed precision
config.mixed_precision = True

# Solution 2: Optimize data loading
train_loader = EventDataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)

# Solution 3: Use efficient model
optimized_model = optimize_memory_usage(model, "moderate")
```

#### Deployment Issues
```bash
# Check memory usage
liquid-vision model-info --architecture base --input-dim 1000

# Test on device
liquid-vision simulate --frames 10 --output test.npz
```

### Performance Tuning

#### Hyperparameter Optimization
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    tau = trial.suggest_float("tau", 5.0, 50.0)
    leak = trial.suggest_float("leak", 0.01, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    # Create model and train
    model = LiquidNet(input_dim, hidden_units, output_dim, tau=tau, leak=leak)
    config = TrainingConfig(learning_rate=lr, epochs=10)
    
    trainer = LiquidTrainer(model, config, train_loader, val_loader)
    history = trainer.fit()
    
    # Return validation accuracy
    return history["val_metrics"][-1]["accuracy"]

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")
```

#### Model Architecture Search
```python
def search_architecture():
    architectures = [
        [32, 16],
        [64, 32], 
        [128, 64, 32],
        [64, 32, 16],
        [128, 64, 32, 16]
    ]
    
    results = {}
    
    for arch in architectures:
        model = LiquidNet(input_dim, arch, output_dim)
        
        # Quick evaluation
        trainer = LiquidTrainer(model, config, train_loader, val_loader)
        history = trainer.fit()
        
        results[str(arch)] = {
            "accuracy": history["val_metrics"][-1]["accuracy"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "memory_mb": sum(p.numel() * 4 for p in model.parameters()) / 1024**2
        }
        
    return results

# Find best architecture
arch_results = search_architecture()
best_arch = max(arch_results.keys(), key=lambda k: arch_results[k]["accuracy"])
print(f"Best architecture: {best_arch}")
```

## Best Practices

### Model Development
1. Start with small models and scale up
2. Use synthetic data for initial development
3. Implement proper validation and testing
4. Monitor memory usage during training
5. Use quantization for deployment

### Production Deployment
1. Always profile before deployment
2. Use appropriate quantization for target hardware
3. Implement proper error handling and logging
4. Test thoroughly on target devices
5. Monitor performance in production

### Performance Optimization
1. Use memory-efficient implementations for large models
2. Leverage distributed training for big datasets
3. Apply quantization and pruning for edge deployment
4. Use appropriate batch sizes for your hardware
5. Profile and benchmark regularly