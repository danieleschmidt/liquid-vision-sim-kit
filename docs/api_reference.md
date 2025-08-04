# API Reference

## Core Components

### LiquidNeuron

The fundamental building block of liquid neural networks.

```python
class LiquidNeuron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int, 
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        recurrent_connection: bool = True,
        dt: float = 1.0,
    )
```

**Parameters:**
- `input_dim`: Input feature dimension
- `hidden_dim`: Hidden state dimension  
- `tau`: Time constant for liquid dynamics
- `leak`: Leak rate for state decay
- `activation`: Activation function ("tanh", "sigmoid", "relu", "swish", "gelu")
- `recurrent_connection`: Whether to include recurrent connections
- `dt`: Integration time step

**Methods:**
- `forward(x, hidden=None, dt=None)`: Forward pass through neuron
- `reset_state()`: Reset internal state

### LiquidNet

Multi-layer liquid neural network.

```python
class LiquidNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        tau: float = 10.0,
        leak: float = 0.1,
        activation: str = "tanh",
        readout_activation: Optional[str] = None,
        dropout: float = 0.0,
        dt: float = 1.0,
    )
```

**Parameters:**
- `input_dim`: Input dimension
- `hidden_units`: List of hidden layer sizes
- `output_dim`: Output dimension
- `tau`: Time constant for all layers
- `leak`: Leak rate for all layers
- `activation`: Activation function for liquid layers
- `readout_activation`: Activation for output layer
- `dropout`: Dropout rate between layers
- `dt`: Integration time step

**Methods:**
- `forward(x, reset_state=False, dt=None)`: Forward pass
- `reset_states()`: Reset all liquid states
- `get_liquid_states()`: Get current states
- `set_time_constants(tau)`: Update time constants

## Event Simulation

### EventSimulator

Base class for event camera simulation.

```python
class DVSSimulator(EventSimulator):
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        contrast_threshold: float = 0.15,
        noise_model: Optional[NoiseModel] = None,
        refractory_period: float = 1.0,
    )
```

**Methods:**
- `simulate_frame(frame, timestamp)`: Generate events from single frame
- `simulate_video(frames, timestamps=None, fps=30.0)`: Simulate video sequence
- `reset()`: Reset simulator state

### SceneGenerator

Synthetic scene generation for training data.

```python
class SceneGenerator:
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        background_color: float = 0.5,
        frame_rate: float = 30.0,
    )
```

**Methods:**
- `add_object(object_type, position, size, velocity, color, motion_pattern, **kwargs)`: Add moving object
- `generate_frame(frame_number)`: Generate single frame
- `generate_sequence(num_frames, return_timestamps=True)`: Generate frame sequence
- `clear_objects()`: Remove all objects

## Training Components

### LiquidTrainer

Main training class for liquid neural networks.

```python
class LiquidTrainer:
    def __init__(
        self,
        model: LiquidNet,
        config: TrainingConfig,
        train_loader: EventDataLoader,
        val_loader: Optional[EventDataLoader] = None,
        logger: Optional[logging.Logger] = None,
    )
```

**Methods:**
- `fit(epochs=None, resume_from=None)`: Train the model
- `evaluate(test_loader)`: Evaluate on test set

### TrainingConfig

Configuration for training parameters.

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    gradient_clip: Optional[float] = 1.0
    dt: float = 1.0
    reset_state_frequency: int = 10
    loss_type: str = "cross_entropy"
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"
    device: str = "auto"
    mixed_precision: bool = False
    quantization_aware: bool = False
```

### EventDataLoader

Data loader for event-based datasets.

```python
class EventDataLoader:
    def __init__(
        self,
        dataset: EventDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        temporal_batching: bool = False,
    )
```

## Event Encoding

### EventEncoder

Base class for event encoding schemes.

```python
def create_encoder(
    encoding_type: str,
    sensor_size: Tuple[int, int] = (640, 480),
    **kwargs
) -> EventEncoder
```

**Encoding Types:**
- `"temporal"`: Temporal surface encoding
- `"spatial"`: Spatial histogram encoding  
- `"timeslice"`: Multi-slice temporal encoding
- `"adaptive"`: Learnable adaptive encoding

### TemporalEncoder

```python
class TemporalEncoder(EventEncoder):
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (640, 480),
        time_window: float = 50.0,
        tau_decay: float = 20.0,
        polarity_separate: bool = True,
    )
```

## Edge Deployment

### EdgeDeployer  

Deploy models to edge devices.

```python
class EdgeDeployer:
    def __init__(
        self,
        target: Union[str, DeploymentTarget] = DeploymentTarget.ESP32,
        optimize_memory: bool = True,
        quantization: str = "int8",
        max_memory_kb: int = 512,
    )
```

**Methods:**
- `export_model(model, output_path, test_input=None, generate_test=True)`: Export for deployment
- `generate_firmware_template(output_dir, project_name)`: Generate firmware template

**Supported Targets:**
- ESP32, ESP32_S3
- CORTEX_M4, CORTEX_M7  
- ARDUINO
- RASPBERRY_PI

## Loss Functions

### LiquidLoss

Specialized loss for liquid networks with regularization.

```python
class LiquidLoss(nn.Module):
    def __init__(
        self,
        base_loss: str = "cross_entropy",
        stability_weight: float = 0.01,
        sparsity_weight: float = 0.001,
        energy_weight: float = 0.005,
        diversity_weight: float = 0.01,
    )
```

### TemporalLoss

Loss function for temporal consistency.

```python
class TemporalLoss(nn.Module):
    def __init__(
        self,
        base_loss: str = "cross_entropy", 
        temporal_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        consistency_weight: float = 0.1,
    )
```

## Quantization

### QuantizedLiquidNet

Quantized version for efficient inference.

```python
class QuantizedLiquidNet(LiquidNet):
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        output_dim: int,
        weight_quantization: str = "uniform",
        activation_quantization: str = "uniform", 
        weight_bits: int = 8,
        activation_bits: int = 8,
        **kwargs
    )
```

### ModelQuantizer

Post-training quantization utilities.

```python
class ModelQuantizer:
    def quantize_model(
        self,
        model: LiquidNet,
        quantization_scheme: str = "uniform",
        weight_bits: int = 8,
        activation_bits: int = 8,
        calibrate: bool = True,
    ) -> QuantizedLiquidNet
```

## Optimization

### MemoryEfficientLiquidNet

Memory-optimized implementation.

```python
class MemoryEfficientLiquidNet(LiquidNet):
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int], 
        output_dim: int,
        memory_efficient: bool = True,
        gradient_checkpointing: bool = True,
        activation_offloading: bool = False,
        **kwargs
    )
```

**Methods:**
- `get_memory_stats()`: Get memory usage statistics
- `clear_memory_cache()`: Clear memory cache
- `memory_efficient_mode()`: Context manager for efficient execution

## Benchmarking

### EnergyProfiler

Profile energy consumption and performance.

```python  
class EnergyProfiler:
    def __init__(self, config: BenchmarkConfig)
    
    def profile_model(
        self,
        model,
        input_generator: Callable,
        model_name: str = "liquid_model",
        **kwargs
    ) -> Dict[str, Any]
```

### BenchmarkConfig

```python
@dataclass
class BenchmarkConfig:
    duration_s: float = 60.0
    warmup_s: float = 5.0
    power_sampling_hz: float = 10.0
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32])
    input_shapes: List[Tuple[int, ...]] = field(default_factory=lambda: [(1, 64, 64)])
    device: str = "cpu"
    save_detailed_logs: bool = True
    output_dir: str = "benchmarks/results"
```

## Factory Functions

### Model Creation

```python
def create_liquid_net(
    input_dim: int,
    output_dim: int,
    architecture: str = "small",
    **kwargs
) -> LiquidNet
```

**Architectures:**
- `"tiny"`: [16] hidden units
- `"small"`: [32, 16] hidden units  
- `"base"`: [64, 32, 16] hidden units
- `"large"`: [128, 64, 32] hidden units

### Simulator Creation

```python
def create_simulator(
    simulator_type: str = "dvs",
    resolution: Tuple[int, int] = (640, 480),
    **kwargs
) -> EventSimulator
```

**Simulator Types:**
- `"dvs"`: Basic DVS simulator
- `"davis"`: DAVIS with APS frames
- `"advanced_dvs"`: DVS with realistic effects

## Command Line Interface

```bash
# Show system information
liquid-vision info

# Generate synthetic events
liquid-vision simulate --frames 100 --resolution 128 96 --output events.npz

# Train a model
liquid-vision train --task classification --epochs 50 --architecture small

# Show model information  
liquid-vision model-info --architecture base --input-dim 1000 --output-dim 10
```

## Examples

### Basic Usage

```python
from liquid_vision import LiquidNet, EventSimulator, SceneGenerator

# Create scene
scene = SceneGenerator(resolution=(128, 96))
scene.add_object(ObjectType.CIRCLE, position=(64, 48), size=10, 
                velocity=(2, 0), color=0.9)

# Generate events
frames, timestamps = scene.generate_sequence(num_frames=50)
simulator = EventSimulator(resolution=(128, 96), contrast_threshold=0.1)
events = simulator.simulate_video(frames, timestamps)

# Create and train model
model = LiquidNet(input_dim=2*128*96, hidden_units=[64, 32], output_dim=3)
# ... training code ...
```

### Gesture Recognition

```python
from liquid_vision.training import LiquidTrainer, TrainingConfig
from examples.gesture_recognition import GestureDataset

# Create dataset
dataset = GestureDataset(num_samples=1000, resolution=(64, 64))
train_loader = EventDataLoader(dataset, batch_size=32)

# Train model
config = TrainingConfig(epochs=25, learning_rate=1e-3)
trainer = LiquidTrainer(model, config, train_loader)
history = trainer.fit()
```

### Edge Deployment

```python
from liquid_vision.deployment import EdgeDeployer, DeploymentTarget

# Deploy to ESP32
deployer = EdgeDeployer(target=DeploymentTarget.ESP32, quantization="int8")
deployment_info = deployer.export_model(model, "firmware/liquid_model.c")

# Generate firmware template
firmware_files = deployer.generate_firmware_template("firmware/", "gesture_classifier")
```