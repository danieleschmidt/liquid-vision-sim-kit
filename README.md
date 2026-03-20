# liquid-vision-sim-kit

Neuromorphic event-camera simulation and liquid state network inference kit for embedded / edge deployment.

## Overview

`liquid-vision-sim-kit` provides a minimal, NumPy-only toolkit for experimenting with bio-inspired, event-driven vision and reservoir computing (echo-state networks) on resource-constrained devices (ARM Cortex-M, ESP32, etc.).

### Components

| Class | Description |
|-------|-------------|
| `EventCameraSimulator` | Simulate a Dynamic Vision Sensor (DVS): emit `(x, y, polarity, timestamp)` events from consecutive grayscale frames |
| `NeuromorphicDataset` | Generate synthetic event-stream datasets for training/evaluation |
| `LiquidStateNetwork` | Echo-state / liquid state reservoir with fixed random reservoir + trainable linear readout |
| `EmbeddedProfiler` | Estimate parameter count, memory footprint (bytes), and Cortex-M clock cycles |
| `CHeaderExporter` | Export weight matrices as C header files for on-device inference |

## Installation

```bash
pip install numpy
pip install -e .
```

## Quick Start

```python
import numpy as np
from liquid_vision_sim_kit import (
    EventCameraSimulator,
    NeuromorphicDataset,
    LiquidStateNetwork,
    EmbeddedProfiler,
    CHeaderExporter,
)

# 1. Simulate event camera output
sim = EventCameraSimulator()
frame1 = np.random.rand(128, 128)
frame2 = np.random.rand(128, 128)
events = sim.generate_events(frame1, frame2, threshold=0.1)
print(f"Generated {len(events)} events")

# 2. Build a synthetic dataset
dataset = NeuromorphicDataset(n_frames=50, height=32, width=32, seed=42)
print(f"Dataset size: {len(dataset)} sequences")

# 3. Train a Liquid State Network
lsn = LiquidStateNetwork(input_dim=4, reservoir_size=256, output_dim=2)
X_train = [dataset[i] for i in range(len(dataset))]
y_train = np.random.randint(0, 2, len(dataset)).astype(float)
lsn.fit_readout(X_train, y_train)

# 4. Inference
output = lsn.forward(dataset[0])
print(f"Output: {output}")

# 5. Profile for embedded deployment
profiler = EmbeddedProfiler()
model_dict = {"W_in": lsn.W_in, "W_res": lsn.W_res, "W_out": lsn.W_out}
stats = profiler.profile(model_dict)
print(stats)

# 6. Export weights as C header
exporter = CHeaderExporter()
exporter.export(model_dict, "model_weights.h")
```

## Architecture

### EventCameraSimulator

Computes pixel-wise frame difference and emits binary events:

```
event = (x, y, polarity, timestamp)
  polarity = +1  if frame2[y,x] - frame1[y,x] >  threshold
  polarity = -1  if frame2[y,x] - frame1[y,x] < -threshold
```

### LiquidStateNetwork

Echo-state network with echo-state property guaranteed by scaling spectral radius of `W_res` to < 1:

```
state[t+1] = tanh(W_res @ state[t] + W_in @ event[t])
output     = W_out @ state[-1]
```

Readout is trained via least-squares pseudo-inverse:

```
W_out = (y^T S)(S^T S)^{-1}
```

### Embedded Deployment

The `EmbeddedProfiler` estimates:
- **param_count**: total float32 parameters
- **memory_bytes**: `param_count × 4`
- **cortex_m_cycles**: `ceil(param_count / ops_per_cycle)`

The `CHeaderExporter` generates a `.h` file with `static const float` arrays ready for inclusion in C/C++ firmware.

## Testing

```bash
python -m pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.21

## License

MIT
