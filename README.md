# KoopmanVAE: Learning Dynamics with Koopman Operators and Variational Autoencoders

A PyTorch-based framework for learning interpretable, linearized representations of complex dynamical systems using Koopman operator theory combined with variational autoencoders.

## Overview

KoopmanVAE implements a hybrid approach that combines:
- **Koopman Operator Theory**: Lifts nonlinear dynamics into a space where they behave linearly
- **Variational Autoencoders (VAE)**: Learn compressed latent representations of system states
- **Control Dynamics**: Models both autonomous and controlled systems with optional control encoding
- **Forcing Functions**: Supports external forcing terms (e.g., weather/external inputs) via sparse parameterization

This framework enables learning of dynamical systems from data with applications in:
- Physics-informed machine learning
- Control system design
- Time-series forecasting
- System identification

## Key Features

- **Flexible Encoder/Decoder Architectures**: MLPs, GRUs, LSTMs, and Transformer-based backbones
- **Koopman Operator Learning**: Learns linear operators in latent space while reconstructing nonlinear dynamics
- **Control-Aware Dynamics**: Optional control encoding and forcing network integration
- **Multiple Loss Functions**: Koopman consistency, reconstruction, KL divergence, spectral stability, and more
- **Exponential Moving Average (EMA)**: Stable training with optional EMA updates
- **Comprehensive Monitoring**: Real-time training metrics, validation, and force decomposition visualization
- **Multi-System Support**: Pre-configured systems including:
  - Lorenz attractor
  - Duffing oscillator
  - Van der Pol oscillator
  - Inverted pendulum
  - Coupled mass systems

## Architecture

### System Components

```
KoopmanVAE System Bundle:
├── Encoder: State → Latent space (outputs μ, log-σ for VAE)
├── Decoder: Latent/Concatenated space → State reconstruction
├── Koopman Operator (A): Learns linear transition in latent space [z, x]_t → [z, x]_{t+1}
├── Forcing Network: Learns sparse forcing functions (HAVOK framework)
├── Control Encoder: (Optional) Encodes control inputs to latent dimension
├── Control Decoder: (Optional) Decodes latent control back to action space
└── Control Matrix (B): Maps control to state transitions
```

### Loss Functions

The training objective combines multiple components:

| Loss Component | Purpose |
|---|---|
| **Koopman Consistency** (γ₁) | Enforces linear dynamics in latent space |
| **Rollout Reconstruction** (γ₂) | Multi-step trajectory reconstruction error |
| **Reconstruction (ε₁, ε₂)** | Initial and full sequence encoder-decoder fidelity |
| **KL Divergence** (βKL) | VAE regularization |
| **Entropy** (αent) | Latent space regularization |
| **Spectral Stability** (δspec) | Constrains Koopman eigenvalue magnitudes |
| **Forcing Sparsity** (λforcing) | Encourages sparse forcing functions |
| **Control Recovery** (ε₃) | Control reconstruction accuracy |

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
cd KoopmanVAE
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install hydra-core omegaconf pytorch-ema python-dotenv numpy scipy matplotlib
```

4. **Configure environment variables**
Create a `.env` file in the project root:
```bash
DATASET_W_METEO_PATH=/path/to/your/datasets
```

## Quick Start

### Basic Training

```bash
python main.py
```

This runs with the default Lorenz system configuration from `conf/model/lorenz.yaml`.

### Training with Custom System

```bash
python main.py --config-name=config model=duffing
python main.py --config-name=config model=cartpole
python main.py --config-name=config model=hvac_linear
```

### Hyperparameter Tuning

Override configuration parameters from the command line:

```bash
python main.py \
  model=lorenz \
  model.dims.latent_dim=16 \
  model.train.lr=1e-3 \
  model.train.batch_size=32 \
  model.train.horizon=20 \
  model.weights.gamma_1=1.5
```

### Generate Synthetic Data

```bash
cd systems
python generate_data.py --system lorenz --duration 1000 --dt 0.01
```

## Configuration

### Config Structure

Configurations are defined in YAML files under `conf/`:

```yaml
model:
  exp_name: "koopman_vae_experiment"
  checkpoint_dir: "./checkpoints"
  
  data:
    dataset: "lorenz_data"
    state_cols: [x, y, z]
    control_cols: []
    forcing_cols: []
    cyclic_features: []
    subseq_len: 100
    stride: 10
    
  dims:
    state_dim: 3
    latent_dim: 8
    control_dim: 0
    forcing_dim: 3
    hidden_dim: 64
    hidden_depth: 3
    
  train:
    encoder_type: "mlp"
    decoder_type: "mlp"
    operator_type: "linear"
    use_forcing: false
    forcing_net: "mlp"
    batch_size: 64
    lr: 1e-3
    horizon: 10
    
  weights:
    gamma_1: 1.0         # Koopman consistency
    gamma_2: 1.0         # Rollout reconstruction
    beta_kl: 0.01        # KL divergence
    delta_spec: 1e-6     # Spectral stability
```

## Project Structure

```
KoopmanVAE/
├── main.py                      # Entry point
├── conf/
│   ├── config.yaml              # Default configuration
│   ├── config.py                # Configuration dataclasses
│   └── model/                   # System-specific configs
│       ├── lorenz.yaml
│       ├── duffing_spring.yaml
│       ├── cartpole.yaml
│       └── ...
├── data/
│   └── loader_factory.py        # Data loading pipeline
├── models/
│   ├── models.py                # Core model definitions
│   ├── model_backbones.py       # Encoder/decoder backbones
│   ├── model_factories.py       # Model instantiation
│   └── model_inits.py           # Weight initialization
├── training/
│   ├── engine.py                # KoopmanTrainer class
│   ├── losses.py                # Loss functions
│   ├── monitor.py               # Metrics tracking
│   └── __init__.py
├── systems/
│   ├── base.py                  # System base class
│   ├── generate_data.py         # Data generation
│   ├── lorenz.py
│   ├── duffing_spring.py
│   ├── cartpole.py
│   └── ...
├── utils/                       # Utility functions
├── HAVOK/                       # HAVOK forcing framework
└── generated_samples/           # Output predictions
```

## Advanced Usage

### Using Different Backbones

```bash
# Transformer encoder/decoder
python main.py model=lorenz model.train.encoder_type=transformer model.train.decoder_type=transformer

# LSTM-based
python main.py model=lorenz model.train.encoder_type=lstm model.train.decoder_type=lstm

# GRU-based
python main.py model=lorenz model.train.encoder_type=gru model.train.decoder_type=gru
```

### Control-Aware Learning

```bash
python main.py model=cartpole \
  model.dims.control_dim=1 \
  model.train.encode_control=true
```

### HAVOK Forcing

```bash
python main.py model=lorenz \
  model.train.use_forcing=true \
  model.train.forcing_net=mlp \
  model.weights.lambda_forcing=0.1
```

### Multi-Step Rollout

Adjust prediction horizon for training:

```bash
python main.py model=lorenz \
  model.train.horizon=50 \
  model.weights.gamma_2=2.0
```

## Training Monitoring

The trainer logs metrics to:
- Console output (real-time loss values)
- TensorBoard (via `monitor.py`)
- Experiment directories under `./experiments/YYYY-MM-DD/HH-MM-SS/`

### Key Metrics

- **train_loss**: Total weighted loss
- **koopman_consistency_loss**: Linear operator accuracy
- **reconstruction_loss**: Encoder-decoder fidelity
- **kl_loss**: VAE regularization
- **val_loss**: Validation set performance
- **entropy**: Latent space entropy

## Results and Outputs

After training completes:

1. **Checkpoints**: Saved at intervals under `checkpoint_dir`
2. **Generated Samples**: Predictions and reconstructions in `generated_samples/`
3. **Force Decomposition**: HAVOK force analysis plots
4. **Experiment Logs**: Full run configuration and metrics

## Extending the Framework

### Adding a New System

1. Create a new file in `systems/` (e.g., `my_system.py`):

```python
from systems.base import DynamicalSystem
import numpy as np

class MySystem(DynamicalSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def dynamics(self, state, control=None, time=None):
        # Implement your system dynamics
        x, y = state
        dx = -y
        dy = x
        return np.array([dx, dy])
```

2. Register in `systems/registry.py`
3. Create config file `conf/model/my_system.yaml`
4. Run: `python main.py model=my_system`

### Custom Loss Functions

Extend `training/losses.py`:

```python
class KoopmanLossManager:
    def compute_custom_loss(self, ...):
        # Implement custom loss
        return loss
```

## Performance Tips

1. **Use EMA**: Stabilizes training (`model.train.use_ema=true`)
2. **Spectral Regularization**: Add spectral stability loss for smooth latent dynamics
3. **Batch Size**: Larger batches (128-256) improve Koopman consistency
4. **Learning Rate**: Start with 1e-3, decay over time
5. **Horizon**: Increase gradually during training for better long-term predictions
6. **GPU**: Ensure CUDA is available for faster training

## References

- Koopman Operator Theory: [E. Kingma et al., "Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114)
- HAVOK Framework: [Brunton et al., "Discovering Governing Equations from Data"](https://arxiv.org/abs/1609.04803)
- Koopman VAE Integration: [Frye et al., "Asymptotically Optimal Learning of Non-Linear Dynamical Systems"](https://arxiv.org/abs/2112.11297)

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues and pull requests to improve the framework.

## Contact

For questions or suggestions, please open an issue on the repository.

---

**Last Updated**: 2024  
**Maintainers**: [Your Name/Organization]
