from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class DataConfig:
    dataset: str
    state_cols: List[str]
    control_cols: List[str]
    forcing_cols: List[str]
    cyclic_features: List[str]
    subseq_len: int
    stride: int
    scale: bool = True
    val_ratio: float = 0.2
    split_mode: str = 'random'
    time_col: str = 'time'

@dataclass(frozen=True)
class ModelDims:
    state_dim: int
    latent_dim: int
    control_dim: int
    forcing_dim: int
    hidden_dim: int
    hidden_depth: int

@dataclass(frozen=True)
class LossWeights:
    gamma_1: float = 1.0        # Koopman Consistency
    gamma_2: float = 1.0        # Rollout State Reconstruction
    rollout_decay: float = 0.9
    epsilon_1: float = 1.0      # Encoder Dencoder Reconstruction init
    epsilon_2: float = 1.0      # Encoder Dencoder Reconstruction all
    beta_kl: float = 0.01       # KL Divergence
    alpha_ent: float = 0.1      # Entropy
    lambda_vamp: float = 1.0    # Kinetic Variance
    delta_spec: float = 1e-6    # Spectral Stability
    zero_gain: float = 0.1      # Origin Constraint
    epsilon_3: float = 1.0      # Control Recovery
    lambda_forcing: float = 0.1 # Sparse Forcing (HAVOK)

@dataclass(frozen=True)
class TrainConfig:
    encoder_type: str
    decoder_type: str
    operator_type: str
    batch_size: int = 64
    lr: float = 1e-3
    backend: str = 'torch'      # 'torch' or 'native'
    use_ema: bool = True
    ema_decay: float = 0.999
    val_interval: int = 5
    save_interval: int = 10
    min_entropy_threshold: float = 1.0
    horizon: int = 10
    device: str = 'cuda'
    plot_forcing_mode : str = "dimensions"
    stochastic: bool = True
    encode_control: bool = False
    concat_true: bool = True

@dataclass(frozen=True)
class KoopmanConfig:
    # Metadata
    exp_name: str
    checkpoint_dir: str
    
    # Nested Groups
    data: DataConfig
    dims: ModelDims
    weights: LossWeights
    train: TrainConfig          # Grouped training params
    
    # Flags
    concat_true: bool = False