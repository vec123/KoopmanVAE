import os
from data.loader_factory import DataPipelineFactory
from conf.config import DataConfig, TrainConfig 
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")

print("--- Testing Data Loading ---")
print(f"Loading from: {DATASET_PATH}")

data_cfg = DataConfig(
    state_cols=['$T_1$', '$T_2$', '$T_{wall}$', '$T_{out}$'],
    control_cols=['u_0'],
    forcing_cols=[],
    cyclic_features=['hour', 'day_of_week'],
    horizon=10,
    subseq_len=20,
    stride=1,
    scale=True,
    val_ratio=0.2,
    split_mode='random'
)

# You need this for batch_size and backend!
train_cfg = TrainConfig(
    batch_size=64,
    backend='torch'
)

# 2. Access the attributes from the correct objects
BATCH_SIZE = train_cfg.batch_size  # Correct path
BACKEND = train_cfg.backend        # Correct path

# 3. Execute the pipeline
train_loader, val_loader, scaler = DataPipelineFactory.create_pipeline(
    directories=[DATASET_PATH], 
    config=data_cfg,
    batch_size=BATCH_SIZE,
    backend=BACKEND,
    verbose=True
)
# 4. Verification
print("\n--- Verification ---")
print(f"Train Loader Type: {type(train_loader)}")
print(f"Val Loader Type:   {type(val_loader)}")

# Sample check to ensure dimensions are correct (4 states + 4 cyclic = 8)
sample_batch = next(iter(train_loader))
# Batch usually returns (X, U, F) or similar depending on your Dataset class
X, U, F = sample_batch 
print(f"Input Shape (B, T, D): {X.shape}") 
print(f"Target Feature Dim:    {X.shape[-1]} (Expected 8)")