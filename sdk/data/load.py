import os
from loader_factory import DataPipelineFactory


DATASET_PATH = os.getenv("DATASET_PATH")

config = {
    'backend': 'torch',
    'time_col': 'time',
    'state_cols': ['$T_1$', '$T_2$', '$T_{wall}$', '$T_{out}$'],
    'control_cols': ['u_0'],
    'forcing_cols': [],
    'cyclic_features': ['hour', 'day_of_week'],
    'subseq_len': 20,
    'stride': 1,
    'batch_size': 64,
    'scale': True,
    'val_ratio': 0.2,       # Added explicitly for the loader
    'split_mode': 'random'  # Added explicitly for the loader
}

# 2. Pass the DATASET_PATH variable instead of the placeholder "./data"
# Ensure it is wrapped in a list as the factory expects 'directories'
train_loader, val_loader, scaler = DataPipelineFactory.create_pipeline(
    directories=[DATASET_PATH], 
    config=config
)

# 3. Verification
print(f"Train batches: {type(train_loader)}")
print(f"Validation batches: {type(val_loader)}")