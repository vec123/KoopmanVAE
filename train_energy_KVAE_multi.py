# train_koopman_all_systems.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import joblib
from systems import SYSTEMS
from models.models import (
    ResidualMLP,
    LinearMatrix,
    TTKoopman
)
from trainers.Multi_TT_KVAE_trainer import KoopmanVAETrainer
from logger.logger import InfoVectorLogger


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def split_into_subsequences(sequences, subseq_len, stride=1):
    subsequences = []
    for traj in sequences:
        T = traj.shape[0]
        for start in range(0, T - subseq_len + 1, stride):
            subsequences.append(traj[start:start + subseq_len])
    return subsequences


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], 0  # dummy id (ignored)


# -------------------------------------------------------
# System metadata
# -------------------------------------------------------

SYSTEM_CONFIG = {
    "van_der_pol": {
        "state_dim": 2,
        "init_sampler": lambda: np.random.uniform(-1, 1, size=2),
        "phase_plots": [(0, 1)],
    },
    "limit_cycle": {
        "state_dim": 2,
        "init_sampler": lambda: np.random.uniform(-1, 1, size=2),
        "phase_plots": [(0, 1)],
    },
    "lorenz": {
        "state_dim": 3,
        "init_sampler": lambda: np.random.uniform(-5, 5, size=3),
        "phase_plots": [(0, 1, 2)],
    },
    "oscillator": {
        "state_dim": 6,
        "init_sampler": lambda: np.random.uniform(0.1, 2.0, size=6),
        "phase_plots": [(3, 4), (0, 3)],  # (p1,p2), (m1,p1)
    },
}


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Generate Single Trajectory
    tensor_train_koopman = False  # Set to True to use TTKoopman, False for LinearMatrix
    log_name = "Multi_KVAE_energy" 


    subseq_len = 24*7
    stride = 10
    dt = 1
    batch_size = 64
    latent_dim = 265
    tt_rank = 32  # Start small (2, 4, or 8)
    tt_shape = [(10, 10), (10, 10)]
    hidden_dim = 128
    horizon = 48

    # --- Load Data ---
    START_DATE = "2025-01-01 00:00:00"
    END_DATE   = "2025-12-31 23:59:59"
    sensor_data = "ES0031405047432001AZ0F"
    DATASET_PATH = f"data_in/SensorData/{sensor_data}.csv"
    TARGET_COL = "value"                
    TIME_COL = "timestamp"                      
    
    # 1. Load Data from CSV
    df = pd.read_csv(DATASET_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    mask = (df[TIME_COL] >= START_DATE) & (df[TIME_COL] <= END_DATE)
    df = df.loc[mask].reset_index(drop=True)

    # Create Cyclic Time Features
    # Hour of day (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df[TIME_COL].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df[TIME_COL].dt.hour / 24.0)
    
    # Day of week (0-6)
    df['day_sin'] = np.sin(2 * np.pi * df[TIME_COL].dt.dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df[TIME_COL].dt.dayofweek / 7.0)

    # Define all columns to be used as state
    FEATURE_COLS = [TARGET_COL, 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    data_raw = df[FEATURE_COLS].values


    # --- Scale  ---
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_raw)

    state_dim = data_scaled.shape[1]
    print(f"Loaded data with shape: {data_scaled.shape} (State Dim: {state_dim})")


     # --- Split sequences CONFIG ---
    sequences = [data_scaled] 
    subsequences = split_into_subsequences(sequences, subseq_len=subseq_len, stride=stride)
    loader = DataLoader(SequenceDataset(subsequences), batch_size=batch_size, shuffle=True)

    # 2. Models (Note: Use MLP for both for now to ensure rollout manifold is learned)
    encoder = ResidualMLP(state_dim, 2 * latent_dim, [128]*3).to(device)
    decoder = ResidualMLP(latent_dim, state_dim, [128]*3).to(device)

    # Replace LinearMatrix with TTKoopman
    if tensor_train_koopman:
        koopman_matrix = TTKoopman(
            latent_dim=latent_dim, 
            tt_rank=tt_rank, 
            tt_shape=tt_shape
        ).to(device)
    else:
        koopman_matrix = LinearMatrix(latent_dim, latent_dim).to(device)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(koopman_matrix.parameters()),
        lr=1e-3
    )

    logger = InfoVectorLogger(log_dir=f"logs/Free_spectrum_{sensor_data}_{log_name}")

    # Save the scaler now
    scaler_path = os.path.join(logger.log_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # 3. Trainer
    trainer = KoopmanVAETrainer(
        encoder=encoder,
        decoder=decoder,
        koopman_matrix=koopman_matrix,
        dataloader=loader,
        optimizer=optimizer,
        latent_dim=latent_dim,
        horizon=horizon,  # Supervise 30 steps of the rollout
        beta=1e-4,       
        gamma=10.0,     
        delta=1e-10,
        device=device,
        logger=logger,
        save_epoch=500
    )

    print("Starting Joint Training (No Stages)...")
    trainer.train(20000)

if __name__ == "__main__":
    main()

