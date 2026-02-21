# train_koopman_all_systems.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
    DATASET = "oscillator"

    system_func = SYSTEMS[DATASET]
    cfg = SYSTEM_CONFIG[DATASET]
    init = cfg["init_sampler"]()

    N_traj = 1  # Number of different initial conditions
    seq_len = 500
    subseq_len = 200
    stride = 10
    dt = 1
    t = np.arange(0, seq_len * dt, dt)
    batch_size = 64
    latent_dim = 265
    tt_rank = 32  # Start small (2, 4, or 8)
    tt_shape = [(10, 10), (10, 10)]
    hidden_dim = 128
    horizon = 150

    sequences = []
    for i in range(N_traj):
        # This samples a DIFFERENT starting point for every trajectory
        init = cfg["init_sampler"]() 
        traj = odeint(system_func, init, t).astype(np.float32)
        sequences.append(traj)

    # This now takes windows from ALL 50 trajectories
    subsequences = split_into_subsequences(sequences, subseq_len=subseq_len, stride=stride)
    loader = DataLoader(SequenceDataset(subsequences), batch_size=batch_size, shuffle=True)

    # 2. Models (Note: Use MLP for both for now to ensure rollout manifold is learned)
    encoder = ResidualMLP(cfg["state_dim"], 2 * latent_dim, [hidden_dim]*3).to(device)
    decoder = ResidualMLP(latent_dim, cfg["state_dim"], [hidden_dim]*3).to(device)
   
    #decoder = LinearMatrix(latent_dim, cfg["state_dim"]).to(device)
   

    # Replace LinearMatrix with TTKoopman
    koopman_matrix = TTKoopman(
        latent_dim=latent_dim, 
        tt_rank=tt_rank, 
        tt_shape=tt_shape
    ).to(device)

    koopman_matrix = LinearMatrix(latent_dim, latent_dim).to(device)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(koopman_matrix.parameters()),
        lr=1e-3
    )

    logger = InfoVectorLogger(log_dir=f"logs/Multihop_KVAE_2_{DATASET}")

    # 3. Trainer
    trainer = KoopmanVAETrainer(
        encoder=encoder,
        decoder=decoder,
        koopman_matrix=koopman_matrix,
        dataloader=loader,
        optimizer=optimizer,
        latent_dim=latent_dim,
        horizon=horizon,      # Supervise 30 steps of the rollout
        beta=1e-4,       # Low KL to keep latent space very tight
        gamma=10.0,      # High Koopman weight to prioritize dynamics
        delta=1e-2,
        device=device,
        logger=logger,
        save_epoch=100
    )

    print("Starting Joint Training (No Stages)...")
    trainer.train(20000)

if __name__ == "__main__":
    main()

