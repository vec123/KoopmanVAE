# loaders.py
import os
import numpy as np
import pandas as pd

class TrajectoryLoader:
    """PHASE 1: Loads CSVs from disk into Pandas DataFrames"""
    def __init__(self, state_cols, control_cols, forcing_cols=None, time_col='time', verbose = False):
        self.state_cols = state_cols
        self.control_cols = control_cols
        self.forcing_cols = forcing_cols or []
        self.time_col = time_col
        self.verbose = verbose
    def fetch_data(self, directories):
        import glob
        all_dfs = []
        for d in directories:
            if  self.verbose:
                print("TrajectoryLoader loading from directoy: ", d)
            files = glob.glob(os.path.join(d, "*.csv"))
            if self.verbose:
                print("TrajectoryLoader found files from directoy: ", files)
            for f in sorted(files):
                all_dfs.append(pd.read_csv(f))
        return all_dfs

    def split_trajectories(self, trajectories, val_ratio=0.2, split_mode='random'):
        if split_mode == 'random':
            import random
            random.shuffle(trajectories)
        split_idx = int(len(trajectories) * (1 - val_ratio))
        return trajectories[:split_idx], trajectories[split_idx:]

class NativeLoader:
    """PHASE 2: Iterates over memory to create Mini-Batches (Numpy/Jax)"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.samples = dataset.samples 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backend = dataset.engine 

    def __iter__(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            
            # Use the backend's stack method (e.g., np.stack or jnp.stack)
            batch_x = self.backend.stack([self.samples[i]['x'] for i in batch_indices])
            batch_u = self.backend.stack([self.samples[i]['u'] for i in batch_indices])
            batch_f = self.backend.stack([self.samples[i]['f'] for i in batch_indices])
            
            yield batch_x, batch_u, batch_f

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size