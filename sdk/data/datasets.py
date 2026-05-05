import torch
import numpy as np
from torch.utils.data import Dataset
from backends import TorchBackend, JaxBackend, NumpyBackend

class ControlledWindowDataset(Dataset):
    def __init__(self, trajectories, state_cols, control_cols, forcing_cols, 
                 subseq_len, stride, scaler=None, backend="torch"):
        
        # Select backend
        if backend == "torch":
            self.engine = TorchBackend()
        elif backend == "jax":
            self.engine = JaxBackend()
        else:
            self.engine = NumpyBackend()

        self.samples = []

        for df in trajectories:
            # Extract raw values - fallback to empty (T, 0) arrays if list is empty
            x_raw = df[state_cols].values
            u_raw = df[control_cols].values if control_cols else np.empty((len(x_raw), 0))
            f_raw = df[forcing_cols].values if forcing_cols else np.empty((len(x_raw), 0))
            
            # Apply individual scaling
            if scaler and scaler.is_fitted:
                x_proc = scaler.x_scaler.transform(x_raw)
                # Only scale if columns actually exist
                u_proc = scaler.u_scaler.transform(u_raw) if u_raw.shape[1] > 0 else u_raw
                f_proc = scaler.f_scaler.transform(f_raw) if f_raw.shape[1] > 0 else f_raw
            else:
                x_proc, u_proc, f_proc = x_raw, u_raw, f_raw
            
            # Sliding window generation
            num_windows = (len(x_proc) - subseq_len) // stride + 1
            for i in range(0, num_windows * stride, stride):
                self.samples.append({
                                    'x': self.engine.convert(x_proc[i : i + subseq_len]),
                                    'u': self.engine.convert(u_proc[i : i + subseq_len]),
                                    'f': self.engine.convert(f_proc[i : i + subseq_len])
                                })
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['x'], s['u'], s['f']