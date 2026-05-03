
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset

def save_training_trajectories(X_raw, U_raw, labels, dataset_name, save_dir="plots", num_figs=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_traj, T_x, dx = X_raw.shape
    # Get T_u specifically from the control array
    T_u = U_raw.shape[1]
    du = U_raw.shape[2]
    
    indices = np.random.choice(n_traj, min(num_figs, n_traj), replace=False)

    for idx in indices:
        fig, axes = plt.subplots(dx + du, 1, figsize=(10, 2 * (dx + du)), sharex=True)
        fig.suptitle(f"Training Trajectory {idx} - {dataset_name}", fontsize=14)

        # 1. Plot States
        for i in range(dx):
            # Use the actual length of the state data for the x-axis
            axes[i].plot(range(T_x), X_raw[idx, :, i], color='tab:blue', linewidth=1.5)
            # Safe label access
            label = labels[i] if i < len(labels) else f"State $x_{i}$"
            axes[i].set_ylabel(label)
            axes[i].grid(True, alpha=0.3)

        # 2. Plot Controls
        for j in range(du):
            ax_u = axes[dx + j] # This moves to the 5th and 6th slots
            ax_u.step(range(T_u), U_raw[idx, :, j], color='tab:red', where='post', linewidth=1.5)
            ax_u.set_ylabel(f"Control $u_{j}$")
            ax_u.grid(True, alpha=0.3)
                    
            # Only show "Time Steps" on the very last subplot
            if j == du - 1:
                ax_u.set_xlabel("Time Steps")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(save_dir, f"train_traj_{dataset_name}_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved training trajectory plot to: {save_path}")

def split_into_subsequences_controlled(x_seqs, u_seqs, subseq_len, stride=1):
    x_sub, u_sub = [], []
    for x_traj, u_traj in zip(x_seqs, u_seqs):
        T = x_traj.shape[0]
        for start in range(0, T - subseq_len + 1, stride):
            x_sub.append(x_traj[start:start + subseq_len])
            u_sub.append(u_traj[start:start + subseq_len])
    return x_sub, u_sub

def apply_trigonometric_embedding(X, is_batch=True):
        """
        Transforms state from [x, x_dot, theta, theta_dot] 
        to [x, x_dot, sin(theta), cos(theta), theta_dot].
        
        Args:
            X: np.ndarray of shape (N, T, 4) if is_batch=True, 
            or (4,) if is_batch=False.
        Returns:
            np.ndarray of shape (N, T, 5) or (5,).
        """
        if is_batch:
            # X shape: (N, T, 4)
            x_pos     = X[:, :, 0:1]
            x_dot     = X[:, :, 1:2]
            theta     = X[:, :, 2:3]
            theta_dot = X[:, :, 3:4]
            
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            
            return np.concatenate([x_pos, x_dot, sin_t, cos_t, theta_dot], axis=-1)
        
        else:
            # X shape: (4,)
            x, x_dot, theta, theta_dot = X
            return np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])
        
class ControlledSequenceDataset(Dataset):
    def __init__(self, x_sequences, u_sequences):
        self.x = [torch.tensor(s, dtype=torch.float32) for s in x_sequences]
        self.u = [torch.tensor(s, dtype=torch.float32) for s in u_sequences]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.u[idx]

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
