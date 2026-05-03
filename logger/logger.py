# logger.py
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

class VectorLogger:
    def __init__(self, log_dir="logs", display_images=False, num_samples=8):
        self.log_dir = log_dir
        self.num_samples = num_samples
        os.makedirs(log_dir, exist_ok=True)

        self.d_losses = []
        self.g_losses = []
        self.display_images = display_images

    # -----------------------------
    def log_loss(self, d_loss, g_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)

    # -----------------------------
    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.d_losses, label="D Loss")
        plt.plot(self.g_losses, label="G Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Progression")
        path = os.path.join(self.log_dir, "loss.png")
        plt.savefig(path)
        if self.display_images:
            plt.show()
        plt.close()

    # -----------------------------
    def log_latent(self, z, x_input, x_output, step=None):
        """
        z: latent vectors
        x_input: real vectors
        x_output: generated vectors
        """
        x_output = x_output.squeeze(1)
        if self.num_samples is not None:
            z = z[:self.num_samples]
            x_input = x_input[:self.num_samples]
            x_output = x_output[:self.num_samples]

        # Save latent info
        latent_path = os.path.join(
            self.log_dir, f"latents_step_{step}.pt" if step else "latents.pt"
        )
        torch.save({"z": z, "x_input": x_input, "x_output": x_output}, latent_path)

        # Save plots only if vectors are 2D
        if x_output.ndim == 2 and x_output.shape[1] == 2:
            self._save_vector_scatter(x_output, f"generated_step_{step}" if step else "generated")
        if x_input.ndim == 2 and x_input.shape[1] == 2:
            self._save_vector_scatter(x_input, f"real_step_{step}" if step else "real")

    # -----------------------------
    def _save_vector_scatter(self, vectors, name):
        if isinstance(vectors, torch.Tensor):
            if vectors.requires_grad:
                vectors = vectors.detach()
            vectors = vectors.cpu().numpy()

        vectors = vectors.squeeze()

        # Only plot if 2D
        if vectors.ndim != 2 or vectors.shape[1] != 2:
            print(f"Skipping plot for {name}, shape={vectors.shape}")
            return

        plt.figure(figsize=(5, 5))
        plt.scatter(vectors[:, 0], vectors[:, 1], alpha=0.6)
        plt.title(name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)

        path = os.path.join(self.log_dir, f"{name}.png")
        plt.savefig(path)
        if self.display_images:
            plt.show()
        plt.close()

    # -----------------------------
    def save_models(self, models_dict, step):
        for name, model in models_dict.items():
            if model is None:
                continue
            path = os.path.join(self.log_dir, f"{name}_{step}.pt")
            torch.save(model.state_dict(), path)

    # -----------------------------
    def save_loss(self):
        torch.save(
            {"d_losses": self.d_losses, "g_losses": self.g_losses},
            os.path.join(self.log_dir, "losses.pt"),
        )



class InfoVectorLogger:
    def __init__(self, log_dir="logs", labels=None, display_images=False):
        self.log_dir = log_dir
        self.labels = labels  # This should be config[system_type]["labels"]
        os.makedirs(log_dir, exist_ok=True)
        self.scalars = {}
        self.display_images = display_images

    def log_scalars(self, scalars_dict, step=None):
        """Store and print multiple scalar values per step"""
        for key, value in scalars_dict.items():
            if key not in self.scalars:
                self.scalars[key] = []
            self.scalars[key].append((step, value))

        if step is not None:
            # Create a clean string for console output
            scalars_str = " | ".join(f"{k}: {v:.4f}" for k, v in scalars_dict.items())
            print(f"[Epoch {step}] {scalars_str}")
    
    def save_models(self, models_dict, step):
        """Save models and Koopman matrix"""
        for name, model in models_dict.items():
            if model is None:
                continue
            path = os.path.join(self.log_dir, f"{name}_{step}.pt")
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), path)
            elif isinstance(model, torch.Tensor):
                torch.save(model, path)
            else:
                print(f"Skipping {name}, type {type(model)} cannot be saved")

    def save_trajectories(self, z_traj, x_rec, x_true=None, step=None):
        def ensure_numpy(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return t

        z_traj = ensure_numpy(z_traj)
        x_rec = ensure_numpy(x_rec)
        if x_true is not None:
            x_true = ensure_numpy(x_true)

        # Fix dimensions: [Batch, Time, Dim] -> [Time, Dim]
        if z_traj.ndim == 3: z_traj = z_traj[0]
        if x_rec.ndim == 3: x_rec = x_rec[0]
        if x_true is not None and x_true.ndim == 3: x_true = x_true[0]

        """ 
        Dz_plot = min(z_traj.shape[1], 8) 
        fig, axes = plt.subplots(Dz_plot, 1, figsize=(10, 1.5 * Dz_plot), sharex=True, squeeze=False)
        for i in range(Dz_plot):
            axes[i, 0].plot(z_traj[:, i], color='tab:blue')
            axes[i, 0].set_ylabel(f"$z_{{{i}}}$")
            axes[i, 0].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"latent_traj_{step}.png"))
        plt.close()
        """
        # 2. Reconstructed vs True (State Dimensions)
        Dx = x_rec.shape[1]
        fig, axes = plt.subplots(Dx, 1, figsize=(10, 2.5 * Dx), sharex=True, squeeze=False)
        
        for i in range(Dx):
            axes[i, 0].plot(x_rec[:, i], label="Reconstructed", color='tab:red', alpha=0.8)
            if x_true is not None:
                axes[i, 0].plot(x_true[:, i], 'k--', label="True", alpha=0.6)
            
            # DYNAMIC LABELS: Use provided labels or fallback to x_i
            ylabel = self.labels[i] if (self.labels and i < len(self.labels)) else f"State {i}"
            axes[i, 0].set_ylabel(ylabel)
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0: axes[i, 0].legend()

        plt.suptitle(f"System State Reconstruction - Step {step}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.log_dir, f"rec_vs_true_{step}.png"))
        plt.close()


    def save_trajectories_w_control(self, z_traj, x_rec, x_true=None, u_true=None, u_rec=None, step=None, prefix=""):
        def ensure_numpy(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return t

        z_traj = ensure_numpy(z_traj)
        x_rec = ensure_numpy(x_rec)
        x_true = ensure_numpy(x_true)
        u_true = ensure_numpy(u_true)
        u_rec = ensure_numpy(u_rec)

        # Remove batch dimension: [1, T, D] -> [T, D]
        if z_traj.ndim == 3: z_traj = z_traj[0]
        if x_rec.ndim == 3: x_rec = x_rec[0]
        if x_true is not None and x_true.ndim == 3: x_true = x_true[0]
        if u_true is not None and u_true.ndim == 3: u_true = u_true[0]
        if u_rec is not None and u_rec.ndim == 3: u_rec = u_rec[0]

        Dx = x_rec.shape[1]
        Du = u_true.shape[1] if u_true is not None else 0
        total_rows = Dx + Du

        fig, axes = plt.subplots(total_rows, 1, figsize=(10, 2.5 * total_rows), sharex=True, squeeze=False)

        # 1. Plot States
        for i in range(Dx):
            if x_true is not None:
                axes[i, 0].plot(x_true[:, i], 'k--', label="True State", alpha=0.6)
            axes[i, 0].plot(x_rec[:, i], label="Rec State", color='tab:red', linewidth=1.5)
            
            ylabel = self.labels[i] if (hasattr(self, 'labels') and self.labels and i < len(self.labels)) else f"State {i}"
            axes[i, 0].set_ylabel(ylabel)
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0: axes[i, 0].legend(loc='upper right')

        # 2. Plot Controls (True vs Reconstructed)
        if u_true is not None:
            for j in range(Du):
                ax_idx = Dx + j
                # Ground Truth Control
                axes[ax_idx, 0].step(range(len(u_true)), u_true[:, j], 'k--', where='post', label="U True", alpha=0.5)
                
                # Reconstructed Control
                if u_rec is not None:
                    axes[ax_idx, 0].step(range(len(u_rec)), u_rec[:, j], color='tab:green', where='post', label="U Rec")
                
                axes[ax_idx, 0].set_ylabel(f"Control $u_{{{j}}}$")
                axes[ax_idx, 0].grid(True, alpha=0.3)
                if j == 0: axes[ax_idx, 0].legend(loc='upper right')

        plt.xlabel("Time Steps")
        plt.suptitle(f"{prefix.upper()} Rollout Analysis - Step {step}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        save_path = os.path.join(self.log_dir, f"{prefix}_combined_rollout_{step}.png")
        plt.savefig(save_path)
        plt.close()

    def save_trajectories_w_control_and_forcing(self, z_traj, x_rec, x_true=None, u_true=None, u_rec=None, f_true=None, step=None, prefix=""):
        def ensure_numpy(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return t

        z_traj = ensure_numpy(z_traj)
        x_rec = ensure_numpy(x_rec)
        x_true = ensure_numpy(x_true)
        u_true = ensure_numpy(u_true)
        u_rec = ensure_numpy(u_rec)
        f_true = ensure_numpy(f_true) # New forcing array

        # Remove batch dimension: [1, T, D] -> [T, D]
        if z_traj.ndim == 3: z_traj = z_traj[0]
        if x_rec.ndim == 3: x_rec = x_rec[0]
        if x_true is not None and x_true.ndim == 3: x_true = x_true[0]
        if u_true is not None and u_true.ndim == 3: u_true = u_true[0]
        if u_rec is not None and u_rec.ndim == 3: u_rec = u_rec[0]
        if f_true is not None and f_true.ndim == 3: f_true = f_true[0]

        Dx = x_rec.shape[1]
        Du = u_true.shape[1] if u_true is not None else 0
        Df = f_true.shape[1] if f_true is not None else 0
        
        # New total rows include States + Controls + Forcing
        total_rows = Dx + Du + Df

        fig, axes = plt.subplots(total_rows, 1, figsize=(10, 2.5 * total_rows), sharex=True, squeeze=False)

        # 1. Plot States
        for i in range(Dx):
            if x_true is not None:
                axes[i, 0].plot(x_true[:, i], 'k--', label="True State", alpha=0.6)
            axes[i, 0].plot(x_rec[:, i], label="Rec State", color='tab:red', linewidth=1.5)
            
            ylabel = self.labels[i] if (hasattr(self, 'labels') and self.labels and i < len(self.labels)) else f"State {i}"
            axes[i, 0].set_ylabel(ylabel)
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0: axes[i, 0].legend(loc='upper right')

        # 2. Plot Controls (True vs Reconstructed)
        if u_true is not None:
            for j in range(Du):
                ax_idx = Dx + j
                axes[ax_idx, 0].step(range(len(u_true)), u_true[:, j], 'k--', where='post', label="U True", alpha=0.5)
                
                if u_rec is not None:
                    axes[ax_idx, 0].step(range(len(u_rec)), u_rec[:, j], color='tab:green', where='post', label="U Rec")
                
                axes[ax_idx, 0].set_ylabel(f"Control $u_{{{j}}}$")
                axes[ax_idx, 0].grid(True, alpha=0.3)
                if j == 0: axes[ax_idx, 0].legend(loc='upper right')

        # 3. Plot External Forcing
        if f_true is not None:
            for k in range(Df):
                ax_idx = Dx + Du + k
                # Forcing is usually continuous (like temp), so we use plot instead of step
                axes[ax_idx, 0].plot(f_true[:, k], color='tab:blue', linewidth=1.5, label="Ext Forcing")
                
                ylabel = f"Forcing $f_{{{k}}}$"
                # If you have specific names for forcing (like 'Temp'), you could map them here
                axes[ax_idx, 0].set_ylabel(ylabel)
                axes[ax_idx, 0].grid(True, alpha=0.3)
                if k == 0: axes[ax_idx, 0].legend(loc='upper right')

        plt.xlabel("Time Steps")
        plt.suptitle(f"{prefix.upper()} Rollout Analysis - Step {step}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        save_path = os.path.join(self.log_dir, f"{prefix}_combined_rollout_{step}.png")
        plt.savefig(save_path)
        plt.close()
     