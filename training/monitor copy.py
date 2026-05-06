import os
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from dataclasses import asdict
from conf.config import KoopmanConfig

class KoopmanMonitor:
    def __init__(self, cfg: KoopmanConfig):
        self.cfg = cfg
        # Path management using typed config properties
        self.run_dir = os.path.join(cfg.checkpoint_dir, cfg.exp_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.history = []
        # Save a copy of the config as a "manifest" for the run
        self._save_config_manifest()

    def _save_config_manifest(self):
        manifest_path = os.path.join(self.run_dir, "config_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)

    def log_metrics(self, epoch: int, metrics: dict):
        log_entry = {"epoch": epoch, **metrics}
        self.history.append(log_entry)
        
        # Save history incrementally
        history_path = os.path.join(self.run_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def plot_loss_progression(self):
        if not self.history: return
        
        plt.figure(figsize=(10, 5))
        epochs = [h['epoch'] for h in self.history]
        
        # Plot all loss components found in metrics
        for key in self.history[0].keys():
            if key.startswith("loss_"):
                values = [h[key] for h in self.history]
                plt.plot(epochs, values, label=key.replace("loss_", ""))
        
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Loss")
        plt.title(f"Training Progression: {self.cfg.exp_name}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(self.run_dir, "loss_curves.png"))
        plt.close()

    
    def save_checkpoint(self, models, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_states': {k: v.state_dict() for k, v in models.items() if v is not None},
            'config': asdict(self.cfg)
        }
        
        name = "best_model.pt" if is_best else f"checkpoint_ep{epoch}.pt"
        torch.save(checkpoint, os.path.join(self.run_dir, name))


    def plot_loss_curves(self):
        if not self.history: return
        
        # 1. Prepare Data
        epochs = [h['epoch'] for h in self.history]
        total_train = [sum(v for k, v in h.items() if k.startswith('train_loss_')) for h in self.history]
        total_val = [sum(v for k, v in h.items() if k.startswith('val_loss_')) for h in self.history]

        # 2. Setup Figure (3 Rows: Shared, Train Only, Val Only)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # --- Subplot 0: Combined (Shared Scale) ---
        axes[0].plot(epochs, total_train, 'b-', label='Train Total', alpha=0.6)
        axes[0].plot(epochs, total_val, 'r--', label='Val Total', alpha=0.8)
        axes[0].set_yscale('log')
        axes[0].set_title("System Progression (Shared Log Scale)")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.2)

        # --- Subplot 1: Train Only (Individual Scale) ---
        axes[1].plot(epochs, total_train, 'b-o', markersize=3)
        axes[1].set_title("Training Detail (Local Scale)")
        axes[1].set_ylabel("Train Loss")
        axes[1].grid(True, alpha=0.3)
        # Force local scaling to see the "bounciness"
        if total_train:
            axes[1].set_ylim(min(total_train)*0.95, max(total_train)*1.05)

        # --- Subplot 2: Val Only (Individual Scale) ---
        axes[2].plot(epochs, total_val, 'r--s', markersize=3)
        axes[2].set_title("Validation Detail (Local Scale)")
        axes[2].set_ylabel("Val Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)
        # Force local scaling
        if total_val:
            axes[2].set_ylim(min(total_val)*0.95, max(total_val)*1.05)

        plt.tight_layout()
        plt.savefig("loss_curves.png")
        plt.close(fig)

    def plot_rollout(self, x_true, x_rec, epoch, prefix="val", u_true=None, u_rec=None, e_true=None):
        """Plots states, and optionally controls and forcing impact."""
        os.makedirs("plots", exist_ok=True)

        # 1. Prepare State Data
        true_np = x_true[0].detach().cpu().numpy()
        rec_np = x_rec[0].detach().cpu().numpy()
        num_x = true_np.shape[-1]
        
        # 2. Determine Control and Forcing rows
        u_np = u_true[0].detach().cpu().numpy() if u_true is not None else None
        u_rec_np = u_rec[0].detach().cpu().numpy() if u_rec is not None else None
        num_u = u_np.shape[-1] if u_np is not None else 0
        
        # Handle Forcing Mode
        e_np = e_true[0].detach().cpu().numpy() if e_true is not None else None
        plot_e_mode = getattr(self.cfg.train, 'plot_forcing_mode', 'magnitude')
        num_e = 0
        if e_np is not None:
            num_e = 1 if plot_e_mode == 'magnitude' else e_np.shape[-1]

        total_rows = num_x + num_u + num_e
        fig, axes = plt.subplots(total_rows, 1, figsize=(12, 2 * total_rows), sharex=True)
        if total_rows == 1: axes = [axes]

        # --- A. Plot States (X) ---
        x_labels = getattr(self.cfg.data, 'state_labels', [])
        for i in range(num_x):
            axes[i].plot(true_np[:, i], 'k-', label='True State', alpha=0.7)
            axes[i].plot(rec_np[:, i], 'r--', label='Koopman Rec')
            ylabel = x_labels[i] if i < len(x_labels) else f"X_{i}"
            axes[i].set_ylabel(ylabel)
            if i == 0: axes[i].legend(loc='upper right')

        # --- B. Plot Controls (U) ---
        if num_u > 0:
            u_labels = getattr(self.cfg.data, 'control_labels', [])
            for j in range(num_u):
                idx = num_x + j
                axes[idx].step(range(len(u_np)), u_np[:, j], 'g-', label='True Control', where='post')
                if u_rec_np is not None:
                    axes[idx].step(range(len(u_rec_np)), u_rec_np[:, j], 'm--', label='Rec Control', where='post')
                
                ylabel = u_labels[j] if j < len(u_labels) else f"U_{j}"
                axes[idx].set_ylabel(ylabel)

        # --- C. Plot Forcing (E) ---
        if num_e > 0:
            start_idx = num_x + num_u
            if plot_e_mode == 'magnitude':
                e_mag = np.linalg.norm(e_np, axis=-1)
                axes[start_idx].fill_between(range(len(e_mag)), e_mag, color='orange', alpha=0.3)
                axes[start_idx].plot(e_mag, color='darkorange', label='Forcing Magnitude')
                axes[start_idx].set_ylabel("|E|_2")
            else:
                for k in range(e_np.shape[-1]):
                    axes[start_idx + k].plot(e_np[:, k], color='orange', label=f'E_{k}')
                    axes[start_idx + k].set_ylabel(f"E_{k}")

        plt.suptitle(f"Epoch {epoch:03d} | {prefix.upper()} Rollout")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots/{prefix}_ep{epoch:03d}.png")
        plt.close(fig)