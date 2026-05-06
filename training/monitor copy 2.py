import os
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process
import threading
from dataclasses import asdict

# Use 'Agg' to ensure no GUI overhead
import matplotlib
matplotlib.use('Agg')

class KoopmanMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir = os.path.join(cfg.checkpoint_dir, cfg.exp_name)
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.history = []
        self._save_config_manifest()

    def _save_config_manifest(self):
        manifest_path = os.path.join(self.run_dir, "config_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)

    def log_metrics(self, epoch: int, metrics: dict):
        """Still uses threading for I/O (JSON is light)."""
        log_entry = {"epoch": epoch, **metrics}
        self.history.append(log_entry)
        
        def save_history():
            with open(os.path.join(self.run_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=4)
        
        threading.Thread(target=save_history, daemon=True).start()

    def plot_loss_curves(self):
        """Launches a separate PROCESS to bypass the GIL."""
        if not self.history: return
        df = pd.DataFrame(self.history)
        
        # We spawn a Process, not a Thread
        p = Process(target=self._exec_loss_plot_standalone, args=(df, self.run_dir))
        p.start()

    @staticmethod
    def _exec_loss_plot_standalone(df, run_dir):
        """
        Static method because Multiprocessing must 'pickle' the target. 
        It cannot easily pickle 'self' if it contains complex torch objects.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        
        try:
            epochs = df['epoch'].values
            train_cols = [c for c in df.columns if 'train_loss_' in c]
            val_cols = [c for c in df.columns if 'val_loss_' in c]
            
            t_total = df[train_cols].sum(axis=1).values
            v_total = df[val_cols].sum(axis=1).values

            fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            for ax in axes:
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(style='plain', axis='y')

            axes[0].plot(epochs, t_total, 'b-', label='Train')
            axes[0].plot(epochs, v_total, 'r--', label='Val')
            axes[0].set_yscale('log')
            axes[0].set_title("Total Loss (Log)")
            axes[0].legend()

            axes[1].plot(epochs, t_total, 'b-')
            axes[1].set_title("Train Detail (Linear)")

            axes[2].plot(epochs, v_total, 'r--')
            axes[2].set_title("Val Detail (Linear)")

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"Background Plotting Error: {e}")

    def plot_rollout(self, x_true, x_rec, epoch, prefix="val", **kwargs):
        """Launches a separate PROCESS for heavy image rendering."""
        # Immediate CPU move to free up GPU memory before the process starts
        data = {
            'x_true': x_true[0].detach().cpu().numpy(),
            'x_rec': x_rec[0].detach().cpu().numpy(),
            'u_true': kwargs['u_true'][0].detach().cpu().numpy() if kwargs.get('u_true') is not None else None,
            'epoch': epoch,
            'prefix': prefix,
            'plot_dir': self.plot_dir
        }

        p = Process(target=self._exec_rollout_standalone, args=(data,))
        p.start()

    @staticmethod
    def _exec_rollout_standalone(d):
        import matplotlib.pyplot as plt
        try:
            true, rec = d['x_true'], d['x_rec']
            fig, axes = plt.subplots(true.shape[-1], 1, figsize=(10, 2*true.shape[-1]), sharex=True)
            if true.shape[-1] == 1: axes = [axes]
            
            for i in range(true.shape[-1]):
                axes[i].plot(true[:, i], 'k', alpha=0.5)
                axes[i].plot(rec[:, i], 'r--')
                axes[i].set_ylabel(f"Dim {i}")
            
            plt.tight_layout()
            path = os.path.join(d['plot_dir'], f"{d['prefix']}_ep{d['epoch']:03d}.png")
            plt.savefig(path)
            plt.close(fig)
        except Exception as e:
            print(f"Rollout Plotting Error: {e}")

    def save_checkpoint(self, models, epoch, is_best=False):
        # Checkpointing should remain in the main thread/thread to ensure 
        # it finishes before the next weight update
        checkpoint = {
            'epoch': epoch,
            'model_states': {k: v.state_dict() for k, v in models.items() if v is not None}
        }
        name = "best_model.pt" if is_best else f"checkpoint_ep{epoch}.pt"
        torch.save(checkpoint, os.path.join(self.run_dir, name))