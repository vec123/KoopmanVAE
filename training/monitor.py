import os
import psutil
import json
import torch
import threading
import pandas as pd
from multiprocessing import Process
from dataclasses import asdict
import random

import matplotlib
matplotlib.use('Agg') # Essential for background processes
import matplotlib.pyplot as plt

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
        """Threaded JSON logging to avoid disk wait."""
        log_entry = {"epoch": epoch, **metrics}
        self.history.append(log_entry)
        
        def _save_json(hist, path):
            with open(path, "w") as f:
                json.dump(hist, f, indent=4)
        
        threading.Thread(
            target=_save_json, 
            args=(list(self.history), os.path.join(self.run_dir, "history.json")), 
            daemon=True
        ).start()

    def save_checkpoint(self, models, epoch, is_best=False):
        """
        Asynchronous Checkpointing: 
        Copies weights instantly, saves to disk in background.
        """
        # 1. Capture state_dicts IMMEDIATELY to avoid saving 'future' weights
        state_dicts = {k: v.state_dict() for k, v in models.items() if v is not None}
        
        checkpoint = {
            'epoch': epoch,
            'model_states': state_dicts,
        }
        
        name = "best_model.pt" if is_best else f"checkpoint_ep{epoch}.pt"
        save_path = os.path.join(self.run_dir, name)

        def _do_save(data, path):
            try:
                torch.save(data, path)
            except Exception as e:
                print(f"\n[CRITICAL] Checkpoint Save Failed: {e}")

        # Use a Thread for disk I/O so the trainer can keep moving
        threading.Thread(target=_do_save, args=(checkpoint, save_path), daemon=True).start()

    def plot_loss_curves(self):
        """Isolated Process for Loss Plotting to bypass GIL."""
        if not self.history: return
        df = pd.DataFrame(self.history)
        
        p = Process(target=self._exec_loss_plot_isolated, args=(df, self.run_dir))
        p.start()


    def plot_rollout(self, x_true, x_rec, epoch, prefix="val", **kwargs):
        """Isolated Process for Rollout Plotting with Random Sampling."""
        # 1. Determine the batch size and pick a random index
        batch_size = x_true.size(0)
        idx = random.randint(0, batch_size - 1)
        
        # 2. Move only the selected random trajectory to CPU
        data = {
            'x_true': x_true[idx].detach().cpu().numpy(),
            'x_rec': x_rec[idx].detach().cpu().numpy(),
            'u_true': kwargs['u_true'][idx].detach().cpu().numpy() if kwargs.get('u_true') is not None else None,
            'f_true': kwargs['f_true'][idx].detach().cpu().numpy() if kwargs.get('f_true') is not None else None,
            'epoch': epoch,
            'prefix': prefix,
            'plot_dir': self.plot_dir,
            'sample_idx': idx  # Useful to log which sample was picked
        }
        
        # 3. Spawn the isolated plotting process
        p = Process(target=self._exec_rollout_isolated, args=(data,))
        p.start()

    @staticmethod
    def _exec_loss_plot_isolated(df, run_dir):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        p = psutil.Process(os.getpid())
        try:
            p.nice(-10) # Lower niceness = Higher priority (Linux/Mac)
        except: pass # Fallback for Windows or permission limits

        try:
            epochs = df['epoch'].values
            t_cols = [c for c in df.columns if 'train_loss_' in c]
            v_cols = [c for c in df.columns if 'val_loss_' in c]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            for ax in axes:
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(style='plain', axis='y')

            axes[0].plot(epochs, df[t_cols].sum(axis=1), 'b-', label='Train')
            axes[0].plot(epochs, df[v_cols].sum(axis=1), 'r--', label='Val')
            axes[0].set_yscale('log')
            axes[0].set_title("Total Loss (Log Scale)")
            axes[0].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=150)
            plt.close('all')
        except Exception as e:
            print(f"Loss Plot Process Error: {e}")

    @staticmethod
    def _exec_rollout_isolated(d):
        p = psutil.Process(os.getpid())
        try: p.nice(-10) 
        except: pass

        import matplotlib.pyplot as plt
        try:
            x_true, x_rec = d['x_true'], d['x_rec']
            u_true, f_true = d['u_true'], d['f_true']
            
            # Determine grid requirements
            x_dims = x_true.shape[-1]
            u_dims = u_true.shape[-1] if u_true is not None else 0
            f_dims = f_true.shape[-1] if f_true is not None else 0
            total_dims = x_dims + u_dims + f_dims

            fig, axes = plt.subplots(total_dims, 1, figsize=(12, 2.0 * total_dims), sharex=True)
            if total_dims == 1: axes = [axes]

            curr_idx = 0

            # 1. Plot States (True vs Rec)
            for i in range(x_dims):
                ax = axes[curr_idx]
                ax.plot(x_true[:, i], 'k-', alpha=0.6, label='True State' if i==0 else "")
                ax.plot(x_rec[:, i], 'r--', label='Rec State' if i==0 else "")
                ax.set_ylabel(f"State {i}")
                if i == 0: ax.legend(loc='upper right')
                curr_idx += 1

            # 2. Plot Controls (if exists)
            if u_true is not None:
                for i in range(u_dims):
                    ax = axes[curr_idx]
                    ax.plot(u_true[:, i], 'g-', label='Control' if i==0 else "")
                    ax.set_ylabel(f"Control {i}")
                    ax.set_facecolor('#f9fff9') # Subtle green tint for inputs
                    if i == 0: ax.legend(loc='upper right')
                    curr_idx += 1

            # 3. Plot Forcing (if exists)
            if f_true is not None:
                for i in range(f_dims):
                    ax = axes[curr_idx]
                    ax.plot(f_true[:, i], 'm-', label='Forcing' if i==0 else "")
                    ax.set_ylabel(f"Forcing {i}")
                    ax.set_facecolor('#fff9ff') # Subtle purple tint for external
                    if i == 0: ax.legend(loc='upper right')
                    curr_idx += 1

            plt.xlabel("Time Steps")
            plt.suptitle(f"Epoch {d['epoch']} | {d['prefix'].upper()} | State Reconstruction & Inputs")
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
            
            save_path = os.path.join(d['plot_dir'], f"{d['prefix']}_ep{d['epoch']:03d}.png")
            plt.savefig(save_path, dpi=120)
            plt.close('all')
            
        except Exception as e:
            print(f"Rollout Plot Process Error: {e}")