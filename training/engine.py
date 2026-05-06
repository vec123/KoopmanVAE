import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from training.monitor import KoopmanMonitor
from training.losses import KoopmanLossManager
from torch_ema import ExponentialMovingAverage

import time

torch.set_float32_matmul_precision('high')

class KoopmanTrainer:
    def __init__(self, system_bundle, config, dataloaders, device="cuda"):
        self.device = device
        self.cfg = config # Standardizing name to match monitor
        self.models = nn.ModuleDict(system_bundle).to(device)
        self.train_loader, self.val_loader = dataloaders
        
        self.loss_manager = KoopmanLossManager(config)
        self.monitor = KoopmanMonitor(config)
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=config.train.lr)
        
        # Fixed Attributes
        self.stochastic = getattr(self.cfg.train, 'stochastic', False)
        self.horizon = config.train.horizon
        self.val_interval = config.train.val_interval 
        self.save_interval = config.train.save_interval 
        self.concat_true = config.train.concat_true 
        self.ema = ExponentialMovingAverage(self.models.parameters(), decay=config.train.ema_decay) if config.train.use_ema else None
        
        self.best_val_loss = float('inf')

    @torch.compile(mode="reduce-overhead")
    def rollout(self, z, x_init, U, E, steps: int):
        B_batch, Dz = z.shape
        Dx = x_init.shape[-1]
        
        # Pre-fetch modules to avoid ModuleDict overhead in loop
        model_A = self.models['A']
        model_B = self.models['B'] if 'B' in self.models else None
        model_E = self.models['E'] if 'E' in self.models else None
        forcing_net = self.models['forcing_net'] if 'forcing_net' in self.models else None

        # Pre-allocate the entire output tensor
        z_preds = torch.empty((B_batch, steps + 1, Dz + Dx), device=z.device, dtype=z.dtype)
        
        if self.concat_true:
            current_state = torch.cat([z, x_init], dim=-1)
        else:
            current_state = z
        z_preds[:, 0] = current_state
        v_r_list = []

        for i in range(steps):
            #  Linear step
            next_state = model_A(current_state)
            
            #  Additive terms (Control/Forcing)
            if model_B is not None and U is not None:
                next_state += model_B(U[:, i])
            
            if model_E is not None and E is not None:
                next_state += model_E(E[:, i])
                
            if forcing_net is not None:
                v = forcing_net(current_state)
                v_r_list.append(v)
                next_state += v
                
            z_preds[:, i + 1] = next_state
            current_state = next_state # Feedback for next step
            
        return z_preds, v_r_list

    @torch.compile(mode="reduce-overhead")
    def rollout_solid(self, z, x_init, U, E, steps: int):
        B_batch, Dz = z.shape
        Dx = x_init.shape[-1]
        
        z_preds = torch.empty((B_batch, steps + 1, Dz + Dx), device=z.device, dtype=z.dtype)
        
        current_z = z
        current_x = x_init
        z_preds[:, 0] = torch.cat([current_z, current_x], dim=-1)

        # FIX: Access ModuleDict correctly
        model_A = self.models['A']
        model_B = self.models['B'] if 'B' in self.models else None
        model_E = self.models['E'] if 'E' in self.models else None
        forcing_net = self.models['forcing_net'] if 'forcing_net' in self.models else None

        v_r_list = []

        for i in range(steps):

            z_combined = torch.cat([current_z, current_x], dim=-1)
            
            # Apply Linear Dynamics
            z_next_combined = model_A(z_combined)
            
            # Apply Control (B)
            if model_B is not None and U is not None:
                z_next_combined = z_next_combined + model_B(U[:, i])
            
            # Apply External Forcing (E)
            if model_E is not None and E is not None:
                z_next_combined = z_next_combined + model_E(E[:, i])
                
            # Apply Internal Forcing (HAVOK)
            if forcing_net is not None:
                v = forcing_net(z_combined)
                v_r_list.append(v)
                z_next_combined = z_next_combined + v
                
            current_z = z_next_combined[:, :Dz]
            current_x = z_next_combined[:, Dz:] 
            
            z_preds[:, i + 1] = z_next_combined
            
        return z_preds, v_r_list

    def forward_logic(self, X, U, E, is_train=True, monitor_time = True):
        B, T, Dx = X.shape
        h = self.horizon if is_train else T - 1
        
        # Window slicing (Keep as is)
        if monitor_time:
            t0 = time.time()
        t_start = torch.randint(0, T - h, (1,)).item() if is_train else 0
        x_win = X[:, t_start : t_start + h + 1]
        u_win = U[:, t_start : t_start + h]
        e_win = E[:, t_start : t_start + h] if E is not None else None

        #  Parallel Processing: Encode states and controls
        if monitor_time:
            t1 = time.time()
        z_enc, mu, logstd = self.encode_states(x_win)
        u_eff, u_rec_3d = self._process_control(x_win, u_win) 

        # Rollout (Optimized version from previous discussion)
        # z_preds: [B, h+1, Dz+Dx]
        if monitor_time:
            t2 = time.time()
        z_preds, v_r = self.rollout(z_enc[:, 0], x_win[:, 0], u_eff, e_win, steps=h)

        # BATCHED DECODING (The Speedup)
        # Combine both paths into one large batch for the decoder
        z_vae_combined = torch.cat([z_enc, x_win], dim=-1) # [B, h+1, Dz+Dx]
        
        # Flatten both to [Batch * Time, Dim] and stack
        # Shape: [2 * B * (h+1), Dz+Dx]
        combined_latents = torch.cat([
            z_preds.reshape(-1, z_preds.shape[-1]),
            z_vae_combined.reshape(-1, z_vae_combined.shape[-1])
        ], dim=0)

        if monitor_time:
            t3 = time.time()
        # Single Decoder Call
        all_recs = self.models['decoder'](combined_latents)

        # Split the results back into their respective paths
        split_idx = B * (h + 1)
        x_rec_rollout_flat = all_recs[:split_idx]
        x_rec_vae_flat = all_recs[split_idx:]
        
        if monitor_time:
            t4 = time.time()
        # 5. Control Loss
        if u_rec_3d is not None:
            loss_ctrl = F.mse_loss(u_rec_3d, u_win)
        else:
            # Use a buffer or pre-allocated zero to avoid sync
            loss_ctrl = torch.tensor(0.0, device=self.device)

        if monitor_time:
            t4 = time.time()

        if monitor_time:
            if (t4-t0) > 0.05:
                print(f"Forward Lags: slicing: {t1-t0:.3f}s | encoding: {t2-t1:.3f}s | rollout: {t3-t2:.3f}s| recon rollout: {t4-t3:.3f}s")
            else:
                "forward fast"
        return {
            'z_preds': z_preds,
            'z_enc': z_vae_combined,
            'x_rec': x_rec_vae_flat, 
            'x_rec_3d': x_rec_rollout_flat.view(B, h + 1, Dx), 
            'x_true': x_win,
            'u_true': u_win,
            'f_true': e_win,
            'u_rec': u_rec_3d,
            'mu': mu, 
            'logstd': logstd, 
            'v_r': v_r,
            'A_op': self.models['A'],
            'encoder': self.models['encoder'], 
            'decoder': self.models['decoder'],
            'loss_control': loss_ctrl
        }

    def forward_logic_old(self, X, U, E, is_train=True):
        B, T, Dx = X.shape
        h = self.horizon if is_train else T - 1
        
        # Window slicing
        t_start = torch.randint(0, T - h, (1,)).item() if is_train else 0
        x_win = X[:, t_start : t_start + h + 1]
        u_win = U[:, t_start : t_start + h]
        e_win = E[:, t_start : t_start + h] if E is not None else None

        # 1. Encode states across the window
        z_enc, mu, logstd = self.encode_states(x_win)

        # 2. Process Control (Encode/Decode)
        # Returns u_eff [B, H, Lu] and u_rec_3d [B, H, Du]
        u_eff, u_rec_3d = self._process_control(x_win, u_win) 

        # 3. Rollout: Pass the first encoded frame and first raw frame
        # z_preds is [B, h+1, Dz + Dx]
        z_preds, v_r = self.rollout(z_enc[:, 0], x_win[:, 0], u_eff, e_win, steps=h)

        # 4. Decoding Logic
        # For Dynamics Path (rec_rollout): Already concatenated in rollout
        x_rec_rollout_flat = self.models['decoder'](z_preds.reshape(-1, z_preds.shape[-1]))
        
        # For VAE Path (rec_full): Manually concat z_enc and x_win to match rollout structure
        z_vae_combined = torch.cat([z_enc, x_win], dim=-1)
        x_rec_vae_flat = self.models['decoder'](z_vae_combined.reshape(-1, z_vae_combined.shape[-1]))
        
        # 5. Calculate Control Loss (if applicable)
        loss_ctrl = F.mse_loss(u_rec_3d, u_win) if u_rec_3d is not None else torch.tensor(0.0, device=self.device)

        return {
            'z_preds': z_preds,         # [B, h+1, Dz+Dx]
            'z_enc': z_vae_combined,    # [B, h+1, Dz+Dx]
            'x_rec': x_rec_vae_flat,    # Flattened for loss_manager
            'x_rec_3d': x_rec_rollout_flat.reshape(B, h + 1, Dx), 
            'x_true': x_win,
            'u_true': u_win,
            'u_rec': u_rec_3d,
            'mu': mu, 
            'logstd': logstd, 
            'v_r': v_r,
            'A_op': self.models['A'],
            'encoder': self.models['encoder'], 
            'decoder': self.models['decoder'],
            'loss_control': loss_ctrl
        }

    def train_step(self, X, U, E, monitor_time = True):
        if monitor_time:
            t0 = time.time()

        self.models.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            res = self.forward_logic(X, U, E, is_train=True)
            if monitor_time:
                t1 = time.time()
            loss, logs = self.loss_manager(res, res['v_r'], res['mu'], res['logstd'])
            if monitor_time:
                t2 = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if monitor_time:
                t3 = time.time()

        if monitor_time:
            if (t3-t0) > 0.2:
                print(f"Train Step Lags: Fwd: {t1-t0:.3f}s | Loss: {t2-t1:.3f}s | Opt: {t3-t2:.3f}s")
                
        if self.ema: 
            self.ema.update()
        return loss.item(), logs, res

    def train_epoch(self):
        epoch_loss, last_logs, last_res = 0, {}, None
        for batch in self.train_loader:
            X, U, E = [b.to(self.device, non_blocking=True) for b in batch]
            loss_val, logs, res = self.train_step(X, U, E )
            epoch_loss += loss_val
            last_logs, last_res = logs, res
        return epoch_loss / len(self.train_loader), last_logs, last_res

    def validate(self, epoch):
        self.models.eval()
        val_logs, last_res = {}, None
        with torch.no_grad():
            for batch in self.val_loader:
                X, U, E = [b.to(self.device, non_blocking=True) for b in batch]
                last_res = self.forward_logic(X, U, E, is_train=False)
                _, step_logs = self.loss_manager(last_res, last_res['v_r'], last_res['mu'], last_res['logstd'])
                for k, v in step_logs.items():
                    val_logs[k] = val_logs.get(k, 0.0) + v

        avg_val = {f"val_{k}": v / len(self.val_loader) for k, v in val_logs.items()}
        
        if last_res:
            # FIX: Explicitly pass control and forcing data to the monitor
            self.monitor.plot_rollout(
                last_res['x_true'], 
                last_res['x_rec_3d'], 
                epoch, 
                prefix="val",
                u_true=last_res.get('u_true'),
                u_rec=last_res.get('u_rec'),
                f_true=last_res.get('f_true')
            )
        return avg_val

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            t_loss, t_logs, t_res = self.train_epoch()
            
            if epoch % self.save_interval == 0:
                    # Filter for active losses (weight > 0)
                    active_losses = [
                        f"{k.replace('loss_', '')}: {v:.4f}" 
                        for k, v in t_logs.items() 
                        if k.startswith('loss_') and v > 0
                    ]
                    loss_str = " | ".join(active_losses)
                    print(f"Epoch {epoch:03d}/{epochs} | Total: {t_loss:.4f} | {loss_str}")

            if epoch % self.val_interval == 0:
                v_metrics = self.validate(epoch)
                
                metrics_to_log = {
                    'epoch': epoch,
                    'loss': t_loss,           
                    'val_loss': v_metrics.get('val_loss', 0.0),
                    **t_logs, 
                    **v_metrics
                }

                if t_res:
                    # FIX: Explicitly pass control and forcing data here too
                    self.monitor.plot_rollout(
                        t_res['x_true'], 
                        t_res['x_rec_3d'], 
                        epoch, 
                        prefix="train",
                        u_true=t_res.get('u_true'),
                        f_true=t_res.get('f_true'),
                        u_rec=t_res.get('u_rec'),
                        e_true=t_res.get('e_true')
                    )
                
                self.monitor.log_metrics(epoch, {**{f"train_{k}": v for k, v in t_logs.items()}, **v_metrics})
                self.monitor.plot_loss_curves()
                
                # Checkpointing logic
                current_mse = v_metrics.get('val_rec', v_metrics.get('val_mse', float('inf')))
                is_best = current_mse < self.best_val_loss
                if is_best: self.best_val_loss = current_mse
                
                if is_best or (epoch % self.save_interval == 0):
                    self.monitor.save_checkpoint(self.models, epoch, is_best=is_best)

    def encode_states(self, X):
        B, T, D = X.shape
        mu, logstd = self.models['encoder'](X.reshape(-1, D))
        
        if self.stochastic:
            # Reparameterization trick: z = mu + sigma * epsilon
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Deterministic mode: just use the mean
            z = mu
            
        return z.reshape(B, T, -1), mu.reshape(B, T, -1), logstd.reshape(B, T, -1)

    # Refactor the transition into a single function to help the compiler
    def transition(self, z, u, e):
        z_next = self.models['A'](z) # Can be a NonLinearNet(nn.Module)
        if 'B' in self.models and u is not None: 
            z_next += self.models['B'](u)
        if 'E' in self.models and e is not None: 
            z_next += self.models['E'](e)
        return z_next

    def _process_control(self, x_win, u_win):
            """
            Deterministic Encode/Decode for control inputs.
            Returns:
                u_eff: Latent control signals for rollout [B, H, L]
                u_rec_3d: Reconstructed control signals [B, H, Du]
            """
            B, H, Du = u_win.shape
            Dx = x_win.shape[-1]
            
            u_eff = u_win
            u_rec_3d = None
            
            if 'control_encoder' in self.models:
                # Prepare input: Concat state x_t and control u_t
                # x_win is [B, H+1, Dx], we take x up to H
                u_in = torch.cat((x_win[:, :H], u_win), dim=-1).reshape(-1, Dx + Du)
                
                # Deterministic Encoding
                u_lat = self.models['control_encoder'](u_in)
                # If the model returns (mu, logstd) despite being deterministic, take mu
                if isinstance(u_lat, (tuple, list)):
                    u_lat = u_lat[0]
                
                u_eff = u_lat.reshape(B, H, -1)
                
                # Deterministic Decoding (if decoder provided)
                if 'control_decoder' in self.models:
                    u_rec_flat = self.models['control_decoder'](u_lat)
                    u_rec_3d = u_rec_flat.reshape(B, H, Du)
                    
            return u_eff, u_rec_3d
