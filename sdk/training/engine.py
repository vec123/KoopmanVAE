import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from .losses import compute_spectral_penalty, compute_vae_kl, compute_forcing_reg

class KoopmanTrainer:
    def __init__(self, system_bundle, config, dataloaders, device="cuda"):
        self.device = device
        self.cfg = config
        self.models = nn.ModuleDict(system_bundle).to(device)
        self.train_loader, self.val_loader = dataloaders
        
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=config.lr)
        self.ema = self._setup_ema() if config.use_ema else None
        self.global_epoch = 0

    def _setup_ema(self):
        ema_fn = get_ema_multi_avg_fn(0.999)
        return {k: AveragedModel(v, multi_avg_fn=ema_fn) for k, v in self.models.items()}

    def _dynamics_step(self, z, u=None, f=None):
        z_next = self.models['A'](z)
        if 'B' in self.models and u is not None:
            z_next += self.models['B'](u)
        if 'E' in self.models and f is not None:
            z_next += self.models['E'](f)
        return z_next

    def train_epoch(self):
        self.models.train()
        total_loss = 0
        for X, U, F in self.train_loader:
            self.optimizer.zero_grad()
            
            # 1. Latent Encoding
            X, U, F = X.to(self.device), U.to(self.device), F.to(self.device)
            enc_out = self.models['encoder'](X.reshape(-1, X.shape[-1]))
            mu, logvar = torch.chunk(enc_out, 2, dim=-1)
            z = mu + torch.randn_like(mu) * torch.exp(logvar)
            
            # 2. Rollout
            z_curr = z.reshape(X.shape[0], X.shape[1], -1)[:, 0]
            z_preds = [z_curr]
            for t in range(self.cfg.horizon):
                z_curr = self._dynamics_step(z_curr, U[:, t], F[:, t])
                z_preds.append(z_curr)
            
            # 3. Losses (Simplified for brevity)
            z_preds_stack = torch.stack(z_preds, dim=1)
            recon = F.mse_loss(self.models['decoder'](z_preds_stack.reshape(-1, z_preds_stack.shape[-1])), 
                               X[:, :self.cfg.horizon+1].reshape(-1, X.shape[-1]))
            
            loss = recon + self.cfg.beta_kl * compute_vae_kl(mu, logvar)
            loss.backward()
            self.optimizer.step()
            
            if self.ema:
                for k in self.models: self.ema[k].update_parameters(self.models[k])
                
            total_loss += loss.item()
        return total_loss / len(self.train_loader)