import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanVAETrainer:
    def __init__(
        self,
        encoder,
        decoder,
        koopman_matrix,
        dataloader,
        optimizer,
        latent_dim,
        device=None,
        logger=None,
        save_epoch=50,
        horizon=20,          # How many steps to look ahead during training
        beta=1e-3,           # KL weight
        gamma=1.0,           # Koopman weight
        delta=1e-18,          # Spectral weight
        init_state=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.K = koopman_matrix.to(self.device)

        self.dataloader = dataloader
        self.optimizer = optimizer
        self.latent_dim = latent_dim
        self.logger = logger
        self.save_epoch = save_epoch
        self.init_state = init_state

        self.horizon = horizon
        self.beta_kl = beta
        self.beta_koop = gamma
        self.beta_spec = delta
        self.global_epoch = 0

    def get_koopman_matrix(self):
        # If using the TT-parameterization
        if hasattr(self.K, 'get_matrix'):
            return self.K.get_matrix()
        # Fallback for standard LinearMatrix
        for attr in ["matrix", "weight", "A", "W"]:
            if hasattr(self.K, attr): return getattr(self.K, attr)
        return list(self.K.parameters())[0]

    def spectral_loss(self):
        Kmat = self.get_koopman_matrix()
        eigvals = torch.linalg.eigvals(Kmat)
        return torch.mean((torch.abs(eigvals) - 1.0) ** 2)

    def train_step(self, X):
        X = X.to(self.device) # [B, T, Dx]
        B, T, Dx = X.shape
        self.optimizer.zero_grad()

        # 1. Random Start Point for the rollout window
        t = np.random.randint(0, T - self.horizon)
        
        # 2. Encode the initial state and future states in the window
        # We process the whole window to get "ground truth" latent targets
        X_window = X[:, t : t + self.horizon + 1]
        mu_all, logstd_all = torch.chunk(self.encoder(X_window.reshape(-1, Dx)), 2, dim=-1)
        mu_all = mu_all.reshape(B, self.horizon + 1, -1)
        logstd_all = logstd_all.reshape(B, self.horizon + 1, -1)

        # 3. Standard VAE Loss (Reconstruction of t=0)
        z_0 = mu_all[:, 0] + torch.randn_like(logstd_all[:, 0]) * torch.exp(logstd_all[:, 0])
        x_rec_0 = self.decoder(z_0)
        loss_rec = F.mse_loss(x_rec_0, X[:, t])
        
        # KL Divergence (Sum over latent dim, mean over batch)
        loss_kl = -0.5 * torch.sum(1 + 2*logstd_all[:, 0] - mu_all[:, 0].pow(2) - torch.exp(2*logstd_all[:, 0]), dim=-1).mean()

        # 4. Multi-Step Koopman Rollout
        # We iterate K and compare to the encoder's future means
        loss_koop = 0
        z_curr = mu_all[:, 0] 
        
        for i in range(1, self.horizon + 1):
            z_curr = self.K(z_curr)
            loss_koop += F.mse_loss(z_curr, mu_all[:, i])
        
        loss_koop /= self.horizon
        loss_spec = self.spectral_loss()

        # 5. Total Loss
        loss = loss_rec + (self.beta_kl * loss_kl) + (self.beta_koop * loss_koop) + (self.beta_spec * loss_spec)

        loss.backward()
        # Clip gradients to prevent "exploding rollouts" during backprop
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.K.parameters()), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "rec": loss_rec.item(),
            "kl": loss_kl.item(),
            "koop": loss_koop.item(),
            "spec": loss_spec.item(),
        }

    def train(self, epochs):
        for epoch in range(epochs):
            self.global_epoch += 1
            logs = {}
            for X, _ in self.dataloader:
                step_out = self.train_step(X)
                for k, v in step_out.items():
                    logs[k] = logs.get(k, 0.0) + v
            
            # Average logs
            avg_logs = {k: v / len(self.dataloader) for k, v in logs.items()}

            if self.global_epoch % 10 == 0:
                print(f"Epoch {self.global_epoch} | Rec: {avg_logs['rec']:.4f} | Koop: {avg_logs['koop']:.4f} | KL: {avg_logs['kl']:.2f}")

            if self.logger:
                self.logger.log_scalars(avg_logs, step=self.global_epoch)
                
                if self.global_epoch % self.save_epoch == 0:
                    self.visualize_and_save(X)
                    
                    models_to_save = {
                        "encoder": self.encoder,
                        "decoder": self.decoder,
                        "koopman_matrix": self.K
                    }
                    self.logger.save_models(models_to_save, step=self.global_epoch)

    @torch.no_grad()
    def visualize_and_save(self, X_batch):
        # Select one trajectory from the batch
        x_true = X_batch[0:1].to(self.device) # [1, T, Dx]
        T = x_true.shape[1]
        
        # 1. Encode the start
        mu_0, _ = torch.chunk(self.encoder(x_true[:, 0]), 2, dim=-1)
        
        # 2. Long Rollout (entire sequence length)
        z_rollout = [mu_0]
        for _ in range(T - 1):
            z_rollout.append(self.K(z_rollout[-1]))
        z_rollout = torch.stack(z_rollout, dim=1).squeeze(2) # [1, T, Latent]
        
        # 3. Decode rollout
        x_rollout = self.decoder(z_rollout.reshape(-1, self.latent_dim)).reshape(1, T, -1)
        
        # 4. Save to logger
        self.logger.save_trajectories(
            z_traj=z_rollout, 
            x_rec=x_rollout, 
            x_true=x_true, 
            step=self.global_epoch
        )
        print(f"--- Visualized rollout at epoch {self.global_epoch} ---")