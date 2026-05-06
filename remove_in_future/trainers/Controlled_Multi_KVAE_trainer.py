import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class ControlledKoopmanVAETrainer:
    def __init__(
        self,
        encoder,
        decoder,
        system_matrix,
        dataloader,
        optimizer,
        latent_dim,
        control_matrix = None,
        external_forcing_matrix = None,
        forcing_network = None,
        control_encoder = None,
        control_decoder = None,
        val_dataloader=None,  
        val_split=0.1,        
        device=None,
        logger=None,
        save_epoch=50,
        val_epochs=10,        
        horizon=20,
        beta=1e-3,
        gamma_1=1.0,
        gamma_2=1.0,
        delta=1e-18,
        alpha=10,
        epsilon_1= 1,
        epsilon_2= 1,
        epsilon_3 =1,
        lambda_forcing = 1,
        zero_structure_gain = 0.0,
        stochastic = False,
        concat_true = False,
        horizon_decay = 1.0,
        use_ema = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stochastic = stochastic
        self.concat_true = concat_true
        self.horizon_decay = horizon_decay

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.A = system_matrix.to(self.device)

        # optional modules
        self.B = control_matrix.to(self.device) if control_matrix else None
        self.external_forcing_matrix = external_forcing_matrix.to(self.device) if external_forcing_matrix else None

        self.forcing_network = forcing_network.to(self.device) if forcing_network else None
        self.lambda_forcing = lambda_forcing

        self.control_encoder = control_encoder.to(self.device) if control_encoder else None
        self.control_decoder = control_decoder.to(self.device) if control_encoder else None

        if val_dataloader is not None:
            self.train_loader = dataloader
            self.val_loader = val_dataloader
        else:
            self.train_loader, self.val_loader = self._split_dataloader(dataloader, val_split)

        self.optimizer = optimizer
        self.latent_dim = latent_dim
        self.logger = logger
        self.save_epoch = save_epoch
        self.val_epochs = val_epochs

        self.horizon = horizon
        self.beta_kl = beta
        self.beta_koop = gamma_1
        self.beta_koop_rec = gamma_2
        self.beta_spec = delta
        self.beta_entropy = alpha
        self.beta_rec_init_single = epsilon_1
        self.beta_rec_init_all = epsilon_2
        self.beta_zero_struct = zero_structure_gain
        self.beta_control_rec = epsilon_3

        self.global_epoch = 0
        self.best_val_loss = float('inf')

        self.use_ema = use_ema  
        self.ema_decay = 0.999
        if self.use_ema:
            self.ema_encoder = AveragedModel(self.encoder, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
            self.ema_decoder = AveragedModel(self.decoder, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
            self.ema_A = AveragedModel(self.A, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
            self.ema_B = AveragedModel(self.B, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
    
    def _split_dataloader(self, dataloader, split_frac):
        dataset = dataloader.dataset
        n_val = int(len(dataset) * split_frac)
        indices = torch.randperm(len(dataset)).tolist()
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=dataloader.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=dataloader.batch_size, shuffle=True)
        return train_loader, val_loader

    def train_step(self, X, U, F_ext, optimize=True):
        """Unified step for train/val. Handles tuple encoder output and 3D reshaping."""
        X, U, F_ext = X.to(self.device), U.to(self.device), F_ext.to(self.device)
        B_size, T, Dx = X.shape
        
        if optimize:
            self.optimizer.zero_grad()

        # 1. Sample a random window
        t = np.random.randint(0, T - self.horizon)
        X_window = X[:, t : t + self.horizon + 1] # Shape: [B, H+1, Dx]

        # 2. Encode: Unpack the tuple (mu, logstd)
        mu_all, logstd_all = self.encoder(X_window.reshape(-1, Dx)) # Shape: [B*(H+1), Latent]
        
        std_all = torch.exp(logstd_all)
        if self.stochastic:
            z_sampled_flat = mu_all + torch.randn_like(std_all) * std_all
        else:            
            z_sampled_flat = mu_all

        # 3. Handle concat_true: Reshape to 3D before concatenating
        if self.concat_true:
            # Move back to [B, H+1, Latent] to match X_window
            z_reshaped = z_sampled_flat.view(B_size, self.horizon + 1, -1)
            z_window_3d = torch.cat((z_reshaped, X_window), dim=-1) # [B, H+1, Latent + Dx]
            # Flatten again for the decoder/rest of processing
            z_sampled_flat = z_window_3d.reshape(-1, z_window_3d.shape[-1])
        else:
            z_window_3d = z_sampled_flat.view(B_size, self.horizon + 1, -1)

        # Actual latent dimension for the decoder input
        current_latent_dim = z_sampled_flat.shape[-1]

        # 4. Reconstruction Losses
        X_window_flat = X_window.reshape(-1, Dx)
        X_rec_all_flat = self.decoder(z_sampled_flat)
        loss_rec_all = F.mse_loss(X_rec_all_flat, X_window_flat)

        X_rec_all = X_rec_all_flat.reshape(B_size, -1, Dx)
        loss_rec_init = F.mse_loss(X_rec_all[:, 0], X_window[:, 0])

        # 5. VAE Losses (using the unpacked logstd)
        entropy_loss = self.entropy_loss(logstd_all, min_entropy_threshold=1.0)
        loss_kl = -0.5 * torch.sum(1 + 2 * logstd_all - mu_all.pow(2) - torch.exp(2 * logstd_all), dim=-1).mean()

        # 6. Koopman Rollout
        z_curr = z_window_3d[:, 0] # Start from t=0 of the window
        z_rollout = [z_curr]
        v_r_collected = []

        # Optional Control Encoding
        loss_rec_control = torch.tensor(0.0, device=self.device)
        if self.control_encoder and self.control_decoder:
            U_true = U[:, t : t + self.horizon]
            X_U_input = torch.cat((X_window[:, :-1], U_true), dim=-1).reshape(-1, Dx + U_true.shape[-1])
            u_enc_flat = self.control_encoder(X_U_input)
            U_latents = u_enc_flat.reshape(B_size, self.horizon, -1)
            U_rec_flat = self.control_decoder(u_enc_flat)
            loss_rec_control = F.mse_loss(U_rec_flat, U_true.reshape(-1, U_true.shape[-1]))

        for i in range(1, self.horizon + 1):
            z_next_pred = self.A(z_curr) 

            if self.B:
                u_prev = U_latents[:, i-1] if self.control_encoder else U[:, t + i - 1]
                z_next_pred += self.B(u_prev)

            if self.external_forcing_matrix:
                z_next_pred += self.external_forcing_matrix(F_ext[:, t + i - 1])

            if self.forcing_network:
                v_r = self.forcing_network(z_curr)
                v_r_collected.append(v_r)
                z_next_pred += v_r

            z_curr = z_next_pred
            z_rollout.append(z_curr)

        # 7. Compute Rollout Losses
        z_rollout_tensor = torch.stack(z_rollout, dim=1) # [B, H+1, current_latent_dim]
        weights = torch.tensor([self.horizon_decay**i for i in range(self.horizon + 1)], device=self.device).view(1, -1, 1)
        
        loss_koop = F.mse_loss(z_rollout_tensor, z_window_3d, reduction='none')
        loss_koop = (loss_koop * weights).mean()

        X_rec_rollout = self.decoder(z_rollout_tensor.reshape(-1, current_latent_dim))
        loss_rec_multi = F.mse_loss(X_rec_rollout, X_window_flat)

        # 8. Structured/Spectral Losses
        loss_spec = self.spectral_loss()
        x_zero = torch.zeros(1, Dx).to(self.device)
        z_zero_mu, _ = self.encoder(x_zero) # Unpack
        if self.concat_true:
            z_zero_mu = torch.cat((z_zero_mu, x_zero), dim=-1)
        loss_zero_struct = torch.mean(self.decoder(z_zero_mu)**2)

        loss_forcing = self.forcing_regularization(v_r_collected) if self.forcing_network else torch.tensor(0.0, device=self.device)

        # 9. Weighted Sum
        total_loss = (self.beta_koop_rec * loss_rec_multi) + \
                    (self.beta_rec_init_single * loss_rec_init) + \
                    (self.beta_rec_init_all * loss_rec_all) + \
                    (self.beta_kl * loss_kl) + \
                    (self.beta_koop * loss_koop) + \
                    (self.beta_spec * loss_spec) + \
                    (self.beta_entropy * entropy_loss) + \
                    (self.beta_zero_struct * loss_zero_struct) + \
                    (self.beta_control_rec * loss_rec_control) + \
                    (self.lambda_forcing * loss_forcing)

        if optimize:
            total_loss.backward()
            self.optimizer.step()
            if self.use_ema:
                self.ema_encoder.update_parameters(self.encoder)
                self.ema_decoder.update_parameters(self.decoder)
                self.ema_A.update_parameters(self.A)
                self.ema_B.update_parameters(self.B)

        return {
            "loss": total_loss.item(),
            "rec init single": loss_rec_init.item(),
            "rec init all": loss_rec_all.item(),
            "rec multi": loss_rec_multi.item(),
            "koop": loss_koop.item(),
            "kl": loss_kl.item(),
            "spectral": loss_spec.item(),
            "entropy": entropy_loss.item(),
            "zero_struct": loss_zero_struct.item(),
            "control_rec": loss_rec_control.item(),
            "loss_forcing": loss_forcing.item()
        }
        
    def train(self, epochs):
            for epoch in range(epochs):
                self.global_epoch += 1
                self.encoder.train(); self.decoder.train(); self.A.train(); self.B.train()
                
                t_logs = {}; last_train = None
                for X, U, K in self.train_loader:
                    step_out = self.train_step(X, U, K, optimize=True)
                    t_logs = {k: t_logs.get(k, 0.0) + v for k, v in step_out.items()}
                    last_train = (X, U, K)
                
                avg_train = {k: v / len(self.train_loader) for k, v in t_logs.items()}

                if self.global_epoch % self.val_epochs == 0:
                    avg_val = self.validate(train_batch=last_train)
                    all_logs = {**avg_train, **avg_val}
                    
                    # Dynamic terminal output for all tracked losses
                    print(f"\n[Epoch {self.global_epoch}]")
                    print(f"  TRAIN >> " + " | ".join(f"{k}: {v:.4f}" for k, v in avg_train.items()))
                    print(f"  VAL   >> " + " | ".join(f"{k}: {v:.4f}" for k, v in avg_val.items()))

                    if self.logger: self.logger.log_scalars(all_logs, step=self.global_epoch)
                    
                    # Checkpoint on best validation loss
                    cur_val_loss = avg_val.get('val_loss', float('inf'))
                    if cur_val_loss < self.best_val_loss:
                        self.best_val_loss = cur_val_loss
                        self._save_checkpoint("best_model")

                if self.global_epoch % self.save_epoch == 0:
                    self._save_checkpoint(f"epoch_{self.global_epoch}")


    def validate(self, train_batch=None):
        # Ensure all modules are in eval mode
        self.encoder.eval(); self.decoder.eval(); self.A.eval()
        if self.B: self.B.eval()
        if self.external_forcing_matrix: self.external_forcing_matrix.eval()
        
        val_logs = {}
        last_val_batch = None

        # Use the context manager to swap weights (if your EMA setup is active)
        with self.ema_scope():
            with torch.no_grad():
                # 1. Unpack X, U, and F from the dataloader
                for X, U, F_ext in self.val_loader:
                    # 2. Pass all three to train_step with optimize=False
                    step_out = self.train_step(X, U, F_ext, optimize=False)
                    
                    for k, v in step_out.items():
                        val_logs[k] = val_logs.get(k, 0.0) + v
                    last_val_batch = (X, U, F_ext)

                # Average validation logs
                avg_val_logs = {f"val_{k}": v / len(self.val_loader) for k, v in val_logs.items()}
                
                # 3. Visualization updates
                if last_val_batch:
                    # Pass the full triplet (X, U, F) to the visualizer
                    self.visualize_and_save(
                        last_val_batch[0], 
                        last_val_batch[1], 
                        last_val_batch[2], 
                        prefix="val_ema"
                    )
                if train_batch:
                    # Ensure your train_batch also contains (X, U, F)
                    self.visualize_and_save(
                        train_batch[0], 
                        train_batch[1], 
                        train_batch[2], 
                        prefix="train_ema"
                    )

        return avg_val_logs


    @torch.no_grad()
    def visualize_and_save(self, X_batch, U_batch, F_batch, prefix="train"):
        self.encoder.eval(); self.decoder.eval(); self.A.eval()
        if self.B: self.B.eval()
        if self.external_forcing_matrix: self.external_forcing_matrix.eval()
        if self.control_encoder: self.control_encoder.eval()
        if self.control_decoder: self.control_decoder.eval()

        # 1. Prepare Data
        x_true = X_batch[0:1].to(self.device)  # [1, T, Dx]
        u_true = U_batch[0:1].to(self.device)  # [1, T, Du]
        f_true = F_batch[0:1].to(self.device)  # [1, T, Df]
        T = x_true.shape[1]
        
        # 2. Initial Latent State - Unpack tuple
        mu_0, _ = self.encoder(x_true[:, 0])
        
        # Dimensionality fix for concat_true
        if self.concat_true:
            z_curr = torch.cat((mu_0, x_true[:, 0]), dim=-1)
        else:
            z_curr = mu_0
        
        current_latent_dim = z_curr.shape[-1]
        z_rollout = [z_curr]
        u_reconstructed_list = []

        # 3. Rollout Loop
        for i in range(T - 1):
            # Handle Control Encoder
            if self.control_encoder:
                xu_pair = torch.cat([x_true[:, i], u_true[:, i]], dim=-1)
                u_input = self.control_encoder(xu_pair)
                if self.control_decoder:
                    u_reconstructed_list.append(self.control_decoder(u_input))
            else:
                u_input = u_true[:, i]

            # Dynamics step
            z_next = self.A(z_rollout[-1])

            if self.B:
                z_next += self.B(u_input)

            if self.external_forcing_matrix:
                z_next += self.external_forcing_matrix(f_true[:, i])
            
            z_rollout.append(z_next)

        # 4. Final Formatting
        z_rollout_tensor = torch.stack(z_rollout, dim=1) # [1, T, current_latent_dim]
        x_rollout = self.decoder(z_rollout_tensor.reshape(-1, current_latent_dim)).reshape(1, T, -1)
        
        u_rec_tensor = None
        if u_reconstructed_list:
            u_rec_tensor = torch.stack(u_reconstructed_list, dim=1)
            # Pad last timestep to match length T
            u_rec_tensor = torch.cat([u_rec_tensor, u_rec_tensor[:, -1:]], dim=1)

        # 5. Log Results
        if self.logger:
            self.logger.save_trajectories_w_control_and_forcing(
                z_traj=z_rollout_tensor, 
                x_rec=x_rollout, 
                x_true=x_true, 
                u_true=u_true,
                u_rec=u_rec_tensor, 
                f_true=f_true,
                step=self.global_epoch,
                prefix=prefix
            )

    def _save_checkpoint(self, name):
        if self.logger:
            models_to_save = {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "system_matrix": self.A,
                "control_matrix": self.B
            }
            self.logger.save_models(models_to_save, step=self.global_epoch)

    def entropy_loss(self, logstd, min_entropy_threshold=1.0):
        """
        Computes the entropy constraint for the Koopman latent space.
        logstd: the second half of the encoder output (cat[mu, logstd])
        """

        avg_entropy = torch.mean(logstd)

        return torch.relu(min_entropy_threshold - avg_entropy)
    
    def spectral_loss(self):
        Amat = self.get_A_matrix_tensor()
        # Spectral radius = max absolute eigenvalue
        eigvals = torch.linalg.eigvals(Amat)
        max_eig = torch.max(torch.abs(eigvals))
        # Penalty: only punish if the radius exceeds 1.0
        return torch.relu(max_eig - 1.0)


    def forcing_regularization(self, v_r_list):
        """
        Regularizes the learned forcing terms to be sparse and small.
        v_r_list: A list of tensors [Batch, Latent] captured during rollout.
        """
        if not v_r_list:
            return torch.tensor(0.0, device=self.device)
        
        # Stack into [Batch, Horizon, Latent]
        v_r_tensor = torch.stack(v_r_list, dim=1)
        
        # 1. Sparsity Penalty (L1): Encourages v_r to be zero most of the time.
        # This is key for HAVOK-style intermittent forcing.
        loss_l1 = torch.norm(v_r_tensor, p=1, dim=-1).mean()
        
        # 2. Magnitude Penalty (L2): Prevents the network from giving huge "kicks".
        loss_l2 = torch.norm(v_r_tensor, p=2, dim=-1).mean()
        
        # Balance them: L1 is usually more important for HAVOK principles
        return 1.0 * loss_l1 + 0.1 * loss_l2
    

    def get_A_matrix_tensor(self):
        """Extracts the system matrix A for spectral regularization."""
        # Check for TT-parameterization or custom getter
        if hasattr(self.A, 'get_matrix'):
            return self.A.get_matrix()
        # Fallback for standard Linear layers or wrappers
        for attr in ["matrix", "weight", "A", "W"]:
            if hasattr(self.A, attr): 
                val = getattr(self.A, attr)
                return val.weight if isinstance(val, nn.Linear) else val
        return list(self.A.parameters())[0]

    from contextlib import contextmanager

    @contextmanager
    def ema_scope(self):
        if not self.use_ema:
            yield
            return

        # 1. Store original parameters
        orig_params = {
            "enc": self.encoder.state_dict(),
            "dec": self.decoder.state_dict(),
            "A": self.A.state_dict(),
            "B": self.B.state_dict()
        }

        try:
            # 2. Load EMA parameters into the "live" models
            self.encoder.load_state_dict(self.ema_encoder.module.state_dict())
            self.decoder.load_state_dict(self.ema_decoder.module.state_dict())
            self.A.load_state_dict(self.ema_A.module.state_dict())
            self.B.load_state_dict(self.ema_B.module.state_dict())
            yield
        finally:
            # 3. Restore original parameters for training
            self.encoder.load_state_dict(orig_params["enc"])
            self.decoder.load_state_dict(orig_params["dec"])
            self.A.load_state_dict(orig_params["A"])
            self.B.load_state_dict(orig_params["B"])

