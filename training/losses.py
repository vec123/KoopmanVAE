import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class KoopmanLossManager(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.loss_keys = [
            "rec_init", "rec_full", "koop", "rec_rollout", 
            "kl", "ent", "vamp", "spec", "forc", "zero", "loss_control"
        ]
        
        self.weight_attr_map = {
            "rec_init": "epsilon_1",    "rec_full": "epsilon_2",       
            "koop": "gamma_1",          "rec_rollout": "gamma_2",           
            "kl": "beta_kl",            "ent": "alpha_ent",          
            "vamp": "lambda_vamp",      "spec": "delta_spec",        
            "forc": "lambda_forcing",   "zero": "zero_gain",
            "loss_control": "epsilon_3"  
        }

        weights = [getattr(cfg.weights, self.weight_attr_map[k], 0.0) for k in self.loss_keys]
        self.register_buffer("weights_vector", torch.tensor(weights, dtype=torch.float32))

    def forward(self, results, v_r_collected, mu, logstd):
        # Infer the device from the current batch to avoid manual passing
        dev = mu.device 
        
        # 1. Internal helper to check weight activity
        def is_active(key):
            idx = self.loss_keys.index(key)
            return self.weights_vector[idx] > 0

        #Conditional Computation (Saves FLOPs and avoids SVD errors)

        loss_list = []
        
        # --- Dynamics & Reconstruction ---
        loss_list.append(F.mse_loss(results['x_rec_3d'][:, 0], results['x_true'][:, 0]) if is_active("rec_init") else torch.tensor(0.0, device=dev))
        loss_list.append(F.mse_loss(results['x_rec'], results['x_true'].reshape(-1, results['x_true'].shape[-1])) if is_active("rec_full") else torch.tensor(0.0, device=dev))
        loss_list.append(F.mse_loss(results['z_preds'], results['z_enc']) if is_active("koop") else torch.tensor(0.0, device=dev))
        loss_list.append(self.compute_decaying_rollout(results['x_rec_3d'], results['x_true']) if is_active("rec_rollout") else torch.tensor(0.0, device=dev))
        
        # --- VAE & Stats ---
        loss_list.append(compute_vae_kl(mu, logstd) if is_active("kl") else torch.tensor(0.0, device=dev))
        loss_list.append(compute_entropy_loss(logstd, self.cfg.train.min_entropy_threshold) if is_active("ent") else torch.tensor(0.0, device=dev))
        
        # --- Heavy Math ---
        loss_list.append(compute_vamp_loss(results['z_enc'][:, 0], results['z_enc'][:, 1]) if is_active("vamp") else torch.tensor(0.0, device=dev))
        loss_list.append(compute_spectral_penalty(results['A_op']) if is_active("spec") else torch.tensor(0.0, device=dev))
        
        # --- Forcing & Controls ---
        loss_list.append(compute_forcing_reg(v_r_collected, dev) if is_active("forc") else torch.tensor(0.0, device=dev))
        loss_list.append(results.get('zero_struct_loss', torch.tensor(0.0, device=dev)) if is_active("zero") else torch.tensor(0.0, device=dev))
        loss_list.append(results.get('loss_control', torch.tensor(0.0, device=dev)) if is_active("loss_control") else torch.tensor(0.0, device=dev))

        # Stack and Multiply
        raw_losses = torch.stack(loss_list)
        
        # CRITICAL FIX: Ensure weights_vector is moved to the data's device
        # This eliminates the "cuda:0 and cpu" RuntimeError
        weighted_losses = raw_losses * self.weights_vector.to(dev)
        
        total_loss = weighted_losses.sum()

        # 4. Metrics Logging
        metrics = {
            f"loss_{self.loss_keys[i]}": weighted_losses[i].item() 
            for i in range(len(self.loss_keys)) 
            if self.weights_vector[i] > 0
        }

        return total_loss, metrics

    def compute_decaying_rollout(self, x_rec_3d, x_true):
        mse_elements = F.mse_loss(x_rec_3d, x_true, reduction='none')
        T = mse_elements.shape[1]
        decay_rate = getattr(self.cfg.weights, 'rollout_decay', 0.9)
        t_steps = torch.arange(T, device=x_rec_3d.device, dtype=x_rec_3d.dtype)
        t_weights = torch.pow(decay_rate, t_steps).view(1, -1, 1) 
        return (mse_elements * t_weights).mean()

# --- Refactored & Optimized Helper Functions ---

def compute_vamp_loss(z_t, z_t_plus_1, epsilon=1e-6):
    """Refactored for numerical stability and graph capture."""
    z_t = z_t - z_t.mean(dim=0, keepdim=True)
    z_t_plus_1 = z_t_plus_1 - z_t_plus_1.mean(dim=0, keepdim=True)
    n = z_t.shape[0]
    
    C00 = (z_t.t() @ z_t) / (n - 1) + epsilon * torch.eye(z_t.shape[1], device=z_t.device)
    C11 = (z_t_plus_1.t() @ z_t_plus_1) / (n - 1) + epsilon * torch.eye(z_t.shape[1], device=z_t.device)
    C01 = (z_t.t() @ z_t_plus_1) / (n - 1)

    def inv_sqrt(mat):
        u, s, v = torch.linalg.svd(mat)
        return u @ torch.diag(1.0 / torch.sqrt(s + 1e-8)) @ v.t()

    vamp_matrix = inv_sqrt(C00) @ C01 @ inv_sqrt(C11)
    return -torch.norm(vamp_matrix, p='fro')**2

def compute_vae_kl(mu, logvar):
    """Vectorized KL Divergence."""
    return -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - torch.exp(2 * logvar), dim=-1).mean()

def compute_spectral_penalty(A_module):
    """Spectral radius penalty with torch.linalg compatibility."""
    weight = A_module.weight if hasattr(A_module, 'weight') else next(A_module.parameters())
    # Note: eigvals can be slow; for ultra-fast training, consider power iteration
    eigvals = torch.linalg.eigvals(weight)
    max_eig = torch.max(torch.abs(eigvals))
    return torch.relu(max_eig - 1.0)

def compute_forcing_reg(v_r_list, device):
    if not v_r_list:
        return torch.tensor(0.0, device=device)
    
    v_r_tensor = torch.stack(v_r_list, dim=1)
    l1 = torch.norm(v_r_tensor, p=1, dim=-1).mean()
    l2 = torch.norm(v_r_tensor, p=2, dim=-1).mean()
    return l1 + 0.1 * l2

def compute_entropy_loss(logstd, min_threshold=1.0):
    """Entropy loss to prevent collapse."""
    return torch.relu(min_threshold - torch.mean(logstd))