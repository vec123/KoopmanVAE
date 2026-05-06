import torch
import torch.nn.functional as F
from conf.config import KoopmanConfig


import torch
import torch.nn.functional as F

class KoopmanLossManager:
    def __init__(self, cfg: KoopmanConfig):
        self.cfg = cfg
        # We'll use the device passed during __call__ to ensure compatibility
        self.weight_map = {
            "rec_init": "epsilon_1",    
            "rec_full": "epsilon_2",       
            "koop": "gamma_1", 
            "rec_rollout": "gamma_2",           
            "kl": "beta_kl",             
            "ent": "alpha_ent",          
            "vamp": "lambda_vamp",       
            "spec": "delta_spec",        
            "forc": "lambda_forcing",    
            "zero": "zero_gain",
            "loss_control": "epsilon_3"  
        }

    def compute_decaying_rollout(self, x_rec_3d, x_true, device):
        """Computes MSE with temporal exponential decay."""
        # Calculate raw squared errors: [B, T, Dx]
        mse_elements = F.mse_loss(x_rec_3d, x_true, reduction='none')
        
        T = mse_elements.shape[1]
        decay_rate = getattr(self.cfg.weights, 'rollout_decay', 0.9)
        
        # Create weights: [1.0, 0.9, 0.81, ...]
        t_steps = torch.arange(T, device=device, dtype=mse_elements.dtype)
        t_weights = torch.pow(decay_rate, t_steps).view(1, -1, 1) 
        
        return (mse_elements * t_weights).mean()

    def __call__(self, results, v_r_collected, mu, logstd, device):
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        # 1. Map results to specific loss functions
        # We pass arguments into the helper functions here
        loss_funcs = {
            "rec_init": lambda: F.mse_loss(results['x_rec_3d'][:, 0], results['x_true'][:, 0]),
            "rec_full": lambda: F.mse_loss(results['x_rec'], results['x_true'].reshape(-1, results['x_true'].shape[-1])),
            "rec_rollout": lambda: self.compute_decaying_rollout(results['x_rec_3d'], results['x_true'], device),
            "koop": lambda: F.mse_loss(results['z_preds'], results['z_enc']),
            "kl": lambda: compute_vae_kl(mu, logstd),
            "ent": lambda: compute_entropy_loss(logstd, self.cfg.train.min_entropy_threshold),
            "spec": lambda: compute_spectral_penalty(results['A_op']),
            "vamp": lambda: compute_vamp_loss(results['z_enc'][:, 0], results['z_enc'][:, 1]),
            "forc": lambda: compute_forcing_reg(v_r_collected),
            "zero": lambda: compute_zero_struct_loss(results['encoder'], results['decoder'], device, getattr(self.cfg, 'concat_true', False)),
            "loss_control": lambda: results.get('loss_control', torch.tensor(0.0, device=device))
        }

        # 2. Universal Loop: applies weights from your config
        for key, weight_attr in self.weight_map.items():
            weight = getattr(self.cfg.weights, weight_attr, 0.0)
            
            if weight != 0:
                # Execution happens only if weight > 0
                loss_val = loss_funcs[key]()
                weighted_loss = weight * loss_val
                total_loss += weighted_loss
                metrics[f"loss_{key}"] = weighted_loss.item()
            else:
                metrics[f"loss_{key}"] = 0.0

        return total_loss, metrics
    
    
#-------------------------------Losses--------------------------------------------------
def compute_vamp_loss(z_t, z_t_plus_1, epsilon=1e-6):
    """Maximizes the kinetic variance captured in the latent transition."""
    # Center data
    z_t = z_t - z_t.mean(dim=0, keepdim=True)
    z_t_plus_1 = z_t_plus_1 - z_t_plus_1.mean(dim=0, keepdim=True)
    
    n = z_t.shape[0]
    # Covariances
    C00 = (z_t.t() @ z_t) / (n - 1) + epsilon * torch.eye(z_t.shape[1], device=z_t.device)
    C11 = (z_t_plus_1.t() @ z_t_plus_1) / (n - 1) + epsilon * torch.eye(z_t.shape[1], device=z_t.device)
    C01 = (z_t.t() @ z_t_plus_1) / (n - 1)

    # VAMP-2 Score calculation via SVD for stability
    def inv_sqrt(mat):
        u, s, v = torch.linalg.svd(mat)
        return u @ torch.diag(1.0 / torch.sqrt(s)) @ v.t()

    vamp_matrix = inv_sqrt(C00) @ C01 @ inv_sqrt(C11)
    return -torch.norm(vamp_matrix, p='fro')**2

def compute_vae_kl(mu, logvar):
    """Standard KL Divergence for VAE Bottleneck."""
    return -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - torch.exp(2 * logvar), dim=-1).mean()

def compute_spectral_penalty(A_module):
    """Penalizes the spectral radius of the system matrix A if it exceeds 1.0."""
    # Handle both nn.Linear and custom Parameter objects
    weight = A_module.weight if hasattr(A_module, 'weight') else next(A_module.parameters())
    eigvals = torch.linalg.eigvals(weight)
    max_eig = torch.max(torch.abs(eigvals))
    return torch.relu(max_eig - 1.0)

def compute_forcing_reg(v_r_list):
    """L1+L2 regularization for intermittent (HAVOK-style) forcing."""
    if not v_r_list:
        return torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    v_r_tensor = torch.stack(v_r_list, dim=1)
    l1 = torch.norm(v_r_tensor, p=1, dim=-1).mean()
    l2 = torch.norm(v_r_tensor, p=2, dim=-1).mean()
    return l1 + 0.1 * l2

def compute_entropy_loss(logstd, min_threshold=1.0):
    """Prevents latent distribution collapse."""
    return torch.relu(min_threshold - torch.mean(logstd))

def compute_zero_struct_loss(encoder, decoder, device, concat_true=False):
    """Ensures that f(0) = 0 (Encoder) and g(0) = 0 (Decoder)."""
   
    in_features = next(encoder.parameters()).shape[1]
    zero_input = torch.zeros((1, in_features), device=device)
    
    # Encoder(0) should be 0
    mu_zero, _ = encoder(zero_input)
    loss_enc = torch.norm(mu_zero, p=2)
    
    # Decoder(0) should be 0
    # Create zero latent matching latent_dim
    latent_dim = next(decoder.parameters()).shape[1]
    zero_latent = torch.zeros((1, latent_dim), device=device)
    x_rec_zero = decoder(zero_latent)
    loss_dec = torch.norm(x_rec_zero, p=2)
    
    return loss_enc + loss_dec