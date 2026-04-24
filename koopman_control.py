import os
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import joblib

# Local project imports
from systems.controlled_systems import cartpole_controlled
from models.models import LinearMatrix, KoopmanEncoder

class KoopmanLQRController_:
    """
    Object-oriented wrapper to handle the Koopman + LQR control loop.
    Manages scaling, encoding, and optimal gain computation.
    """
    def __init__(self, base_path, device="cpu", checkpoint_step=10000, 
                 state_dim = 4, latent_dim = 128, hidden_dim = 256,
                 concat_true = True):
        self.device = device
        self.concat_true = concat_true
        # 1. Load Scalers
        self.x_scaler = joblib.load(os.path.join(base_path, "x_scaler.pkl"))
        self.u_scaler = joblib.load(os.path.join(base_path, "u_scaler.pkl"))
        
        # 2. Re-initialize and Load Models
        # Ensure dimensions match your VARIANT/Configuration
        self.state_dim, self.latent_dim, self.hidden_dim = state_dim, latent_dim, hidden_dim
        self.encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim).to(device)
        if concat_true:
            latent_dim += state_dim
        self.decoder = LinearMatrix(latent_dim, state_dim).to(device)
        self.system_net = LinearMatrix(latent_dim, latent_dim).to(device)
        self.control_net = LinearMatrix(1, latent_dim).to(device)

        models = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "system_matrix": self.system_net,
            "control_matrix": self.control_net
        }
        
        for name, model in models.items():
            path = os.path.join(base_path, f"{name}_{checkpoint_step}.pt")
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()

        # 3. Extract Linear Matrices for LQR
        self.A = self._get_weight(self.system_net)
        self.B = self._get_weight(self.control_net)
        if self.B.shape[0] == 1: self.B = self.B.T # Match [Latent, Act]
        self.C = self._get_weight(self.decoder)

        # 4. Pre-compute LQR Gain
        self.K = self._compute_lqr(latent_dim)
        
        # 5. Pre-compute Normalized Target (Equilibrium at 0,0,0,0)
        self.z_target = self._get_z_target()

    def _get_weight(self, model):
        """Helper to extract weights regardless of specific wrapper."""
        for attr in ["matrix", "weight", "W"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                return val.weight.detach().cpu().numpy() if isinstance(val, torch.nn.Linear) else val.detach().cpu().numpy()
        return list(model.parameters())[0].detach().cpu().numpy()

    def _compute_lqr(self, latent_dim):
        # Physical weights (Bryson's Rule)
        # Penalizing Angle (index 2) and Angular Velocity (index 3)
        Q_phys_diag = np.array([100.0, 0.0, 1000.0, 0.0])
        Q_phys = np.diag(Q_phys_diag)
        
        # Project Physical Q into Latent space via C (the decoder)
        # C predicts normalized state, so we multiply by x_scaler.scale_ to get physical units
        C_phys = self.C * self.x_scaler.scale_.reshape(-1, 1)
        Q_z = C_phys.T @ Q_phys @ C_phys + np.eye(latent_dim) * 1e-6 # Tikhonov reg
        
        R_lqr = np.eye(1) * 100 # Control cost
        
        P = scipy.linalg.solve_discrete_are(self.A, self.B, Q_z, R_lqr)
        K = np.linalg.inv(R_lqr + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K

    def _get_z_target(self, x_target=np.zeros((1, 4))):
        with torch.no_grad():
            x_target = self.x_scaler.transform(x_target)
            x_target_t = torch.as_tensor(x_target, dtype=torch.float32).to(self.device)
            z_target, _ = torch.chunk(self.encoder(x_target_t), 2, dim=-1)

            if self.concat_true:
                z_target = torch.cat((z_target, x_target_t), dim=-1)

            return z_target.cpu().numpy().reshape(-1, 1)

    def get_control(self, x_phys):
        """Standard control law: u = -K(z - z_target)"""
        x_norm = self.x_scaler.transform(x_phys.reshape(1, -1))
        x_t = torch.as_tensor(x_norm, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            z_curr, _ = torch.chunk(self.encoder(x_t), 2, dim=-1)
            z_curr = z_curr.cpu().numpy().reshape(-1, 1)

        if self.concat_true:
            z_curr = np.concatenate((z_curr, x_norm.reshape(-1, 1)), axis=0)

        u_norm = self.K @ (z_curr - self.z_target)
        
        # Denormalize for environment
        u_phys = (u_norm.item() * self.u_scaler.scale_[0]) + self.u_scaler.mean_[0]
        return np.clip(u_phys, -25.0, 25.0)


class KoopmanLQRController:
    """
    Object-oriented wrapper to handle the Koopman + LQR control loop.
    Works with standard scalers or defaults to Identity if files are missing.
    """
    def __init__(self, base_path, device="cpu", checkpoint_step=10000, 
                 state_dim=4, latent_dim=128, hidden_dim=256,
                 concat_true=True):
        self.device = device
        self.concat_true = concat_true
        
        # 1. Load Scalers with Fallback
        self.x_scaler = self._load_scaler(base_path, "x_scaler.pkl", state_dim)
        self.u_scaler = self._load_scaler(base_path, "u_scaler.pkl", 1)
        
        # 2. Re-initialize and Load Models
        self.state_dim, self.latent_dim, self.hidden_dim = state_dim, latent_dim, hidden_dim
        self.encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim).to(device)
        
        actual_latent_dim = latent_dim + state_dim if concat_true else latent_dim
        
        self.decoder = LinearMatrix(actual_latent_dim, state_dim).to(device)
        self.system_net = LinearMatrix(actual_latent_dim, actual_latent_dim).to(device)
        self.control_net = LinearMatrix(1, actual_latent_dim).to(device)

        models = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "system_matrix": self.system_net,
            "control_matrix": self.control_net
        }
        
        for name, model in models.items():
            path = os.path.join(base_path, f"{name}_{checkpoint_step}.pt")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
            else:
                raise ValueError(f"Error: Checkpoint {path} not found.")

        # 3. Extract Matrices
        self.A = self._get_weight(self.system_net)
        self.B = self._get_weight(self.control_net)
        if self.B.shape[0] == 1: self.B = self.B.T 
        self.C = self._get_weight(self.decoder)

        # 4. Pre-compute
        self.K = self._compute_lqr(actual_latent_dim)
        self.z_target = self._get_z_target()

    def _load_scaler(self, base_path, name, dim):
        """Loads scaler if exists, otherwise creates an Identity Scaler."""
        path = os.path.join(base_path, name)
        if os.path.exists(path):
            return joblib.load(path)
        
        # Mock Scaler for robustness
        class IdentityScaler:
            def __init__(self, d):
                self.scale_ = np.ones(d)
                self.mean_ = np.zeros(d)
            def transform(self, x): return x
            def inverse_transform(self, x): return x
            
        print(f"Info: {name} not found. Proceeding without scaling.")
        return IdentityScaler(dim)

    def _get_weight(self, model):
        for attr in ["matrix", "weight", "W"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                return val.weight.detach().cpu().numpy() if isinstance(val, torch.nn.Linear) else val.detach().cpu().numpy()
        return list(model.parameters())[0].detach().cpu().numpy()

    def _compute_lqr(self, actual_dim):
        # Physical weights
        Q_phys_diag = np.array([100.0, 0.0, 1000.0, 0.0])
        Q_phys = np.diag(Q_phys_diag)
        
        # Safe scale multiplication (handles IdentityScaler)
        C_phys = self.C * self.x_scaler.scale_.reshape(-1, 1)
        Q_z = C_phys.T @ Q_phys @ C_phys + np.eye(actual_dim) * 1e-6 
        
        R_lqr = np.eye(1) * 100 
        
        P = scipy.linalg.solve_discrete_are(self.A, self.B, Q_z, R_lqr)
        K = np.linalg.inv(R_lqr + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K

    def _get_z_target(self, x_target=None):
        if x_target is None:
            x_target = np.zeros((1, self.state_dim))
            
        with torch.no_grad():
            x_norm = self.x_scaler.transform(x_target)
            x_t = torch.as_tensor(x_norm, dtype=torch.float32).to(self.device)
            # Handle potential tuple return from encoder (z, mean, logvar etc)
            enc_out = self.encoder(x_t)
            mu_all, logstd_all = torch.chunk(enc_out, 2, dim=-1)
            z_target = mu_all 
            print("z_target.shape: ", z_target.shape)
            if self.concat_true:
                z_target = torch.cat((z_target, x_t), dim=-1)

            return z_target.cpu().numpy().reshape(-1, 1)

    def get_control(self, x_phys):
        x_norm = self.x_scaler.transform(x_phys.reshape(1, -1))
        x_t = torch.as_tensor(x_norm, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            enc_out = self.encoder(x_t)
            enc_out = self.encoder(x_t)
            mu_all, logstd_all = torch.chunk(enc_out, 2, dim=-1)
            z_curr = mu_all
            z_curr = z_curr.cpu().numpy().reshape(-1, 1)

        if self.concat_true:
            z_curr = np.concatenate((z_curr, x_norm.reshape(-1, 1)), axis=0)

        u_norm = -self.K @ (z_curr - self.z_target) # Standard u = -Kx
        
        # Denormalize
        u_phys = (u_norm.item() * self.u_scaler.scale_[0]) + self.u_scaler.mean_[0]
        return np.clip(u_phys, -25.0, 25.0)
# -------------------------------------------------------
# Execution Script
# -------------------------------------------------------

def main():
    # Update this path to your specific log folder
    concat_true = True

    log_path = "logs/Linear_controlled_KVAE_v2_test_norm_True_concat_False_free_spectrum_cartpole"
    
    controller = KoopmanLQRController(log_path, 
                                      checkpoint_step=9500,
                                      state_dim = 4, 
                                      latent_dim = 128, 
                                      hidden_dim = 256,
                                      concat_true = False
                 )
    
    # Simulation Settings
    dt = 0.1
    max_steps = 500
    x_curr = np.array([0.0, 0.0, 0.2, 0.0]) # Start with 0.2 rad tilt
    
    history = {"x": [], "u": [], "t": []}

    print(f"Starting simulation from state: {x_curr}")
    for i in range(max_steps):
        # 1. Get Koopman Action
        u_val = controller.get_control(x_curr)
        
        # 2. Log
        history["x"].append(x_curr.copy())
        history["u"].append(u_val)
        history["t"].append(i * dt)
        
        # 3. Physics Step (Sub-stepping for stability)
        for _ in range(10):
            dx = cartpole_controlled(x_curr, None, u_val)
            x_curr += np.array(dx) * (dt / 10.0)
            
        if i % 50 == 0:
            print(f"Time: {i*dt:.2f}s | Angle: {x_curr[2]:.4f} | Control: {u_val:.2f}")

    # Plotting
    hist_x = np.array(history["x"])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(history["t"], hist_x[:, 0], label="Position (x)")
    axs[0].set_ylabel("Pos (m)")
    
    axs[1].plot(history["t"], hist_x[:, 2], label="Angle (theta)", color='orange')
    axs[1].axhline(0, color='red', linestyle='--')
    axs[1].set_ylabel("Angle (rad)")
    
    axs[2].step(history["t"], history["u"], label="Control (u)", color='green')
    axs[2].set_ylabel("Force (N)")
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("simulation_results.png")

if __name__ == "__main__":
    main()