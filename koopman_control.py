import os
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import joblib
import cvxpy as cp
# Local project imports
from systems.controlled_systems import cartpole_controlled, simple_independent_linear, simple_nonlinear_spring, complex_nonlinear_system
from models.models import LinearMatrix, KoopmanEncoder, LinearKoopmanEncoder

def apply_trigonometric_embedding(X, is_batch=True):
        """
        Transforms state from [x, x_dot, theta, theta_dot] 
        to [x, x_dot, sin(theta), cos(theta), theta_dot].
        
        Args:
            X: np.ndarray of shape (N, T, 4) if is_batch=True, 
            or (4,) if is_batch=False.
        Returns:
            np.ndarray of shape (N, T, 5) or (5,).
        """
        if is_batch:
            # X shape: (N, T, 4)
            x_pos     = X[:, :, 0:1]
            x_dot     = X[:, :, 1:2]
            theta     = X[:, :, 2:3]
            theta_dot = X[:, :, 3:4]
            
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            
            return np.concatenate([x_pos, x_dot, sin_t, cos_t, theta_dot], axis=-1)
        
        else:
            # X shape: (4,)
            x, x_dot, theta, theta_dot = X
            return np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])


class KoopmanLQRController:
    """
    Object-oriented wrapper to handle the Koopman + LQR control loop.
    Works with standard scalers or defaults to Identity if files are missing.
    """
    def __init__(self, base_path, x_target, Q, R, device="cpu", checkpoint_step=10000, 
                 state_dim=4, u_dim =1, latent_dim=128, hidden_dim=256,
                 concat_true=True):
        self.device = device
        self.concat_true = concat_true
        
        # 1. Load Scalers with Fallback
        self.x_scaler = self._load_scaler(base_path, "x_scaler.pkl", state_dim)
        self.u_scaler = self._load_scaler(base_path, "u_scaler.pkl", u_dim)
        
        # 2. Re-initialize and Load Models
        self.state_dim, self.u_dim, self.latent_dim, self.hidden_dim = state_dim, u_dim, latent_dim, hidden_dim
        self.encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim, hidden_depth=5).to(device)
        #self.encoder = LinearKoopmanEncoder(state_dim, latent_dim).to(device)
        
        actual_latent_dim = latent_dim + state_dim if concat_true else latent_dim
        
        self.decoder = LinearMatrix(actual_latent_dim, state_dim).to(device)
        self.system_net = LinearMatrix(actual_latent_dim, actual_latent_dim).to(device)
        self.control_net = LinearMatrix(u_dim, actual_latent_dim).to(device)

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
        if self.B.shape[0] == u_dim: self.B = self.B.T 
        self.C = self._get_weight(self.decoder)


        self.x_target = x_target
        self.Q_phys_diag = Q 
        self.Q_phys = np.diag(self.Q_phys_diag)
        if self.x_scaler:
         C_phys = self.C * self.x_scaler.scale_.reshape(-1, 1)
        else:
         C_phys = self.C

        self.Q_latent = C_phys.T @ self.Q_phys @ C_phys + np.eye(actual_latent_dim) * 1e-6 

        self.R_diag = R
        self.R = np.diag(self.R_diag)

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

        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q_latent, self.R)
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        return K

    def _get_z_target(self):
        x_target =  self.x_target
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
            # 1. Handle Trigonometric Embedding for Cartpole if necessary
            # x_phys starts as [x, x_dot, theta, theta_dot]
            if x_phys.shape[-1] == 4 and self.x_scaler.mean_.shape[0] == 5:
                x, x_dot, theta, theta_dot = x_phys.flatten()
                x_phys = np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

            # 2. Normalize and move to device
            x_norm = self.x_scaler.transform(x_phys.reshape(1, -1))
            x_t = torch.as_tensor(x_norm, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                enc_out = self.encoder(x_t)
                # Split mu and logstd (assuming VAE encoder output)
                mu_all, _ = torch.chunk(enc_out, 2, dim=-1)
                z_curr = mu_all.cpu().numpy().reshape(-1, 1) # Force column vector

            # 3. Handle state concatenation if used in training
            if self.concat_true:
                z_curr = np.concatenate((z_curr, x_norm.reshape(-1, 1)), axis=0)

            # 4. LQR Control Calculation
            # Ensure z_target is a column vector
            z_target_vec = self.z_target.reshape(-1, 1)
            u_norm = -self.K @ (z_curr - z_target_vec)
            
            # 5. Safe Denormalization
            # u_norm is now a vector/matrix, we take the first element for u_phys
            # Flatten u_norm to ensure it's a 1D array [u1, u2, ...]
            u_vec = u_norm.flatten()
            u_phys = (u_vec * self.u_scaler.scale_) + self.u_scaler.mean_
            return np.clip(u_phys, -25.0, 25.0)


class KoopmanMPCController(KoopmanLQRController): # Inherit to keep loading logic
    def __init__(self, *args, horizon=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        
        self.u_limit = 1.0 # Normalized limit (approx based on your u_scaler)
        self.upper_u = 50
        self.lower_u = -50
    def get_control(self, x_phys):
        # 1. Transform and Normalize (Keep your existing logic)
        if x_phys.shape[-1] == 4 and self.x_scaler.mean_.shape[0] == 5:
            x, x_dot, theta, theta_dot = x_phys.flatten()
            x_phys = np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

        x_norm = self.x_scaler.transform(x_phys.reshape(1, -1))
        x_t = torch.as_tensor(x_norm, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            enc_out = self.encoder(x_t)
            mu_all, _ = torch.chunk(enc_out, 2, dim=-1)
            z_curr = mu_all.cpu().numpy().reshape(-1)

        if self.concat_true:
            z_curr = np.concatenate((z_curr, x_norm.reshape(-1)), axis=0)

        # 2. Solve MPC Optimization Problem
        u_opt = self._solve_mpc(z_curr)
        
        # 3. Denormalize the first control action
        u_phys = (u_opt * self.u_scaler.scale_) + self.u_scaler.mean_
        return np.clip(u_phys, -2500.0, 2500.0)

    def _solve_mpc(self, z_initial):
        z_dim = self.A.shape[0]
        u_dim = self.u_dim
        H = self.horizon
        
        # Variables
        z = cp.Variable((z_dim, H + 1))
        u = cp.Variable((u_dim, H))
        
        # Target
        z_ref = self.z_target.flatten()
        
        # Cost and Constraints
        cost = 0
        constraints = [z[:, 0] == z_initial]

        for t in range(H):
            # Cost: State error + Control effort
            cost += cp.quad_form(z[:, t] - z_ref, self.Q_latent )
            cost += cp.quad_form(u[:, t], np.eye(u_dim) * self.R)
            
            # Dynamics Constraint
            constraints += [z[:, t+1] == self.A @ z[:, t] + self.B @ u[:, t]]
            
            # Control Limits (Normalized Space)
            # You can get these precisely from your u_scaler limits
            constraints += [u[:, t] <= self.upper_u, u[:, t] >= self.lower_u] 

        # Final terminal cost (optional but helps stability)
        cost += cp.quad_form(z[:, H] - z_ref, self.Q_latent )
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
            return np.zeros(u_dim)
            
        return u[:, 0].value # Return only the first action (Receding Horizon)
# -------------------------------------------------------
# Execution Script
# -------------------------------------------------------

def main():
    # Update this path to your specific log folder
    concat_true = True
    system_type = "complex_nonlinear_system"
    #system_type = "simple_independent_linear"
    log_path = "logs/Test_underactuated_nonlinear_encoder_normalized_False_state_concat_True_control_enc_False_spec_free_latent_dim_32_complex_nonlinear_system"
    #log_path = "logs/Test_linear_system_nonlinear_encoder_normalized_False_state_concat_True_control_enc_False_spec_free_latent_dim_4_simple_independent_linear"
    
    # Simulation Settings
    dt = 0.1
    max_steps = 500
    x_curr = np.array([0.0, 0.0, 0.0, 0.0]) # Start with 0.2 rad tilt
    #x_target = np.zeros((1,4))
    x_target = np.array([1.0, 0.0, 1.0, 0.0])
     
    Q_diag = np.array([0.0, 0.0, 1000.0, 0.0])
   # Q_diag = np.array([10.0, 0.0, 1000.0, 1000.0, 0.0])

    R_diag = np.array([0.01 ])
    #R_diag = np.array([0.01])
    controller = KoopmanMPCController(log_path, 
                                      x_target,
                                      Q_diag,
                                      R_diag,
                                      checkpoint_step=5000,
                                      state_dim = 4, 
                                      u_dim = 1,
                                      latent_dim = 32, 
                                      hidden_dim = 128,
                                      concat_true = concat_true
                 )
    

    history = {"x": [], "u": [], "t": []}

    print(f"Starting simulation from state: {x_curr}")
    for i in range(max_steps):
        if system_type == "cartpole":
            #print("x_curr.shape: ", x_curr.shape)
            x_embed = apply_trigonometric_embedding(x_curr, is_batch=False)
            #print("x_embed: ", x_embed)
            labels = [r"$x$", r"$\dot{x}$", r"$\sin(\theta)$", r"$\cos(\theta)$", r"$\dot{\theta}$"]
            u_val = controller.get_control(x_embed)
        else:
            u_val = controller.get_control(x_curr)
       # print("u_val: ", u_val)
        # 2. Log
        history["x"].append(x_curr.copy())
        history["u"].append(u_val)
        history["t"].append(i * dt)
        
        # 3. Physics Step (Sub-stepping for stability)
        for _ in range(10):
            if system_type == "cartpole":
                dx = cartpole_controlled(x_curr, None, u_val)
            elif system_type == "simple_independent_linear" :
             dx = simple_independent_linear(x_curr, None, u_val)
            elif system_type == "simple_nonlinear_spring":
                dx = simple_nonlinear_spring(x_curr, None, u_val)
            elif system_type == "complex_nonlinear_system":
                dx = complex_nonlinear_system(x_curr, None, u_val)
            else:
                raise ValueError("System not supported")
            x_curr += np.array(dx) * (dt / 10.0)
        
        if i % 50 == 0:
                # Format control values: handles 1D (scalar) or 2D (array)
                if isinstance(u_val, (np.ndarray, list)):
                    u_str = ", ".join([f"{v:.2f}" for v in u_val])
                else:
                    u_str = f"{u_val:.2f}"
                
                print(f"Time: {i*dt:.2f}s | State: {x_curr} | Control: [{u_str}]")


    def plot_results_dynamic(history, state_labels, control_labels=None):
        """
        Infers dimensionality from hist_x and hist_u and plots everything.
        
        Args:
            history: dict containing 'x', 'u', and 't'
            state_labels: list of strings, e.g., [r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
            control_labels: list of strings, e.g., [r"$F_x$"]
        """
        hist_x = np.array(history["x"]) # Shape: (N_steps, state_dim)
        hist_u = np.array(history["u"]) # Shape: (N_steps, u_dim)
        t_axis = np.array(history["t"])
        
        # Ensure u is 2D
        if hist_u.ndim == 1:
            hist_u = hist_u.reshape(-1, 1)
            
        state_dim = hist_x.shape[1]
        u_dim = hist_u.shape[1]
        total_plots = state_dim + u_dim
        
        fig, axs = plt.subplots(total_plots, 1, figsize=(10, 2.5 * total_plots), sharex=True)
        
        # If there is only 1 plot, matplotlib returns a single axis object instead of an array
        if total_plots == 1:
            axs = [axs]

        # 1. Plot State Dimensions
        for i in range(state_dim):
            label = state_labels[i] if i < len(state_labels) else f"State {i}"
            axs[i].plot(t_axis, hist_x[:, i], label=label, color=f"C{i}") # Use default color cycle
            axs[i].set_ylabel(label)
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc="upper right")

        # 2. Plot Control Dimensions
        for j in range(u_dim):
            ax_idx = state_dim + j
            # Try to get a specific label, otherwise default to u{j}
            if control_labels and j < len(control_labels):
                u_label = control_labels[j]
            else:
                u_label = f"Control $u_{j}$"
                
            axs[ax_idx].step(t_axis, hist_u[:, j], where='post', label=u_label, color='black', alpha=0.8)
            axs[ax_idx].set_ylabel(u_label)
            axs[ax_idx].grid(True, alpha=0.3)
            axs[ax_idx].legend(loc="upper right")

        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("simulation_results.png")

    labels = [r"$x$", r"$\dot{x}$"]
    plot_results_dynamic(history, labels, [r"Force"])
   
if __name__ == "__main__":
    main()