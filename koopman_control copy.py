import os
import torch
import numpy as np
import cvxpy as cp
import scipy.linalg
import matplotlib.pyplot as plt

# Local imports
from system_plotters.controlled_systems import cartpole_controlled
from models.models import LinearMatrix, TTKoopman, KoopmanEncoder

def get_matrix(model):
    if hasattr(model, 'get_matrix'):
        return model.get_matrix().detach().cpu().numpy()
    for attr in ["matrix", "weight", "A", "W"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (torch.nn.Parameter, torch.Tensor)):
                return val.detach().cpu().numpy()
            if isinstance(val, torch.nn.Linear):
                return val.weight.detach().cpu().numpy()
    return list(model.parameters())[0].detach().cpu().numpy()

class KoopmanMPC:
    def __init__(self, A, B, Q, R, horizon, u_limit=25.0):
        self.A, self.B, self.Q, self.R = A, B, Q, R
        self.N, self.nx, self.nu = horizon, A.shape[0], B.shape[1]
        self.u_limit = u_limit

        self.z_init = cp.Parameter((self.nx, 1))
        self.z_target = cp.Parameter((self.nx, 1))
        self.u_seq = cp.Variable((self.nu, self.N))
        self.z_seq = cp.Variable((self.nx, self.N + 1))

        cost = 0
        constraints = [self.z_seq[:, 0:1] == self.z_init]
        for k in range(self.N):
            cost += cp.quad_form(self.z_seq[:, k:k+1] - self.z_target, self.Q)
            cost += cp.quad_form(self.u_seq[:, k:k+1], self.R)
            constraints += [self.z_seq[:, k+1:k+2] == self.A @ self.z_seq[:, k:k+1] + self.B @ self.u_seq[:, k:k+1]]
            constraints += [cp.abs(self.u_seq[:, k]) <= self.u_limit]

        cost += cp.quad_form(self.z_seq[:, self.N:] - self.z_target, self.Q)
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, z_curr, z_target):
        self.z_init.value = z_curr.reshape(-1, 1)
        self.z_target.value = z_target.reshape(-1, 1)
        self.prob.solve(solver=cp.OSQP, warm_start=True)
        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return 0.0
        return self.u_seq.value[0, 0]

def main():
    device = "cpu"
    checkpoint_step = 1800
    base_path = "logs/Controlled_KVAE_v2_noise_free_horizon_15_full_A_cartpole"

    dt = 0.01
    sub_steps = 10

    # 1. Load Models
    state_dim, latent_dim, hidden_dim = 4, 128, 256
    encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim).to(device) 
    decoder = LinearMatrix(latent_dim, state_dim)
    #system_matrix = TTKoopman(latent_dim, 16, [(8, 8), (16, 16)])
    system_matrix =  LinearMatrix(latent_dim, latent_dim).to(device)
    control_matrix = LinearMatrix(1, latent_dim)

    for name, m in {"encoder": encoder, "decoder": decoder, "system_matrix": system_matrix, "control_matrix": control_matrix}.items():
        m.load_state_dict(torch.load(f"{base_path}/{name}_{checkpoint_step}.pt", map_location=device))
    encoder.eval()

    # 2. Extract Matrices
    A = get_matrix(system_matrix)
    B = get_matrix(control_matrix) * 50.0 
    if B.shape[0] == 1: B = B.T 
    C = get_matrix(decoder)

    # Weights: Penalty on physical states projected into latent space
    Q_x = np.diag([5.0, 1.0, 10000.0, 1.0])
    # Initial construction
    Q_z_raw = C.T @ Q_x @ C
    Q_z_sym = (Q_z_raw + Q_z_raw.T) / 2.0 # Ensure perfect symmetry
    
    # Robust PSD projection:
    # We decompose Q_z, take the eigenvalues, and force them to be >= 1e-6
    vals, vecs = np.linalg.eigh(Q_z_sym)
    vals = np.maximum(vals, 1e-6) 
    Q_z = vecs @ np.diag(vals) @ vecs.T
    
    # Double check symmetry for the solver
    Q_z = (Q_z + Q_z.T) / 2.0
    
    # MPC Horizon: Foresight for longer trajectories
    mpc = KoopmanMPC(A, B, Q_z, np.eye(1)*0.001, horizon=40)
    
    with torch.no_grad():
        z_target = torch.chunk(encoder(torch.zeros(1, 4)), 2, dim=-1)[0].numpy()

    # 3. Simulation
    # Reducing initial tilt slightly to see if stabilization is possible
    x_curr = np.array([0.0, 0.0, 0.15, 0.0]) 
    hist_x, hist_u = [], []

    print("Running simulation (Target: 10 seconds)...")
    max_steps = 1000 # 10 seconds at dt=0.01
    
    for i in range(max_steps):
        x_t = torch.from_numpy(x_curr).float().view(1, 4)
        with torch.no_grad():
             z_curr = torch.chunk(encoder(x_t), 2, dim=-1)[0].numpy()
        print("solving")
        u_opt = mpc.solve(z_curr, z_target)
        print("solved")
        hist_x.append(x_curr.copy()); hist_u.append(u_opt)
        
        # APPLY CONTROL: Removed the 0* multiplier
        x_next = x_curr.copy()
        for _ in range(sub_steps):
            dx = cartpole_controlled(x_next, None, u_opt)
            x_next += np.array(dx) * (dt / sub_steps)
        x_curr = x_next
        
        # Exit if pole falls
        #if np.abs(x_curr[2]) > np.pi/2: 
        #    print(f"Failed at {i*0.01:.2f}s")
         #   break
    else:
        print("Success! Completed full trajectory.")

    # 4. Final Plotting (Same logic as before)
    hist_x, hist_u = np.array(hist_x), np.array(hist_u)
    t = np.arange(len(hist_x)) * 0.01
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    labels = ["Position (m)", "Velocity (m/s)", "Angle (rad)", "Ang. Vel (rad/s)", "Control (N)"]
    data = [hist_x[:,0], hist_x[:,1], hist_x[:,2], hist_x[:,3], hist_u]
    colors = ['#1f77b4', '#17becf', '#ff7f0e', '#d62728', '#2ca02c']

    for i in range(5):
        axs[i].plot(t, data[i], color=colors[i], lw=2)
        axs[i].set_title(labels[i])
        axs[i].grid(True, alpha=0.3)
        d_range = np.max(data[i]) - np.min(data[i])
        margin = d_range * 0.1 if d_range > 0 else 0.1
        axs[i].set_ylim(np.min(data[i]) - margin, np.max(data[i]) + margin)

    plt.tight_layout()
    plt.savefig("final_report.png")
    print("Report saved to final_report.png")

if __name__ == "__main__":
    main()