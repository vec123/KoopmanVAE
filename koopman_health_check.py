import os
import torch
import numpy as np
import scipy.linalg
import joblib
from models.models import ResidualMLP, KoopmanEncoder, LinearMatrix, TTKoopman

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

def check_physical_controllability(A, B, C, x_scaler=None, u_scaler=None):
    state_dim = C.shape[0]
    
    # --- Scaler Adjustment Logic ---
    # If scaled, we must adjust B and C matrices to reflect raw units
    # A_raw = A (Dynamics in latent space are unitless)
    # B_raw = B / u_std
    # C_raw = C * x_std
    
    B_phys = B.copy()
    C_phys = C.copy()
    
    if u_scaler is not None:
        # B relates normalized U to Latent. 
        # To get raw U -> Latent, we divide by the standard deviation.
        u_std = u_scaler.scale_[0]
        B_phys = B / u_std
        print(f"[Scaler] Adjusting B by 1/{u_std:.4f} (Control Scale)")

    if x_scaler is not None:
        # C relates Latent to normalized X.
        # To get Latent -> raw X, we multiply by the standard deviation.
        x_std = x_scaler.scale_.reshape(-1, 1)
        C_phys = C * x_std
        print(f"[Scaler] Adjusting C by state std devs: {x_scaler.scale_}")

    # Construct the physical controllability matrix: [C*B, C*A*B, C*A^2*B, ...]
    cols = []
    for i in range(state_dim):
        term = C_phys @ np.linalg.matrix_power(A, i) @ B_phys
        cols.append(term)
    
    Q_phys = np.hstack(cols)
    phys_rank = np.linalg.matrix_rank(Q_phys, tol=1e-5)
    
    print(f"[Physical] State-Space Rank: {phys_rank} / {state_dim}")
    s = np.linalg.svd(Q_phys, compute_uv=False)
    print(f"[Physical] Control Authority (Singular Values): {s}")
    
    return phys_rank, B_phys, C_phys

def diagnostic_check_desko():
    device = "cpu"
    dataset_name = "cartpole"
    checkpoint_step = 10000 
    
    # Path setup
    log_name = "Controlled_KVAE_v1_noise_free_horizon_15_full_A_latent_dim_50_scale"
    base_path = f"logs/{log_name}_{dataset_name}"

    # 1. Load Models
    state_dim, latent_dim, hidden_dim = 4, 50, 256
    encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim).to(device)  
    decoder = LinearMatrix(latent_dim, state_dim).to(device)
    system_matrix = LinearMatrix(latent_dim, latent_dim).to(device)
    control_matrix = LinearMatrix(1, latent_dim).to(device)

    # 2. Load Scalers
    x_scaler, u_scaler = None, None
    x_scaler_path = os.path.join(base_path, "x_scaler.pkl")
    u_scaler_path = os.path.join(base_path, "u_scaler.pkl")
    
    if os.path.exists(x_scaler_path):
        x_scaler = joblib.load(x_scaler_path)
        u_scaler = joblib.load(u_scaler_path)
        print("Scalers loaded successfully.")

    try:
        for name, m in {"encoder": encoder, "decoder": decoder, 
                          "system_matrix": system_matrix, "control_matrix": control_matrix}.items():
            path = os.path.join(base_path, f"{name}_{checkpoint_step}.pt")
            m.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        print(f"Error loading models: {e}"); return

    # 3. Extract Matrices
    A = get_matrix(system_matrix)      
    B_raw = get_matrix(control_matrix) 
    C = get_matrix(decoder)            
    B = B_raw.T if B_raw.shape[0] == 1 else B_raw

    print(f"\n{'='*20} DeSKO Scaled Health Check {'='*20}")

    # --- STEP 1: Spectral Stability ---
    max_eig = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"[Spectral] Spectral Radius: {max_eig:.6f}")

    # --- STEP 2: Physical Matrix Adjustment ---
    # We pass the scalers to get the "Physical" version of B and C
    phys_rank, B_phys, C_phys = check_physical_controllability(A, B, C, x_scaler, u_scaler)

    # --- STEP 3: DARE (LQR) Stabilizability ---
    # Use Bryson's Rule for Q and R weights based on raw units
    # (1 / max_allowable_error^2)
    Q_weights = np.array([1/1.0**2, 1/1.0**2, 1/0.1**2, 1/1.0**2]) # Cart, Vel, Theta, AngVel
    Q_x = np.diag(Q_weights)
    
    # Project raw weights into Latent Space using Physical C
    Q_z = C_phys.T @ Q_x @ C_phys + np.eye(latent_dim) * 1e-4
    R = np.eye(1) * (1 / 25.0**2) # Penalize relative to max force 25N
    R = np.eye(1)*0.01
    try:
        P = scipy.linalg.solve_discrete_are(A, B, Q_z, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        print(f"[DARE]     LQR Success. Gain Norm: {np.linalg.norm(K):.4f}")
    except:
        print("[DARE]     FAILED. System is not stabilizable.")

    print(f"{'='*50}\n")

if __name__ == "__main__":
    diagnostic_check_desko()