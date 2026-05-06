import os
import torch
import numpy as np
import joblib
from models.models import LinearMatrix

def main():
    # 1. Configuration
    SENSOR_NAME ="ES0031405047432001AZ0F"
    LOG_NAME = f"Multi_KVAE_energy_{SENSOR_NAME}"
    LOG_DIR = f"logs/{LOG_NAME}"
    epoch = 5000
    KOOPMAN_PATH = os.path.join(LOG_DIR, f"system_matrix_{epoch}.pt")
    SAVE_PATH = os.path.join(LOG_DIR, f"koopman_eigendecomp_{epoch}.pkl")
    device = "cpu" # Eigendecomposition is usually faster on CPU for this size

    if not os.path.exists(KOOPMAN_PATH):
        print(f"Error: Could not find model at {KOOPMAN_PATH}")
        return

    # 2. Load the Koopman Weights
    print(f"Loading Koopman Matrix from {KOOPMAN_PATH}...")
    K_obj = torch.load(KOOPMAN_PATH, map_location=device)
    
    # Handle the LinearMatrix class structure (linear.weight)
    if isinstance(K_obj, dict):
        if 'linear.weight' in K_obj:
            K_mat = K_obj['linear.weight'].detach().numpy()
        elif 'weight' in K_obj:
            K_mat = K_obj['weight'].detach().numpy()
        else:
            raise KeyError(f"Weight key not found. Available: {K_obj.keys()}")
    else:
        K_mat = K_obj.detach().numpy()

    # 3. Perform Eigendecomposition
    # K*V = V*L -> z = V*alpha -> alpha = V_inv * z
    print(f"Computing decomposition for {K_mat.shape} matrix...")
    eigenvalues, eigenvectors = np.linalg.eig(K_mat)
    
    # Pre-calculate the inverse for projection
    # pinv is safer for high-dimensional latent spaces (265 dim)
    v_inv = np.linalg.pinv(eigenvectors)

    # 4. Save Metadata
    # We save as a dictionary for easy access in other scripts
    decomp_data = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "v_inv": v_inv,
        "latent_dim": K_mat.shape[0]
    }

    joblib.dump(decomp_data, SAVE_PATH)
    
    print("-" * 30)
    print(f"SUCCESS: Eigendecomposition saved to {SAVE_PATH}")
    print(f"Number of modes: {len(eigenvalues)}")
    
    # Quick Analysis
    n_stable = np.sum(np.abs(eigenvalues) <= 1.001)
    print(f"Stable modes (|λ| <= 1): {n_stable} / {len(eigenvalues)}")
    print("-" * 30)

        
    # 4. Detailed Analysis of Spectral Modes
    print("-" * 30)
    print(f"SUCCESS: Eigendecomposition saved to {SAVE_PATH}")
    print(f"Number of modes: {len(eigenvalues)}")
    
    # Check for stable modes (on or inside unit circle)
    mags = np.abs(eigenvalues)
    n_stable = np.sum(mags <= 1.001)
    
    # NEW: Check for modes specifically near 1 (Periodic/Constant behaviors)
    # We use a tolerance (atol) to find modes effectively on the unit circle
    is_near_unit_circle = np.isclose(mags, 1.0, atol=1e-3)
    n_unit_circle = np.sum(is_near_unit_circle)
    
    # NEW: Check for the "Identity" mode (λ = 1 + 0j)
    # This represents the steady-state mean/bias of the system
    is_identity = np.isclose(eigenvalues, 1.0 + 0j, atol=1e-3)
    n_identity = np.sum(is_identity)

    print(f"Stable modes (|λ| <= 1):      {n_stable} / {len(eigenvalues)}")
    print(f"Periodic modes (|λ| ≈ 1):    {n_unit_circle} / {len(eigenvalues)}")
    print(f"Identity modes (λ ≈ 1+0j):   {n_identity}")
    
    # Optional: Print the top 5 periodic frequencies (angles of λ)
    if n_unit_circle > 0:
        unit_eigenvals = eigenvalues[is_near_unit_circle]
        # Sort by angle to find primary frequencies
        angles = np.angle(unit_eigenvals)
        print(f"Top 5 Periodic angles (rad): {np.sort(np.unique(np.abs(angles)))[:5]}")
    print("-" * 30)

    

if __name__ == "__main__":
    main()