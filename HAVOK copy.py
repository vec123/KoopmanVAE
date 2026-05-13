import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
def sequential_threshold_least_squares(Theta, dXdt, thres, niter=10):
    """SINDy optimization: sequential threshold least squares."""
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
    for k in range(niter):
        mask_under_thres = (np.abs(Xi) < thres)
        Xi[mask_under_thres] = 0
        for j in range(Xi.shape[1]):
            mask_above_thres = ~mask_under_thres[:, j]
            Xi[mask_above_thres, j] = np.linalg.lstsq(Theta[:, mask_above_thres], dXdt[:, j], rcond=None)[0]
    return Xi

def run_havok_comparison_plot():
    # --- 1. Data Loading ---
    # Assuming the same structure as your environment
    dataset_path = os.getenv("DATASET_PATH", "./data") 
    files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    if not files:
        print("No files found.")
        return

    df = pd.read_csv(files[0])
    df.columns = df.columns.str.replace('$', '', regex=False).str.strip()
    
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    limit = 50000 
    x_signal = df['x'].values[:limit]
    t = df['time'].values[:limit]

    # --- 2. Build Hankel Matrix & SVD ---
    q = 100 # Number of delays
    m = len(x_signal)
    H = np.zeros((q, m - q + 1))
    for i in range(q):
        H[i, :] = x_signal[i : (m - q + 1 + i)]
    
    u, sigma, vh = np.linalg.svd(H, full_matrices=False)
    # V is the embedded coordinates (normalized)
    V = vh.T 

    # --- 3. Linear Regression (HAVOK Model) ---
    r = 15 # Rank as per paper logic
    V_star = V[:, :r]
    
    # Compute derivative using 4th order central difference
    dv = np.zeros((V_star.shape[0] - 5, r))
    for i in range(2, V_star.shape[0] - 3):
        dv[i-2, :] = (1/(12*dt)) * (-V_star[i+2, :] + 8*V_star[i+1, :] - 8*V_star[i-1, :] + V_star[i-2, :])
    
    # Trim V_star to match dv dimensions
    V_trimmed = V_star[2 : (dv.shape[0] + 2), :]
    
    # SINDy library (Linear terms + Bias)
    Theta = np.column_stack([np.ones(len(V_trimmed)), V_trimmed])
    
    # Regression for the first r-1 components
    # The r-th component (v_r) acts as the forcing term
    lamb = 0.0 # Threshold (0 for pure least squares as in your target example)
    Xi = sequential_threshold_least_squares(Theta, dv[:, :r-1], lamb)
    
    # Extract A (dynamics) and B (forcing)
    # Xi[0,:] is bias, Xi[1:r, :] is A, Xi[r, :] is B
    A = Xi[1:r, :].T
    B = Xi[r, :].reshape(-1, 1)

    # --- 4. Reconstruction/Simulation ---
    # To reconstruct the signal, we use the linear system: dv = Av + Bvr
    # We'll use the original forcing v_r to see how well the linear part matches v1
    v_forcing = V_trimmed[:, r-1]
    v_recon = np.zeros((len(V_trimmed), r-1))
    v_recon[0, :] = V_trimmed[0, :r-1]
    
    # Simple Euler integration for demonstration (matching the notebooks logic)
    for i in range(len(V_trimmed) - 1):
        v_recon[i+1, :] = v_recon[i, :] + (A @ v_recon[i, :] + B.flatten() * v_forcing[i]) * dt

    # --- 5. Plotting ---
    n_plot = 10000
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    n_available = len(V_trimmed) 

    # Ensure n_plot does not exceed available data
    n_plot = min(10000, n_available)

    # Slice the time array to match the START and LENGTH of V_trimmed
    # The derivative window starts at index 2, and we have n_plot points.
    t_plot = t[q + 2 : q + 2 + n_plot]

    # Now the dimensions will match (n_plot,)
    ax[0].plot(t_plot, V_trimmed[:n_plot, 0], color='gray', alpha=0.5, lw=3, label='True (SVD)')
    ax[0].plot(t_plot, v_recon[:n_plot, 0], color='gold', lw=1.5, label='HAVOK Linear Recon')
    ax[0].set_ylabel('$v_1$ Amplitude')
    ax[0].legend()
    ax[0].set_title("HAVOK Reconstruction of Embedded Dynamics")

    # Plot forcing term Activity
    forcing = V_trimmed[:n_plot, r-1]
    threshold = 0.002
    active = np.abs(forcing) > threshold
    
    ax[1].plot(t[q:q+n_plot], forcing, color='silver', label='Forcing Term $v_r$')
    # Highlight activity
    ax[1].plot(t[q:q+n_plot], np.where(active, forcing, np.nan), color='red', label='Active Forcing')
    ax[1].axhline(threshold, color='black', linestyle='--', alpha=0.3)
    ax[1].axhline(-threshold, color='black', linestyle='--', alpha=0.3)
    ax[1].set_ylabel('$v_r$ (Forcing)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("havok_fixed_comparison.png", dpi=300)
    print("Fixed HAVOK plot saved.")

if __name__ == "__main__":
    run_havok_comparison_plot()