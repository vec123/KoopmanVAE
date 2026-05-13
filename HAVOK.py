import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def run_havok_analysis_plot():
    # --- 1. Data Loading ---
    dataset_path = os.getenv("DATASET_PATH", "./data") 
    files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    if not files:
        print("No files found.")
        return

    df = pd.read_csv(files[0])
    df.columns = df.columns.str.replace('$', '', regex=False).str.strip()
    
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    limit = 50000 
    true_xyz = df[['x', 'y', 'z']].values[:limit]
    t = df['time'].values[:limit]

    # --- 2. Build Hankel Matrix & SVD (Embedding) ---
    q = 100 
    x_signal = true_xyz[:, 0]
    m = len(x_signal)
    H = np.zeros((q, m - q + 1))
    for i in range(q):
        H[i, :] = x_signal[i : (m - q + 1 + i)]
    
    u, sigma, vh = np.linalg.svd(H, full_matrices=False)
    V = vh.T 
    r = 15
    V_star = V[:, :r]

    # --- 3. Index Alignment for Trajectories ---
    # Hankel starts at q-1. Derivative (Central Diff) starts 2 steps later.
    # Total shift = q + 1
    t_offset = q + 1
    n_plot = 3*15000 # Number of points for trajectory plots
    t_plot = t[t_offset : t_offset + n_plot]
    
    # Slice data to match t_plot exactly
    true_plot = true_xyz[t_offset : t_offset + n_plot]
    # v_r (forcing) is the r-th column (index r-1)
    # Because V starts at index q-1, we offset by 2 to align with t_plot
    forcing = V_star[2:2+n_plot, r-1]

    # --- 4. Plotting ---
    fig = plt.figure(figsize=(18, 12))
    
    # A. True Lorenz Attractor (3D)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(true_xyz[:,0], true_xyz[:,1], true_xyz[:,2], color='black', lw=0.5, alpha=0.6)
    ax1.set_title("True Lorenz Attractor", fontweight='bold')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    # B. Embedded Attractor (v1, v2, v3)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(V_star[:,0], V_star[:,1], V_star[:,2], color='teal', lw=0.7)
    ax2.set_title("Embedded Attractor ($v_1, v_2, v_3$)", fontweight='bold')
    ax2.set_xlabel("$v_1$"); ax2.set_ylabel("$v_2$"); ax2.set_zlabel("$v_3$")

    # C. X Trajectory
    ax3 = fig.add_subplot(4, 3, 3)
    ax3.plot(t_plot, true_plot[:, 0], color='tab:blue', lw=1)
    ax3.set_ylabel("X")
    ax3.grid(True, alpha=0.3)

    # D. Y Trajectory
    ax4 = fig.add_subplot(4, 3, 6)
    ax4.plot(t_plot, true_plot[:, 1], color='tab:green', lw=1)
    ax4.set_ylabel("Y")
    ax4.grid(True, alpha=0.3)

    # E. Z Trajectory
    ax5 = fig.add_subplot(4, 3, 9)
    ax5.plot(t_plot, true_plot[:, 2], color='tab:red', lw=1)
    ax5.set_ylabel("Z")
    ax5.grid(True, alpha=0.3)

    # F. Forcing Term (v_r)
    ax6 = fig.add_subplot(4, 3, 12)
    threshold = 0.002
    active = np.abs(forcing) > threshold
    ax6.plot(t_plot, forcing, color='silver', lw=1)
    ax6.plot(t_plot, np.where(active, forcing, np.nan), color='red', lw=1, label="Bursting")
    ax6.set_ylabel("$v_{15}$ (Forcing)")
    ax6.set_xlabel("Time (s)")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "havok_analysis_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"Analysis saved to {save_path}")

if __name__ == "__main__":
    run_havok_analysis_plot()