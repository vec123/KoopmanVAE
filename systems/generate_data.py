import os
import sys
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from registry import SYSTEM_REGISTRY
from simulators import simulate_euler_maruyama

load_dotenv()

def run(target_systems=None):
    """
    Generates trajectory data and plots for specified systems.
    If target_systems is None, generates data for all systems in the registry.
    """
    base_dir = os.getenv("DATASET_FOLDER")
    os.makedirs(f"{base_dir}/csv", exist_ok=True)
    os.makedirs(f"{base_dir}/plots", exist_ok=True)

    # Filter registry based on targets
    if target_systems:
        active_systems = {k: v for k, v in SYSTEM_REGISTRY.items() if k in target_systems}
        if not active_systems:
            print(f"Error: None of the requested systems {target_systems} found in registry.")
            return
    else:
        active_systems = SYSTEM_REGISTRY

    print(f"Generating data for {len(active_systems)} systems: {list(active_systems.keys())}")

    for name, data in active_systems.items():
        # Using sys_obj to avoid conflict with 'sys' module for CLI args
        sys_obj = data["instance"]
        cfg = data["config"]
        sys_obj.u_scale = cfg["u_scale"]

        print(f" -> Processing {name}...")

        #  Simulate: Returns X (N_traj, T, D_obs), U (N_traj, T, D_control)
        try:
            X, U = simulate_euler_maruyama(
                sys_obj, 
                n_traj=10, 
                seq_len=cfg["seq_len"], 
                dt=cfg["dt"], 
                substeps=cfg["substeps"],
                u_time_scale=cfg.get("u_time_scale", (0.0, 0.0))
            )
        except Exception as e:
            print(f"    [ERROR] Simulation failed for {name}: {e}")
            continue

        # Prepare directory structure
        sys_csv_dir = f"{base_dir}/csv/{name}"
        sys_plot_dir = f"{base_dir}/plots/{name}"
        os.makedirs(sys_csv_dir, exist_ok=True)
        os.makedirs(sys_plot_dir, exist_ok=True)

        n_traj, t_steps, d_obs = X.shape
        d_control = U.shape[2]
        
        # Define labels for CSV
        u_labels = [f"u_{j}" for j in range(d_control)]
        cols = ["time"] + sys_obj.labels + u_labels

        #Summary plot (overlapping first state of all trajectories)
        #fig_sum, ax_sum = plt.subplots(figsize=(12, 6))

        for i in range(n_traj):
            times = np.arange(t_steps) * cfg["dt"]
            
            # --- Save Individual CSV ---
            traj_matrix = np.hstack([times.reshape(-1, 1), X[i], U[i]])
            df_traj = pd.DataFrame(traj_matrix, columns=cols)
            df_traj.to_csv(f"{sys_csv_dir}/traj_{i:02d}.csv", index=False)

            # --- Individual Plot (Stacked Subplots per Trajectory) ---
            # Rows = Number of observations + 1 for control
            n_rows = d_obs + 1
            fig_ind, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows), sharex=True)
            
            # Handle case where n_rows is 1 (though unlikely here)
            if n_rows == 1: axes = [axes]

            # Plot each Observation/State dimension
            for d in range(d_obs):
                axes[d].plot(times, X[i, :, d], color='steelblue', lw=1.5)
                axes[d].set_ylabel(sys_obj.labels[d])
                axes[d].grid(True, alpha=0.3)

            # Plot Controls (as a step function)
            for j in range(d_control):
                axes[-1].step(times, U[i, :, j], where='post', label=u_labels[j], alpha=0.8, color='crimson')
            
            axes[-1].set_ylabel("Control (u)")
            axes[-1].set_xlabel("Time (s)")
            axes[-1].grid(True, alpha=0.3)
            if d_control > 1:
                axes[-1].legend(loc='upper right', fontsize='x-small')

            plt.suptitle(f"SYSTEM: {name.upper()} | TRAJECTORY: {i:02d}", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fig_ind.savefig(f"{sys_plot_dir}/traj_{i:02d}.png")
            plt.close(fig_ind)

            # --- Add to Overlap Summary ---
            #ax_sum.plot(times, X[i, :, 0], alpha=0.4)

        # Finalize and save the summary plot
        #ax_sum.set_title(f"Comparison: {name} (Primary Dimension: {sys_obj.labels[0]})")
        #ax_sum.set_xlabel("Time (s)")
        #ax_sum.set_ylabel(sys_obj.labels[0])
        #ax_sum.grid(True, alpha=0.3)
        #fig_sum.savefig(f"{base_dir}/plots/{name}_SUMMARY_OVERLAP.png")
        #plt.close(fig_sum)

    print("\nSuccess. All files and plots generated in the 'outputs/' directory.")

if __name__ == "__main__":
    # Check for command line arguments to target specific systems
    # Usage: python generate_dataset.py cartpole lorenz
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    target_systems=["hvac_linear"]
    run(target_systems=target_systems)

    """ 
    python
    from generate_dataset import run
    run(target_systems=["hvac_nonlinear"])
    """