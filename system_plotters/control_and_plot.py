
import matplotlib.pyplot as plt
import os
from .controlled_simulators import generate_trajectories_odeint, generate_trajectories_euler_maruyama

os.makedirs("system_trajectory_plots", exist_ok=True)

if __name__ == "__main__":
    # Metadata for plotting
    config = {
        "pendulum": {
            "labels": [r"$\theta$ (Angle)", r"$\omega$ (Ang Vel)"],
            "colors": ["royalblue", "navy"]
        },
        "cartpole": {
            "labels": [r"$x$ (Pos)", r"$\dot{x}$ (Vel)", r"$\theta$ (Angle)", r"$\dot{\theta}$ (Ang Vel)"],
            "colors": ["forestgreen", "darkgreen", "crimson", "darkred"]
        }
    }

    for system_type in [ "cartpole"]:
        for sim_method in [ "euler"]:
            T = 5000
            dt = 0.002
            noise_lvl = 0.1
            if sim_method == "odeint":
                X, U, t = generate_trajectories_odeint(system_type, n_traj=1, seq_len=T, dt=dt,  noise_lvl=noise_lvl)
            else:
                X, U, t = generate_trajectories_euler_maruyama(system_type, n_traj=1, seq_len=T, dt=dt,  noise_lvl=[0.0,0.0], control=True)
            n_states = X.shape[2]
            
            # Create a subplot for each state + one for control
            fig, axes = plt.subplots(n_states + 1, 1, figsize=(10, 2 * (n_states + 1)), sharex=True)
            
            # Plot each state
            for i in range(n_states):
                axes[i].plot(t, X[0, :, i], color=config[system_type]["colors"][i], label=config[system_type]["labels"][i])
                axes[i].set_ylabel("Magnitude") 
                axes[i].legend(loc="upper right")
                axes[i].grid(True, alpha=0.3)
            
            # Plot Control signal at the bottom
            axes[-1].step(t, U[0, :, 0], color='gray', alpha=0.7, where='post', label='Control (u)')
            axes[-1].set_ylabel("Force/Torque")
            axes[-1].set_xlabel("Time (s)")
            axes[-1].legend(loc="upper right")
            axes[-1].grid(True, alpha=0.3)
            
            plt.suptitle(f"Complete State Trajectories: {system_type.capitalize()}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = f"system_trajectory_plots/{system_type}_all_states_{sim_method}.png"
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.show()