# simulate_and_plot.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from systems import SYSTEMS

def simulate_system(system_name, initial_condition, T=20, dt=0.01, save_path="trajectory.png"):
    if system_name not in SYSTEMS:
        raise ValueError(f"System '{system_name}' not found. Available: {list(SYSTEMS.keys())}")

    system = SYSTEMS[system_name]
    t = np.arange(0, T, dt)

    # Solve ODE
    traj = odeint(system, initial_condition, t)

    # Plot trajectory
    plt.figure(figsize=(6,6))
    if traj.shape[1] == 2:
        plt.plot(traj[:,0], traj[:,1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{system_name} trajectory")
    elif traj.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.plot3D(traj[:,0], traj[:,1], traj[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{system_name} trajectory")
    else:
        raise ValueError("System dimension not supported")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Trajectory saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Choose system and initial condition
    simulate_system("van_der_pol", initial_condition=[2, 0], T=20, save_path="vdp.png")
    simulate_system("limit_cycle", initial_condition=[0.5, 0.5], T=20, save_path="lc.png")
    simulate_system("lorenz", initial_condition=[1,1,1], T=50, save_path="lorenz.png")
