# simulate_and_plot.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from systems import SYSTEMS

def simulate_system(system_name, initial_conditions, T=20, dt=0.01, save_path="trajectory.png"):
    if system_name not in SYSTEMS:
        raise ValueError(f"System '{system_name}' not found. Available: {list(SYSTEMS.keys())}")

    system = SYSTEMS[system_name]
    t = np.arange(0, T, dt)

    # Handle single initial condition (wrap in list)
    if isinstance(initial_conditions[0], (int, float)):
        initial_conditions = [initial_conditions]

    dim = len(initial_conditions[0])

    if dim == 3:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

    for idx, ic in enumerate(initial_conditions):
        traj = odeint(system, ic, t)
        if dim == 2:
            plt.plot(traj[:,0], traj[:,1], lw=0.8, alpha=0.6)
        elif dim == 3:
            ax.plot3D(traj[:,0], traj[:,1], traj[:,2], lw=0.5, alpha=0.3, color='blue')
        else:
            raise ValueError("System dimension not supported")

    # Labels & titles
    if dim == 2:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{system_name} phase portrait")
        plt.axis('equal')
    elif dim == 3:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{system_name} Lorenz cloud")
        ax.view_init(elev=30, azim=120)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Phase portrait saved to {save_path}")

# ----------------------------------
# Example usage
# ----------------------------------
if __name__ == "__main__":
    # 2D systems
    grid_x = np.linspace(-3, 3, 10)
    grid_y = np.linspace(-3, 3, 10)
    initial_conditions_2d = [[x, y] for x in grid_x for y in grid_y]
    simulate_system("van_der_pol", initial_conditions_2d, T=150, save_path="vdp_mass.png")
    simulate_system("limit_cycle", initial_conditions_2d, T=150, save_path="lc_mass.png")

    # 3D Lorenz: small random cloud of initial conditions
    np.random.seed(42)
    initial_conditions_3d = np.random.uniform(low=-0.5, high=0.5, size=(200,3)) + np.array([1,1,1])
    simulate_system("lorenz", initial_conditions_3d, T=150, save_path="lorenz_cloud.png")
