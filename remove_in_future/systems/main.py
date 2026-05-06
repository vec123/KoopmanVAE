
from simulators import generate_trajectories_euler_maruyama
from systems.registry import SYSTEM_REGISTRY
import torch

DATASET = "hvac_nonlinear"
system = SYSTEM_REGISTRY[DATASET]
state_dim = system.state_dim
control_dim = system.control_dim
n_traj = 10
seq_len = 50
dt = 1
substeps = 10

process_noise = torch.zeros(state_dim)
control_noise = torch.zeros(control_dim)

X, U = generate_trajectories_euler_maruyama(
        system,
        n_traj=n_traj,
        seq_len=seq_len,
        dt=dt,
        process_noise=process_noise,
        control_noise=control_noise,
        substeps=substeps,
        control=True
    )

x.to_csv()
u.to_csv()