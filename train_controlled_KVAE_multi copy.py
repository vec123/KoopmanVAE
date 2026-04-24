import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Assuming these are in your local project files
from system_plotters.controlled_simulators import generate_trajectories_euler_maruyama
from models.models import ResidualMLP, LinearMatrix, TTKoopman
from trainers.Controlled_Multi_KVAE_trainer import ControlledKoopmanVAETrainer
from logger.logger import InfoVectorLogger

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def save_training_trajectories(X_raw, U_raw, labels, dataset_name, save_dir="plots", num_figs=5):
    """
    Saves figures showing the trajectories the model is trained on.
    
    Args:
        X_raw: Numpy array of shape [N, T, Dx]
        U_raw: Numpy array of shape [N, T, Du]
        labels: List of strings for state labels
        dataset_name: String name of the dataset (for filenames)
        save_dir: Directory to save plots
        num_figs: Number of random trajectories to plot
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_traj, T, dx = X_raw.shape
    du = U_raw.shape[2]
    
    # Select indices to plot (clamped to available trajectories)
    indices = np.random.choice(n_traj, min(num_figs, n_traj), replace=False)

    for idx in indices:
        # Create a figure with a subplot for each state + one for control
        fig, axes = plt.subplots(dx + du, 1, figsize=(10, 2 * (dx + du)), sharex=True)
        fig.suptitle(f"Training Trajectory {idx} - {dataset_name}", fontsize=14)

        # 1. Plot States
        for i in range(dx):
            axes[i].plot(X_raw[idx, :, i], color='tab:blue', linewidth=1.5)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, alpha=0.3)

        # 2. Plot Control (U)
        # We assume control is the last subplot
        for j in range(du):
            ax_u = axes[dx + j]
            ax_u.step(range(T), U_raw[idx, :, j], color='tab:red', where='post', linewidth=1.5)
            ax_u.set_ylabel(f"Control $u_{j}$")
            ax_u.set_xlabel("Time Steps")
            ax_u.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(save_dir, f"train_traj_{dataset_name}_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved training trajectory plot to: {save_path}")

def split_into_subsequences_controlled(x_seqs, u_seqs, subseq_len, stride=1):
    x_sub, u_sub = [], []
    for x_traj, u_traj in zip(x_seqs, u_seqs):
        T = x_traj.shape[0]
        for start in range(0, T - subseq_len + 1, stride):
            x_sub.append(x_traj[start:start + subseq_len])
            u_sub.append(u_traj[start:start + subseq_len])
    return x_sub, u_sub

class ControlledSequenceDataset(Dataset):
    def __init__(self, x_sequences, u_sequences):
        self.x = [torch.tensor(s, dtype=torch.float32) for s in x_sequences]
        self.u = [torch.tensor(s, dtype=torch.float32) for s in u_sequences]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.u[idx]

# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    plot_training_trajs = False
    # --- Configuration ---
    DATASET = "cartpole"  # "cartpole" or "inverted_pendulum"
    log_name = "Controlled_KVAE_v5_noise_free"
    
    n_traj = 5
    seq_len = 1000
    subseq_len = 501
    stride = 5
    dt = 0.001
    batch_size = 64*4
    latent_dim = 128
    hidden_dim = 128
    horizon = 500
    control_dim = 1
    
    # New Noise Configuration
    # For cartpole: [x_noise, theta_noise]
    # For pendulum: just a float (applied to omega)
    noise_lvl = [0.0, 0.00] if DATASET == "cartpole" else 0.00

    # --- System Specific Meta-Setup ---
    if DATASET == "cartpole":
        state_dim = 4
        labels = [r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
        system_key = "cartpole" # for the EM function
    else:
        state_dim = 2
        labels = [r"$\theta$", r"$\omega$"]
        system_key = "pendulum" # for the EM function

    # --- 1. Generate Data using Euler-Maruyama ---
    print(f"Generating {n_traj} trajectories for {DATASET} via Euler-Maruyama...")
    
    # We use your imported EM function which handles the internal loop and sub-stepping
    X_raw, U_raw, t_axis = generate_trajectories_euler_maruyama(
        system_type=system_key,
        n_traj=n_traj,
        seq_len=seq_len,
        dt=dt,
        noise_lvl=noise_lvl,
        sub_steps=10,
        control=True
    )
    if plot_training_trajs:
        save_training_trajectories(
            X_raw=X_raw, 
            U_raw=U_raw, 
            labels=labels, 
            dataset_name=DATASET,
            num_figs=5
        )
        
    # Convert lists/numpy to sequences for the dataset
    # EM function returns [N, T, D]
    x_sequences = [X_raw[i] for i in range(n_traj)]
    u_sequences = [U_raw[i] for i in range(n_traj)]

    # Create sub-windows
    x_sub, u_sub = split_into_subsequences_controlled(x_sequences, u_sequences, subseq_len, stride)
    loader = DataLoader(ControlledSequenceDataset(x_sub, u_sub), batch_size=batch_size, shuffle=True)

    # 2. Models
    class KoopmanEncoder(torch.nn.Module):
        def __init__(self, state_dim, latent_dim, hidden_dim_int, hidden_depth=2):
            super().__init__()
            
            # We ensure the second argument (out_dim) is an INTEGER.
            # We pass the list of hidden layers to the third argument.
            self.backbone = ResidualMLP(
                state_dim,               # in_dim
                hidden_dim_int,          # out_dim (MUST BE INT)
                [hidden_dim_int] * hidden_depth # hidden_channels (LIST)
            )
            
            self.fc_mu = torch.nn.Linear(hidden_dim_int, latent_dim)
            self.fc_logstd = torch.nn.Linear(hidden_dim_int, latent_dim)

        def forward(self, x):
                h = self.backbone(x)
                mu = self.fc_mu(h)
                raw_std_output = self.fc_logstd(h)

                std = torch.nn.functional.softplus(raw_std_output) + 1e-7
                
                logstd = torch.log(std)
                
                return torch.cat([mu, logstd], dim=-1)
            
    encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim).to(device)  
    #decoder = ResidualMLP(latent_dim, state_dim, [hidden_dim]*3).to(device)

    decoder = LinearMatrix(latent_dim, state_dim).to(device)
    # System dynamics (A) and Control mapping (B)
    #system_matrix =  LinearMatrix(latent_dim, latent_dim).to(device)
    system_matrix =  TTKoopman(latent_dim=latent_dim, tt_rank=16, tt_shape=[(8, 8), (16, 16)]).to(device)
    control_matrix = LinearMatrix(control_dim, latent_dim).to(device)
    # --- STABILITY INITIALIZATION ---
    with torch.no_grad():
        for param in control_matrix.parameters():
            param.data.mul_(0.00)


    optimizer = optim.Adam(
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(system_matrix.parameters()) + 
        list(control_matrix.parameters()),
        lr=1e-3
    )

    logger = InfoVectorLogger(log_dir=f"logs/{log_name}_{DATASET}")

    # 3. Trainer
    trainer = ControlledKoopmanVAETrainer(
        encoder=encoder,
        decoder=decoder,
        system_matrix=system_matrix,
        control_matrix=control_matrix,
        dataloader=loader,
        optimizer=optimizer,
        latent_dim=latent_dim,
        horizon=horizon,
        beta=1e-80,          # KL
        gamma_1=1.0,         # koopman dynamics weight in latent space
        gamma_2=1.0,         # koopman reduction loss weight
        delta=1e-30,         # Spectral loss weight
        alpha=1e-30,        # Entropy loss weight
        epsilon_1 = 1e-30,  #Initial reconstruction loss weight
        epsilon_2 = 1e-30,   # All-time reconstruction loss weight
        device=device,
        logger=logger,
        save_epoch=100,
        val_epochs = 100
    )

    print("Starting Controlled Joint Training...")
    trainer.train(5000)

if __name__ == "__main__":
    main()