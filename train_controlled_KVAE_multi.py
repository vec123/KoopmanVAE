import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

from models.models import ResidualMLP, LinearMatrix, TTKoopman, KoopmanEncoder, LinearKoopmanEncoder
from systems.simulators import generate_trajectories_euler_maruyama
from trainers.Controlled_Multi_KVAE_trainer import ControlledKoopmanVAETrainer
from logger.logger import InfoVectorLogger
from systems.registry import SYSTEM_REGISTRY


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def save_training_trajectories(X_raw, U_raw, labels, dataset_name, save_dir="plots", num_figs=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_traj, T_x, dx = X_raw.shape
    # Get T_u specifically from the control array
    T_u = U_raw.shape[1]
    du = U_raw.shape[2]
    
    indices = np.random.choice(n_traj, min(num_figs, n_traj), replace=False)

    for idx in indices:
        fig, axes = plt.subplots(dx + du, 1, figsize=(10, 2 * (dx + du)), sharex=True)
        fig.suptitle(f"Training Trajectory {idx} - {dataset_name}", fontsize=14)

        # 1. Plot States
        for i in range(dx):
            # Use the actual length of the state data for the x-axis
            axes[i].plot(range(T_x), X_raw[idx, :, i], color='tab:blue', linewidth=1.5)
            # Safe label access
            label = labels[i] if i < len(labels) else f"State $x_{i}$"
            axes[i].set_ylabel(label)
            axes[i].grid(True, alpha=0.3)

        # 2. Plot Controls
        for j in range(du):
            ax_u = axes[dx + j] # This moves to the 5th and 6th slots
            ax_u.step(range(T_u), U_raw[idx, :, j], color='tab:red', where='post', linewidth=1.5)
            ax_u.set_ylabel(f"Control $u_{j}$")
            ax_u.grid(True, alpha=0.3)
                    
            # Only show "Time Steps" on the very last subplot
            if j == du - 1:
                ax_u.set_xlabel("Time Steps")

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

def apply_trigonometric_embedding(X, is_batch=True):
        """
        Transforms state from [x, x_dot, theta, theta_dot] 
        to [x, x_dot, sin(theta), cos(theta), theta_dot].
        
        Args:
            X: np.ndarray of shape (N, T, 4) if is_batch=True, 
            or (4,) if is_batch=False.
        Returns:
            np.ndarray of shape (N, T, 5) or (5,).
        """
        if is_batch:
            # X shape: (N, T, 4)
            x_pos     = X[:, :, 0:1]
            x_dot     = X[:, :, 1:2]
            theta     = X[:, :, 2:3]
            theta_dot = X[:, :, 3:4]
            
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            
            return np.concatenate([x_pos, x_dot, sin_t, cos_t, theta_dot], axis=-1)
        
        else:
            # X shape: (4,)
            x, x_dot, theta, theta_dot = X
            return np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])
        
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

    # --- Configuration ---
    plot_training_trajs = True
    normalize_scale = False
    concat_true = True
    encode_control = False
    remove_theta_acc = False

    
    DATASET = "hvac_nonlinear"
    log_name = "TestEMA_onoff_nonlinear_encoder"
    num_epochs = 20000
    save_every = 500

    n_traj = 10
    seq_len = 50
    subseq_len = 20
    stride = 1
    dt = 1
    substeps = 10
    batch_size = 64*4
    latent_dim = 128
    hidden_dim = 128*2
    horizon = 15
    
    log_name = (
        f"{log_name}_normalized_"
        f"{normalize_scale}_state_concat_{concat_true}_"
        f"control_enc_{encode_control}_spec_free_latent_dim_{latent_dim}"
    )


    # --- 1. Generate Data by Simulating the System with EM ---
    print(f"Generating {n_traj} trajectories for {DATASET} via Euler-Maruyama...")
    
    system = SYSTEM_REGISTRY[DATASET]
    state_dim = system.state_dim
    control_dim = system.control_dim
    labels = system.labels

    process_noise = [0.0, 0.00] if DATASET == "cartpole" else 0.00
    control_noise = 0.0

    X_raw, U_raw = generate_trajectories_euler_maruyama(
        system,
        n_traj=n_traj,
        seq_len=seq_len,
        dt=dt,
        process_noise=process_noise,
        control_noise=control_noise,
        substeps=substeps,
        control=True
    )
    remove_states = True
    if remove_states:
       X_raw = X_raw[:, :, 0:3]
       state_dim = X_raw.shape[2]

    if plot_training_trajs:
        save_training_trajectories(
            X_raw=X_raw, 
            U_raw=U_raw, 
            labels=labels, 
            save_dir="generated_samples",
            dataset_name=DATASET,
            num_figs=5
        )

    # -------Prepare and Normalize (Optional) Training Data  ---
    if normalize_scale:
        X_flat = X_raw.reshape(-1, state_dim)
        U_flat = U_raw.reshape(-1, control_dim)

        x_scaler = StandardScaler()
        u_scaler = StandardScaler()

        X_scaled_flat = x_scaler.fit_transform(X_flat)
        U_scaled_flat = u_scaler.fit_transform(U_flat)

        # Reshape back to [N, T, D]
        X_train = X_scaled_flat.reshape(n_traj, seq_len, state_dim)
        U_train = U_scaled_flat.reshape(n_traj, seq_len, control_dim)

        # SAVE SCALERS for use in the controller/inference script
        log_dir = f"logs/{log_name}_{DATASET}"
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        joblib.dump(x_scaler, os.path.join(log_dir, "x_scaler.pkl"))
        joblib.dump(u_scaler, os.path.join(log_dir, "u_scaler.pkl"))
        print(f"Scalers saved to {log_dir}")

    else:
        X_train, U_train =X_raw, U_raw

    x_sequences = [X_train[i] for i in range(n_traj)]
    u_sequences = [U_train[i] for i in range(n_traj)]

    x_sub, u_sub = split_into_subsequences_controlled(x_sequences, u_sequences, subseq_len, stride)
    loader = DataLoader(ControlledSequenceDataset(x_sub, u_sub), batch_size=batch_size, shuffle=True)

    #----------Modules
    #encoder = LinearKoopmanEncoder(state_dim, latent_dim).to(device)  
    encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim, hidden_depth=5).to(device)  
   
    if concat_true:
        latent_dim = latent_dim + state_dim 
    #decoder = ResidualMLP(latent_dim, state_dim, [hidden_dim]*3).to(device)
    decoder = LinearMatrix(latent_dim , state_dim).to(device)

    system_matrix =  LinearMatrix(latent_dim, latent_dim).to(device)
    #system_matrix =  TTKoopman(latent_dim=latent_dim, tt_rank=16, tt_shape=[(8, 8), (16, 16)]).to(device)

    control_matrix = LinearMatrix(control_dim, latent_dim).to(device)
    if encode_control:
        control_encoder = ResidualMLP(control_dim+state_dim, latent_dim, [hidden_dim]*3).to(device)
        control_decoder =  ResidualMLP(latent_dim, control_dim, [hidden_dim]*3).to(device)
        control_matrix = LinearMatrix(latent_dim, latent_dim).to(device)
    else:
        control_encoder = None
        control_decoder = None
        control_matrix = LinearMatrix(control_dim, latent_dim).to(device)

    with torch.no_grad():
        for param in control_matrix.parameters():
            param.data.mul_(0.1)
        #for param in system_matrix.parameters():
        #    param.data.mul_(1.00)

    optimizer = optim.Adam(
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(system_matrix.parameters()) + 
        list(control_matrix.parameters()),
        lr=5e-4
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
        control_encoder=control_encoder,
        control_decoder = control_decoder,
        latent_dim=latent_dim,
        horizon=horizon,
        beta=1e-80,          # KL
        gamma_1=10.0,         # koopman dynamics weight in latent space
        gamma_2=10.0,         # koopman dynamics reconstruction loss weight
        delta=1.e-10,         # Spectral loss weight
        alpha=0.0,        # Entropy loss weight
        epsilon_1 = 10.00,  #Initial reconstruction loss weight
        epsilon_2 = 10.00,   # All-time reconstruction loss weight
        zero_structure_gain = 0.0, # Zero-structure loss weight
        device=device,
        logger=logger,
        save_epoch = save_every,
        val_epochs = save_every,
        stochastic = False,
        concat_true = concat_true,
        horizon_decay = 1
    )

    print("Starting Controlled Joint Training...")
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()