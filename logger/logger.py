# logger.py
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

class VectorLogger:
    def __init__(self, log_dir="logs", display_images=False, num_samples=8):
        self.log_dir = log_dir
        self.num_samples = num_samples
        os.makedirs(log_dir, exist_ok=True)

        self.d_losses = []
        self.g_losses = []
        self.display_images = display_images

    # -----------------------------
    def log_loss(self, d_loss, g_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)

    # -----------------------------
    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.d_losses, label="D Loss")
        plt.plot(self.g_losses, label="G Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Progression")
        path = os.path.join(self.log_dir, "loss.png")
        plt.savefig(path)
        if self.display_images:
            plt.show()
        plt.close()

    # -----------------------------
    def log_latent(self, z, x_input, x_output, step=None):
        """
        z: latent vectors
        x_input: real vectors
        x_output: generated vectors
        """
        x_output = x_output.squeeze(1)
        if self.num_samples is not None:
            z = z[:self.num_samples]
            x_input = x_input[:self.num_samples]
            x_output = x_output[:self.num_samples]

        # Save latent info
        latent_path = os.path.join(
            self.log_dir, f"latents_step_{step}.pt" if step else "latents.pt"
        )
        torch.save({"z": z, "x_input": x_input, "x_output": x_output}, latent_path)

        # Save plots only if vectors are 2D
        if x_output.ndim == 2 and x_output.shape[1] == 2:
            self._save_vector_scatter(x_output, f"generated_step_{step}" if step else "generated")
        if x_input.ndim == 2 and x_input.shape[1] == 2:
            self._save_vector_scatter(x_input, f"real_step_{step}" if step else "real")

    # -----------------------------
    def _save_vector_scatter(self, vectors, name):
        if isinstance(vectors, torch.Tensor):
            if vectors.requires_grad:
                vectors = vectors.detach()
            vectors = vectors.cpu().numpy()

        vectors = vectors.squeeze()

        # Only plot if 2D
        if vectors.ndim != 2 or vectors.shape[1] != 2:
            print(f"Skipping plot for {name}, shape={vectors.shape}")
            return

        plt.figure(figsize=(5, 5))
        plt.scatter(vectors[:, 0], vectors[:, 1], alpha=0.6)
        plt.title(name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)

        path = os.path.join(self.log_dir, f"{name}.png")
        plt.savefig(path)
        if self.display_images:
            plt.show()
        plt.close()

    # -----------------------------
    def save_models(self, models_dict, step):
        for name, model in models_dict.items():
            if model is None:
                continue
            path = os.path.join(self.log_dir, f"{name}_{step}.pt")
            torch.save(model.state_dict(), path)

    # -----------------------------
    def save_loss(self):
        torch.save(
            {"d_losses": self.d_losses, "g_losses": self.g_losses},
            os.path.join(self.log_dir, "losses.pt"),
        )


class InfoVectorLogger:
    def __init__(self, log_dir="logs", display_images=False, num_samples=None):
        self.log_dir = log_dir
        self.num_samples = num_samples  # how many trajectories to visualize
        os.makedirs(log_dir, exist_ok=True)

        self.scalars = {}  # {key: [(step, value)]}
        self.display_images = display_images

    # -----------------------------
    def log_scalars(self, scalars_dict, step=None):
        """Store multiple scalar values per step"""
        for key, value in scalars_dict.items():
            if key not in self.scalars:
                self.scalars[key] = []
            self.scalars[key].append((step, value))

        if step is not None:
            scalars_str = " | ".join(f"{k}: {v:.4f}" for k, v in scalars_dict.items())
            print(f"[Step {step}] {scalars_str}")

    # -----------------------------
    def plot_scalars(self):
        """Plot all stored scalars over training steps"""
        for key, values in self.scalars.items():
            steps, vals = zip(*values)
            plt.figure(figsize=(6, 4))
            plt.plot(steps, vals, label=key)
            plt.xlabel("Step")
            plt.ylabel(key)
            plt.title(f"{key} over time")
            plt.grid(True)
            plt.legend()
            path = os.path.join(self.log_dir, f"{key}.png")
            plt.savefig(path)
            if self.display_images:
                plt.show()
            plt.close()

    # -----------------------------
    def save_models(self, models_dict, step):
        """Save models and Koopman matrix"""
        for name, model in models_dict.items():
            if model is None:
                continue
            path = os.path.join(self.log_dir, f"{name}_{step}.pt")
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), path)
            elif isinstance(model, torch.Tensor):
                torch.save(model, path)
            else:
                print(f"Skipping {name}, type {type(model)} cannot be saved")

    # -----------------------------
    def save_trajectories_(self, z_traj, x_rec, x_true=None, step=None):
        """
        Save latent and reconstructed trajectories as images.
        z_traj: [B, T, D] latent sequence
        x_rec: [B, T, Dx] stochastic reconstruction (sampled z)
        x_true: [B, T, Dx] true trajectory
        step: current epoch for naming
        """

        def ensure_numpy(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return t

        z_traj = ensure_numpy(z_traj)
        x_rec = ensure_numpy(x_rec)
        if x_true is not None:
            x_true = ensure_numpy(x_true)

        B, T, D = z_traj.shape
        Dx = x_rec.shape[2]

        # Use first trajectory
        z_traj = z_traj[0]
        x_rec = x_rec[0]
        if x_true is not None:
            x_true = x_true[0]

        # ---------------------- Deterministic reconstruction ----------------------
        # Decode latent mean for deterministic reconstruction
        x_det = self.decoder(torch.tensor(z_traj, dtype=torch.float32)).detach().cpu().numpy()

        # ---------------------- 1. Latent trajectory ----------------------
        plt.figure(figsize=(10, 6))
        for i in range(D):
            plt.plot(z_traj[:, i], alpha=0.6, label=f"$z_{i}$" if i==0 else None)
        plt.xlabel("Time step")
        plt.ylabel("Latent value")
        plt.title(f"Latent Trajectories at step {step}")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"latent_traj_step_{step}.png"))
        plt.close()

        # ---------------------- 2. Reconstructed vs True ----------------------
        plt.figure(figsize=(10, 6))
        for i in range(Dx):
            plt.plot(x_det[:, i], 'b-', alpha=0.8, label=f"Deterministic $x_{i}$" if i==0 else None)
            plt.plot(x_rec[:, i], 'r--', alpha=0.5, label=f"Stochastic $x_{i}$" if i==0 else None)
        if x_true is not None:
            for i in range(Dx):
                plt.plot(x_true[:, i], 'k-', linewidth=2, label=f"True $x_{i}$" if i==0 else None)
        plt.xlabel("Time step")
        plt.ylabel("State value")
        plt.title(f"Reconstructed vs True Trajectories at step {step}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"rec_vs_true_step_{step}.png"))
        plt.close()

        # ---------------------- 3. Phase portrait ----------------------
        if Dx == 2:
            plt.figure(figsize=(6, 6))
            plt.plot(x_det[:, 0], x_det[:, 1], 'b-', alpha=0.8, label="Deterministic traj")
            plt.plot(x_rec[:, 0], x_rec[:, 1], 'r--', alpha=0.5, label="Stochastic traj")
            if x_true is not None:
                plt.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=2, label="True traj")
            plt.xlabel("$x_0$")
            plt.ylabel("$x_1$")
            plt.title(f"Phase Portrait at step {step}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"phase_step_{step}.png"))
            plt.close()
        elif Dx == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(x_det[:, 0], x_det[:, 1], x_det[:, 2], 'b-', alpha=0.8, label="Deterministic traj")
            ax.plot(x_rec[:, 0], x_rec[:, 1], x_rec[:, 2], 'r--', alpha=0.5, label="Stochastic traj")
            if x_true is not None:
                ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], 'k-', linewidth=2, label="True traj")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            ax.set_title(f"Phase Portrait at step {step}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"phase3d_step_{step}.png"))
            plt.close()

    def save_trajectories(self, z_traj, x_rec, x_true=None, step=None):
        """
        Save latent and reconstructed trajectories as images
        z_traj: [T, D] or [B, T, D] latent
        x_rec: [T, Dx] or [B, T, Dx] reconstructed
        x_true: optional true trajectory [T, Dx]
        step: current epoch for naming
        """
        def ensure_numpy(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return t

        z_traj = ensure_numpy(z_traj)
        x_rec = ensure_numpy(x_rec)
        if x_true is not None:
            x_true = ensure_numpy(x_true)

        # Use first trajectory if multiple samples
        if z_traj.ndim == 3:
            z_traj = z_traj[0]
        if x_rec.ndim == 3:
            x_rec = x_rec[0]
        if x_true is not None and x_true.ndim == 3:
            x_true = x_true[0]

        # 1. Latent trajectory plot
        plt.figure(figsize=(10, 6))
        for i in range(z_traj.shape[1]):
            plt.plot(z_traj[:, i], alpha=0.6, label=f"$z_{i}$" if i == 0 else None)
        plt.xlabel("Time step")
        plt.ylabel("Latent value")
        plt.title(f"Latent Trajectories at step {step}")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"latent_traj_step_{step}.png"))
        plt.close()

        # 2. Reconstructed vs true trajectory
        plt.figure(figsize=(10, 6))
        for i in range(x_rec.shape[1]):
            plt.plot(x_rec[:, i], alpha=0.6, label=f"Reconstructed $x_{i}$")
        if x_true is not None:
            for i in range(x_true.shape[1]):
                plt.plot(x_true[:, i], 'k--', linewidth=2, label=f"True $x_{i}$" if i == 0 else None)
        plt.xlabel("Time step")
        plt.ylabel("State value")
        plt.title(f"Reconstructed vs True Trajectories at step {step}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"rec_vs_true_step_{step}.png"))
        plt.close()

        # 3. Phase portrait if 2D or 3D
        Dx = x_rec.shape[1]
        if Dx == 2:
            plt.figure(figsize=(6, 6))
            plt.plot(x_rec[:, 0], x_rec[:, 1], alpha=0.7, label="Reconstructed traj")
            if x_true is not None:
                plt.plot(x_true[:, 0], x_true[:, 1], 'k--', linewidth=2, label="True traj")
            plt.xlabel("$x_0$")
            plt.ylabel("$x_1$")
            plt.title(f"Phase Portrait at step {step}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"phase_step_{step}.png"))
            plt.close()
        elif Dx == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(7,6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(x_rec[:,0], x_rec[:,1], x_rec[:,2], alpha=0.7, label="Reconstructed traj")
            if x_true is not None:
                ax.plot(x_true[:,0], x_true[:,1], x_true[:,2], 'k--', linewidth=2, label="True traj")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            ax.set_title(f"Phase Portrait at step {step}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"phase3d_step_{step}.png"))
            plt.close()


    # -----------------------------
    def save_rollout(self, encoder, decoder, koopman_matrix, init_state, num_steps=150, step=None, device=None):
        """
        Generate a deterministic Koopman rollout from init_state and save plots.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoder.eval()
        decoder.eval()
        koopman_matrix = koopman_matrix.to(device)

        # Prepare initial state
        if isinstance(init_state, np.ndarray):
            x0 = torch.tensor(init_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,Dx]
        elif isinstance(init_state, torch.Tensor):
            x0 = init_state.to(device).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("init_state must be np.ndarray or torch.Tensor")
        
        def reparameterize(mu, logvar, K=1):
            """
            Returns z of shape (B, K, latent_dim)
            """
            B, D = mu.shape

            mu = mu.unsqueeze(1).expand(B, K, D)
            logvar = logvar.unsqueeze(1).expand(B, K, D)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        # Encode initial latent
        with torch.no_grad():
            out = encoder(x0) 
            #print("out.shape: ", out.shape)
            assert out.shape[2]%2 == 0.0
            latent_dim = int(out.shape[2]/2)
            z0_mu = out[:,:,latent_dim:]
            std = out[:,:,latent_dim:]
            std = torch.nn.functional.softplus(std)
            z0_logstd = torch.log(std + 1e-8)
            #print("z0_mu.shape: ", z0_mu.shape)
            B,T, Dz = z0_mu.shape
            z = reparameterize( z0_mu.reshape(B*T,Dz), z0_logstd.reshape(B*T,Dz))
            #print("z.shape: ", z.shape)
            z = z.reshape(B,T,-1)
            z = z.squeeze(1)
            #print("z.shape: ", z.shape)
            z_traj = [z.clone()]

            # Rollout using Koopman matrix
            for _ in range(num_steps - 1):
                z = koopman_matrix(z)  # [1, latent_dim]
                z_traj.append(z.clone())

            z_traj = torch.stack(z_traj)# [T, latent_dim]

            # Decode to state space
            x_rec = decoder(z_traj)  # [num_steps, Dx]
            x_rec = x_rec.cpu().numpy()
            z_traj = z_traj.cpu().numpy()

        # ------------------------ Save Plots ------------------------

        # 1. Latent trajectory
        plt.figure(figsize=(10,6))
        for i in range(z_traj.shape[1]):
            plt.plot(z_traj[:, i], alpha=0.6, label=f"$z_{i}$" if i==0 else None)
        plt.xlabel("Time step")
        plt.ylabel("Latent value")
        plt.title(f"Latent Rollout (Step {step})")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"latent_rollout_step_{step}.png"))
        plt.close()

        # 2. Reconstructed states
        plt.figure(figsize=(10,6))
        for i in range(x_rec.shape[1]):
            plt.plot(x_rec[:, i], alpha=0.6, label=f"$x_{i}$")
        plt.xlabel("Time step")
        plt.ylabel("State value")
        plt.title(f"Reconstructed Rollout (Step {step})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"x_rec_rollout_step_{step}.png"))
        plt.close()

        # 3. Phase portrait if 2D or 3D
        Dx = x_rec.shape[1]
        if Dx == 2:
            plt.figure(figsize=(6,6))
            plt.plot(x_rec[:,0], x_rec[:,1], alpha=0.7, label="Rollout")
            plt.xlabel("$x_0$")
            plt.ylabel("$x_1$")
            plt.title(f"Phase Portrait Rollout (Step {step})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"phase_rollout_step_{step}.png"))
            plt.close()
        elif Dx == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(7,6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(x_rec[:,0], x_rec[:,1], x_rec[:,2], alpha=0.7, label="Rollout")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            ax.set_title(f"Phase Portrait Rollout (Step {step})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"phase3d_rollout_step_{step}.png"))
            plt.close()
