import torch


def entropy_loss(self, logstd, min_entropy_threshold=1.0):
        """
        Computes the entropy constraint for the Koopman latent space.
        logstd: the second half of the encoder output (cat[mu, logstd])
        """

        avg_entropy = torch.mean(logstd)

        return torch.relu(min_entropy_threshold - avg_entropy)
    
def spectral_loss(self):
        Amat = self.get_A_matrix_tensor()
        # Spectral radius = max absolute eigenvalue
        eigvals = torch.linalg.eigvals(Amat)
        max_eig = torch.max(torch.abs(eigvals))
        # Penalty: only punish if the radius exceeds 1.0
        return torch.relu(max_eig - 1.0)


def forcing_regularization(self, v_r_list):
        """
        Regularizes the learned forcing terms to be sparse and small.
        v_r_list: A list of tensors [Batch, Latent] captured during rollout.
        """
        if not v_r_list:
            return torch.tensor(0.0, device=self.device)
        
        # Stack into [Batch, Horizon, Latent]
        v_r_tensor = torch.stack(v_r_list, dim=1)
        
        # 1. Sparsity Penalty (L1): Encourages v_r to be zero most of the time.
        # This is key for HAVOK-style intermittent forcing.
        loss_l1 = torch.norm(v_r_tensor, p=1, dim=-1).mean()
        
        # 2. Magnitude Penalty (L2): Prevents the network from giving huge "kicks".
        loss_l2 = torch.norm(v_r_tensor, p=2, dim=-1).mean()
        
        # Balance them: L1 is usually more important for HAVOK principles
        return 1.0 * loss_l1 + 0.1 * loss_l2
    