import torch
import torch.nn as nn
import torch.nn.functional as F

from  models.model_backbones import  *


#-----------------------Factory pattern for the Encoder


class BaseKoopmanEncoder(nn.Module):
    def __init__(self, backbone, feature_dim, latent_dim, stochastic=True):
        super().__init__()
        self.backbone = backbone
        self.stochastic = stochastic
        
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logstd = nn.Linear(feature_dim, latent_dim) if stochastic else None

    def forward(self, x):
        # Handle 2D input (MLP) vs 3D input (RNN/Transformer)
        is_mlp = not isinstance(self.backbone, (nn.GRU, nn.LSTM, TransformerBackbone, GRUBackbone))
        
        if is_mlp and x.dim() == 3:
            # Flatten time for MLP: [B, T, D] -> [B*T, D]
            B, T, D = x.shape
            features = self.backbone(x.reshape(-1, D)).view(B, T, -1)
        else:
            features = self.backbone(x) # Expects [B, T, D]

        mu = self.fc_mu(features)
        logstd = self.fc_logstd(features) if self.stochastic else torch.zeros_like(mu)
        return mu, logstd
    
    
#-----------------------The Koopman Encoder 
class KoopmanEncoder(torch.nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dim_int, hidden_depth=5, bounded=True, stochastic=True):
        super().__init__()
        
        self.bounded = bounded
        self.stochastic = stochastic
        
        # Note: pass bounded=False to the backbone because we want the 
        # features (h) to be rich/unconstrained before the final heads.
        self.backbone = ResidualMLP(
            state_dim,
            hidden_dim_int,
            [hidden_dim_int] * hidden_depth,
            bounded= False 
        )
        
        self.fc_mu = torch.nn.Linear(hidden_dim_int, latent_dim)
        
        if self.stochastic:
            self.fc_logstd = torch.nn.Linear(hidden_dim_int, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.fc_mu(h)

        # Handle Bounding for the Latent Mean
        if self.bounded:
            mu = torch.tanh(mu) # Bounds mu to (-1, 1)

        # Handle Deterministic vs Stochastic Output
        if not self.stochastic:
            # Returns a tuple with None or an empty tensor as the second element
            return mu, torch.tensor([]) 
        
        # Stochastic logic
        raw_std_output = self.fc_logstd(h)
        std = torch.nn.functional.softplus(raw_std_output) + 1e-7
        logstd = torch.log(std)
        
        return mu, logstd
    
#-----------------------A linear Encoder         
class LinearKoopmanEncoder(torch.nn.Module):
        def __init__(self, state_dim, latent_dim):
            super().__init__()
            
            # We ensure the second argument (out_dim) is an INTEGER.
            # We pass the list of hidden layers to the third argument.

            self.fc_mu = LinearMatrix(state_dim, latent_dim)
            self.fc_logstd =LinearMatrix(state_dim, latent_dim)

        def forward(self, x):
                
                mu = self.fc_mu(x)

                raw_std_output = self.fc_logstd(x)
                std = torch.nn.functional.softplus(raw_std_output) + 1e-7
                logstd = torch.log(std)
                
                return mu, logstd
            
#-----------------------A Tensor Train Model
class LinearMatrix(nn.Module):
    """
    Linear layer as a full matrix, no bias by default.
    Can be interpreted as a Koopman matrix or linear decoder.
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.is_mlp = False
                
        self.input_dim = in_dim
        self.output_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        
    def forward(self, x):
        """
        x: [B, T, in_dim] or [B, in_dim]
        returns: same shape with last dim = out_dim
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B*T, D)
            out = self.linear(x_flat)
            return out.view(B, T, -1)
        else:
            return self.linear(x)


class QuadraticMatrix(nn.Module):
    """
    Computes A_quad * (z ⊗ z), where ⊗ is the Kronecker product.
    If z has dim D, z ⊗ z has dim D^2.
    """
    def __init__(self, dim, bias=False):
        super().__init__()
        self.dim = dim
        # The input to this linear layer is the flattened outer product
        self.linear = nn.Linear(dim * dim, dim, bias=bias)

    def forward(self, z):
        # z: [B, D]
        # Compute outer product: [B, D, 1] * [B, 1, D] -> [B, D, D]
        z_outer = torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        
        # Flatten to [B, D*D]
        z_quad = z_outer.view(z.shape[0], -1)
        
        # Apply A_quad
        return self.linear(z_quad)

class LinearQuadraticOperator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_part = LinearMatrix(dim, dim)
        self.quad_part = QuadraticMatrix(dim)

    def forward(self, z):
        return self.linear_part(z) + self.quad_part(z)
    

class BlockDiagonal(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.dim = latent_dim
        self.n_blocks = latent_dim // 2
        self.has_real_tail = (latent_dim % 2 != 0)
        
        # Initialize as identity-ish (a=1, b=0)
        self.a = nn.Parameter(torch.ones(self.n_blocks))
        self.b = nn.Parameter(torch.zeros(self.n_blocks))
        
        if self.has_real_tail:
            self.real_lambda = nn.Parameter(torch.ones(1))

    def forward(self, z):
        # z: [Batch, latent_dim]
        z_vecs = z[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
        
        # Broadcasting parameters across the batch
        # a, b are [n_blocks], we unsqueeze to [1, n_blocks, 1]
        a = self.a.view(1, -1, 1)
        b = self.b.view(1, -1, 1)
        
        z1 = z_vecs[..., 0:1]
        z2 = z_vecs[..., 1:2]
        
        res_1 = a * z1 + b * z2
        res_2 = -b * z1 + a * z2
        
        out_main = torch.cat([res_1, res_2], dim=-1).view(z.shape[0], -1)
        
        if self.has_real_tail:
            res_tail = z[:, -1:] * self.real_lambda
            return torch.cat([out_main, res_tail], dim=-1)
        
        return out_main
    

class LearnableComplexBlockDiagonal(nn.Module):
       
    def __init__(self, backbone, feature_dim, latent_dim):
        """
        backbone: A network (MLP) that maps z -> features.
        feature_dim: The output size of the backbone.
        latent_dim: The dimension of the Koopman state z.
        """
        super().__init__()
        self.backbone = backbone
        self.n_blocks = latent_dim // 2
        self.has_real_tail = (latent_dim % 2 != 0)
        
        # Maps backbone features to the (a, b) entries of the blocks
        self.n_params = (self.n_blocks * 2) + (1 if self.has_real_tail else 0)
        self.param_head = nn.Linear(feature_dim, self.n_params)

    def forward(self, z):
            # 1. Faster Backbone Inference
            features = self.backbone(z)
            params = self.param_head(features) 

            # 2. Extract and Broadcast
            # Instead of multiple views, use one split
            p_blocks = params[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
            a, b = p_blocks[..., 0], p_blocks[..., 1]
            
            # 3. Explicit Vectorized Multiplication (Faster for Inductor)
            z_vecs = z[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
            z1, z2 = z_vecs[..., 0], z_vecs[..., 1]
            
            # (a + ib)(z1 + iz2) 
            # Using unsqueeze here avoids the stack/view overhead in some versions
            res_real = a * z1 + b * z2
            res_imag = -b * z1 + a * z2
            
            # 4. Flatten and Concat
            out_main = torch.stack([res_real, res_imag], dim=-1).flatten(1)
            
            if self.has_real_tail:
                # Slicing z[:, -1:] is faster than indexing for cat
                return torch.cat([out_main, z[:, -1:] * params[:, -1:]], dim=-1)
            
            return out_main


class LearnableComplexBlockDiagonal_(nn.Module):
       
    def __init__(self, backbone, feature_dim, latent_dim):
        """
        backbone: A network (MLP) that maps z -> features.
        feature_dim: The output size of the backbone.
        latent_dim: The dimension of the Koopman state z.
        """
        super().__init__()
        self.backbone = backbone
        self.n_blocks = latent_dim // 2
        self.has_real_tail = (latent_dim % 2 != 0)
        
        # Maps backbone features to the (a, b) entries of the blocks
        self.n_params = (self.n_blocks * 2) + (1 if self.has_real_tail else 0)
        self.param_head = nn.Linear(feature_dim, self.n_params)

    def forward(self, z):
        # 1. Generate the matrix entries D(z)
        features = self.backbone(z)
        params = self.param_head(features) 

        # 2. Extract a(z) and b(z) for each 2x2 block
        p_blocks = params[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
        a = p_blocks[..., 0:1] # Real part (Stability/Damping)
        b = p_blocks[..., 1:2] # Imaginary part (Frequency)

        # 3. Apply D(z) to z
        # Reshape z to [Batch, n_blocks, 2] to match the blocks
        z_vecs = z[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
        z1, z2 = z_vecs[..., 0:1], z_vecs[..., 1:2]
        
        # Local transformation: [a(z)  b(z)] [z1]
        #                      [-b(z) a(z)] [z2]
        res_1 = a * z1 + b * z2
        res_2 = -b * z1 + a * z2
        
        # 4. Interleave back to [Batch, latent_dim]
        # Using stack + view is the most stable for torch.compile
        out_main = torch.stack([res_1, res_2], dim=-1).view(z.shape[0], -1)
        
        if self.has_real_tail:
            res_tail = z[:, -1:] * params[:, -1:]
            return torch.cat([out_main, res_tail], dim=-1)
        
        return out_main

    
#-----------------------A Tensor Train Model
class TTKoopman(nn.Module):
    def __init__(self, latent_dim, tt_rank, tt_shape=[(4, 4), (8, 8)]):
        """
        latent_dim: Total size (e.g., 32)
        tt_shape: Factors of latent_dim. Prod of in_dims and out_dims must equal latent_dim.
        """
        super().__init__()
        self.tt_shape = tt_shape
        self.num_cores = len(tt_shape)
        self.latent_dim = latent_dim
        
        self.cores = nn.ParameterList()
        for i in range(self.num_cores):
            in_d, out_d = tt_shape[i]
            r_left = 1 if i == 0 else tt_rank
            r_right = 1 if i == self.num_cores - 1 else tt_rank
            
            # Initialize cores to produce something close to an Identity matrix
            # We use a small mean for diagonal-like behavior and tiny noise
            core = torch.randn(r_left, in_d, out_d, r_right) * 0.01
            if in_d == out_d:
                # Add 1.0 to the "diagonal" of the core to bias towards Identity
                for j in range(min(in_d, out_d)):
                    core[0, j, j, 0] += 1.0 if (i == 0 or i == self.num_cores-1) else 1.0
            
            self.cores.append(nn.Parameter(core))
            
        # --- Count and Print Parameters ---
        tt_params = sum(p.numel() for p in self.cores)
        dense_params = latent_dim ** 2
        compression = dense_params / tt_params if tt_params > 0 else 0
        
        print(f"--- TTKoopman Initialized ---")
        print(f"Latent Dim: {latent_dim}")
        print(f"TT Shape:   {tt_shape} | Rank: {tt_rank}")
        print(f"TT Params:  {tt_params}")
        print(f"Dense Equiv: {dense_params}")
        print(f"Compression: {compression:.2f}x")
        print(f"-----------------------------")

    def get_matrix(self):
        """Reconstruct the full matrix from TT-cores."""
        res = self.cores[0]
        for i in range(1, self.num_cores):
            res = torch.tensordot(res, self.cores[i], dims=([-1], [0]))
        
        res = res.squeeze(0).squeeze(-1)
        
        # Reorder dimensions from interleaved (in1, out1, in2, out2...) 
        # to grouped (in1, in2..., out1, out2...)
        num_dims = res.dim()
        permute_idx = list(range(0, num_dims, 2)) + list(range(1, num_dims, 2))
        res = res.permute(*permute_idx)
        
        return res.reshape(self.latent_dim, self.latent_dim)

    def forward(self, z):
        # z: [Batch, Latent_dim]
        # Weight matrix: [Latent_dim, Latent_dim]
        return z @ self.get_matrix()