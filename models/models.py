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
        """
        A version where the block diagonal matrix A is constant (learned parameters),
        rather than being predicted from the state z.
        
        latent_dim: The dimension of the Koopman state z.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_blocks = latent_dim // 2
        self.has_real_tail = (latent_dim % 2 != 0)
        
       
        self.complex_params = nn.Parameter(torch.randn(self.n_blocks, 2) * 0.1)
        
        if self.has_real_tail:
            self.real_param = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, z):
        # We apply a constraint (like tanh) to ensure stability if needed
        # This mirrors your original logic: params = torch.tanh(params)
        params = torch.tanh(self.complex_params)
        a = params[:, 0] # Shape: (n_blocks)
        b = params[:, 1] # Shape: (n_blocks)
        
        # Prepare z: shape (Batch, n_blocks, 2)
        z_vecs = z[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
        z1, z2 = z_vecs[..., 0], z_vecs[..., 1]
        
        # Complex multiplication: (a + ib)(z1 + iz2)
        # Real: a*z1 - b*z2 | Imag: b*z1 + a*z2
        # Note: I adjusted the signs to match standard complex multiplication 
        # (Your original had res_real = a*z1 + b*z2, which is fine if intended)
        res_real = a * z1 - b * z2
        res_imag = b * z1 + a * z2
        
        # Flatten and Concat
        out_main = torch.stack([res_real, res_imag], dim=-1).flatten(1)
        
        if self.has_real_tail:
            tail = z[:, -1:] * torch.tanh(self.real_param)
            return torch.cat([out_main, tail], dim=-1)
        
        return out_main
    

class LearnableComplexBlockDiagonal_v2(nn.Module):
       
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
            # Run backbone and head
            features = self.backbone(z)
            params = self.param_head(features) 
            params = torch.tanh(params) * 1

            # Vectorized Complex Math (Highly "Fusable" for Inductor)
            # We use reshape + view_as_complex to avoid the overhead of manual stacking
            z_main = z[:, :self.n_blocks*2].reshape(-1, self.n_blocks, 2)
            p_main = params[:, :self.n_blocks*2].reshape(-1, self.n_blocks, 2)
            
            # view_as_complex requires the tensor to be contiguous
            z_c = torch.view_as_complex(z_main.contiguous())
            p_c = torch.view_as_complex(p_main.contiguous())
            
            # This single line replaces all the manual res_real/res_imag logic
            out_c = z_c * p_c 
            
            out_main = torch.view_as_real(out_c).flatten(1)
            
            #  Handle the real tail if it exists
            if self.has_real_tail:
                # params[:, -1:] maps the last param to the last state element
                tail = z[:, -1:] * params[:, -1:]
                return torch.cat([out_main, tail], dim=-1)
                
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
            # Backbone Inference
            features = self.backbone(z)
            params = self.param_head(features) 
            params = torch.tanh(params) * 1

            # Extract and Broadcast
            # Instead of multiple views, use one split
            p_blocks = params[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
            a, b = p_blocks[..., 0], p_blocks[..., 1]
            
            # Explicit Vectorized Multiplication (Faster for Inductor)
            z_vecs = z[:, :self.n_blocks*2].view(-1, self.n_blocks, 2)
            z1, z2 = z_vecs[..., 0], z_vecs[..., 1]
            
            # (a + ib)(z1 + iz2) 
            res_real = a * z1 + b * z2
            res_imag = -b * z1 + a * z2
            
            # Flatten and Concat
            out_main = torch.stack([res_real, res_imag], dim=-1).flatten(1)
            
            if self.has_real_tail:
                # Slicing z[:, -1:] is faster than indexing for cat
                return torch.cat([out_main, z[:, -1:] * params[:, -1:]], dim=-1)
            
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