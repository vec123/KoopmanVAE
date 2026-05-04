import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """
    GRU-based encoder for sequences.
    
    Args:
        input_dim: dimension of each timestep (state dimension)
        hidden_dim: GRU hidden dimension
        latent_dim: dimension of latent vector z
        num_layers: number of GRU layers
        bidirectional: whether to use a bidirectional GRU
        use_vae: if True, outputs mean and logvar for VAE
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, 
                 bidirectional=False, use_vae=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_vae = use_vae

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # If bidirectional, hidden dims are doubled
        final_hidden_dim = hidden_dim * (2 if bidirectional else 1)

        # Linear layers to produce latent vector
        if use_vae:
            self.fc_mu = nn.Linear(final_hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(final_hidden_dim, latent_dim)
        else:
            self.fc_latent = nn.Linear(final_hidden_dim, latent_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns:
            if use_vae: mu, logvar
            else: latent vector z
        """
        batch_size = x.size(0)
        out, h_n = self.gru(x)  # out: [B, T, H], h_n: [num_layers*directions, B, H]

        # Take the last hidden state (or mean over directions/layers)
        if self.bidirectional:
            # Concatenate last forward and backward hidden states
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, H*2]
        else:
            h_last = h_n[-1]  # [B, H]

        if self.use_vae:
            mu = self.fc_mu(h_last)
            logvar = self.fc_logvar(h_last)
            return mu, logvar
        else:
            z = self.fc_latent(h_last)
            return z

import torch
import torch.nn as nn

class GRUEncoderSeq(nn.Module):
    """
    GRU Encoder that outputs mu and logvar for each timestep.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        factor = 2 if bidirectional else 1
        self.mu_layer = nn.Linear(hidden_dim * factor, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim * factor, latent_dim)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        Returns:
            mu: [B, T, latent_dim]
            logvar: [B, T, latent_dim]
        """
        out, _ = self.gru(x)  # out: [B, T, hidden_dim*factor]
        mu = self.mu_layer(out)
        logvar = self.logvar_layer(out)
        return mu, logvar


class GRUEncoderResidual(nn.Module):
    """
    GRU-based encoder with skip/residual connections across layers.
    
    Args:
        input_dim: dimension of each timestep (state dimension)
        hidden_dim: GRU hidden dimension
        latent_dim: dimension of latent vector z
        num_layers: number of GRU layers
        bidirectional: whether to use a bidirectional GRU
        use_vae: if True, outputs mean and logvar for VAE
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, 
                 bidirectional=False, use_vae=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_vae = use_vae

        self.gru_layers = nn.ModuleList()
        self.projections = nn.ModuleList()  # for skip connections

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * (2 if bidirectional else 1)
            self.gru_layers.append(
                nn.GRU(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,  # each layer is 1 GRU
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )
            # Projection if input/output dims differ
            if in_dim != hidden_dim * (2 if bidirectional else 1):
                self.projections.append(nn.Linear(in_dim, hidden_dim * (2 if bidirectional else 1)))
            else:
                self.projections.append(nn.Identity())

        # Final linear layers to latent space
        final_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        if use_vae:
            self.fc_mu = nn.Linear(final_hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(final_hidden_dim, latent_dim)
        else:
            self.fc_latent = nn.Linear(final_hidden_dim, latent_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns:
            if use_vae: mu, logvar
            else: latent vector z
        """
        out = x
        for gru, proj in zip(self.gru_layers, self.projections):
            # Pass through GRU layer
            residual = proj(out)
            out_layer, _ = gru(out)  # output: [B, T, H]
            # Add skip/residual connection
            out = out_layer + residual

        # Take last hidden state for latent representation
        # last timestep across sequence dimension
        h_last = out[:, -1, :]  # [B, H]

        if self.use_vae:
            mu = self.fc_mu(h_last)
            logvar = self.fc_logvar(h_last)
            return mu, logvar
        else:
            z = self.fc_latent(h_last)
            return z


#----MLP
class MLP(nn.Module):
    """
    Generic MLP given a list of layer sizes:
    [in_dim, hidden1, hidden2, ..., out_dim]
    Uses ReLU activations except final layer.
    """
    def __init__(self, layers):
        super().__init__()
        self.is_mlp = True

        assert len(layers) >= 2, "At least input and output needed."

        modules = []
        for i in range(len(layers) - 1):
            in_dim  = layers[i]
            out_dim = layers[i + 1]

            modules.append(nn.Linear(in_dim, out_dim))

            # Add ReLU except for final layer
            if i < len(layers) - 2:
                modules.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class MLPResidualBlock(nn.Module):
    """
    A 2-layer residual block:
        x -> Linear -> ReLU -> Linear -> +skip
    If dimensions differ, a linear projection is used for the skip.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

        # Project skip if needed
        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        return self.act(out + identity)

class ResidualMLP(nn.Module):
    """
    MLP with residual connections. Can be initialized as:
        ResidualMLP(in_dim, out_dim, hidden_channels=[...])
    """
    def __init__(self, in_dim, out_dim, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = []

        # Build full layer list: [in_dim, h1, h2, ..., out_dim]
        layers = [in_dim] + hidden_channels

        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(MLPResidualBlock(layers[i], layers[i + 1]))

        self.net = nn.Sequential(*blocks)
        self.final = nn.Linear(hidden_channels[-1], out_dim)
    def forward(self, x):
        h = self.net(x)
        out = self.final(h)
        return out
    
class BottleneckedResidualMLP(nn.Module):
    """
    MLP with residual blocks and a centralized bottleneck.
    """
    def __init__(self, in_dim, out_dim, hidden_channels=None, bottleneck_dim=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 128]
        
        # 1. Encoder/Compression path
        enc_layers = []
        curr_dim = in_dim
        for h in hidden_channels:
            enc_layers.append(MLPResidualBlock(curr_dim, h))
            curr_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # 2. The Bottleneck
        # If no bottleneck_dim is provided, we default to a small value (e.g., 2 or 4)
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim else max(1, out_dim // 8)
        self.to_bottleneck = nn.Linear(hidden_channels[-1], self.bottleneck_dim)
        
        # 3. Decoder/Expansion path
        self.from_bottleneck = nn.Linear(self.bottleneck_dim, hidden_channels[-1])
        
        dec_layers = []
        curr_dim = hidden_channels[-1]
        for h in reversed(hidden_channels):
            dec_layers.append(MLPResidualBlock(curr_dim, h))
            curr_dim = h
        self.decoder = nn.Sequential(*dec_layers)

        # 4. Final Projection
        self.final = nn.Linear(hidden_channels[-1], out_dim)

    def forward(self, x):
        # Compress
        h = self.encoder(x)
        
        # Pass through bottleneck (the "sparse" signal)
        b = torch.tanh(self.to_bottleneck(h)) 
        
        # Expand
        h = self.from_bottleneck(b)
        h = self.decoder(h)
        
        # Final adjustment
        out = self.final(h)
        return out
    
class KoopmanEncoder(torch.nn.Module):
        def __init__(self, state_dim, latent_dim, hidden_dim_int, hidden_depth=5):
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
                
                return torch.cat([mu, logstd], dim=-1)
            
# --------------------
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

#tt models
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