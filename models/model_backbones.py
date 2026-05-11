
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T, D]
        # Causal mask to prevent looking into the future
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        return self.transformer(self.embedding(x), mask=mask)

class GRUBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out # [B, T, H]
    

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
    def __init__(self, in_dim, out_dim, hidden_channels=None, bounded=False):
        super().__init__()

       
        if hidden_channels is None:
            hidden_channels = []

        # Build full layer list: [in_dim, h1, h2, ..., out_dim]
        layers = [in_dim] + hidden_channels

        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(MLPResidualBlock(layers[i], layers[i + 1]))

        self.bounded = bounded
        self.net = nn.Sequential(*blocks)
        self.final = nn.Linear(hidden_channels[-1], out_dim)
    def forward(self, x):
        h = self.net(x)
        out = self.final(h)
        return torch.sigmoid(out) if self.bounded else out
    
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