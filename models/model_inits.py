import torch
import torch.nn as nn
from models.models import LinearMatrix, BlockDiagonal, LearnableComplexBlockDiagonal

def init_weights_near_zero(m, std=0.001):
    """Initializes weights with a very small Gaussian distribution."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_gaussian(m, std=0.02):
    """Standard Xavier/Kaiming-style Gaussian initialization."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_koopman_small_diagonal(m, std=0.01):
    """Initializes a Linear weight matrix as an Identity + small noise."""
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            # Start with Identity (1.0 on diagonal)
            nn.init.eye_(m.weight)
            # Add small noise to prevent symmetry
            m.weight.add_(torch.randn_like(m.weight) * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

def apply_system_init(system_bundle, mode='gaussian', koopman_cfg=None, scale=0.95):
    for name, module in system_bundle.items():
        if name == 'A':
            if isinstance(module, LearnableComplexBlockDiagonal):
                module.backbone.apply(init_weights_gaussian)
                if hasattr(module, 'param_head'):
                    nn.init.constant_(module.param_head.weight, 0.0)
                    with torch.no_grad():
                        nn.init.constant_(module.param_head.bias, 0.0)
                        for i in range(0, module.n_params - (1 if module.has_real_tail else 0), 2):
                            module.param_head.bias[i] = scale
                        if module.has_real_tail:
                            module.param_head.bias[-1] = scale
            
            # This is where the error was happening
            else:
                # We iterate through submodules to find the actual Linear layer
                for sub in module.modules():
                    if isinstance(sub, nn.Linear):
                        with torch.no_grad():
                            nn.init.eye_(sub.weight)
                            sub.weight.mul_(scale)
                            if sub.bias is not None:
                                nn.init.constant_(sub.bias, 0.0)

        elif name in ['B', 'E', 'forcing_net']:
            # Use .apply() to safely traverse any LinearMatrix or Sequential wrapper
            module.apply(lambda m: init_weights_near_zero(m, std=0.0001))

        else:
            # Encoders/Decoders
            module.apply(init_weights_gaussian)