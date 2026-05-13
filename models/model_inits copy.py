import torch
import torch.nn as nn

def init_weights_near_zero(m, std=0.01):
    """Initializes weights with a very small Gaussian distribution."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_uniform(m, limit=1.0):
    """Initializes weights uniformly between [-limit, limit]."""
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -limit, limit)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_gaussian(m, std=0.02):
    """Standard Gaussian initialization around 0."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_koopman_small_diagonal(m, std=0.01):
    """
    Initializes the matrix A as a diagonal matrix with small random 
    values close to zero.
    """
    if isinstance(m, nn.Linear):
        # 1. Start with zeros
        nn.init.constant_(m.weight, 0.0)
        
        # 2. Fill the diagonal with small Gaussian values
        diag_len = min(m.weight.shape[0], m.weight.shape[1])
        diag_values = torch.randn(diag_len) * std
        
        with torch.no_grad():
            for i in range(diag_len):
                m.weight[i, i] = diag_values[i]
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def apply_system_init(system_bundle, mode='gaussian'):
    """
    Applies the chosen initialization to the entire system_bundle.
    """
    for name, module in system_bundle.items():
        if name == 'A':
            # Updated to use small diagonal initialization
            module.apply(lambda m: init_koopman_small_diagonal(m, std=0.01))
        
        elif name == 'B':
            # Control matrix B usually starts very small
            module.apply(lambda m: init_weights_near_zero(m, std=0.001))
            
        else:
            # Encoders, Decoders, and Control Encoders/Decoders
            if mode == 'gaussian':
                module.apply(init_weights_gaussian)
            elif mode == 'uniform':
                module.apply(lambda m: init_weights_uniform(m, limit=1.0))
            elif mode == 'near_zero':
                module.apply(init_weights_near_zero)