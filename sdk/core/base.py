# koopman_sdk/core/base.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


class BaseSystem(ABC):
    def __init__(self, name, state_dim, control_dim, labels, params, dt, init_mean, init_range):
        self.name = name
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.labels = labels
        self.params = params
        self.dt = dt
        self.init_mean = np.array(init_mean)
        self.init_range = np.array(init_range)

    @abstractmethod
    def ode(self, state, t, u, process_noise):
        pass

    def get_initial_state(self):
        """Standardized initial state sampler."""
        return self.init_mean + np.random.uniform(-self.init_range, self.init_range)
    
class KoopmanOperator(nn.Module, ABC):
    @abstractmethod
    def forward(self, z): pass
    
    @abstractmethod
    def get_matrix(self):
        """Required for spectral loss calculation."""
        pass

# koopman_sdk/core/dynamics.py
class LinearSystem(KoopmanOperator):
    def __init__(self, dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.eye(dim))
    
    def forward(self, z):
        return z @ self.matrix.t()

    def get_matrix(self):
        return self.matrix