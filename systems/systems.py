import numpy
from .systems_logic import *


class BaseSystem:
    def __init__(self, name, state_dim, control_dim, labels, params, u_scale, 
                 init_mean=None, init_range=None, obs_dim=None):
        self.name = name
        self.state_dim = state_dim 
        self.obs_dim = obs_dim or state_dim 
        self.control_dim = control_dim
        self.labels = labels
        self.u_scale = u_scale
        
        # Physics Parameters
        self.params = params 
        
        # Initial State Distribution (Defaults to 0 mean, 0.1 range if not provided)
        self.init_mean = np.array(init_mean) if init_mean is not None else np.zeros(state_dim)
        self.init_range = np.array(init_range) if init_range is not None else np.ones(state_dim) * 0.1

    def get_initial_state(self):
        """Generates a state: mean + uniform(-range, range)"""
        offsets = np.random.uniform(-self.init_range, self.init_range)
        return self.init_mean + offsets
    
    def format_control(self, u):
        return u

    def observe(self, state):
        """Standard observation is just the state. Override for Cartpole."""
        return np.array(state)

    def get_sigma(self, noise_lvl):
            """
            Maps noise_lvl to state dimensions.
            - If list/ndarray: returns as is (must match state_dim).
            - If float: applies to 'derivative' indices (1, 3, ...) as default.
            """
            if isinstance(noise_lvl, (list, np.ndarray)):
                if len(noise_lvl) != self.state_dim:
                    raise ValueError(f"Noise list length {len(noise_lvl)} must match state_dim {self.state_dim}")
                return np.array(noise_lvl)
            
            # Default behavior for scalar: noise on velocity/derivatives only
            sig = np.zeros(self.state_dim)
            for i in range(1, self.state_dim, 2):
                sig[i] = noise_lvl
            return sig



class DuffingSpringSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        p = params or (0.1, 1.0, 5.0)
        m = init_mean or [0.0, 0.0]
        r = init_range or [1.0, 0.5]
        super().__init__("spring", 2, 1, [r"$x$", r"$\dot{x}$"], p, 5.0, m, r)

    def ode(self, state, t, u, process_noise):
        return simple_nonlinear_spring(state, t, u, *self.params, process_noise=process_noise)

class CoupledMassSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        p = params or (1.0, 1.0, 0.1, 0.1, 1.0, 5.0)
        m = init_mean or [0.0, 0.0, 0.0, 0.0]
        r = init_range or [1.0, 0.0, 1.0, 0.0]
        super().__init__("complex_mass", 4, 1, [r"$x_1$", r"$\dot{x}_1$", r"$x_2$", r"$\dot{x}_2$"], p, 5.0, m, r)

    def ode(self, state, t, u, process_noise):
        return complex_nonlinear_system(state, t, u, *self.params, process_noise=process_noise)

class HVACSystem(BaseSystem):
    def __init__(self, nonlinear=True, params=None, init_mean=None, init_range=None):
        self.state_dim = 4
        self.control_dim = 2
        if nonlinear:
            p = params or (0.05, 0.8, 5.67e-8, 5000.0)
            name = "hvac_nonlinear"
        else:
            p = params or (0.1, 0.9, 5.0)
            name = "hvac_linear"
            
        m = init_mean or [298.15, 295.15, 293.15, 295.15] # ~25C, 22C, 20C, 22C
        r = init_range or [1.0, 0.5, 0.5, 0.5]
        super().__init__(name, 4, 1, [r"$T_1$", r"$T_2$", r"$T_{wall}$", r"$T_{out}$"], p, 1.0, m, r)
        self.is_nonlinear = nonlinear

    def format_control(self, u):
        return 1.0 if u > 0.5 else 0.0

    def ode(self, state, t, u, process_noise):
        f = nonlinear_hvac_system if self.is_nonlinear else hvac_on_off_system
        return f(state, t, u, *self.params, process_noise=process_noise)

class IndependentLinearSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        m = init_mean or [0.0, 0.0, 0.0, 0.0]
        r = init_range or [0.5, 0.5, 0.5, 0.5]
        super().__init__("independent_linear", 4, 2, [r"$x$", r"$\dot{x}$", r"$z$", r"$\dot{z}$"], (), 5.0, m, r)

    def ode(self, state, t, u, process_noise):
        return simple_independent_linear(state, t, u, process_noise=process_noise)
