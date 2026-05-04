import numpy as np
class BaseSystem:
    def __init__(self, name, state_dim, control_dim, labels, params, dt, init_mean, init_range):
            
        self.name = name
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.labels = labels
        self.params = params
        self.dt = dt  # <--- This was likely missing

        self.u_scale = 1.0 
        
        
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
