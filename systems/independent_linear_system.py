import numpy as np
from base import BaseSystem

class IndependentLinearSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # params: could be scaling factors for the actuators (default to 1.0)
        p = params or (1.0, 1.0) 
        
        m_init = init_mean or [0.0, 0.0, 0.0, 0.0]
        r_init = init_range or [0.5, 0.5, 0.5, 0.5]
        
        super().__init__(
            name="independent_linear", 
            state_dim=4, 
            control_dim=2, # Two independent actuators
            labels=[r"$x$", r"$\dot{x}$", r"$z$", r"$\dot{z}$"], 
            params=p, 
            dt=0.01, 
            init_mean=m_init, 
            init_range=r_init
        )

    def ode(self, state, t, u, process_noise, control_noise):
        """
        Two independent double-integrators.
        State: [x, x_dot, z, z_dot]
        u: [force_x, force_z]
        """
        x, x_dot, z, z_dot = state
        scale_x, scale_z = self.params

        # 1. Apply Control & Control Noise
        # u is expected to be [u0, u1]
        u0 = u[0] if hasattr(u, "__len__") else u
        u1 = u[1] if hasattr(u, "__len__") and len(u) > 1 else 0.0
        
        # control_noise is a vector of size (control_dim,) sampled by simulator
        u_noisy_x = (u0 + control_noise[0]) * scale_x
        u_noisy_z = (u1 + control_noise[1]) * scale_z
        
        # 2. Physics (Simple double integrator: a = F)
        # x_acc = u_x
        # z_acc = u_z
        x_acc = u_noisy_x
        z_acc = u_noisy_z
        
        # 3. Apply Process Noise
        # Maps to the derivatives of the velocities (indices 1 and 3)
        x_acc += process_noise[1]
        z_acc += process_noise[3]
        
        return [x_dot, x_acc, z_dot, z_acc]