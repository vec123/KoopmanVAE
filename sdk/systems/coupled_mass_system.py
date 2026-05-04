import numpy as np
from base import BaseSystem

class CoupledMassSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # Unpack or default params: m1, m2, d1, d2, k_lin, k_cub
        p = params or (1.0, 1.0, 0.1, 0.1, 1.0, 5.0)
        m_init = init_mean or [0.0, 0.0, 0.0, 0.0]
        r_init = init_range or [1.0, 0.0, 1.0, 0.0]
        
        super().__init__(
            name="complex_mass", 
            state_dim=4, 
            control_dim=1, 
            labels=[r"$x_1$", r"$\dot{x}_1$", r"$x_2$", r"$\dot{x}_2$"], 
            params=p, 
            dt=0.01, 
            init_mean=m_init, 
            init_range=r_init
        )

    def ode(self, state, t, u, process_noise, control_noise):
        """
        Underactuated Dynamics for two coupled masses.
        u is a scalar force applied ONLY to mass 1.
        """
        x1, v1, x2, v2 = state
        m1, m2, d1, d2, k_lin, k_cub = self.params

        # 1. Control Logic (Scalar Force on Mass 1)
        u_val = u[0] if hasattr(u, "__len__") else u
        # Pre-sampled control_noise[0] added here
        u1_noisy = u_val + control_noise[0]
        
        # 2. Coupling Force (Non-linear Spring Interaction)
        rel_dist = x2 - x1
        f_coupling = k_lin * rel_dist + k_cub * (rel_dist**3)
        
        # 3. Physics Equations (Drift)
        # Mass 1: Driven by Control + Coupling - Damping - Anchor Spring
        a1 = (u1_noisy + f_coupling - d1 * v1 - k_lin * x1) / m1
        
        # Mass 2: Driven ONLY by Coupling - Damping - Anchor Spring (No Actuator)
        a2 = (-f_coupling - d2 * v2 - k_lin * x2) / m2
        
        # 4. Apply Process Noise (Pre-sampled disturbances to accelerations)
        # We map process_noise indices to the acceleration derivatives
        # State index 1 is v1 (derivative is a1), index 3 is v2 (derivative is a2)
        a1 += process_noise[1]
        a2 += process_noise[3]
        
        return [v1, a1, v2, a2]