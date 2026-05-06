import numpy as np
from base import BaseSystem

class LorenzSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # params: (sigma, rho, beta) 
        # Standard chaotic values: (10.0, 28.0, 8/3)
        p = params or (10.0, 28.0, 2.667)
        
        m_init = init_mean or [0.1, 0.0, 0.0]
        r_init = init_range or [0.5, 0.5, 0.5]
        
        super().__init__(
            name="lorenz", 
            state_dim=3, 
            control_dim=1, # Control typically added to the y or z equation
            labels=[r"$x$", r"$y$", r"$z$"], 
            params=p, 
            dt=0.005, # Lorenz requires small time steps for stability
            init_mean=m_init, 
            init_range=r_init
        )

    def ode(self, state, t, u, process_noise, control_noise):
        """
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y + u
        dz/dt = x * y - beta * z
        """
        x, y, z = state
        sigma, rho, beta = self.params

        # 1. Control & Noise
        u_val = u[0] if hasattr(u, "__len__") else u
        u_noisy = u_val + control_noise[0]

        # 2. Physics Equations (Drift)
        dx = sigma * (y - x)
        dy = x * (rho - z) - y + u_noisy
        dz = x * y - beta * z

        # 3. Apply Process Noise
        # In chaos studies, even tiny noise can lead to massive divergence (the Butterfly Effect)
        derivs = [dx, dy, dz]
        return [d + n for d, n in zip(derivs, process_noise)]