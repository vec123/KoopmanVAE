import numpy as np
from base import BaseSystem

class VanDerPolSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # params: (mu,) where mu controls the nonlinearity/damping strength
        p = params or (1.0,)
        
        m_init = init_mean or [0.5, 0.0]
        r_init = init_range or [1.5, 0.5]
        
        super().__init__(
            name="vanderpol", 
            state_dim=2, 
            control_dim=1, 
            labels=[r"$x$", r"$\dot{x}$"], 
            params=p, 
            dt=0.01, 
            init_mean=m_init, 
            init_range=r_init
        )

    def ode(self, state, t, u, process_noise, control_noise):
        """
        x'' - mu(1 - x^2)x' + x = u
        State: [x, x_dot]
        """
        x, x_dot = state
        mu = self.params[0]

        # 1. Control & Noise
        u_val = u[0] if hasattr(u, "__len__") else u
        u_noisy = u_val + control_noise[0]

        # 2. Physics: x_acc = mu * (1 - x^2) * x_dot - x + u
        x_acc = mu * (1.0 - x**2) * x_dot - x + u_noisy

        # 3. Process Noise (on acceleration)
        x_acc += process_noise[1]

        return [x_dot, x_acc]