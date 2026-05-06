import numpy as np
from base import BaseSystem

class DuffingSpringSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # Unpack or default params: delta (damping), alpha (linear), beta (cubic)
        p = params or (0.1, 1.0, 5.0)
        m_init = init_mean or [0.0, 0.0]
        r_init = init_range or [1.0, 0.5]
        
        super().__init__(
            name="spring", 
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
        A 1D Nonlinear Oscillator (Duffing Equation).
        Calculates the derivative of the state [x, x_dot].
        """
        x, x_dot = state
        delta, alpha, beta = self.params

        # 1. Control Logic (Scalar Force)
        u_val = u[0] if hasattr(u, "__len__") else u
        # Apply pre-sampled control noise
        u_noisy = u_val + control_noise[0]
        
        # 2. Physics Equations (Drift)
        # x_acc = Force - Damping - Linear_Spring - Cubic_Spring
        # Note: In the Duffing equation, mass is typically normalized to 1.0
        x_acc = u_noisy - delta * x_dot - alpha * x - beta * (x**3)
        
        # 3. Apply Process Noise (Pre-sampled disturbances to acceleration)
        # We add the noise vector provided by the simulator to the velocity derivative
        # state[0] = x -> derivative is x_dot
        # state[1] = x_dot -> derivative is x_acc
        x_acc += process_noise[1]
        
        return [x_dot, x_acc]