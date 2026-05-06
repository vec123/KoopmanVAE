import numpy as np
from base import BaseSystem

class InvertedPendulumSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # Default params: m, l, g, b
        p = params or (1.0, 1.0, 9.81, 0.1)
        m_init = init_mean or [0.0, 0.0]
        r_init = init_range or [np.pi, 0.0]
        
        # Note: state_dim=2, control_dim=1
        super().__init__(
            name="pendulum", 
            state_dim=2, 
            control_dim=1, 
            labels=[r"$\theta$", r"$\omega$"], 
            params=p, 
            dt=0.01, # Recommended internal dt
            init_mean=m_init, 
            init_range=r_init
        )

    def ode(self, state, t, u, process_noise, control_noise):
        """
        Calculates the derivative of the state.
        
        Args:
            state: [theta, omega]
            t: Current time
            u: Control torque (scalar or array-like)
            process_noise: Vector of shape (state_dim,) - already sampled by simulator
            control_noise: Vector of shape (control_dim,) - already sampled by simulator
        """
        theta, omega = state
        m, l, g, b = self.params # Unpack the stored parameters
        
        # 1. Apply Control Noise (Actuator jitter)
        # u is typically a vector, so we take u[0] for scalar physics
        u_val = u[0] if isinstance(u, (list, np.ndarray)) else u
        u_noisy = u_val + control_noise[0]
        
        # 2. Physics calculation
        dtheta = omega
        
        # Acceleration (d_omega / dt)
        # Formula: tau = I*alpha + b*omega + m*g*l*sin(theta)
        # alpha = (tau - b*omega - m*g*l*sin(theta)) / (m * l^2)
        domega = (u_noisy - b * omega - m * g * l * np.sin(theta)) / (m * l**2)
        
        # 3. Apply Process Noise (Environmental disturbance to acceleration)
        # We add the noise vector provided by the simulator
        domega += process_noise[1] # Applied to the velocity derivative

        return [dtheta, domega]