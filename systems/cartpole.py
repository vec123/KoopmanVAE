import numpy as np
from base import BaseSystem

class CartpoleSystem(BaseSystem):
    def __init__(self, linearized=False, params=None, init_mean=None, init_range=None):
        # Unpack or default params: mc, mp, l, g, b
        p = params or (1.0, 0.1, 0.5, 9.81, 0.1)
        self.is_linearized = linearized
        
        m_init = init_mean or [0.0, 0.0, (1.57 if linearized else 0.0), 0.0]
        r_init = init_range or [0.1, 0.1, 0.2, 0.1]
        
        if linearized:
            name = "cartpole_linear"
            labels = [r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
        else:
            name = "cartpole"
            # Note: 5 observables for 4 underlying states
            labels = [r"$x$", r"$\dot{x}$", r"$\sin\theta$", r"$\cos\theta$", r"$\dot{\theta}$"]
            
        super().__init__(
            name=name,
            state_dim=4, 
            control_dim=1,
            labels=labels,
            params=p, 
            dt=0.01,
            init_mean=m_init, 
            init_range=r_init
        )

    def observe(self, state):
        """Lifts the raw physical state to the observation space."""
        if self.is_linearized:
            return np.array(state)
        
        # Non-linear observation: x, x_dot, sin(theta), cos(theta), theta_dot
        x, x_dot, theta, theta_dot = state
        return np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

    def ode(self, state, t, u, process_noise, control_noise):
        """
        Deterministic Drift calculation. 
        Noise vectors are pre-sampled and passed in.
        """
        x, x_dot, theta, theta_dot = state
        mc, mp, l, g, b = self.params

        # 1. Periodicity
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # 2. Control & Control Noise
        u_val = u[0] if hasattr(u, "__len__") else u
        # u_noisy is the actual force applied to the cart
        u_noisy = u_val + control_noise[0] - b * x_dot
        
        # 3. Physics (Non-linear Cartpole Equations)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        total_m = mc + mp
        
        # Intermediate calculations for the standard cartpole ODE
        temp = (u_noisy + mp * l * (theta_dot**2) * sin_t) / total_m
        denom = l * (4.0/3.0 - (mp * (cos_t**2)) / total_m)
        
        theta_acc = (g * sin_t - cos_t * temp) / denom
        x_acc = temp - (mp * l * theta_acc * cos_t) / total_m

        # 4. Apply Process Noise (Disturbances to accelerations)
        # Assuming process_noise[0] is cart x_acc noise, [1] is pole theta_acc noise
        x_acc += process_noise[1]  # Maps to index 1 of state_dim derivative
        theta_acc += process_noise[3] # Maps to index 3 of state_dim derivative
        
        return [x_dot, x_acc, theta_dot, theta_acc]