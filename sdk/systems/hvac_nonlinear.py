import numpy as np
from base import BaseSystem

class NonlinearHVACSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # params: (k_conv, eps, sigma, heater_p, C_air, C_wall)
        # Real-world university room values (approximated)
        p = params or (0.05, 0.85, 5.67e-8, 500.0, 80000.0, 400000.0)
        
        super().__init__(
            name="hvac_nonlinear",
            state_dim=4,
            control_dim=1,
            labels=[r"$T_{air,1}$", r"$T_{air,2}$", r"$T_{wall}$", r"$T_{out}$"],
            params=p,
            dt=60.0, 
            init_mean=init_mean or [293.15, 293.15, 291.15, 285.15],
            init_range=init_range or [2.0, 1.0, 1.0, 0.0]
        )

    def format_control(self, u):
        """Force inputs to be strictly 0.0 (Off) or 1.0 (On)."""
        u = np.atleast_1d(u)
        return np.where(u >= 0.5, 1.0, 0.0)

    def ode(self, state, t, u, process_noise, control_noise):
        T1, T2, Tw, Tout = state
        k_conv, epsilon, sigma, heater_p, C_air, C_wall = self.params

        # Binary Heater Power (0 or 500 Watts)
        q_heater = u[0] * heater_p

        # Fluxes (Watts)
        q_rad = epsilon * sigma * (T1**4 - Tw**4)
        q_conv_air = k_conv * (T1 - T2)
        q_leak_air = k_conv * (T2 - Tout)
        q_leak_wall = (k_conv/2) * (Tw - Tout)

        # dT = Total Energy Flux / Thermal Capacity
        # Large C values make these derivatives small and stable
        dT1 = (q_heater - q_conv_air - q_rad) / C_air
        dT2 = (q_conv_air - q_leak_air) / C_air
        dTw = (q_rad - q_leak_wall) / C_wall
        dTout = 0.0

        return [d + n for d, n in zip([dT1, dT2, dTw, dTout], process_noise)]