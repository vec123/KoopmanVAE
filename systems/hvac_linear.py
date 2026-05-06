import numpy as np
from base import BaseSystem

class HVACSystem(BaseSystem):
    def __init__(self, params=None, init_mean=None, init_range=None):
        # Default params: insulation, coupling, heater_power
        p = params or (0.1, 0.9, 5.0)
        
        m_init = init_mean or [298.15, 295.15, 293.15, 295.15]
        r_init = init_range or [1.0, 0.5, 0.5, 0.5]
        
        super().__init__(
            name="hvac_linear",
            state_dim=4,
            control_dim=1,
            labels=[r"$T_1$", r"$T_2$", r"$T_{wall}$", r"$T_{out}$"],
            params=p,
            dt=1.0, # Thermal systems often use larger time steps
            init_mean=m_init,
            init_range=r_init
        )

    def format_control(self, u):
        """Ensure return is always a numpy array (vector)."""
        u = np.atleast_1d(u)
        return np.where(u > 0.5, 1.0, 0.0)

    def ode(self, state, t, u, process_noise, control_noise):
        """
        4-State Thermal System Logic.
        state: [T1, T2, Twall, Tout]
        u: Scalar binary control [0 or 1]
        """
        T1, T2, Twall, Tout = state
        insulation, coupling, heater_power = self.params

        u_noisy = u[0] + control_noise[0] 
        heater_on = 1.0 if u_noisy > 0.5 else 0.0

        # 2. Physics: Newton's Law of Cooling
        # Room 1: Gains from heater, exchange with Room 2 and Wall
        dT1 = (heater_on * heater_power) - \
              (insulation * (T1 - Twall)) - \
              (coupling * (T1 - T2))
        
        # Room 2: Exchange with Room 1 and Wall
        dT2 = (coupling * (T1 - T2)) - \
              (insulation * (T2 - Twall))
        
        # Wall: Interaction with both rooms and the outdoor environment
        dTwall = (insulation * (T1 - Twall)) + \
                 (insulation * (T2 - Twall)) - \
                 (insulation * (Twall - Tout))
        
        # Outside: Deterministic drift (diurnal cycle simulation)
        dTout = 0.01 * np.sin(t / 100)

        # 3. Apply Process Noise
        # In thermal systems, noise typically represents sensor jitter or 
        # unmodeled drafts/occupancy.
        derivs = [dT1, dT2, dTwall, dTout]
        
        # Vectorized addition of pre-sampled noise
        # Note: Tout (derivs[3]) usually has 0 noise if it's a forced boundary condition,
        # but the simulator can provide a value if desired.
        for i in range(len(derivs)):
            derivs[i] += process_noise[i]
            
        return derivs