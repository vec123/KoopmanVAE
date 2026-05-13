# registry.py
import numpy as np
from cartpole import CartpoleSystem
from inverted_pendulum import InvertedPendulumSystem
from duffing_spring import DuffingSpringSystem
from coupled_mass_system import CoupledMassSystem
from hvac_nonlinear import NonlinearHVACSystem
from hvac_linear import HVACSystem
from independent_linear_system import IndependentLinearSystem
from van_der_pool import VanDerPolSystem
from lorenz import LorenzSystem

SYSTEM_REGISTRY = {
    "cartpole": {
        "instance": CartpoleSystem(
            linearized=False,
            params=(1.0, 0.1, 0.5, 9.81, 0.1),
            init_mean=[0.0, 0.0, 0.0, 0.0],
            init_range=[0.1, 0.1, 0.2, 0.1]
        ),
        "config": {"dt": 0.02, "seq_len": 500, "substeps": 10, "u_scale": 10.0}
    },

    "cartpole_linear": {
        "instance": CartpoleSystem(
            linearized=True,
            params=(1.0, 0.1, 0.5, 9.81, 0.1),
            init_mean=[0.0, 0.0, 1.57, 0.0],
            init_range=[0.05, 0.05, 0.1, 0.05]
        ),
        "config": {"dt": 0.02, "seq_len": 400, "substeps": 10, "u_scale": 5.0}
    },

    "pendulum": {
        "instance": InvertedPendulumSystem(
            params=(1.0, 1.0, 9.81, 0.1),
            init_mean=[0.0, 0.0],
            init_range=[np.pi, 0.5]
        ),
        "config": {"dt": 0.05, "seq_len": 200, "substeps": 5, "u_scale": 3.0}
    },

    "spring": {
        "instance": DuffingSpringSystem(
            params=(0.1, 1.0, 5.0),
            init_mean=[0.0, 0.0],
            init_range=[1.0, 0.5]
        ),
        "config": {"dt": 0.01, "seq_len": 1000, "substeps": 2, "u_scale": 0.5}
    },

    "complex_mass": {
        "instance": CoupledMassSystem(
            params=(1.0, 1.0, 0.5, 0.5, 1.0, 0.5), # Reduced k_cub from 5.0 to 0.5, increased damping
            init_mean=[0.0, 0.0, 0.0, 0.0],
            init_range=[0.2, 0.1, 0.2, 0.1]       # Smaller initial displacement
        ),
        "config": {"dt": 0.01, "seq_len": 1000, "substeps": 10, "u_scale": 1.0} # Faster dt, more substeps
    },

    "hvac_nonlinear": {
        "instance": NonlinearHVACSystem(
            # High thermal mass (100k and 500k) prevents numerical overshoot
            params=(0.08, 0.9, 5.67e-8, 600.0, 100000.0, 500000.0),
            init_mean=[292.0, 292.0, 290.0, 285.0],
        ),
        "config": {
            "dt": 60.0,                # 1 minute snapshots
            "seq_len": 1440,           # 24 hours total
            "substeps": 200,           # High resolution for T^4 stability
            "u_scale": 1.0, 
            "u_time_scale": (7200.0, 28800.0) # On/Off cycles of 2 to 8 hours
        }
    },

    "hvac_linear": {
        "instance": HVACSystem(
            params=(0.1, 0.9, 5.0),
            init_mean=[298.15, 295.15, 293.15, 295.15],
            init_range=[2.0, 1.0, 1.0, 2.0]
        ),
        "config": {
                "dt": 60.0,                # 1 minute snapshots
                "seq_len": 1440,           # 24 hours
                "substeps": 500,           # High-precision internal integration
                "u_scale": 1.0, 
                "u_time_scale": (3600.0, 14400.0) # HVAC on/off for 1 to 4 hours
            }
    },

    "independent_linear": {
        "instance": IndependentLinearSystem(
            params=(1.0, 1.0),
            init_mean=[0.0, 0.0, 0.0, 0.0],
            init_range=[0.5, 0.5, 0.5, 0.5]
        ),
        "config": {"dt": 0.1, "seq_len": 200, "substeps": 1, "u_scale": 0.0,  "u_time_scale": (5.0, 15.0) }
    },

    "vanderpol": {
        "instance": VanDerPolSystem(
            params=(1.5,),
            init_mean=[1.0, 0.0],
            init_range=[2.0, 1.0]
        ),
        "config": {"dt": 0.1, "seq_len": 500, "substeps": 5, "u_scale": 0.0,  "u_time_scale": (1.0, 5.0)}
    },

    "lorenz": {
        "instance": LorenzSystem(
            params=(10.0, 28.0, 2.667),
            init_mean=[1.0, 1.0, 20.0],
            init_range=[5.0, 5.0, 5.0]
        ),
        "config": {"dt": 0.001, "seq_len": 50000, "substeps": 1, "u_scale": 0.0,  "u_time_scale": (0.1, 3.0)}
    }
}