from .systems import *

SYSTEM_REGISTRY = {
    # --- Mechanical Systems ---
    "cartpole": CartpoleSystem(
        linearized=False,
        params=(1.0, 0.1, 0.5, 9.81, 0.1),         # mc, mp, l, g, b
        init_mean=[0.0, 0.0, 0.0, 0.0],            # x, x_dot, theta, theta_dot
        init_range=[0.1, 0.1, 0.2, 0.1]            # +/- variation
    ),
    
    "cartpole_linear": CartpoleSystem(
        linearized=True,
        params=(1.0, 0.1, 0.5, 9.81, 0.1),         # mc, mp, l, g, b
        init_mean=[0.0, 0.0, 1.57, 0.0],           # x, x_dot, theta (upright), theta_dot
        init_range=[0.1, 0.1, 0.2, 0.1]
    ),
    
    "pendulum": InvertedPendulumSystem(
        params=(1.0, 1.0, 9.81, 0.1),              # m, l, g, b
        init_mean=[0.0, 0.0],                      # theta, omega
        init_range=[np.pi, 0.0]                    # Start anywhere on the circle
    ),

    # --- Oscillator Systems ---
    "spring": DuffingSpringSystem(
        params=(0.1, 1.0, 5.0),                    # delta, alpha, beta
        init_mean=[0.0, 0.0],                      # x, x_dot
        init_range=[1.0, 0.5]
    ),
    
    "complex_mass": CoupledMassSystem(
        params=(1.0, 1.0, 0.1, 0.1, 1.0, 5.0),     # m1, m2, d1, d2, k_lin, k_cub
        init_mean=[0.0, 0.0, 0.0, 0.0],            # x1, v1, x2, v2
        init_range=[1.0, 0.0, 1.0, 0.0]
    ),

    # --- Thermal Systems (Values in Kelvin) ---
    "hvac_nonlinear": HVACSystem(
        nonlinear=True,
        params=(0.05, 0.8, 5.67e-8, 5000.0),       # k_conv, epsilon, sigma_sb, heater
        init_mean=[298.15, 295.15, 293.15, 295.15],# T1, T2, Tw, Tout
        init_range=[1.0, 0.5, 0.5, 0.5]
    ),
    
    "hvac_linear": HVACSystem(
        nonlinear=False,
        params=(0.1, 0.9, 5.0),                    # insulation, coupling, heater
        init_mean=[298.15, 295.15, 293.15, 295.15],
        init_range=[1.0, 0.5, 0.5, 0.5]
    ),

    # --- Baseline ---
    "independent_linear": IndependentLinearSystem(
        params=(),                                 # No physical constants for basic linear
        init_mean=[0.0, 0.0, 0.0, 0.0],            # x, dx, z, dz
        init_range=[0.5, 0.5, 0.5, 0.5]
    )
}