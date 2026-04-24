# systems.py
import numpy as np

def van_der_pol(state, t, mu=1.0):
    """
    Van der Pol oscillator
    dx/dt = y
    dy/dt = mu*(1-x^2)*y - x
    """
    x, y = state
    dxdt = y
    dydt = mu*(1 - x**2)*y - x
    return [dxdt, dydt]

def limit_cycle_system(state, t):
    """
    2D system with a limit cycle at x^2 + y^2 = 1
    """
    x, y = state
    dxdt = x - y - x*(x**2 + y**2)
    dydt = x + y - y*(x**2 + y**2)
    return [dxdt, dydt]

def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    """
    Classic Lorenz system
    """
    x, y, z = state
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]


def oscillator_system(
    state,
    t,
    c1=1.6,
    c2=0.16,
    c3=0.16,
    c4=0.06,
    b1=5.0,
    b2=5.0,
    b3=5.0,
    K=1.0,
    u=(0.0, 0.0, 0.0),
):
    """
    6D nonlinear oscillator cascade (uncontrolled by default)

    state = [m1, m2, m3, p1, p2, p3]
    """

    m1, m2, m3, p1, p2, p3 = state
    u1, u2, u3 = u

    m1_dot = c1 / (K + p3**2) - c2*m1 + b1*u1
    p1_dot = c3*m1 - c4*p1

    m2_dot = c1 / (K + p1**2) - c2*m2 + b2*u2
    p2_dot = c3*m2 - c4*p2

    m3_dot = c1 / (K + p2**2) - c2*m3 + b3*u3
    p3_dot = c3*m3 - c4*p3

    return [
        m1_dot, m2_dot, m3_dot,
        p1_dot, p2_dot, p3_dot
    ]


# Dictionary for easy access
SYSTEMS = {
    "van_der_pol": van_der_pol,
    "limit_cycle": limit_cycle_system,
    "oscillator": oscillator_system,
    "lorenz": lorenz_system,
}
