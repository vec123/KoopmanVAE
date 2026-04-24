import numpy as np

def inverted_pendulum_controlled(state, t, u, m=1.0, l=1.0, g=9.81, b=0.1, 
                                 process_noise=[0.0], control_noise=0.0):
    """
    State: [theta, omega]
    u: Torque applied at the joint
    """
    theta, omega = state
    
    # 1. Apply Control Noise (Actuator jitter)
    u_noisy = u + np.random.normal(0, control_noise)
    
    dtheta = omega
    
    # 2. Physics calculation
    domega = (u_noisy - b * omega - m * g * l * np.sin(theta)) / (m * l**2)
    
    # 3. Apply Process Noise (Environmental disturbances to acceleration)
    domega += np.random.normal(0, process_noise[0])
    
    return [dtheta, domega]

def cartpole_controlled(state, t, u, mc=1.0, mp=0.1, l=0.5, g=9.81, b=0.1, 
                        process_noise = [ 0.0,0.0 ], control_noise=0.0):
    """
    State: [x, x_dot, theta, theta_dot]
    u: Force applied to the cart
    """
    x, x_dot, theta, theta_dot = state
    
    # 1. Apply Control Noise
    u_noisy = u + np.random.normal(0, control_noise)
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Denominator for acceleration calculations
    temp = (u_noisy + mp * l * theta_dot**2 * sin_t) / (mc + mp)
    theta_acc = (g * sin_t - cos_t * temp) / (l * (4.0/3.0 - mp * cos_t**2 / (mc + mp)))
    x_acc = temp - mp * l * theta_acc * cos_t / (mc + mp)

    # 2. Apply Process Noise to both accelerations
    # x_acc and theta_acc get random hits from the environment
    x_acc += np.random.normal(0, process_noise[0])
    theta_acc += np.random.normal(0,  process_noise[1])

    return [x_dot, x_acc, theta_dot, theta_acc]

def cartpole_linear(state, t, u, mc=1.0, mp=0.1, l=0.5, g=9.81):
    """
    Linearized State: [x, x_dot, theta, theta_dot]
    Simplified for theta near 0 (upright).
    """
    x, x_dot, theta, theta_dot = state
    
    # Common denominator for the linearized physics
    det = l * (4.0/3.0 - mp / (mc + mp))
    
    # Linearized accelerations
    # These are derived by setting sin(theta)=theta and cos(theta)=1
    theta_acc = (g * theta - (u / (mc + mp))) / det
    x_acc = (u / (mc + mp)) - (mp * l * theta_acc / (mc + mp))
    
    return [x_dot, x_acc, theta_dot, theta_acc]