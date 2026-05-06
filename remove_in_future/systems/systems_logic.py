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
    omega = (u_noisy - b * omega - m * g * l * np.sin(theta)) / (m * l**2)
     
    # 3. Apply Process Noise (Environmental disturbances to acceleration)
    domega += np.random.normal(0, process_noise[0])
    
    return [dtheta, domega]

def cartpole_controlled(state, t, u, mc=1.0, mp=0.1, l=0.5, g=9.81, b=0.1, 
                        process_noise=[0.0, 0.0], control_noise=0.0):
    x, x_dot, theta, theta_dot = state
    
    # --- Move periodicity here ---
    # Normalize theta to [-pi, pi] before calculating physics
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    u_val = u[0] if hasattr(u, "__len__") else u
    u_noisy = u_val + np.random.normal(0, control_noise) - b * x_dot
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    total_m = mc + mp
    temp = (u_noisy + mp * l * theta_dot**2 * sin_t) / total_m
    denom = l * (4.0/3.0 - (mp * cos_t**2) / total_m)
    
    theta_acc = (g * sin_t - cos_t * temp) / denom
    x_acc = temp - (mp * l * theta_acc * cos_t) / total_m

    x_acc += np.random.normal(0, process_noise[0])
    theta_acc += np.random.normal(0, process_noise[1])
    
    x =  [x_dot, x_acc, theta_dot, theta_acc]
    return x 

def cartpole_linear(state, t, u, mc=1.0, mp=0.1, l=0.5, g=9.81, process_noise=[0.0, 0.0], control_noise=0.0):
    """
    Linearized State: [x, x_dot, theta, theta_dot]
    Simplified for theta near 0 (upright).
    process_noise: [x_acc_noise, theta_acc_noise]
    """
    x, x_dot, theta, theta_dot = state
    
    # Apply control noise to the scalar input u
    u_noisy = u + control_noise
    
    # Common denominator for the linearized physics
    det = l * (4.0/3.0 - mp / (mc + mp))
    
    # Linearized accelerations
    theta_acc = (g * theta - (u_noisy / (mc + mp))) / det + process_noise[1]
    x_acc = (u_noisy / (mc + mp)) - (mp * l * theta_acc / (mc + mp)) + process_noise[0]
    
    return [x_dot, x_acc, theta_dot, theta_acc]

def simple_independent_linear(state, t, u, process_noise=[0.0, 0.0], control_noise=0.0):
    """
    State: [x, x_dot, theta, theta_dot]
    u: [force_x, torque_theta]
    process_noise: [x_acc_noise, theta_acc_noise]
    control_noise: scalar or [u_x_noise, u_theta_noise]
    """
    x, x_dot, theta, theta_dot = state
    
    # Handle control noise (scalar or vector)
    if isinstance(control_noise, (list, np.ndarray)):
        u_noisy = [u[0] + control_noise[0], u[1] + control_noise[1]]
    else:
        u_noisy = [u[0] + control_noise, u[1] + control_noise]
    
    # Accelerations with added process noise
    x_acc = u_noisy[0] + process_noise[0]
    theta_acc = u_noisy[1] + process_noise[1]
    
    return [x_dot, x_acc, theta_dot, theta_acc]

def simple_nonlinear_spring(state, t, u, delta=0.1, alpha=1.0, beta=5.0, 
                            process_noise=[0.0, 0.0], control_noise=0.0):
    """
    A 1D Nonlinear Oscillator (Duffing-like).
    State: [x, x_dot]
    u: [force]
    """
    x, x_dot = state
    
    # Apply control noise to the input force
    u_val = u[0] if hasattr(u, "__len__") else u
    u_noisy = u_val + np.random.normal(0, control_noise)
    
    # The Physics:
    # x_acc = Force - Damping - Linear_Spring - Cubic_Spring
    x_acc = u_noisy - delta * x_dot - alpha * x - beta * (x**3)
    
    # Add process noise to acceleration
    x_acc += np.random.normal(0, process_noise[0])
    
    return [x_dot, x_acc]

def complex_nonlinear_system(state, t, u, m1=1.0, m2=1.0, d1=0.1, d2=0.1, 
                                 k_lin=1.0, k_cub=5.0, 
                                 process_noise=[0.0, 0.0, 0.0, 0.0], 
                                 control_noise=0.0):
    x1, v1, x2, v2 = state
    
    # UNDERACTUATED LOGIC:
    # u is now a scalar (force on m1 only). 
    # Even if a vector is passed, we ignore the second element.
    u_val = u[0] if hasattr(u, "__len__") else u
    u1_noisy = u_val + np.random.normal(0, control_noise)
    u2_noisy = 0.0  # <--- Mass 2 has no actuator!
    
    # 1. Coupling Force (Interaction between m1 and m2)
    rel_dist = x2 - x1
    f_coupling = k_lin * rel_dist + k_cub * (rel_dist**3)
    
    # 2. Physics Equations
    # Mass 1: Pushed by YOU, pulled by Mass 2
    a1 = (u1_noisy + f_coupling - d1 * v1 - k_lin * x1) / m1
    
    # Mass 2: Pulled ONLY by the coupling spring and the wall
    a2 = (-f_coupling - d2 * v2 - k_lin * x2) / m2
    
    # 3. Process Noise
    p_noise = process_noise
    a1 += np.random.normal(0, p_noise[1] if len(p_noise) == 4 else p_noise[0])
    a2 += np.random.normal(0, p_noise[3] if len(p_noise) == 4 else p_noise[1])
    
    return [v1, a1, v2, a2]

def hvac_on_off_system(state, t, u, 
                       insulation=0.05, coupling=0.02, heater_power=5.0,
                       process_noise=[0.01, 0.01, 0.0, 0.0], control_noise=0.0):
    """
    4-State Thermal System with Binary Control.
    u: [0 or 1] (Heater Off or On)
    """
    T1, T2, Twall, Tout = state
    
    # Apply On/Off logic: u is treated as a binary trigger
    # We use a threshold to handle continuous-valued solvers trying to 'guess'
    heater_on = 1.0 if u > 0.5 else 0.0
    
    # Physics: Newton's Law of Cooling (dT/dt = -k * delta_T)
    # The 'Nonlinearity' can be added by making k dependent on temperature (convection)
    
    # Room 1: Gains from heater, loses to Room 2 and Wall
    dT1 = heater_on * heater_power - insulation * (T1 - Twall) - coupling * (T1 - T2)
    
    # Room 2: Gains from Room 1, loses to Wall
    dT2 = coupling * (T1 - T2) - insulation * (T2 - Twall)
    
    # Wall: Interaction with both rooms and the harsh outside
    dTwall = insulation * (T1 - Twall) + insulation * (T2 - Twall) - insulation * (Twall - Tout)
    
    # Outside: Simplified as a constant or slow sine wave (drift)
    dTout = 0.01 * np.sin(t / 100) 

    # Combine into derivative vector
    derivs = [dT1, dT2, dTwall, dTout]
    
    # Add process noise
    for i in range(len(derivs)):
        derivs[i] += np.random.normal(0, process_noise[i])
        
    return derivs

def hvac_on_off_system_(state, t, u, 
                           insulation=0.05, heater_power=5.0, damping=0.1,
                           process_noise=[0.01, 0.01, 0.01, 0.01], control_noise=0.0):
    """
    State vector x: [T1, T2, Tw, Tout, V1, V2, Vw, Vout]
    - T: Temperatures (Positions)
    - V: Rate of change (Velocities)
    """
    T1, T2, Tw, Tout, V1, V2, Vw, Vout = state
    
    # Binary control logic
    heater_on = 1.0 if u > 0.5 else 0.0
    
    # 1. Kinematics: The change in temperature is the current velocity
    dT1 = V1
    dT2 = V2
    dTw = Vw
    dTout = Vout
    
    # 2. Dynamics: The change in velocity (Acceleration)
    # Each equation follows: Thermal_Force - Damping + Random_Noise
    
    # Room 1 acceleration
    dV1 = (heater_on * heater_power - insulation * (T1 - Tw)) - (damping * V1)
    
    # Room 2 acceleration
    dV2 = (insulation * (T1 - T2) - insulation * (T2 - Tw)) - (damping * V2)
    
    # Wall mass acceleration
    dVw = (insulation * (T1 - Tw) + insulation * (T2 - Tw) - insulation * (Tw - Tout)) - (damping * Vw)
    
    # Outside drift: Modeling it as a random walk or a state-driven drift
    # Removing np.sin(t) makes it mathematically autonomous if 'u' is internalized.
    dVout = -0.001 * (Tout - 15.0) # Restoring force toward 15 degrees
    
    # Apply Process Noise to the VELOCITY states (Random Forces)
    accelerations = [dV1, dV2, dVw, dVout]
    for i in range(len(accelerations)):
        accelerations[i] += np.random.normal(0, process_noise[i])
        
    return [dT1, dT2, dTw, dTout] + accelerations

def nonlinear_hvac_system(state, t, u, 
                          k_conv=0.05,  # Convective coefficient
                          epsilon=0.9,  # Emissivity (0 to 1)
                          sigma_sb=5.67e-8, # Stefan-Boltzmann constant
                          heater_power=5000.0,
                          process_noise=[0.1, 0.1, 0.1, 0.1],
                           control_noise=0.0):
    """
    Nonlinear Thermal System with T^4 Radiation and On/Off Control.
    State: [T1, T2, Twall, Tout] in Kelvin
    """
    T1, T2, Twall, Tout = state
    
    # Binary Control Logic
    heater_on = 1.0 if u > 0.5 else 0.0
    
    # 1. Nonlinear Convection (dT is proportional to delta_T^1.25)
    # This models natural airflow better than a linear constant.
    def convection(Ta, Tb):
        delta = Ta - Tb
        return k_conv * np.sign(delta) * (np.abs(delta)**1.25)

    # 2. Radiative Heat Loss (T^4)
    # Heat exchange between the rooms and the thermal mass (wall)
    def radiation(Ta, Tb):
        return epsilon * sigma_sb * (Ta**4 - Tb**4)

    # 3. Physics Equations
    # Room 1: Heater + Convection from Room 2 - Radiation to Wall
    dT1 = (heater_on * heater_power / 1000.0) - convection(T1, T2) - radiation(T1, Twall)
    
    # Room 2: Convection from Room 1 - Radiation to Wall
    dT2 = convection(T1, T2) - radiation(T2, Twall)
    
    # Wall: Absorbs radiation from rooms, loses to outside
    dTwall = radiation(T1, Twall) + radiation(T2, Twall) - convection(Twall, Tout)
    
    # Outside: Slow oscillation (diurnal cycle)
    dTout =0 # 0.05 * np.cos(t / 24)

    derivs = [dT1, dT2, dTwall, dTout]
    
    # Add process noise
    derivs = [d + np.random.normal(0, n) for d, n in zip(derivs, process_noise)]
    
    return derivs