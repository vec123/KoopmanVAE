import numpy as np
from scipy.integrate import odeint



def generate_trajectories_euler_maruyama(
    sys, 
    n_traj=10, 
    seq_len=100, 
    dt=0.02,
    substeps=5,
    process_noise=0.01, 
    control_noise=0.0,
    control=True
):
    inner_dt = dt / substeps
    sigma_p = sys.get_sigma(process_noise)
    
    all_x, all_u = [], []
    for _ in range(n_traj):
        state = np.array(sys.get_initial_state())
        # Ensure raw control is always 2D: (seq_len, control_dim)
        u_raw_traj = (np.random.rand(seq_len, sys.control_dim) - 0.5) * 2 * sys.u_scale
        
        obs_history = []
        u_applied_history = []
        
        for i in range(seq_len):
            obs_history.append(sys.observe(state))
            
            # 1. Format and handle control toggle
            u_nominal = sys.format_control(u_raw_traj[i])
            if not control:
                u_nominal = np.zeros(sys.control_dim) # Use np.zeros to keep dimensions
            
            # 2. FORCE u_nominal to be an array so the shape is (control_dim,)
            u_nominal = np.atleast_1d(u_nominal)
            u_applied_history.append(u_nominal)
            
            # 3. Physics Substepping
            for _ in range(substeps):
                u_noisy = u_nominal + np.random.normal(0, control_noise, size=sys.control_dim)
                derivs = sys.ode(state, i*dt, u_noisy, process_noise=[0]*sys.state_dim)
                
                state += np.array(derivs) * inner_dt + sigma_p * np.random.normal(0, np.sqrt(inner_dt), sys.state_dim)
            
        all_x.append(np.array(obs_history))
        all_u.append(np.array(u_applied_history))
        
    # Final check: return as (N, T, D)
    return np.array(all_x), np.array(all_u)



"""
def generate_trajectories_odeint(system_type="pendulum", n_traj=5, seq_len=500, dt=0.02, noise_lvl=0.05, control=True):
    t = np.linspace(0, seq_len * dt, seq_len)
    all_x = []
    all_u = []

    for i in range(n_traj):
        if system_type == "pendulum":
            init = [np.random.uniform(-np.pi, np.pi), 0]
            func = inverted_pendulum_controlled
            u_scale = 2.0
        elif system_type == "cartpole":
            init = [0, 0, 1.57, 0]
            func = cartpole_controlled
            u_scale = 10.0

        if control:
            u_seq = np.repeat(np.random.uniform(-u_scale, u_scale, size=(seq_len//10 + 1)), 10)[:seq_len]
        else:
            u_seq = 0*np.repeat(np.random.uniform(-u_scale, u_scale, size=(seq_len//10 + 1)), 10)[:seq_len]
        
        traj = [np.array(init)]
        curr_state = init
        for step in range(seq_len - 1):
            # Pass noise levels to the function
            sol = odeint(func, curr_state, [0, dt], 
                         args=(u_seq[step], 1.0, 0.1, 0.5, 9.81, 0.1, noise_lvl, noise_lvl))
            curr_state = sol[-1]
            traj.append(curr_state)
            
        all_x.append(np.array(traj))
        all_u.append(u_seq.reshape(-1, 1))

    return np.array(all_x), np.array(all_u), t

def generate_trajectories_euler_maruyama_old(system_type="pendulum", n_traj=5, seq_len=500, 
                          dt=0.02, noise_lvl=0.05, sub_steps=10, control=True):
    t = np.linspace(0, seq_len * dt, seq_len)
    all_x = []
    all_u = []
    sub_dt = dt / sub_steps

    for i in range(n_traj):
        
        if system_type == "pendulum":
            init = [np.random.uniform(-np.pi, np.pi), 0.0]
            func = inverted_pendulum_controlled
            u_scale = 2.0
            params = (1.0, 1.0, 9.81, 0.1) 
            # Ensure noise_lvl is a list for consistency [theta_noise, omega_noise]
            # Usually we only apply noise to omega (index 1)
            sigma = [0.0, noise_lvl] if isinstance(noise_lvl, float) else noise_lvl

        elif system_type == "cartpole":
            # Define ranges (adjust these based on your needs)
            pos_range = 0.1    # +/- 0.5 meters
            vel_range = 0.1    # +/- 0.1 m/s
            ang_range = 0.2    # +/- 0.2 radians (~11 degrees)
 
            ang_vel_range = 0.1 # +/- 0.1 rad/s
            u_dim = 2 

            init = [
                np.random.uniform(-pos_range, pos_range),
                np.random.uniform(-vel_range, vel_range),
                np.random.uniform(-ang_range, ang_range), # Centered around 1.57
                np.random.uniform(-ang_vel_range, ang_vel_range)
            ]
            
            #init = [0, 0, 1.57, 0]
            func = cartpole_controlled
            u_scale = 15.0
            params = (1.0, 0.1, 0.5, 9.81, 0.1)
            # Match noise to [x, x_dot, theta, theta_dot]
            # If user passed [noise_x, noise_theta], we map to velocity indices 1 and 3
            if isinstance(noise_lvl, list) and len(noise_lvl) == 2:
                sigma = [0.0, noise_lvl[0], 0.0, noise_lvl[1]]
            else:
                sigma = [0.0, noise_lvl, 0.0, noise_lvl]

        elif system_type == "cartpole_linear":
            # Define ranges (adjust these based on your needs)
            pos_range = 0.1    # +/- 0.5 meters
            vel_range = 0.1    # +/- 0.1 m/s
            ang_range = 0.2    # +/- 0.2 radians (~11 degrees)
            ang_vel_range = 0.1 # +/- 0.1 rad/s
            u_dim = 1 
            init = [
                np.random.uniform(-pos_range, pos_range),
                np.random.uniform(-vel_range, vel_range),
                1.57 + np.random.uniform(-ang_range, ang_range), # Centered around 1.57
                np.random.uniform(-ang_vel_range, ang_vel_range)
            ]
            #init = [0, 0, 1.57, 0]
            func = cartpole_linear
            u_scale = 15.0
            params = (1.0, 0.1, 0.5, 9.81, 0.1)
            # Match noise to [x, x_dot, theta, theta_dot]
            # If user passed [noise_x, noise_theta], we map to velocity indices 1 and 3
            if isinstance(noise_lvl, list) and len(noise_lvl) == 2:
                sigma = [0.0, noise_lvl[0], 0.0, noise_lvl[1]]
            else:
                sigma = [0.0, noise_lvl, 0.0, noise_lvl]

        elif system_type == "simple_independent_linear":
            # State: [x, x_dot, theta, theta_dot]
            init = [np.random.uniform(-0.5, 0.5) for _ in range(4)]
            func = simple_independent_linear # The new function
            u_scale = 5.0
            u_dim = 2 # TWO control inputs
            params = () # No physical constants needed for the simplest version
            sigma = [0.0, noise_lvl, 0.0, noise_lvl]
        elif system_type == "cartpole":
            # ... (your existing cartpole init code)
            u_dim = 1
            # Match noise to [x, x_dot, theta, theta_dot]
            if isinstance(noise_lvl, list) and len(noise_lvl) == 2:
                sigma = [0.0, noise_lvl[0], 0.0, noise_lvl[1]]
            else:
                sigma = [0.0, noise_lvl, 0.0, noise_lvl]
                
        elif system_type == "simple_nonlinear_spring":
            # State: [x, x_dot]
            pos_range = 1.0
            vel_range = 0.5
            init = [
                np.random.uniform(-pos_range, pos_range),
                np.random.uniform(-vel_range, vel_range)
            ]
            
            func = simple_nonlinear_spring
            u_scale = 5.0
            u_dim = 1 
            # Parameters: (delta, alpha, beta)
            params = (0.1, 1.0, 5.0) 
            
            # Match noise to [x, x_dot] -> typically noise is on acceleration (index 1)
            sigma = [0.0, noise_lvl] if isinstance(noise_lvl, float) else noise_lvl

        elif system_type == "complex_nonlinear_system" :
                    # State: [x1, x1_dot, x2, x2_dot]
                    pos_range = 1.0
                    vel_range = 0.5
                    
                    # Initialize all 4 states
                    init = [
                        np.random.uniform(-pos_range, pos_range), # x1
                        np.random.uniform(-vel_range, vel_range), # x1_dot
                        np.random.uniform(-pos_range, pos_range), # x2
                        np.random.uniform(-vel_range, vel_range)  # x2_dot
                    ]
                    
                    func = complex_nonlinear_system
                    u_dim = 1 # Usually 2 for coupled systems, one per mass
                    u_scale = 5.0
                    
                    # Parameters for the 4-state system
                    # Mapping: (m1, m2, d1, d2, k_lin, k_cub)
                    params = (1.0, 1.0, 0.1, 0.1, 1.0, 5.0) 
                    
                    # Process noise applied to the velocity derivatives (indices 1 and 3)
                    if isinstance(noise_lvl, float):
                        sigma = [0.0, noise_lvl, 0.0, noise_lvl]
                    else:
                        sigma = noise_lvl # Assumes noise_lvl is already a 4-element list
        
        elif system_type == "hvac_on_off_system" :
                    T_1_start = 25 + 273.15
                    T_1_range = 1.0
                    T_2_start = 22 + 273.15
                    T_2_range = 0.5
                    T_wall_start = 20 + 273.15
                    T_wall_range = 0.5
                    T_outer_start = 22 + 273.15
                    T_outer_range = 0.5
                    # Initialize all 4 states
                    init = [
                        T_1_start + np.random.uniform(-T_1_range, T_1_range), # x1
                        T_2_start + np.random.uniform(-T_2_range, T_2_range), # x1_dot
                        T_wall_start + np.random.uniform(-T_wall_range, T_wall_range), # x2
                        T_outer_start + np.random.uniform(-T_outer_range, T_outer_range),  # x2_dot
   
                    ]
                    
                    func = hvac_on_off_system
                    u_dim = 1 # Usually 2 for coupled systems, one per mass
                    u_scale = 1.0
                    
                    # Parameters for the 4-state system
                    # Mapping: (m1, m2, d1, d2, k_lin, k_cub)
                    params = (0.1, 0.9, 5.0) 
                    process_noise = [0.0, 0.0, 0.0, 0.0]
                    # Process noise applied to the velocity derivatives (indices 1 and 3)
                    if isinstance(noise_lvl, float):
                        sigma = [0.0, 0.0, 0.0, 0.0]
                    else:
                        sigma = noise_lvl 
        elif system_type == "nonlinear_hvac_system" :
                    T_1_start = 25 + 273.15
                    T_1_range = 1.0 
                    T_2_start = 22 + 273.15
                    T_2_range = 0.5 
                    T_wall_start = 20 + 273.15
                    T_wall_range = 0.5
                    T_outer_start = 22 + 273.15
                    T_outer_range = 0.5
                    # Initialize all 4 states
                    init = [
                        T_1_start + np.random.uniform(-T_1_range, T_1_range), # x1
                        T_2_start + np.random.uniform(-T_2_range, T_2_range), # x1_dot
                        T_wall_start + np.random.uniform(-T_wall_range, T_wall_range), # x2
                        T_outer_start + np.random.uniform(-T_outer_range, T_outer_range)  # x2_dot
                    ]
                    
                    func = nonlinear_hvac_system
                    u_dim = 1 # Usually 2 for coupled systems, one per mass
                    u_scale = 1.0
                    
                    # Parameters for the 4-state system
                    # Mapping: (m1, m2, d1, d2, k_lin, k_cub)
                    params = (0.05, 0.8, 5.67e-8, 500.0)
                    process_noise = [0.0, 0.0, 0.0, 0.0]
                    # Process noise applied to the velocity derivatives (indices 1 and 3)
                    if isinstance(noise_lvl, float):
                        sigma = [noise_lvl, noise_lvl, noise_lvl, noise_lvl]
                    else:
                        sigma = noise_lvl 

        if control:
            u_seq = np.zeros((seq_len, u_dim))
            
            # Loop through each control dimension independently
            for d in range(u_dim):
                current_idx = 0
                while current_idx < seq_len:
                    pulse_len = np.random.randint(5, 15) 
                    
                    u_val = np.random.uniform(-u_scale, u_scale)
                    if system_type == "nonlinear_hvac_system" or system_type == "hvac_on_off_system":
                         if u_val < 0.5:
                              u_val = 0
                         else:
                              u_val = 1.0
                    end_idx = min(current_idx + pulse_len, seq_len)
                    u_seq[current_idx:end_idx, d] = u_val
                    
                    current_idx = end_idx
        else:
            u_seq = np.zeros((seq_len, u_dim))
        
        traj = []
        curr_state = np.array(init, dtype=np.float64)
        
        for step in range(seq_len):
            traj.append(curr_state.copy())
            
            if step < seq_len - 1:
                u_curr = u_seq[step]
                for _ in range(sub_steps):
                    # 1. Deterministic Physics (Drift)
                    derivs = func(curr_state, 0, u_curr, *params, 
                                  process_noise=process_noise, control_noise=0.0)
                    
                    # 2. Stochastic Physics (Diffusion)
                    # np.random.normal(0, sigma) works per-element if sigma is a list
                    dw = np.random.normal(0, sigma) * np.sqrt(sub_dt)
                    
                    curr_state += np.array(derivs) * sub_dt + dw
                    
                        
        all_x.append(np.array(traj))
        all_u.append(u_seq)
        # all_u.append(u_seq.reshape(-1, 1))

    return np.array(all_x), np.array(all_u), t
 """