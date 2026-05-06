import numpy as np
from scipy.integrate import odeint



def simulate_euler_maruyama(
    sys, 
    n_traj=10, 
    seq_len=100, 
    dt=0.02,
    substeps=5,
    u_time_scale=(0.0, 0.0), # (min_hold_time, max_hold_time)
    process_noise_std=0.01, 
    control_noise_std=0.01,
    control=True
):
    inner_dt = dt / substeps
    all_x, all_u = [], []
    
    min_hold, max_hold = u_time_scale

    for _ in range(n_traj):
        state = np.array(sys.get_initial_state())
        obs_history = []
        u_applied_history = []
        
        # Control persistence variables
        current_u = np.zeros(sys.control_dim)
        hold_timer = 0.0
        
        for i in range(seq_len):
            # 1. Update Control based on timescale
            if control:
                if hold_timer <= 0:
                    # Time to pick a new control value
                    current_u = (np.random.rand(sys.control_dim) - 0.5) * 2 * sys.u_scale
                    # Pick how long to hold it (in seconds)
                    hold_duration = np.random.uniform(min_hold, max_hold)
                    hold_timer = hold_duration
                
                hold_timer -= dt # Decrement timer by the step size
            else:
                current_u = np.zeros(sys.control_dim)

            # 2. Observe and Store
            obs_history.append(sys.observe(state))
            u_nominal = sys.format_control(current_u)
            u_applied_history.append(u_nominal)
            
            # 3. Physics Substepping
            for _ in range(substeps):
                p_noise = np.random.normal(0, process_noise_std, size=sys.state_dim) * np.sqrt(inner_dt)
                c_noise = np.random.normal(0, control_noise_std, size=sys.control_dim)
                
                derivs = sys.ode(
                    state, 
                    i * dt, 
                    u_nominal, 
                    process_noise=p_noise / inner_dt, 
                    control_noise=c_noise
                )
                
                state += np.array(derivs) * inner_dt
            
        all_x.append(np.array(obs_history))
        all_u.append(np.array(u_applied_history))
        
    return np.array(all_x), np.array(all_u)