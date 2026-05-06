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


