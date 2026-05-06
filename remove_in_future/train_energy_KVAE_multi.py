# train_koopman_all_systems.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import joblib

from models.models import (
    ResidualMLP,
    BottleneckedResidualMLP,
    LinearMatrix,
    TTKoopman,
    KoopmanEncoder
)
from trainers.Multi_TT_KVAE_trainer import KoopmanVAETrainer
from trainers.Controlled_Multi_KVAE_trainer import ControlledKoopmanVAETrainer
from logger.logger import InfoVectorLogger
from utils.utils import (
    save_training_trajectories,
    save_training_trajectories_w_forcing,
    split_into_subsequences_controlled, 
    split_into_subsequences_controlled_with_forcing,
    ControlledSequenceDataset
    )



# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

        
    plot_training_trajs = True
    normalize_scale = False
    concat_true = True
    encode_control = False
    use_forcing = True
    use_control = False
    learn_forcing = False

    tensor_train_koopman = False  
    stochastic = True

    log_name = "Test" 
    
    DATASET = "ES0031405047432001AZ0F"
    DATASET = "ES0031405047432001AZ0F"
    DATASET = "ES0031405049538001ML0F"
   
    METEO = "meteo_barcelona"
    num_epochs = 5000
    save_every = 100

    subseq_len = 24*7
    stride = 10
    batch_size = 64
    latent_dim = 32
    tt_rank = 32  
    tt_shape = [(10, 10), (10, 10)]
    hidden_dim = 128
    horizon = 24

    
    # --- Load Data by Loading the .CSV ---
    START_DATE = "2010-01-01 00:00:00"
    END_DATE   = "2030-12-31 23:59:59"
    DATASET_PATH = f"data_in/SensorData/{DATASET}.csv"
    EXTERNAL_FORCING_PATH = f"data_in/TemperatureData/{METEO}.csv"

    TARGET_COL = "value"  
     
                  
    TIME_COL = "timestamp"                      

    FORCING_COLS = ["tm_C"]
    FORCING_TIME_COL = "date"

    
    #  Load Data from CSV
    n_traj = 1
    df = pd.read_csv(DATASET_PATH)
    df_external_forcing =  pd.read_csv(EXTERNAL_FORCING_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    mask = (df[TIME_COL] >= START_DATE) & (df[TIME_COL] <= END_DATE)
    df = df.loc[mask].reset_index(drop=True)

    df_external_forcing =  pd.read_csv(EXTERNAL_FORCING_PATH)
    df_external_forcing[FORCING_TIME_COL] = pd.to_datetime(df_external_forcing[FORCING_TIME_COL])
    mask = (df_external_forcing[FORCING_TIME_COL] >= START_DATE) & (df_external_forcing[FORCING_TIME_COL] <= END_DATE)
    df_external_forcing = df_external_forcing.loc[mask].reset_index(drop=True)


    # Add  Cyclic Time Features
    # Hour of day (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df[TIME_COL].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df[TIME_COL].dt.hour / 24.0)
    
    # Day of week (0-6)
    df['day_sin'] = np.sin(2 * np.pi * df[TIME_COL].dt.dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df[TIME_COL].dt.dayofweek / 7.0)

    #  Month of year (1-12)
    # Subtract 1 so it ranges from 0-11 for the calculation
    df['month_sin'] = np.sin(2 * np.pi * (df[TIME_COL].dt.month - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df[TIME_COL].dt.month - 1) / 12.0)

    # Season
    # 0: Winter, 1: Spring, 2: Summer, 3: Autumn
    # Using (month % 12 // 3) is a standard way to align Dec-Jan-Feb as Winter
    df['season'] = (df[TIME_COL].dt.month % 12 // 3)
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4.0)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4.0)

    df['u'] = 0*np.cos(2 * np.pi * df[TIME_COL].dt.dayofweek / 7.0)
    # Define all columns to be used as state
    FEATURE_COLS = [TARGET_COL, 'hour_sin', 'hour_cos', 'day_sin', 'day_cos','month_sin', 'month_cos' ]


    # We use a 'left' merge on timestamps so the result matches df exactly
    df = pd.merge(
        df, 
        df_external_forcing[[FORCING_TIME_COL] + FORCING_COLS], 
        left_on=TIME_COL, 
        right_on=FORCING_TIME_COL, 
        how='left'
    )

    # 3. Handle resolution gaps or missing values (e.g., if weather is hourly and sensor is 15-min)
    # Forward fill then backward fill to ensure no NaNs remain
    df[FORCING_COLS] = df[FORCING_COLS].ffill().bfill()

    # 4. Now extract the tensors from the synchronized dataframe
    X_raw = torch.tensor(df[FEATURE_COLS].values).float().unsqueeze(0)
    if use_control:
        control = df['u'].values
    else:
        control = 0*df['u'].values
    U_raw = torch.tensor(control).float().unsqueeze(1).unsqueeze(0)

    if use_forcing:
        forcing = df[FORCING_COLS].values
    else:
         forcing = 0*df[FORCING_COLS].values
    F_raw = torch.tensor(forcing).float().unsqueeze(0)

    N, seq_len, state_dim = X_raw.shape
    N, seq_len, control_dim = U_raw.shape
    N, seq_len, forcing_dim = F_raw.shape

    if plot_training_trajs:
        save_training_trajectories_w_forcing(
            X_raw=X_raw, 
            U_raw=U_raw, 
            F_raw=F_raw,
            labels=["x_1", "x_2","x_3","x_4","x_5", "x_6", "x_7" ], 
            save_dir="generated_samples",
            dataset_name=DATASET,
            num_figs=5
        )
    
    # -------Prepare and Normalize (Optional) Training Data  ---
    if normalize_scale:
        X_flat = X_raw.reshape(-1, state_dim)
        U_flat = U_raw.reshape(-1, control_dim)
        F_flat = F_raw.reshape(-1, control_dim)

        x_scaler = StandardScaler()
        u_scaler = StandardScaler()
        f_scaler = StandardScaler()

        X_scaled_flat = x_scaler.fit_transform(X_flat)
        U_scaled_flat = u_scaler.fit_transform(U_flat)
        F_scaled_flat = u_scaler.fit_transform(F_flat)

        # Reshape back to [N, T, D]
        X_train = X_scaled_flat.reshape(n_traj, seq_len, state_dim)
        U_train = U_scaled_flat.reshape(n_traj, seq_len, control_dim)
        F_train = F_scaled_flat.reshape(n_traj, seq_len, forcing_dim)

        # SAVE SCALERS for use in the controller/inference script
        log_dir = f"logs/{log_name}_{DATASET}"
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        joblib.dump(x_scaler, os.path.join(log_dir, "x_scaler.pkl"))
        joblib.dump(u_scaler, os.path.join(log_dir, "u_scaler.pkl"))
        joblib.dump(f_scaler, os.path.join(log_dir, "f_scaler.pkl"))
        print(f"Scalers saved to {log_dir}")

    else:
        X_train, U_train, F_train = X_raw, U_raw ,F_raw

    x_sequences = [X_train[i] for i in range(n_traj)]
    u_sequences = [U_train[i] for i in range(n_traj)]
    f__sequences = [F_train[i] for i in range(n_traj)]

    print(f"Number of sequences: {len(x_sequences)}") 
    if len(x_sequences) > 0:
        print(f"Shape of first sequence: {x_sequences[0].shape}")

    x_sub, u_sub, f_sub = split_into_subsequences_controlled_with_forcing(x_sequences, u_sequences,f__sequences, subseq_len, stride)

    print(f"Number of subsequences: {len(x_sub)}") 
    if len(x_sub) > 0:
        print(f"Shape of first subsequence: {x_sub[0].shape}")
    loader = DataLoader(ControlledSequenceDataset(x_sub, u_sub, f_sub), batch_size=batch_size, shuffle=True)

    #----------Modules
    #encoder = LinearKoopmanEncoder(state_dim, latent_dim).to(device)  
    encoder = KoopmanEncoder(state_dim, latent_dim, hidden_dim, hidden_depth=5).to(device)  
   
    if concat_true:
        latent_dim = latent_dim + state_dim 
    decoder = ResidualMLP(latent_dim, state_dim, [hidden_dim]*3).to(device)
    #decoder = LinearMatrix(latent_dim , state_dim).to(device)

    system_matrix =  LinearMatrix(latent_dim, latent_dim).to(device)
    #system_matrix =  TTKoopman(latent_dim=latent_dim, tt_rank=16, tt_shape=[(8, 8), (16, 16)]).to(device)

    if use_forcing:
        external_forcing_matrix = LinearMatrix(forcing_dim, latent_dim).to(device)
    else:
        external_forcing_matrix = None
    control_matrix = LinearMatrix(control_dim, latent_dim).to(device)

    if learn_forcing:
        forcing_bottleneck_dim = 3
        forcing_network = BottleneckedResidualMLP(
            in_dim=latent_dim, 
            out_dim=latent_dim, 
            hidden_channels=[hidden_dim]*3,
            bottleneck_dim=forcing_bottleneck_dim
        ).to(device)

        with torch.no_grad():
            for m in forcing_network.modules():
                if isinstance(m, torch.nn.Linear):
                    # Scale weights to be very small (e.g., 1e-4)
                    m.weight.data.mul_(0.001)
                    # Initialize biases to zero to ensure zero output initially
                    if m.bias is not None:
                        m.bias.data.zero_()
    else:
        forcing_network = None

    if encode_control:
        control_encoder = ResidualMLP(control_dim+state_dim, latent_dim, [hidden_dim]*3).to(device)
        control_decoder =  ResidualMLP(latent_dim, control_dim, [hidden_dim]*3).to(device)
        control_matrix = LinearMatrix(latent_dim, latent_dim).to(device)
    else:
        control_encoder = None
        control_decoder = None
        control_matrix = LinearMatrix(control_dim, latent_dim).to(device)

    with torch.no_grad():
        for param in control_matrix.parameters():
            param.data.mul_(0.1)
        #for param in system_matrix.parameters():
        #    param.data.mul_(1.00)

    # --- Dynamic Parameter Collection ---
    params_to_optimize = []
    
    # Core modules (always present)
    params_to_optimize += list(encoder.parameters())
    params_to_optimize += list(decoder.parameters())
    params_to_optimize += list(system_matrix.parameters())

    # Optional modules (added only if they exist)
    if use_forcing and external_forcing_matrix:
        params_to_optimize += list(external_forcing_matrix.parameters())
        
    if learn_forcing and forcing_network:
        params_to_optimize += list(forcing_network.parameters())
        
    if use_control and control_matrix:
        params_to_optimize += list(control_matrix.parameters())

    if encode_control:
        if control_encoder: params_to_optimize += list(control_encoder.parameters())
        if control_decoder: params_to_optimize += list(control_decoder.parameters())

    # Initialize Optimizer with the combined list
    optimizer = optim.Adam(params_to_optimize, lr=1e-3)

    logger = InfoVectorLogger(log_dir=f"logs/{log_name}_{DATASET}")

    # 3. Trainer
    trainer = ControlledKoopmanVAETrainer(
        encoder=encoder,
        decoder=decoder,
        system_matrix=system_matrix,
        dataloader=loader,
        optimizer=optimizer,
        control_matrix=control_matrix,
        external_forcing_matrix = external_forcing_matrix,
        forcing_network = forcing_network,
        control_encoder=control_encoder,
        control_decoder = control_decoder,
        latent_dim=latent_dim,
        horizon=horizon,
        beta=1e-80,          # KL
        gamma_1=10.0,         # koopman dynamics weight in latent space
        gamma_2=10.0,         # koopman dynamics reconstruction loss weight
        delta=1.e-10,         # Spectral loss weight
        alpha=0.0001,        # Entropy loss weight
        epsilon_1 = 10.00,  #Initial reconstruction loss weight
        epsilon_2 = 10.00,   # All-time reconstruction loss weight
        zero_structure_gain = 0.0, # Zero-structure loss weight
        lambda_forcing = 0.1, # Forcing term regularization
        device=device,
        logger=logger,
        save_epoch = save_every,
        val_epochs = save_every,
        stochastic = stochastic,
        concat_true = concat_true,
        horizon_decay = 1,
        use_ema = True
    )

    print("Starting Controlled Joint Training...")
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()