import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.models import KoopmanEncoder

def export_clusters():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # --- Configuration ---
    SENSOR_NAME ="ES0031405047432001AZ0F"
    LOG_NAME = f"Multi_KVAE_energy_{SENSOR_NAME}"
    LOG_DIR = f"logs/{LOG_NAME}"
    DATASET_PATH = F"data_in/SensorData/{SENSOR_NAME}.csv"
    epoch = 5000
    OUTPUT_CSV = os.path.join(LOG_DIR, "clustering_results.csv")
    NUM_CLUSTERS = 2
    # Load Artifacts
    try:
        scaler = joblib.load(os.path.join(LOG_DIR, "scaler.pkl"))
        use_scaler = True
    except:
        use_scaler = False
    
    decomp = joblib.load(os.path.join(LOG_DIR, f"koopman_eigendecomp_{epoch}.pkl"))
    v_inv = decomp["v_inv"]
    latent_dim = decomp["latent_dim"]
    eigenvalues = decomp["eigenvalues"]

    
    encoder = KoopmanEncoder(5, 32, 128, hidden_depth=5)
    encoder.load_state_dict(torch.load(os.path.join(LOG_DIR, f"encoder_{epoch}.pt"), map_location=device))
    encoder.eval()

    # Prepare Data
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering (Keep consistent with training)
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)

    feat_cols = ['value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    if use_scaler:
        X_scaled = torch.tensor(scaler.transform(df[feat_cols].values), dtype=torch.float32).to(device)
    else:
        X_scaled = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(device)
    # Project to Koopman Eigenspace
    print("Projecting to Eigenspace...")
    with torch.no_grad():
        z_mu = encoder(X_scaled)[:, :latent_dim].cpu().numpy()
    
    alpha_mag = np.abs(z_mu @ v_inv.T)

    # Define Masks
    mags = np.abs(eigenvalues)
    angles = np.abs(np.angle(eigenvalues))
    masks = {
        "cluster_all": (mags > -1e5),
        "cluster_stable": (mags > 0.925) & (mags < 1.01),
        "cluster_decaying": (mags < 0.925),
        "cluster_slow": (mags > 0.925) & (angles < 0.2),
        "cluster_fast": (mags > 0.925) & (angles > 3.0)
    }

    # Run Clustering for each mask and add to DataFrame
    for name, msk in masks.items():
        print(f"Clustering: {name}...")
        X_sub = alpha_mag[:, msk]
        if X_sub.shape[1] == 0: continue
        
        X_sub = StandardScaler().fit_transform(X_sub)
        km = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=42)
        df[name] = km.fit_predict(X_sub)
    print("saving results to CSV...")
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved clustering to {OUTPUT_CSV}")

if __name__ == "__main__":
    export_clusters()