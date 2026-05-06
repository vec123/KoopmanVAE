import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def examine_clusters(csv_path, start_date, end_date, selected_masks=None):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Load Data
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Filter by Time Range
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    df_plot = df.loc[mask].copy()

    if df_plot.empty:
        print(f"No data found between {start_date} and {end_date}")
        return

    # 3. Identify Cluster Columns
    # If no masks specified, find all columns starting with 'cluster_'
    if selected_masks is None:
        cluster_cols = [c for c in df.columns if c.startswith('cluster_')]
    else:
        cluster_cols = [f"cluster_{m}" if not m.startswith('cluster_') else m for m in selected_masks]

    # 4. Plotting
    n_plots = len(cluster_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), sharex=True)
    
    # Handle single plot case
    if n_plots == 1: axes = [axes]

    for i, col in enumerate(cluster_cols):
        ax = axes[i]
        
        # Plot the underlying energy signal in light gray
        ax.plot(df_plot['timestamp'], df_plot['value'], color='lightgray', alpha=0.4, zorder=1)
        
        # Overlay the clusters
        scatter = ax.scatter(
            df_plot['timestamp'], 
            df_plot['value'], 
            c=df_plot[col], 
            cmap='tab10',   # Distinct colors for categorical data
            s=20, 
            alpha=0.8,
            zorder=2
        )
        
        # Formatting
        clean_title = col.replace('cluster_', '').replace('_', ' ').title()
        ax.set_title(f"Spectral Mask: {clean_title}", fontsize=14, loc='left')
        ax.set_ylabel("Energy Value")
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar for cluster IDs
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label('Cluster ID')

    # X-axis date formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    plt.suptitle(f"Cluster Analysis: {start_date} to {end_date}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save option
    save_path = csv_path.replace('.csv', '_plot.png')
    plt.savefig(save_path, dpi=200)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    SENSOR_NAME ="ES0031405047432001AZ0F"
    LOG_NAME = f"Multi_KVAE_energy_{SENSOR_NAME}"
    LOG_DIR = f"logs/{LOG_NAME}"
    FILE_PATH = os.path.join(LOG_DIR, "clustering_results.csv")
    
    # Define your zoom window here
    START = "2019-05-15"
    END = "2030-06-14"
    
    # Optional: specify only certain masks to keep the plot clean
    # e.g., ["cluster_slow", "cluster_fast"]
    MY_MASKS = None 

    examine_clusters(FILE_PATH, START, END, selected_masks=MY_MASKS)