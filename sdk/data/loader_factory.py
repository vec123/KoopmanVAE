from torch.utils.data import DataLoader
from loaders import TrajectoryLoader, NativeLoader
from processors import TimeSeriesProcessor, ScalingProcessor
from datasets import ControlledWindowDataset
import pandas as pd
import numpy as np

class DataPipelineFactory:
    @staticmethod
    def create_pipeline(directories, config):
        # 1. Fetching logic - uses TrajectoryLoader
        t_col = config.get('time_col', 'time')  # Define it here!
        s_cols = config['state_cols']
        c_cols = config.get('control_cols', [])
        f_cols = config.get('forcing_cols', [])

        # 2. FETCH DATA
        loader = TrajectoryLoader(
            state_cols=s_cols, 
            control_cols=c_cols,
            forcing_cols=f_cols,
            time_col=t_col
        )
        raw_trajs = loader.fetch_data(directories)

        # 3. Handle Time
        start_date = pd.Timestamp("2024-01-01 00:00:00")
        time_processor = TimeSeriesProcessor()
        for df in raw_trajs:
            df[t_col] = start_date + pd.to_timedelta(df[t_col], unit='s')
        
        processed_trajs = [
            time_processor.add_cyclic_features(df, t_col, config['cyclic_features']) 
            for df in raw_trajs
        ]
        
        # 4. Feature Aggregation
        cyclic_names = time_processor.get_feature_names(config['cyclic_features'])
        extended_states = s_cols + cyclic_names
        # Combine Control and Forcing into a single "Action/External" vector for the model
        external_inputs = c_cols + f_cols 

        # 5. Split and Scale
        train_dfs, val_dfs = loader.split_trajectories(processed_trajs, config.get('val_ratio', 0.2))
        
        # Scaling (Refined for three-way split) ---
        scaler = ScalingProcessor()
        if config.get('scale', True):
            # Extract lists safely
            c_cols = config.get('control_cols', [])
            f_cols = config.get('forcing_cols', [])
            
            all_x = np.concatenate([df[extended_states].values for df in train_dfs])
            
            # Extract values only if columns exist, else pass None/Empty to scaler
            all_u = np.concatenate([df[c_cols].values for df in train_dfs]) if c_cols else None
            all_f = np.concatenate([df[f_cols].values for df in train_dfs]) if f_cols else None
            
            # Your ScalingProcessor.fit should now take (x, u, f)
            scaler.fit(all_x, all_u, all_f)

        # --- 6. Dataset creation (Symmetrical Fix) ---
        # Define common parameters to avoid repetition errors
        ds_kwargs = {
            "state_cols": extended_states,
            "control_cols": config.get('control_cols', []),
            "forcing_cols": config.get('forcing_cols', []),
            "subseq_len": config['subseq_len'],
            "stride": config['stride'],
            "scaler": scaler
        }

        train_ds = ControlledWindowDataset(trajectories=train_dfs, **ds_kwargs)
        val_ds   = ControlledWindowDataset(trajectories=val_dfs, **ds_kwargs)

        # 2. Batching logic - chooses between Torch and Native
        if config.get('backend') == 'torch':
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
        else:
            # Swap to NativeLoader for Numpy/Jax
            train_loader = NativeLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
            val_loader = NativeLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

        return train_loader, val_loader, scaler