import pandas as pd
import numpy as np

from data.loaders import TrajectoryLoader, NativeLoader
from data.processors import TimeSeriesProcessor, ScalingProcessor
from data.datasets import ControlledWindowDataset

class DataPipelineFactory:
    @staticmethod
    def create_pipeline(directories, config, batch_size, backend='torch', num_workers = 1, pin_memory = True,  verbose=False):
        """
        Args:
            directories: List of paths to CSVs
            config: The DataConfig dataclass instance (cfg.data)
            batch_size: Passed from the root config
            backend: 'torch' or 'native'
        """
        # 1. Fetching logic - Using Dataclass attributes
        t_col = getattr(config, 'time_col', 'time') 
        s_cols = config.state_cols
        c_cols = config.control_cols
        f_cols = config.forcing_cols

        # 2. FETCH DATA
        loader = TrajectoryLoader(
            state_cols=s_cols, 
            control_cols=c_cols,
            forcing_cols=f_cols,
            time_col=t_col,
            verbose=verbose
        )
        raw_trajs = loader.fetch_data(directories)

        # 3. Handle Time
        start_date = pd.Timestamp("2024-01-01 00:00:00")
        time_processor = TimeSeriesProcessor()

        for df in raw_trajs:
            if verbose:
                print("raw df: ", df)
            # Check if the column is already strings/datetimes or numeric
            if pd.api.types.is_numeric_dtype(df[t_col]):
                # Handle numeric offsets (0.0, 60.0...)
                df[t_col] = start_date + pd.to_timedelta(df[t_col], unit='s')
            else:
                # Handle string timestamps ("2024-05-01...")
                df[t_col] = pd.to_datetime(df[t_col])
            if verbose:
                print("dataset with datetime: ", df)

        processed_trajs = [
            time_processor.add_cyclic_features(df, t_col, config.cyclic_features) 
            for df in raw_trajs
        ]
        
        # 4. Feature Aggregation
        cyclic_names = time_processor.get_feature_names(config.cyclic_features)
        extended_states = s_cols + cyclic_names


        scaler = ScalingProcessor()
        # --- 5. Prepare Dataset Params ---
        ds_kwargs = {
            "state_cols": extended_states,
            "control_cols": c_cols,
            "forcing_cols": f_cols,
            "subseq_len": config.subseq_len,
            "stride": config.stride,
            "scaler": scaler
        }

        # --- 6. Handle Split Logic ---
        split_mode = getattr(config, 'split_mode', 'trajectory')

        if split_mode == "trajectory":
            # OPTION A: Split by file (The original way)
            train_dfs, val_dfs = loader.split_trajectories(processed_trajs, config.val_ratio)
            
            # Catch the "Zero Samples" error before it crashes
            if not train_dfs:
                raise ValueError("Train split is empty! If you only have one file, set split_mode: 'window'")
                
            train_ds = ControlledWindowDataset(trajectories=train_dfs, **ds_kwargs)
            val_ds   = ControlledWindowDataset(trajectories=val_dfs, **ds_kwargs)

        elif split_mode == "window":
            # OPTION B: Split by sequences/windows (The new way)
            # 1. Create one big dataset from all files
            full_dataset = ControlledWindowDataset(trajectories=processed_trajs, **ds_kwargs)
            
            # 2. Calculate split sizes
            dataset_size = len(full_dataset)
            val_size = int(config.val_ratio * dataset_size)
            train_size = dataset_size - val_size
            
            # 3. Randomly split the windows
            from torch.utils.data import random_split
            train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
            print(f"Split: Window mode - Train: {train_size}, Val: {val_size}")

        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        # --- 7. Final Dataloader Creation ---
        if backend == 'torch':
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory = pin_memory,  num_workers =num_workers)
            val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory = pin_memory, num_workers =num_workers)
        else:
            train_loader = NativeLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = NativeLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, scaler