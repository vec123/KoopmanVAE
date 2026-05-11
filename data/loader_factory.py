import pandas as pd
import numpy as np

from data.loaders import TrajectoryLoader, NativeLoader
from data.processors import TimeSeriesProcessor, ScalingProcessor
from data.datasets import ControlledWindowDataset
import pandas as pd
from torch.utils.data import DataLoader, random_split
from typing import List, Tuple, Dict, Any

class DataPipelineFactory:
    """Orchestrates data loading, feature engineering, and splitting."""

    @staticmethod
    def create_pipeline(
        directories: List[str], 
        config: Any, 
        batch_size: int, 
        backend: str = 'torch', 
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        
        # 1. Component Initialization
        time_proc = TimeSeriesProcessor()
        scaler = ScalingProcessor()
        loader = TrajectoryLoader(
            state_cols=config.state_cols,
            control_cols=config.control_cols,
            forcing_cols=config.forcing_cols,
            time_col=getattr(config, 'time_col', 'time'),
            verbose=kwargs.get('verbose', False)
        )

        # 2. Data Ingestion & Time Standardization
        raw_trajs = loader.fetch_data(directories)
        processed_trajs = DataPipelineFactory._apply_time_logic(
            raw_trajs, time_proc, config
        )

        # 3. Dynamic Feature Discovery
        cyclic_names = time_proc.get_feature_names(config.cyclic_features)
        extended_states = config.state_cols + cyclic_names
        scale_cols = extended_states + config.control_cols + config.forcing_cols

        # 4. Data Splitting & Fitting
        train_ds, val_ds = DataPipelineFactory._handle_split_and_scaling(
            processed_trajs, config, scale_cols, scaler, extended_states
        )

        # 5. Loader Generation
        loaders = DataPipelineFactory._build_loaders(
            train_ds, val_ds, batch_size, backend, kwargs
        )

        return loaders[0], loaders[1], scaler

    @staticmethod
    def _apply_time_logic(trajs: List[pd.DataFrame], proc: Any, config: Any) -> List[pd.DataFrame]:
        """Standardizes timestamps and adds cyclic time features."""
        start_date = pd.Timestamp("2020-01-01 00:00:00")
        t_col = getattr(config, 'time_col', 'time')
        
        output = []
        for df in trajs:
            df = df.dropna(subset=[t_col]).copy()
            # Handle numeric vs string time
            if pd.api.types.is_numeric_dtype(df[t_col]):
                df[t_col] = start_date + pd.to_timedelta(df[t_col], unit='s')
            else:
                df[t_col] = pd.to_datetime(df[t_col], format='mixed')
            
            # Add Sin/Cos embeddings for Koopman periodicity
            df = proc.add_cyclic_features(df, t_col, config.cyclic_features)
            output.append(df)
        return output

    @staticmethod
    def _handle_split_and_scaling(trajs, config, scale_cols, scaler, estates):
        if not trajs:
            raise ValueError("No trajectories were loaded. Check your dataset path.")

        mode = getattr(config, 'split_mode', 'trajectory')

        # 1. Identify training dataframes
        if mode == "trajectory":
            # Ensure at least 1 trajectory goes to train if only one exists
            if len(trajs) == 1:
                print("Warning: Only one trajectory found. Using it for both training and validation logic.")
                train_raw, val_raw = trajs, trajs
            else:
                # Use your existing split logic
                train_raw, val_raw = TrajectoryLoader.static_split(trajs, config.val_ratio)
                
            # Final safety check
            if not train_raw:
                train_raw = trajs  # Fallback
        else:
            # Window mode: Fit on a portion of the files, or all files if count is low
            split_idx = max(1, int(len(trajs) * (1 - config.val_ratio)))
            train_raw = trajs[:split_idx]
            val_raw = trajs

        # 2. Extract and concatenate for fitting
        # Now we are certain train_raw is not empty
        df_concat = pd.concat(train_raw)
        
        x_data = df_concat[estates].values
        u_data = df_concat[config.control_cols].values if config.control_cols else None
        f_data = df_concat[config.forcing_cols].values if config.forcing_cols else None

        # 3. Fit the scaler
        scaler.fit(x_data=x_data, u_data=u_data, f_data=f_data)

        # 4. Prepare Dataset objects
        ds_kwargs = {
            "state_cols": estates, 
            "control_cols": config.control_cols,
            "forcing_cols": config.forcing_cols, 
            "subseq_len": config.subseq_len,
            "stride": config.stride, 
            "scaler": scaler
        }

        if mode == "trajectory":
            return (
                ControlledWindowDataset(trajectories=train_raw, **ds_kwargs),
                ControlledWindowDataset(trajectories=val_raw, **ds_kwargs)
            )
        else:
            full_ds = ControlledWindowDataset(trajectories=trajs, **ds_kwargs)
            val_size = int(config.val_ratio * len(full_ds))
            train_size = len(full_ds) - val_size
            return random_split(full_ds, [train_size, val_size])
        
    @staticmethod
    def _build_loaders(train_ds, val_ds, batch_size, backend, kwargs):
        """Factory for different dataloader backends."""
        if backend == 'torch':
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                           num_workers=kwargs.get('num_workers', 1), 
                           pin_memory=kwargs.get('pin_memory', True)),
                DataLoader(val_ds, batch_size=batch_size, 
                           num_workers=kwargs.get('num_workers', 1))
            )
        return (
            NativeLoader(train_ds, batch_size=batch_size, shuffle=True),
            NativeLoader(val_ds, batch_size=batch_size, shuffle=False)
        )