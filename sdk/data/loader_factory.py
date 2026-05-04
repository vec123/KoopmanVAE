import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .processors import TimeSeriesProcessor
from .datasets import KoopmanDataset

class KoopmanDataFactory:
    def __init__(self, config):
        """
        config: A dictionary or object containing:
            - state_cols, forcing_cols, target_col
            - window_size, stride, batch_size, val_split
        """
        self.cfg = config
        self.processor = TimeSeriesProcessor(config.target_col)

    def create_loaders(self, sensor_path, weather_path):
        # 1. Load CSVs
        df_sensor = pd.read_csv(sensor_path)
        df_weather = pd.read_csv(weather_path)

        # 2. Synchronize Time
        df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
        df_weather['date'] = pd.to_datetime(df_weather['date'])

        # Left join sensor data with weather data on time
        df = pd.merge_asof(
            df_sensor.sort_values('timestamp'),
            df_weather.sort_values('date'),
            left_on='timestamp',
            right_on='date',
            direction='nearest'
        )

        # 3. Feature Engineering
        df = self.processor.add_cyclic_date_features(df, 'timestamp')
        
        # Handle resolution gaps
        df = df.ffill().bfill()

        # 4. Extraction & Scaling
        # We scale the state and forcing, but usually keep control 'u' as is 
        # or scale it separately if it's continuous.
        x_raw = df[self.cfg.state_cols].values
        f_raw = df[self.cfg.forcing_cols].values
        
        # If 'u' is not in CSV, we assume it's a zero-vector or a specific column
        u_raw = df[['u']].values if 'u' in df.columns else np.zeros((len(df), 1))

        # Fit and transform
        x_scaled = self.processor.scaler.fit_transform(x_raw)
        f_scaled = self.processor.f_scaler.fit_transform(f_raw) # Optional scaling for forcing

        # 5. Dataset Creation
        full_dataset = KoopmanDataset(
            x_data=x_scaled,
            u_data=u_raw,
            f_data=f_scaled,
            window_size=self.cfg.window_size,
            stride=self.cfg.stride
        )

        # 6. Train/Val Split
        n_val = int(len(full_dataset) * self.cfg.val_split)
        indices = torch.randperm(len(full_dataset)).tolist()
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_loader = DataLoader(
            Subset(full_dataset, train_idx), 
            batch_size=self.cfg.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx), 
            batch_size=self.cfg.batch_size, 
            shuffle=False
        )

        return train_loader, val_loader