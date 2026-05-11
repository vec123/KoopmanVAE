import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class TimeSeriesProcessor:
    """Modular engine for temporal feature engineering."""
    CYCLE_DEFS = {
        'second': (lambda t: t.dt.second, 60),
        'minute': (lambda t: t.dt.minute, 60),
        'hour': (lambda t: t.dt.hour, 24),
        'day_of_week': (lambda t: t.dt.dayofweek, 7),
        'day_of_month': (lambda t: t.dt.day - 1, 31),
        'month': (lambda t: t.dt.month - 1, 12),
        'year': (lambda t: t.dt.dayofyear - 1, 366)
    }

    def add_cyclic_features(self, df, time_col, enabled_features):
        df = df.copy()
        t = pd.to_datetime(df[time_col])
        for feature in enabled_features:
            if feature in self.CYCLE_DEFS:
                getter, period = self.CYCLE_DEFS[feature]
                values = getter(t)
                df[f'{feature}_sin'] = np.sin(2 * np.pi * values / period)
                df[f'{feature}_cos'] = np.cos(2 * np.pi * values / period)
        return df

    def get_feature_names(self, enabled_features):
        """Returns a list of all sin/cos column names generated."""
        names = []
        for f in enabled_features:
            if f in self.CYCLE_DEFS:
                names.extend([f"{f}_sin", f"{f}_cos"])
        return names

class ScalingProcessor:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.x_scaler = StandardScaler()
        self.u_scaler = StandardScaler()
        self.f_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, x_data, u_data=None, f_data=None):
        # Always fit X
        self.x_scaler.fit(x_data)
        
        # Check if u_data is a valid array with features
        if u_data is not None and len(u_data.shape) > 1 and u_data.shape[1] > 0:
            self.u_scaler.fit(u_data)
            
        # Check if f_data is a valid array with features
        if f_data is not None and len(f_data.shape) > 1 and f_data.shape[1] > 0:
            self.f_scaler.fit(f_data)
            
        self.is_fitted = True

    def transform(self, x, u=None, f=None):
        if not self.is_fitted:
            return x, u, f
        
        x_scaled = self.x_scaler.transform(x)
        
        # Check shapes before transforming to avoid sklearn errors on empty features
        u_scaled = self.u_scaler.transform(u) if (u is not None and u.shape[-1] > 0) else u
        f_scaled = self.f_scaler.transform(f) if (f is not None and f.shape[-1] > 0) else f
        
        return x_scaled, u_scaled, f_scaled