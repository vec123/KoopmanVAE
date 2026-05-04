import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesProcessor:
    """
    Modular engine for temporal feature engineering. 
    Maps time components to sin/cos transformations.
    """
    CYCLE_DEFS = {
        'second':       (lambda t: t.dt.second, 60),
        'minute':       (lambda t: t.dt.minute, 60),
        'hour':         (lambda t: t.dt.hour, 24),
        'day_of_week':  (lambda t: t.dt.dayofweek, 7),
        'day_of_month': (lambda t: t.dt.day - 1, 31),
        'month':        (lambda t: t.dt.month - 1, 12),
        'season':       (lambda t: (t.dt.month % 12 // 3), 4),
        'year':         (lambda t: t.dt.dayofyear - 1, 366)
    }

    def __init__(self, target_col):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.f_scaler = StandardScaler()

    def add_cyclic_features(self, df, time_col, enabled_features):
        """
        Calculates sin/cos pairs for all enabled time cycles.
        """
        df = df.copy()
        # Ensure timestamp is datetime objects
        t = pd.to_datetime(df[time_col])

        for feature in enabled_features:
            if feature not in self.CYCLE_DEFS:
                print(f"Warning: Temporal feature '{feature}' not recognized.")
                continue
                
            getter, period = self.CYCLE_DEFS[feature]
            values = getter(t)
            
            # Use standard naming convention for predictable column access
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