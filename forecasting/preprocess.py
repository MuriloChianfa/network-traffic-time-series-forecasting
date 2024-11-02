import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def setup(df, timeslot='5min', diff=0):
    df_freq_mod = df.copy()
    if timeslot != '5min':
        df_freq_mod['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')
        df_freq_mod.set_index('date')

        df_freq_mod = df_freq_mod.groupby(pd.Grouper(key='date', freq=timeslot))["bps"].mean().reset_index().fillna(0)[:-1]

    X_diff = df_freq_mod['bps']
    if diff != 0:
        for _ in range(0, diff):
            X_diff = np.diff(X_diff, axis=0)
        X_diff = pd.DataFrame(X_diff)

    bps_scaled, bps_scaler = trace_scaler(X_diff)
    return df_freq_mod, bps_scaled, bps_scaler

def trace_scaler(trace):
    bps_values = trace.values.reshape(-1, 1)
    bps_scaler = MinMaxScaler()
    bps_scaled = bps_scaler.fit_transform(bps_values)
    return bps_scaled, bps_scaler
