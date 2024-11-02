import io
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from sklearn.preprocessing import MinMaxScaler

formatter = EngFormatter(unit='bps')

def get_dataset(name):
    df = None

    if name == 'MK-VLAN33':
        df = pd.read_csv('datasets/rt-mk-vlan33-bps-from-19-to-25-october.csv')
    else:
        df = pd.read_csv('datasets/rt-hw-ne8k-link-level3-bps-inbound-from-20-to-26-of-october-2024.csv')

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

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

def plot_timeseries_network_traffic(date, trace, timestep, traceLabel, title=None, format='bar'):
    buf = io.BytesIO()
    fig = plt.figure(figsize=(27, 9))
    if title:
        plt.title(title, fontdict={'fontsize': 36})
    if format == 'bar':
        plt.bar(date, trace, width=datetime.timedelta(minutes=timestep), label=traceLabel)
    else:
        plt.plot(date, trace, label=traceLabel)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.savefig(buf, format="png")
    return buf
