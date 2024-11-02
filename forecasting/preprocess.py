import datetime
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from sklearn.preprocessing import MinMaxScaler

def get_dataset(name):
    df = None

    if name == 'MK-VLAN33':
        df = pd.read_csv('../datasets/rt-mk-vlan33-bps-from-19-to-25-october.csv')
    else:
        df = pd.read_csv('../datasets/rt-hw-ne8k-link-level3-bps-inbound-from-20-to-26-of-october-2024.csv')

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def trace_scaler(trace):
    bps_values = trace.values.reshape(-1, 1)
    bps_scaler = MinMaxScaler()
    bps_scaled = bps_scaler.fit_transform(bps_values)
    return bps_scaled, bps_scaler

def plot_timeseries(date, trace, timestep, traceLabel, title=None):
    fig = plt.figure(figsize=(27, 9))
    if title:
        plt.title(title, fontdict={'fontsize': 36})
    plt.bar(date, trace, width=datetime.timedelta(minutes=timestep), label=traceLabel)
    formatter = EngFormatter(unit='bps')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    return fig
