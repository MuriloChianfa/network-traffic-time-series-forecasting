import pandas as pd
from prophet import Prophet
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt

def run(df, timeslot='5min'):
    df_freq_mod = df.copy()
    if timeslot != '5min':
        df_freq_mod['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')
        df_freq_mod.set_index('date')

        df_freq_mod = df_freq_mod.groupby(pd.Grouper(key='date', freq=timeslot))["bps"].mean().reset_index().fillna(0)[:-1]

    df_prophet = df[['date', 'bps']].rename(columns={'date': 'ds', 'bps': 'y'})

    model = Prophet()
    model.fit(df_prophet)
    return model

def forecast_the_future(model, timeslot='5min', num_of_future_steps=288):
    future = model.make_future_dataframe(periods=num_of_future_steps, freq=timeslot)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    formatter = EngFormatter(unit='bps')
    plt.gca().yaxis.set_major_formatter(formatter)
    return fig
