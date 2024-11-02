import pandas as pd
from prophet import Prophet
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt

def run(df):
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
