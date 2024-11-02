import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt

def run(df, parameters):
    df_prophet = df[['date', 'bps']].rename(columns={'date': 'ds', 'bps': 'y'})

    model = Prophet(changepoint_prior_scale=parameters['changepoint_prior_scale'], seasonality_prior_scale=parameters['seasonality_prior_scale'])
    model.fit(df_prophet)
    return model

def evaluate(model):
    df_cv = cross_validation(model, horizon='1 day')
    df_metrics = performance_metrics(df_cv)

    mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
    mse = mean_squared_error(df_cv['y'], df_cv['yhat'])
    rmse = np.sqrt(mse)

    return f"Mean Absolute Error: {mae:.2f}<br>Mean Squared Error: {mse:.2f}<br>Root Mean Squared Error: {rmse:.2f}"

def forecast_the_future(model, timeslot='5min', num_of_future_steps=288):
    future = model.make_future_dataframe(periods=num_of_future_steps, freq=timeslot)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    formatter = EngFormatter(unit='bps')
    plt.gca().yaxis.set_major_formatter(formatter)
    return fig, forecast
