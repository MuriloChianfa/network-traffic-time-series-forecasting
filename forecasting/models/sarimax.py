import datetime
import numpy as np
import pandas as pd
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.api as sm

def adfuller_test(series):
    result = adfuller(series)
    print('p-value: %f' % result[1])
    return result[1]

def acf_and_pacf_test(series):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))

    plot_acf(series, ax=axes[0])
    plot_pacf(series, ax=axes[1])

    return fig

def preflight_analysis(series):
    pvalue = adfuller_test(series)
    fig = acf_and_pacf_test(series)
    return pvalue, fig

def run(series, parameters, model=None):
    p, d, q = parameters['p'], parameters['d'], parameters['q']
    P, D, Q, s = parameters['p'], parameters['d'], parameters['q'], parameters['s']

    if not model:
        model = sm.tsa.statespace.SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    print(results.summary())
    return model, results
    

def forecast_the_future(results, num_of_future_steps=72):
    pred_uc = results.get_forecast(steps=num_of_future_steps)
    pred_ci = pred_uc.conf_int()
    pred_ci = pd.DataFrame(pred_ci)

    forecasted_values = pred_uc.predicted_mean

    return pred_ci, forecasted_values

def plot_forecasted(df, pred_ci, forecasted_values, timestep=60, num_of_future_steps=72):
    last_date = df['date'].values[-1]
    last_date = pd.to_datetime([last_date], format='%Y%m%d%H%M')[0]
    future_dates = [last_date + datetime.timedelta(minutes=timestep * i) for i in range(1, num_of_future_steps + 1)]
    future_dates = pd.to_datetime(future_dates)

    df_future = pd.DataFrame({
        'date': future_dates,
        'predicted': forecasted_values
    })

    df_combined = pd.concat([
        df[['date', 'bps']].rename(columns={'bps': 'actual'}),
        df_future.rename(columns={'predicted': 'actual'})
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(27, 9))
    plt.plot(df_combined['date'], df_combined['actual'], label='Forecasted Data', color='orange')
    plt.fill_between(future_dates, 0, pred_ci.iloc[:, 1], color='k', alpha=0.1)

    formatter = EngFormatter(unit='bps')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    return fig, df_future, ax
