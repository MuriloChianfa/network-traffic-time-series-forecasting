import pickle
import streamlit as st

import models
import dataset
import preprocess

import models.lstm as lstm
import models.prophet as prophet
import models.sarimax as sarimax

st.spinner()
st.set_page_config(layout="wide", page_title=r'Network Traffic Timeseries Forecasting', initial_sidebar_state= 'expanded')
st.title('Pattern Recognition - Final class project')
st.write(f"### Author: Murilo Chianfa | UEL")

sb = st.sidebar
sb.header("Model Parameters")

dataset_name = sb.selectbox('Select Dataset', ('HW-Link-Level3', 'MK-VLAN33'))
st.write(f"#### Using \"{dataset_name}\" Dataset")

model_choice = sb.selectbox('Select forecasting model', ('SARIMAX', 'Prophet', 'LSTM'))
@st.cache_data
def get_df(name):
    return dataset.get_dataset(name)
df = get_df(dataset_name)

parameters = models.hyperparams_by_model(model_choice, sb)

@st.cache_data
def get_fig(df, format='bar'):
    return dataset.plot_timeseries_network_traffic(df['date'], df['bps'], 5, 'Incoming traffic', format=format)

if model_choice == 'SARIMAX':
    df, bps_scaled, bps_scaler = preprocess.setup(df, f"{parameters['timeslot']}min", parameters['d'])
    st.write('Shape of dataset:', df.shape)
    st.image(get_fig(df, format='plot'))
    pvalue, fig = sarimax.preflight_analysis(bps_scaled)
    formatted_string = "{:.6f}".format(pvalue)
    pvalue = float(formatted_string)
    st.write('## ACF and PACF test:')
    st.write(f"> Augmented Dickey-Fuller p-value: {pvalue}")
    st.pyplot(fig)
else:
    df, bps_scaled, bps_scaler = preprocess.setup(df, f"{parameters['timeslot']}min")
    st.write('Shape of dataset:', df.shape)
    st.image(get_fig(df))

def trigger_sarimax(df, results, parameters):
    st.write('## Model summary:')
    st.write(results.summary())
    pred_ci, forecasted_values = sarimax.forecast_the_future(results, parameters['future_steps'])
    fig, df_future = sarimax.plot_forecasted(df, pred_ci, forecasted_values, parameters['timeslot'], parameters['future_steps'])
    st.write('## Forecasting:')
    st.pyplot(fig)
    return df_future, fig

def trigger_prophet(df, model, parameters):
    st.write(prophet.evaluate(model), unsafe_allow_html=True)
    fig, df_future = prophet.forecast_the_future(model, f"{parameters['timeslot']}min", parameters['future_steps'])
    st.write('## Forecasting:')
    st.pyplot(fig)
    return df_future, fig

def trigger_lstm(df, model, bps_scaled, bps_scaler, parameters):
    fig, df_future = lstm.forecast_the_future(df, model, bps_scaled, bps_scaler, parameters['future_steps'], parameters['timeslot'])
    st.write('## Forecasting:')
    st.pyplot(fig)
    return df_future, fig

df_future = df.copy()
option1, _, option2, option3 = st.columns([.4, .05, .1, .45])

option1.write('')
option1.write('')
if option1.button(f"Fit a new model using {model_choice} and predict {parameters['future_steps']} steps towards future", icon="🔥", use_container_width=True, ):
    if model_choice == 'LSTM':
        model, fig, fig_model = lstm.run(df, bps_scaled, bps_scaler, parameters)
        st.write('## Model evaluation:')
        st.pyplot(fig)
        df_future, fig = trigger_lstm(df, model, bps_scaled, bps_scaler, parameters)
    elif model_choice == 'Prophet':
        model = prophet.run(df, parameters)
        df_future, fig = trigger_prophet(df, model, parameters)
    elif model_choice == 'SARIMAX':
        model, results = sarimax.run(df['bps'], parameters)
        df_future, fig = trigger_sarimax(df, results, parameters)
        model = results
    else:
        st.warning('Invalid model choosed')

    models.download_options(st, model, df_future, fig)

option2.write('## OR')

uploaded_file = option3.file_uploader("Upload your existing model to save time")

if uploaded_file is not None:
    model = pickle.loads(uploaded_file.read())

    if model_choice == 'SARIMAX':
        df_future, fig = trigger_sarimax(df, model, parameters)
    elif model_choice == 'Prophet':
        df_future, fig = trigger_prophet(df, model, parameters)
    elif model_choice == 'LSTM':
        df_future, fig = trigger_lstm(df, model, bps_scaled, bps_scaler, parameters)
    else:
        st.warning('Invalid model uploaded')
        st.stop()

    models.download_options(st, model, df_future, fig)
