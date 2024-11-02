import streamlit as st

import preprocess
import models
from models.lstm import run as lstm, forecast_the_future

st.spinner()
st.set_page_config(layout="wide", page_title=r'Network Timeseries Forecasting', initial_sidebar_state= 'expanded')
st.title('Pattern Recognition - Final class project')
st.write(f"## Author: Murilo Chianfa | UEL")

sb = st.sidebar
sb.header("Model Parameters")

dataset_name = sb.selectbox('Select Dataset', ('HW-Link-Level3', 'MK-VLAN33'))
st.write(f"### Using \"{dataset_name}\" Dataset")

model_choice = sb.selectbox('Select forecasting model', ('LSTM', 'Propeth', 'SARIMAX'))
@st.cache_data
def get_df(name):
    return preprocess.get_dataset(name)
df = get_df(dataset_name)
st.write('Shape of dataset:', df.shape)

@st.cache_data
def get_fig(df):
    return preprocess.plot_timeseries_network_traffic(df['date'], df['bps'], 5, 'Incoming traffic')
st.image(get_fig(df))

parameters = models.hyperparams_by_model(model_choice, sb)

if st.button(f"Fit model using {model_choice} and Predict {parameters['future_steps']} steps towards future"):
    if model_choice == 'LSTM':
        model, bps_scaled, bps_scaler, fig = lstm(df, parameters)
        st.write('## Model evaluation:')
        st.pyplot(fig)
        fig = forecast_the_future(df, model, bps_scaled, bps_scaler, parameters['future_steps'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    else:
        print('invalid model')
