import streamlit as st

import preprocess
import models
from models.lstm import run as lstm

st.spinner()
st.set_page_config(layout="wide", page_title=r'Network Timeseries Forecasting', initial_sidebar_state= 'expanded')
st.title('Pattern Recognition - Final class project')
st.write(f"## Author: Murilo Chianfa | UEL")

sb = st.sidebar
sb.header("Parameters")

dataset_name = sb.selectbox('Select Dataset', ('HW-Link-Level3', 'MK-VLAN33'))
st.write(f"### Using \"{dataset_name}\" Dataset")

model_choice = sb.selectbox('Select forecasting model', ('LSTM', 'Propeth', 'SARIMAX'))
@st.cache_data
def get_df(name):
    return preprocess.get_dataset(name)
df = get_df(dataset_name)
st.write('Shape of dataset:', df.shape)

fig = preprocess.plot_timeseries(df['date'], df['bps'], 5, 'Incoming traffic')
st.pyplot(fig)

parameters = models.hyperparams_by_model(model_choice, sb)

if sb.button("Fit and Evaluate Model"):
    if model_choice == 'LSTM':
        fig = lstm(df, parameters)
        st.write('## Model evaluation:')
        st.pyplot(fig)
    else:
        print('invalid model')
