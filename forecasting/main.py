import streamlit as st

import preprocess
import models
import models.lstm as lstm
import models.prophet as prophet
import models.sarimax as sarimax

st.spinner()
st.set_page_config(layout="wide", page_title=r'Network Timeseries Forecasting', initial_sidebar_state= 'expanded')
st.title('Pattern Recognition - Final class project')
st.write(f"## Author: Murilo Chianfa | UEL")

sb = st.sidebar
sb.header("Model Parameters")

dataset_name = sb.selectbox('Select Dataset', ('HW-Link-Level3', 'MK-VLAN33'))
st.write(f"### Using \"{dataset_name}\" Dataset")

model_choice = sb.selectbox('Select forecasting model', ('LSTM', 'Prophet', 'SARIMAX'))
@st.cache_data
def get_df(name):
    return preprocess.get_dataset(name)
df = get_df(dataset_name)

parameters = models.hyperparams_by_model(model_choice, sb)

@st.cache_data
def get_fig(df, format='bar'):
    return preprocess.plot_timeseries_network_traffic(df['date'], df['bps'], 5, 'Incoming traffic', format=format)

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

if st.button(f"Fit model using {model_choice} and Predict {parameters['future_steps']} steps towards future"):
    if model_choice == 'LSTM':
        model, fig, fig_model = lstm.run(df, bps_scaled, bps_scaler, parameters)
        st.write('## Model evaluation:')
        st.pyplot(fig_model)
        st.pyplot(fig)
        fig = lstm.forecast_the_future(df, model, bps_scaled, bps_scaler, parameters['future_steps'], parameters['timeslot'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    elif model_choice == 'Prophet':
        model = prophet.run(df)
        fig = prophet.forecast_the_future(model, f"{parameters['timeslot']}min", parameters['future_steps'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    elif model_choice == 'SARIMAX':
        model, results = sarimax.run(df['bps'], parameters)
        pred_ci, forecasted_values = sarimax.forecast_the_future(results, parameters['future_steps'])
        fig = sarimax.plot_forecasted(df, pred_ci, forecasted_values, parameters['timeslot'], parameters['future_steps'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    else:
        print('invalid model')
