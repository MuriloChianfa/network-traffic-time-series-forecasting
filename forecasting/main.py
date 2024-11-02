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

df_future = df.copy()

if st.button(f"Fit model using {model_choice} and Predict {parameters['future_steps']} steps towards future"):
    if model_choice == 'LSTM':
        model, fig, fig_model = lstm.run(df, bps_scaled, bps_scaler, parameters)
        st.write('## Model evaluation:')
        # st.pyplot(fig_model)
        st.pyplot(fig)
        fig, df_future = lstm.forecast_the_future(df, model, bps_scaled, bps_scaler, parameters['future_steps'], parameters['timeslot'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    elif model_choice == 'Prophet':
        model = prophet.run(df)
        fig, df_future = prophet.forecast_the_future(model, f"{parameters['timeslot']}min", parameters['future_steps'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    elif model_choice == 'SARIMAX':
        model, results = sarimax.run(df['bps'], parameters)
        pred_ci, forecasted_values = sarimax.forecast_the_future(results, parameters['future_steps'])
        fig, df_future = sarimax.plot_forecasted(df, pred_ci, forecasted_values, parameters['timeslot'], parameters['future_steps'])
        st.write('## Forecasting:')
        st.pyplot(fig)
    else:
        print('invalid model')

    col1, col2, col3 = st.columns(3)

    csv = df_future.to_csv().encode('utf-8')
    col1.download_button(label="Download forecasted values as CSV", data=csv, file_name='future_network_traffic.csv', mime='text/csv')

    fig_name = "network_traffic_future_forecasting.png"
    fig.savefig(fig_name, format='png', dpi=300, bbox_inches='tight')
    with open(fig_name, "rb") as img:
        col2.download_button(label="Download forecasted image", data=img, file_name=fig_name, mime="image/png")

    col3.download_button(
        "Download fitted model",
        data=pickle.dumps(model),
        file_name=f"{model_choice}_model.pkl",
    )
