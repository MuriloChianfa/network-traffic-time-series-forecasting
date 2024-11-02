import pickle
from streamlit_vertical_slider import vertical_slider

def hyperparams_by_model(model, ui):
    params = dict()

    if model == 'LSTM':
        params['timeslot'] = ui.slider("Choose the timeslot for dataset", 5, 60, 5, 5, help='The timeslot is defined in minutes (m)')
        start, end = ui.select_slider("Train, Valid, Test split", options=[.1, .2, .3, .4, .5, .6, .7, .8, .9], value=(0.6, 0.8))
        params['train_valid_test_split'] = [start, end]
        col1, col2, col3 = ui.columns(3)
        with col1:
            params['epochs'] = vertical_slider("Epochs", min_value=10, max_value=64, default_value=14, step=1, value_always_visible=True)
        with col2:
            params['neurons'] = vertical_slider("Neurons", min_value=10, max_value=256, default_value=62, step=1, value_always_visible=True)
        with col3:
            params['layers'] = vertical_slider("Hidden Layers", min_value=1, max_value=32, default_value=4, step=1, value_always_visible=True)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 320, 64, 1)
        return params

    if model == 'Prophet':
        params['timeslot'] = ui.slider("Choose the timeslot for dataset", 5, 60, 5, 5, help='The timeslot is defined in minutes (m)')
        params['changepoint_prior_scale'] = ui.slider("Changepoint priority scale", 0.01, 0.1, 0.05, 0.01)
        params['seasonality_prior_scale'] = ui.slider("Seasonality priority scale", 1, 30, 10, 1)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 864, 288, 1)
        return params

    if model == 'SARIMAX':
        params['timeslot'] = ui.slider("Choose the timeslot for dataset", 60, 240, 60, 5, help='The timeslot is defined in minutes (m)')
        ui.header("Order Parameters")
        col1, col2, col3, col4 = ui.columns(4)
        with col1:
            params['p'] = vertical_slider("P", min_value=0, max_value=48, default_value=3, step=1, value_always_visible=True)
        with col2:
            params['d'] = vertical_slider("D", min_value=0, max_value=10, default_value=0, step=1, value_always_visible=True)
        with col3:
            params['q'] = vertical_slider("Q", min_value=0, max_value=48, default_value=1, step=1, value_always_visible=True)
        with col4:
            params['s'] = vertical_slider("S", min_value=1, max_value=360, default_value=24, step=1, value_always_visible=True)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 144, 48, 1)
        return params

    return params

def download_options(ui, model, df_future, fig):
    col1, col2, col3, _ = ui.columns([.2, 0.17, 0.17, .46])

    csv = df_future.to_csv().encode('utf-8')
    col1.download_button(label="Download forecasted values as CSV", data=csv, file_name='future_network_traffic.csv', mime='text/csv')

    fig_name = "network_traffic_future_forecasting.png"
    fig.savefig(fig_name, format='png', dpi=300, bbox_inches='tight')
    with open(fig_name, "rb") as img:
        col2.download_button(label="Download forecasted image", data=img, file_name=fig_name, mime="image/png")

    col3.download_button(
        "Download fitted model",
        data=pickle.dumps(model),
        file_name="model.pkl",
    )
