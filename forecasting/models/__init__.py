def hyperparams_by_model(model, ui):
    params = dict()

    if model == 'LSTM':
        start, end = ui.select_slider("Train, Valid, Test split", options=[.1, .2, .3, .4, .5, .6, .7, .8, .9], value=(0.6, 0.8))
        params['train_valid_test_split'] = [start, end]
        params['epochs'] = ui.slider("Epochs", 10, 64, 14, 1)
        params['neurons'] = ui.slider("Neurons", 10, 256, 62, 1)
        params['layers'] = ui.slider("Hidden Layers", 1, 32, 4, 1)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 320, 64, 1)
    elif model == 'Prophet':
        params['timeslot'] = ui.slider("Timeslot", 5, 60, 5, 5)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 864, 288, 1)
    elif model == 'SARIMAX':
        K = ui.slider('K', 1, 15)
        params['K'] = K
    else:
        params = dict()

    return params
