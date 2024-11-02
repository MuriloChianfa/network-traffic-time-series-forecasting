def hyperparams_by_model(model, ui):
    params = dict()

    if model == 'LSTM':
        params['timeslot'] = ui.slider("Timeslot (in minutes)", 5, 60, 5, 5)
        start, end = ui.select_slider("Train, Valid, Test split", options=[.1, .2, .3, .4, .5, .6, .7, .8, .9], value=(0.6, 0.8))
        params['train_valid_test_split'] = [start, end]
        params['epochs'] = ui.slider("Epochs", 10, 64, 14, 1)
        params['neurons'] = ui.slider("Neurons", 10, 256, 62, 1)
        params['layers'] = ui.slider("Hidden Layers", 1, 32, 4, 1)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 320, 64, 1)
        return params

    if model == 'Prophet':
        params['timeslot'] = ui.slider("Timeslot (in minutes)", 5, 60, 5, 5)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 864, 288, 1)
        return params

    if model == 'SARIMAX':
        params['timeslot'] = ui.slider("Timeslot (in minutes)", 60, 240, 60, 5)
        ui.header("Order Parameters")
        params['p'] = ui.slider("P", 0, 48, 3, 1)
        params['d'] = ui.slider("d", 0, 10, 0, 1)
        params['q'] = ui.slider("q", 0, 48, 1, 1)
        params['s'] = ui.slider("s", 1, 360, 24, 1)
        ui.header("Forecasting Parameters")
        params['future_steps'] = ui.slider("Steps towards future", 1, 144, 48, 1)
        return params

    return params
