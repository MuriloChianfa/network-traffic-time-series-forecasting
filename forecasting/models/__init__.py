def hyperparams_by_model(model, ui):
    params = dict()

    if model == 'LSTM':
        start, end = ui.select_slider("Train, Valid, Test split", options=[.1, .2, .3, .4, .5, .6, .7, .8, .9], value=(0.6, 0.8))
        params['train_valid_test_split'] = [start, end]
        params['epochs'] = ui.slider("Epochs", 10, 100, 20, 1)
        params['neurons'] = ui.slider("Neurons", 10, 1000, 46, 1)
    elif model == 'Propeth':
        K = ui.slider('K', 1, 15)
        params['K'] = K
    elif model == 'SARIMAX':
        K = ui.slider('K', 1, 15)
        params['K'] = K
    else:
        params = dict()

    return params