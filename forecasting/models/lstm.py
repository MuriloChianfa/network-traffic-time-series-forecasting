import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional

from dataset import prepare_df_split, setup_train_valid_test_split
from preprocess import trace_scaler
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt
import numpy as np
import datetime

def simpleLSTM(X_train, y_train, X_valid, y_valid, epochs, neurons, num_layers=2, num_of_future_forecasts=1):
    if X_train.ndim == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    if X_valid.ndim == 2:
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = Sequential()
    model.add(Bidirectional(LSTM(neurons, use_cudnn=False, return_sequences=(num_layers > 1), input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.1))

    for _ in range(1, num_layers):
        model.add(Bidirectional(LSTM(neurons, return_sequences=(num_layers > 1))))
        model.add(Dropout(0.1))

    model.add(Dense(num_of_future_forecasts))
    model.compile(optimizer='adam', loss='mse')

    print("Training LSTM Model...")
    history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))

    return model, history

def run(df, bps_scaled, bps_scaler, parameters):
    X, y = prepare_df_split(bps_scaled)
    train_len = parameters['train_valid_test_split'][0]
    valid_len = parameters['train_valid_test_split'][1] - train_len
    X_train, y_train, X_valid, y_valid, X_test, y_test, dates_train, dates_valid, dates_test = setup_train_valid_test_split(df['date'], X, y, train_len=train_len, valid_len=valid_len)

    model, history = simpleLSTM(X_train, y_train, X_valid, y_valid, parameters['epochs'], parameters['neurons'], parameters['layers'], num_of_future_forecasts=1)

    fig_model = plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.show()

    train_yhat = model.predict(X_train)
    valid_yhat = model.predict(X_valid)
    test_yhat = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, r2_score
    import pandas as pd

    actual_train = bps_scaler.inverse_transform(y_train)
    actual_valid = bps_scaler.inverse_transform(y_valid)
    actual_test = bps_scaler.inverse_transform(y_test)

    predicted_train = bps_scaler.inverse_transform(train_yhat.reshape((train_yhat.shape[0], -1)) if train_yhat.ndim == 3 else train_yhat)
    predicted_valid = bps_scaler.inverse_transform(valid_yhat.reshape((valid_yhat.shape[0], -1)) if valid_yhat.ndim == 3 else valid_yhat)
    predicted_test = bps_scaler.inverse_transform(test_yhat.reshape((test_yhat.shape[0], -1)) if test_yhat.ndim == 3 else test_yhat)

    def create_dataframe(dates, actual, predicted, label):
        actual_flat = actual.flatten()
        predicted_flat = predicted.flatten()

        min_length = min(len(dates), len(actual_flat), len(predicted_flat))

        return pd.DataFrame({
            'date': dates[:min_length],
            'actual': actual_flat[:min_length],
            'predicted': predicted_flat[:min_length],
            'set': label
        })

    df_train = create_dataframe(dates_train, actual_train, predicted_train, 'train')
    df_valid = create_dataframe(dates_valid, actual_valid, predicted_valid, 'validation')
    df_test = create_dataframe(dates_test, actual_test, predicted_test, 'test')

    df_all = pd.concat([df_train, df_valid, df_test], axis=0)
    dates_all = pd.to_datetime(df_all['date'])

    from datetime import timedelta
    base_time = pd.to_datetime([df['date'][0]], format='%Y%m%d%H%M')[0]

    len_dates_train = len(dates_train)
    train_line = [base_time + timedelta(minutes=parameters['timeslot'] * i) for i in range(len_dates_train)][-1]

    len_dates_valid = len(dates_valid)
    valid_line = [base_time + timedelta(minutes=parameters['timeslot'] * i) for i in range(len_dates_train + len_dates_valid)][-1]

    len_dates_test = len(dates_test)
    test_line = [base_time + timedelta(minutes=parameters['timeslot'] * i) for i in range(len_dates_train + len_dates_valid + len_dates_test)][-1]

    len_all_dates = len(dates_all)

    vertical_lines_positions = pd.to_datetime([
        train_line,
        valid_line
    ], format='%Y%m%d%H%M')

    print(vertical_lines_positions)

    date_ranges = [
        (base_time, train_line),
        (train_line, valid_line),
        (valid_line, '2035-10-25 23:55:55')
    ]

    colors = ['red', 'green', 'blue']
    legends = ['Train', 'Valid', 'Test']
    legend_positions = [
        train_len / 2,
        (train_len + valid_len) - (valid_len / 2),
        train_len + valid_len + ((((train_len + valid_len) - 1) * -1) / 2)]

    fig = plt.figure(figsize=(27, 9))
    plt.plot(df_all['date'], df_all['actual'], label='Actual')

    for _, ((start_date_str, end_date_str), color, legend) in enumerate(zip(date_ranges, colors, legends)):
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        mask = (df_all['date'] >= start_date) & (df_all['date'] <= end_date)

        plt.plot(
            df_all.loc[mask, 'date'],
            df_all.loc[mask, 'predicted'],
            color=color, label='Predicted ' + legend
        )

    for x in vertical_lines_positions:
        plt.axvline(x=x, color='k', linestyle='--', linewidth=1)

    formatter = EngFormatter(unit='bps')
    plt.gca().yaxis.set_major_formatter(formatter)

    for legend, position in zip(legends, legend_positions):
        plt.text(
            pd.to_datetime([[base_time + timedelta(minutes=parameters['timeslot'] * i) for i in range(int(len_all_dates * position))][-1]], format='%Y%m%d%H%M')[0],
            0.95 * plt.ylim()[1],
            legend, verticalalignment='top', fontdict={'fontsize': 'xx-large'}
        )

    plt.legend()
    #plt.show()
    return model, fig, fig_model

def forecast_the_future(df, model, bps_scaled, bps_scaler, future=144, timeslot=5, seq_length=10):
    future_steps = future

    last_sequence = bps_scaled[-seq_length:]
    future_predictions_scaled = []
    input_seq = last_sequence.copy()

    for _ in range(future_steps):
        input_seq_reshaped = input_seq.reshape(1, seq_length, 1)
        yhat = model.predict(input_seq_reshaped, verbose=0)
        future_predictions_scaled.append(yhat[0, 0])
        input_seq = np.append(input_seq[1:], yhat[0, 0])

    future_predictions = bps_scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

    last_date = df['date'].iloc[-1]
    future_dates = [last_date + datetime.timedelta(minutes=timeslot * i) for i in range(1, future_steps + 1)]

    df_future = pd.DataFrame({
        'date': future_dates,
        'predicted': future_predictions.flatten()
    })

    df_combined = pd.DataFrame({
        'date': pd.concat([df['date'], df_future['date']]),
        'bps': pd.concat([df['bps'], pd.Series([np.nan]*len(df_future))]),
        'predicted': pd.concat([pd.Series([np.nan]*len(df)), df_future['predicted']])
    })

    fig = plt.figure(figsize=(27, 9))

    plt.plot(df_combined['date'], df_combined['bps'], label='Actual')
    plt.plot(df_combined['date'], df_combined['predicted'], label='Forecasted', color='orange')

    formatter = EngFormatter(unit='bps')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend()
    #plt.show()
    return fig, df_future
