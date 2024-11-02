from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def simpleLSTM(X_train, y_train, X_valid, y_valid, epochs, neurons, num_of_future_forecasts=1):
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = Sequential()
    model.add(LSTM(neurons, use_cudnn=False, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.1))
    model.add(Dense(num_of_future_forecasts))
    model.compile(optimizer='adam', loss='mse')

    #print("Training LSTM Model...")
    model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))

    return model

from dataset import prepare_df_split, setup_train_valid_test_split
from preprocess import trace_scaler
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt

def run(df, parameters):
    bps_scaled, bps_scaler = trace_scaler(df['bps'])
    X, y = prepare_df_split(bps_scaled)
    train_len = parameters['train_valid_test_split'][0]
    valid_len = parameters['train_valid_test_split'][1] - train_len
    X_train, y_train, X_valid, y_valid, X_test, y_test, dates_train, dates_valid, dates_test = setup_train_valid_test_split(df['date'], X, y, train_len=train_len, valid_len=valid_len)

    model = simpleLSTM(X_train, y_train, X_valid, y_valid, parameters['epochs'], parameters['neurons'], num_of_future_forecasts=1)

    train_yhat = model.predict(X_train)
    valid_yhat = model.predict(X_valid)
    test_yhat = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, r2_score
    import pandas as pd

    # Inverse transform the predictions and actual values
    actual_train = bps_scaler.inverse_transform(y_train)
    actual_valid = bps_scaler.inverse_transform(y_valid)
    actual_test = bps_scaler.inverse_transform(y_test)

    predicted_train = bps_scaler.inverse_transform(train_yhat)
    predicted_valid = bps_scaler.inverse_transform(valid_yhat)
    predicted_test = bps_scaler.inverse_transform(test_yhat)

    # Create DataFrames
    df_train = pd.DataFrame({
        'date': dates_train,
        'actual': actual_train.flatten(),
        'predicted': predicted_train.flatten(),
        'set': 'train'
    })

    df_valid = pd.DataFrame({
        'date': dates_valid,
        'actual': actual_valid.flatten(),
        'predicted': predicted_valid.flatten(),
        'set': 'validation'
    })

    df_test = pd.DataFrame({
        'date': dates_test,
        'actual': actual_test.flatten(),
        'predicted': predicted_test.flatten(),
        'set': 'test'
    })

    df_all = pd.concat([df_train, df_valid, df_test], axis=0)
    dates_all = pd.to_datetime(df_all['date'])

    from datetime import timedelta
    base_time = pd.to_datetime([df['date'][0]], format='%Y%m%d%H%M')[0]

    len_dates_train = len(dates_train)
    train_line = [base_time + timedelta(minutes=5 * i) for i in range(len_dates_train)][-1]

    len_dates_valid = len(dates_valid)
    valid_line = [base_time + timedelta(minutes=5 * i) for i in range(len_dates_train + len_dates_valid)][-1]

    len_dates_test = len(dates_test)
    test_line = [base_time + timedelta(minutes=5 * i) for i in range(len_dates_train + len_dates_valid + len_dates_test)][-1]

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

    # PLOT
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
            pd.to_datetime([[base_time + timedelta(minutes=5 * i) for i in range(int(len_all_dates * position))][-1]], format='%Y%m%d%H%M')[0],
            0.95 * plt.ylim()[1],
            legend, verticalalignment='top', fontdict={'fontsize': 'xx-large'}
        )

    plt.legend()
    #plt.show()
    return fig

