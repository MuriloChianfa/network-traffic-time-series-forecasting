import numpy as np

def prepare_df_split(df, max_len=10):
    X = []
    y = []

    for i in range(len(df) - max_len):
        X.append(df[i:i+max_len])
        y.append(df[i+max_len])

    return np.array(X), np.array(y)

def setup_train_valid_test_split(dates, X, y, max_len=10, train_len=.6, valid_len=.2):
    # Split the data
    train_size = int(len(X) * train_len)
    valid_size = int(len(X) * valid_len)
    test_size = len(X) - train_size - valid_size

    # Setup datasets
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_valid = X[train_size:train_size+valid_size]
    y_valid = y[train_size:train_size+valid_size]

    X_test = X[train_size+valid_size:]
    y_test = y[train_size+valid_size:]

    # Corresponding dates (adjusted for the sequence length)
    dates = dates.values[max_len:]
    dates_train = dates[:train_size]
    dates_valid = dates[train_size:train_size+valid_size]
    dates_test = dates[train_size+valid_size:]

    # Reshape y to be 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, dates_train, dates_valid, dates_test