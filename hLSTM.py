from statsmodels.tsa.arima_model import ARIMA
from ConvLSTM import ConvLSTM
from matplotlib import pyplot as plt
from Utils import *


if __name__ == '__main__':
    # Data pre-process (1)
    sequence = separe_column(r"monitoramento-processo-d (1).txt", 'vmrss')
    sequence, s_min, s_max = normalize(sequence)
    train, test = split_sets(sequence, 0.8)

    # MA block
    model = ARIMA(train, order=(0, 0, 1))
    model_fit = model.fit()
    X = model_fit.predict(start=49, end=len(train) - 1)

    # Data pre-process (2)
    n_steps = 4

    X_error, y_error = split_sequence(X.tolist(), n_steps)
    X, y = split_sequence(train.tolist(), n_steps)
    X_test, y_test = split_sequence(test.tolist(), n_steps)

    X_error = X_error.astype(np.float)
    y_error = y_error.astype(np.float)
    X = X.astype(np.float)
    y = y.astype(np.float)
    X_test = X_test.astype(np.float)
    y_test = y_test.astype(np.float)

    n_features = 1
    n_seq = 2
    n_steps = 2

    X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
    X_error = X_error.reshape((X_error.shape[0], n_seq, 1, n_steps, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_seq, 1, n_steps, n_features))

    # Conv-LSTM block
    model = ConvLSTM(n_steps, n_features, n_seq, 1e-3, 'mse', ['mape', 'mse', 'mae'])
    history = model.fit(X_error[1:], y_error[1:], validation_data=(X_test, y_test), epochs=100, verbose=1)
    pred_x = model.predict(X)
    pred_x_test = model.predict(X_test)

    # Plots
    dif = (len(np.concatenate((y, y_test), axis=0)) - len(pred_x_test))
    axis_x_test = [(i + dif) for i in range(len(pred_x_test))]
    y_true = sequence

    plt.plot(y_true[1:], label='Original Set', color='blue')
    plt.plot(pred_x, label='Predicted Train Set', color='red', linestyle='-.')
    plt.plot(axis_x_test, pred_x_test, label='Predicted Test Set', color='green', linestyle='-.')

    plt.xlabel('Time (min)')
    plt.ylabel('Memory Used')
    plt.legend()

    plt.show()

