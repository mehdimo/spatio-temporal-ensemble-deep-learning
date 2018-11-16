import numpy as np

def __mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diffs = y_true - y_pred
    diffs = np.abs(diffs)
    errors = np.true_divide(diffs, y_true) * 100
    return np.mean(errors), np.std(errors), errors

def __mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diffs = np.abs(y_true - y_pred)
    return np.mean(diffs), np.std(diffs), diffs

def __mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diffs = y_true - y_pred
    diffs = diffs ** 2
    return np.mean(diffs), np.std(diffs), diffs

def report_errors(y_true, y_pred, type='mape'):
    if type == 'mape':
        return __mean_absolute_percentage_error(y_true, y_pred)
    elif type == 'mae':
        return __mean_absolute_error(y_true, y_pred)
    elif type == 'mse':
        return __mean_squared_error(y_true, y_pred)
