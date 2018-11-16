import os
import numpy as np
import argparse
from numpy import concatenate
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from dataLoader import DataLoader
from errorReport import report_errors

modelsNo = 3
models_output_folder = 'data/output/individual_models/'

errorType = {'BLE': 'mae', 'traffic': 'mae', 'electricity': 'mape'}
pastObserves = {'BLE': 1, 'traffic': 24, 'electricity': 48}


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def create_model(parameters, pastObserve=0):
    np.random.seed(11)
    # create model
    input_dim_size = (parameters - 1) + pastObserve
    neuronNo = 120
    model = Sequential()
    model.add(Dense(neuronNo, input_dim=input_dim_size, init='uniform', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def learn_test(expr):
    loader = DataLoader()
    dataset = loader.loadData(dataset=expr)  # dataset options: electricity, traffic, BLE
    pastObserve = pastObserves[expr]
    o_columns = dataset.columns
    predCol = o_columns[-1]
    lenAll = len(dataset)
    lenx = int(lenAll * .75)

    test_orig = []
    mean_errors = []
    error_stds = []
    all_errors = []

    all_predictions = []

    values = dataset.values
    origData = values

    # normalize
    parameters = dataset.values.shape[1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, pastObserve, 1)

    # drop columns we don't want to predict
    droppings = []
    for i in range(1, pastObserve + 1):
        x = [a for a in range(parameters * (i - 1), parameters * i - 1)]
        droppings.extend(x)
    reframed.drop(reframed.columns[droppings], axis=1, inplace=True)
    valuesTrans = reframed.values
    test = valuesTrans

    # split into input and outputs
    train_X_all, train_y_all = valuesTrans[:, :-1], valuesTrans[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    trainingModels = []
    for i in range(modelsNo):
        deepModel = create_model(parameters, pastObserve)
        trainingModels.append(deepModel)

    dy = 0
    sparsity = 3
    for model in trainingModels:
        # fit network
        partsLen = int(len(train_X_all) / sparsity) * sparsity
        a = np.arange(partsLen)
        a = a.reshape(sparsity, int(partsLen / sparsity))
        ixs = []
        # just consider part of dataset not all of that
        for t in range(sparsity):
            if (t == dy):
                ixs.append(a[t])
        # ixs.append(a[t+1]) # for considering 40% sparsity
        # ixs.append(a[t+2]) # for considering 60% sparsity
        ixs = np.array(ixs)
        train_ixs = ixs.flatten()
        train_X, train_y = train_X_all[train_ixs], train_y_all[train_ixs]
        model.fit(train_X, train_y, epochs=20, batch_size=20, verbose=2)
        dy += 1
        # calculate predictions
        predictions = model.predict(test_X)
        predictions = predictions.reshape((len(predictions), 1))

        pads = np.zeros(len(test_y) * (parameters - 1))
        pads = pads.reshape(len(test_y), parameters - 1)

        inv_yhat = concatenate((pads, predictions), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, -1]
        inv_yhat = np.around(inv_yhat, decimals=2)

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_test = concatenate((test_X[:, pastObserve:], test_y), axis=1)
        test_orig = scaler.inverse_transform(inv_test)

        origY = test_orig[:, -1]
        meanErr, std, errors = report_errors(origY, inv_yhat, errorType[expr])

        mean_errors.append(meanErr)
        error_stds.append(std)

        all_errors.append(errors)
        all_predictions.append(inv_yhat)

        print(min(origY), max(origY))
        print(min(inv_yhat), max(inv_yhat))
        print('Test Mean Error: %.3f ' % meanErr)

    p_cols = []
    df = DataFrame(test_orig, columns=o_columns)
    for k in range(len(all_predictions)):
        colName = 'predict_' + str(k + 1)
        p_cols.append(colName)
        df[colName] = all_predictions[k]
    for k in range(len(all_predictions)):
        errName = 'error_' + str(k + 1)
        df[errName] = all_errors[k]

    print(errorType[expr])
    print(mean_errors)

    if not os.path.exists(models_output_folder):
        os.makedirs(models_output_folder)

    outDetails_filename = models_output_folder + 'predictions_details_%s.csv' % expr
    out_filename = models_output_folder + 'predictions_output_%s.csv' % expr

    df.to_csv(outDetails_filename, index=False)

    models_prediction_cols = p_cols
    models_prediction_cols.append(predCol)
    df_modelOutput = df[models_prediction_cols]
    df_modelOutput.to_csv(out_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploiting Spatio-Temporal patterns in data")
    parser.add_argument('--dataset', type=str, default='electricity', \
                        help="name of dataset to train the model (e.g., electricity, traffic, BLE)")

    args = parser.parse_args()
    dataset = args.dataset
    learn_test(dataset)
