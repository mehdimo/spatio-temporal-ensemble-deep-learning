import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from numpy.random import choice
import matplotlib.pyplot as plt
import argparse
from dataLoader import DataLoader
import individualModels
from bloomFilterUtil import TemporalBloomFilter
from errorReport import report_errors

errorType = {'BLE': 'mse', 'traffic': 'mse', 'electricity': 'mape'}
BFsNo = 3

predictions_path = 'data/output/predictions/'
figures_path = 'data/output/figures/'


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


def create_deep(parameters):
    np.random.seed(11)
    # create model
    input_dim_size = (parameters - 1)
    neuronNo = 300
    model = Sequential()
    model.add(Dense(neuronNo, input_dim=input_dim_size, init='uniform', activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def inverse_y(scaler, ypred, colsNo):
    """Creates pads, a matrix of zeros with the shape len(ypred) * colsNo-1. Then inverse ypred to its original form."""
    pads = np.zeros(len(ypred) * (colsNo - 1))
    pads = pads.reshape(len(ypred), colsNo - 1)

    ypred = ypred.reshape((len(ypred), 1))
    inv_yhat2 = concatenate((pads, ypred), axis=1)
    inv_yhat2 = scaler.inverse_transform(inv_yhat2)
    inv_yhat2 = inv_yhat2[:, -1]
    inv_yhat2 = np.around(inv_yhat2, decimals=2)
    return inv_yhat2


def main(expriment_dataset):
    expr = expriment_dataset
    loader = DataLoader()
    data = loader.loadModelsOutput(dataset=expr)
    o_columns = data.columns
    predCol = o_columns[-1]
    values = data.values
    lenx = int(len(data) * .85)
    parameters = data.values.shape[1]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    train = scaled[:lenx, :]
    test = scaled[lenx:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    model = create_deep(parameters)
    model.fit(train_X, train_y, epochs=20, batch_size=20, verbose=2)
    predictions = model.predict(test_X)
    inv_yhat = inverse_y(scaler, predictions, parameters)
    test_orig = inverse_y(scaler, test_y, parameters)
    meanErr_deep, std_deep, errors_deep = report_errors(test_orig, inv_yhat, errorType[expr])

    ################ hybrid of bloom and deep ###################

    modelsNo = parameters - 1
    rev_test = scaler.inverse_transform(test)
    df = DataFrame(rev_test, columns=data.columns)

    deltas = [1, 2, 4]
    deltas = np.array(deltas, dtype=float)

    # create bloom filters
    tbf = TemporalBloomFilter(BFsNo, deltas)

    models_hit_pattern = np.array([np.zeros(len(test_orig)) for i in range(modelsNo)])
    models_weight_pattern = np.array([np.zeros(len(test_orig)) for i in range(modelsNo)])
    hit_bloom = np.zeros(len(test_orig))
    ewma_hist = []

    bloomData = data.ix[:, predCol]
    tbf.fill(bloomData)

    # check temporal data
    ensembleValueChoice = []
    ensembleValueChoice_single = []
    p_cols = data.columns[:-1]
    for i in range(BFsNo):  # e.g., [0,1,2] for the first 3 items of data, we make a weighted choice of model
        ch = choice(modelsNo, 1)
        wm = choice(10, modelsNo)
        wm_sgl = np.argmax(wm)
        ewma_hist.append(np.zeros(modelsNo))
        wm = np.true_divide(wm, np.sum(wm))
        models_weight_pattern[:, i] = wm

        p = df.ix[i, p_cols]
        wm_sgl = p[wm_sgl]
        p = p * wm
        p = np.sum(p)

        ensembleValueChoice.append(p)
        ensembleValueChoice_single.append(wm_sgl)

    for i in range(BFsNo, len(test_orig)):
        models_Bf_hit, hit_pattern, hit_ = tbf.check(df, i)
        models_hit_pattern[:, i] = hit_pattern
        hit_bloom[i] = hit_

        weights = np.array(models_Bf_hit)
        weight_single = np.argmax(weights)

        ewma_hist.append(weights)
        points = df.ix[i, p_cols]
        currentPoint_single = points[weight_single]
        if (np.sum(weights) == 0):
            weights = np.array([1] * modelsNo)

        model_w = np.true_divide(weights, np.sum(weights))
        models_weight_pattern[:, i] = model_w

        points = points * model_w
        currentPoint_w = np.sum(points)
        ensembleValueChoice.append(currentPoint_w)
        ensembleValueChoice_single.append(currentPoint_single)

    ensembleValueChoice = np.array(ensembleValueChoice)
    ensembleValueChoice_single = np.array(ensembleValueChoice_single)

    hybrid = []
    for i in range(len(hit_bloom)):
        if hit_bloom[i] == 1:
            hybrid.append(inv_yhat[i])
        else:
            hybrid.append(ensembleValueChoice[i])
    hybrid = np.array(hybrid)
    print("bloom hit: %s" % sum(1 for i in hit_bloom if i > 0))
    print("total: %s" % len(hit_bloom))

    ################ end of using bloom filter ##################

    avgs = data[p_cols].mean(axis=1)

    meanErr1, std1, errors1 = report_errors(test_orig, data['predict_1'][lenx:], errorType[expr])
    meanErr2, std2, errors2 = report_errors(test_orig, data['predict_2'][lenx:], errorType[expr])
    meanErr3, std3, errors3 = report_errors(test_orig, data['predict_3'][lenx:], errorType[expr])
    meanErr_avg, std_avg, errors_avg = report_errors(test_orig, avgs[lenx:], errorType[expr])
    meanErr_bloom, std_bloom, errors_bloom = report_errors(test_orig, ensembleValueChoice, errorType[expr])
    meanErr_hybrid, std_hybrid, errors_hybrid = report_errors(test_orig, hybrid, errorType[expr])
    meanErr_ensembleSingle, std_ensembleSingle, errors_ensembleSingle = \
        report_errors(test_orig, ensembleValueChoice_single, errorType[expr])

    dfr = DataFrame()
    dfr['original'] = test_orig
    dfr['deep'] = inv_yhat
    dfr['bloom'] = ensembleValueChoice
    dfr['hybrid'] = hybrid
    dfr['bloom_hit'] = hit_bloom
    dfr['err_deep'] = errors_deep
    dfr['err_M1'] = errors1
    dfr['err_M2'] = errors2
    dfr['err_M3'] = errors3
    dfr['err_avg'] = errors_avg
    dfr['err_bloom'] = errors_bloom
    dfr['err_hybrid'] = errors_hybrid
    dfr['err_ensemble_single'] = errors_ensembleSingle

    dfr['Cerr_deep'] = np.cumsum(errors_deep)
    dfr['Cerr_M1'] = np.cumsum(errors1)
    dfr['Cerr_M2'] = np.cumsum(errors2)
    dfr['Cerr_M3'] = np.cumsum(errors3)
    dfr['Cerr_avg'] = np.cumsum(errors_avg)
    dfr['Cerr_bloom'] = np.cumsum(errors_bloom)
    dfr['Cerr_hybrid'] = np.cumsum(errors_hybrid)

    all_errors = [errors1, errors2, errors3, errors_avg, errors_hybrid]

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    pred_file = 'temp_{}_result.csv'.format(expr)
    pred_path = os.path.join(predictions_path, pred_file)
    dfr.to_csv(pred_path, index=False)

    print('M1 error: %s, std: %s' % (meanErr1, std1))
    print('M2 error: %s, std: %s' % (meanErr2, std2))
    print('M3 error: %s, std: %s' % (meanErr3, std3))
    print('SimpleAvg error: %s, std: %s' % (meanErr_avg, std_avg))
    print('Deep error: %s, std: %s' % (meanErr_deep, std_deep))
    print('Bloom error: %s, std: %s' % (meanErr_bloom, std_bloom))
    print('Ensemble error: %s, std: %s' % (meanErr_hybrid, std_hybrid))
    mean_errors = [meanErr1, meanErr2, meanErr3, meanErr_avg, meanErr_hybrid]

    objects = []
    for k in range(3):
        m = 'M' + str(k + 1)
        objects.append(m)
    objects.append('SimpleAvg')
    objects.append('Ensemble')

    ################### Plotting the results ###################################

    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    #################################################
    ### plot horizontal cumulative error over time

    pointsNo = 24
    x = np.arange(pointsNo)
    lineStyles = ['-', '--', '-.', ':']
    markers = ['o', 'v', '^', 's', 'x', '+', '*', 'd', 'P']
    ensObjects = objects

    pltList = []
    i = 0
    j = 0
    fig, ax = plt.subplots()
    for maeList in all_errors:
        maeList2 = maeList[pointsNo:2 * pointsNo]
        maeList2 = np.cumsum(maeList2)
        apt, = ax.plot(x, maeList2, lineStyles[1], linewidth=1, marker=markers[j], markersize=2)
        i = (i + 1) % len(lineStyles)
        j = (j + 1) % len(markers)
        pltList.append(apt)

    start, end = 0, pointsNo + 1
    ax.xaxis.set_ticks(np.arange(start, end, 4))
    xlab = 'Time (hour)'
    if expr == 'BLE':
        xlab = 'Sequence of Trace'
    plt.xlabel(xlab)
    plt.ylabel('Cumulative Error (%s)' % errorType[expr].upper())

    plt.legend(pltList, ensObjects, loc='upper center', bbox_to_anchor=(0., 1.02, 1., .102), ncol=len(ensObjects))
    plt.tight_layout()
    fileName = 'Figure_{}_horiz_error.png'.format(expr)
    file_path = os.path.join(figures_path, fileName)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    #################################################
    ### plot bloom filters hit pattern over time

    x = np.arange(pointsNo)
    lineStyles = ['-', '--', '-.', ':']
    markers = ['o', 'v', 'd', '^', 's', 'p', '*', '+', 'x']

    pltList = []
    mstart, mend = 1, modelsNo + 1
    leg = ['M' + str(i) for i in np.arange(mstart, mend, 1)]
    i = 0
    j = 0
    start, end = 0, pointsNo + 1
    fig, ax = plt.subplots()
    hitPattern = models_hit_pattern
    yrange = range(0, BFsNo + 2)
    for m in range(modelsNo):
        plt.subplot(modelsNo, 1, m + 1)
        apt, = plt.step(x, hitPattern[m, pointsNo:2 * pointsNo], linestyle=lineStyles[i], marker=markers[j],
                        markersize=4, markerfacecolor='None')
        plt.box(on=None)
        plt.yticks(yrange)  # , ('No Hit', '2', '3', '4') )
        plt.xticks(np.arange(start, end, 6))
        plt.gca().yaxis.grid(True)
        if m < modelsNo - 1:
            plt.gca().xaxis.set_visible(False)
        i = (i + 1) % len(lineStyles)
        j = (j + 1) % len(markers)
        pltList.append(apt)

    plt.xlabel('Time (hour)')
    fig.legend(pltList, leg, loc='upper center', bbox_to_anchor=(0., 1.02, 1., .01), ncol=modelsNo)
    plt.tight_layout()
    fileName = 'Figure_{}_BF_hit.png'.format(expr)
    file_path = os.path.join(figures_path, fileName)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    ##### Average error rate for Weighted Sum algorithm
    y_pos = np.arange(len(objects))
    fig, ax = plt.subplots()
    plt.bar(y_pos, mean_errors, align='center', edgecolor='black')
    plt.xticks(y_pos, objects)
    plt.xlabel('Models')
    plt.ylabel('Average Error (%s)' % errorType[expr].upper())
    fileName = 'Figure_{}_average_error.png'.format(expr)
    file_path = os.path.join(figures_path, fileName)
    fig.savefig(file_path)
    # plt.show()
    plt.close()

    ### plot horizontal cumulative error over 7 days time

    mult = 7
    if expr == 'BLE': mult = 2
    pointsNo7 = 24 * mult
    x = np.arange(pointsNo7)
    lineStyles = ['-', '--', '-.', ':']
    markers = ['o', 'v', '^', 's', 'x', '+', '*', 'd', 'P']
    ensObjects = objects

    pltList = []
    i = 0
    j = 0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for maeList in all_errors:
        maeList2 = maeList[pointsNo7:2 * pointsNo7]
        maeList2 = np.cumsum(maeList2)
        apt, = ax.plot(x, maeList2, linestyle=lineStyles[i], linewidth=1, marker=markers[i], markersize=1.5)
        i = (i + 1) % len(lineStyles)
        j = (j + 1) % len(markers)
        pltList.append(apt)

    start, end = 0, pointsNo7 + 1
    ax.xaxis.set_ticks(np.arange(start, end, 24))
    xlab = 'Time (hour)'
    if expr == 'BLE':
        xlab = 'Sequence of Trace'
    plt.xlabel(xlab)
    plt.ylabel('Cumulative Error (%s)' % errorType[expr].upper())

    plt.legend(pltList, ensObjects, loc='upper center', bbox_to_anchor=(0., 1.02, 1., .102), ncol=len(ensObjects))
    plt.tight_layout()
    fileName = 'Figure_{}_horiz_error7days.png'.format(expr)
    file_path = os.path.join(figures_path, fileName)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exploiting Spatio-Temporal patterns in data")
    parser.add_argument('--dataset', type=str, default='electricity', \
                        help="name of dataset to train the model (e.g., electricity, traffic, BLE)")
    parser.add_argument('--i', action='store_true', dest="individuals", \
                        help="Train individual models to produce their predictions. \
                        If not set, the predictions of previous run is considered for the ensemble model.")

    args = parser.parse_args()
    dataset = args.dataset
    run_individuals = args.individuals
    if run_individuals:
        individualModels.learn_test(dataset)
    main(dataset)
