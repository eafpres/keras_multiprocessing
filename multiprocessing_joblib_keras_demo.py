# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:50:18 2020

@author: bbate
"""
#
#%% libraries
#
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
import multiprocessing
#
# if desired to disable GPU for parallel training
#
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
#
#%% timestamp
#
def ts(ms = False, real = False):
    from datetime import datetime
    ts_str = str(datetime.now())
    timestamp = (ts_str[0:10] + '_' +
                 ts_str[11:13] + '_' +
                 ts_str[14:16] + '_' +
                 ts_str[17:19])
    if ms:
        timestamp = (timestamp + '_' +
                     str(int(round(int(ts_str[20:30]) / 1000, 0))).zfill(3))
    if real:
        time = (int(timestamp[11:13]) * 60 * 60 * 1000 +
                int(timestamp[14:16]) * 60 * 1000 +
                int(timestamp[17:19]) * 1000)
        if ms:
            time = time + int(timestamp[20:])
        timestamp = time
    return timestamp
#
#%% train ANN
#
def train_ANN(x, y, epochs, y_offset, pid):
#
    DNN_model = Sequential()
    DNN_model.add(Dense(units = 1, activation = 'elu'))
    DNN_model.add(Dense(units = 20, activation = 'elu'))
    DNN_model.add(Dense(units = 20, activation = 'elu'))
    DNN_model.add(Dense(units = 1))
#
    DNN_model.compile(loss = MeanSquaredError())
#
    DNN_model.fit(np.array(x).reshape(-1, 1), np.array(y) + y_offset,
                  epochs = epochs,
                  verbose = 0)
#
    DNN_model.save('keras_parallel_' + ts(ms = True) +
                   str(pid).zfill(10) + '.hdf5')
#
#%% parallel ANN train
#
def parallel_ANN(n_trials, x, y, epochs, value):
#
    pid = multiprocessing.current_process().pid
    for trial in range(n_trials):
        train_ANN(x, y, epochs, value, pid)
    return pid
#
#%% display results
#
def plot_results(y, x_range, pids, plot_individual = None):
    files = os.listdir()
    files = [files[i]
             for i in range(len(files))
             if 'hdf5' in files[i]]
    files = [files[i]
             for i in range(len(files))
             if any(pid in files[i]
                    for pid in pd.Series([str(pids[j]).zfill(10)
                                          for j in range(len(pids))]))]
    predictions = []
    for file in files:
        model = load_model(file)
        predictions = \
            predictions + [model.predict(np.array(x).reshape(-1, 1))]
#
    x_values = np.linspace(x_range[0], x_range[1], 1000)
#
    if (plot_individual):
        for i in range(len(predictions)):
            fig, ax = plt.subplots()
            for model_preds in predictions[i:(i+1)]:
                kde = gaussian_kde(y + model_preds[:, 0].mean())
                y_density = kde(x_values)
                kde = gaussian_kde(model_preds[:, 0])
                density = kde(x_values)
                ax.fill(x_values, density, alpha = 0.3)
            ax.plot(x_values, y_density, lw = 1, color = 'red')
            plt.show()
    else:
        fig, ax = plt.subplots()
#        print('initialized plot')
        all_preds = [predictions[i:(i+1)][j][:, 0].mean()
                     for i in range(len(predictions))
                     for j in range(len(predictions[i:(i+1)]))]
        kde = gaussian_kde(y + pd.Series(all_preds).mean())
        y_density = kde(x_values)
#        print('obtained mean density')
        trace = 0
        for model_preds in predictions:
            kde = gaussian_kde(model_preds[:, 0])
            density = kde(x_values)
            ax.fill(x_values, density, alpha = 0.3)
#            print('filled trace: ', trace)
            trace += 1
        ax.plot(x_values, y_density, lw = 1, color = 'red')
        plt.show()
#
#%% parallel keras demo
#
if __name__ == '__main__':
    os.chdir('c:/eaf llc/aa-analytics and bi/joblib_keras_optuna_multiprocessing')
#
# data
#
    np.random.seed(42)
    x = pd.Series(np.random.normal(loc = 0, scale = 1, size = 1000))
    y = 2 * x + np.sin(2 * np.pi * x)
#
    parallel_processes = 12
    trials_per_process = 3
    epochs = 500
    y_offsets = list(range(parallel_processes))
#
    start = datetime.now()
    ANN_parallel_results = \
        (Parallel(n_jobs = -1)([delayed(parallel_ANN)(trials_per_process,
                                                      x, y, epochs, offset)
                                for offset in y_offsets]))
    run_time = (datetime.now() - start).total_seconds()
    x_range = [y.mean() - 6 * y.std(), y.mean() + 6 * y.std() + max(y_offsets)]
    plot_results(y, x_range, ANN_parallel_results, plot_individual = False)
    print('\n\nparallel time required: ', run_time)
#
#%% serial test
#
if __name__ == '__main__':
#
# data
#
    np.random.seed(42)
    x = pd.Series(np.random.normal(loc = 0, scale = 1, size = 1000))
    y = 2 * x + np.sin(2 * np.pi * x)
#
    parallel_processes = 12
    trials_per_process = 3
    epochs = 500
    pid = '__SERIAL__'
#
    y_offsets = list(range(parallel_processes))
#
    start = datetime.now()
    for run in range(parallel_processes * trials_per_process):
        train_ANN(x, y, epochs, np.random.choice(y_offsets), pid)
    run_time = (datetime.now() - start).total_seconds()
    x_range = [y.mean() - 6 * y.std(), y.mean() + 6 * y.std() + max(y_offsets)]
    plot_results(y, x_range, ([pid] * parallel_processes * trials_per_process))
    print('\n\nserial time required: ', run_time)
