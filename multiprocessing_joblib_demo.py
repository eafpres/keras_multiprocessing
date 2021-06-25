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
#
#%% function def
#
def slow_function():
    for rep in range(5):
        x = pd.Series(np.random.normal(loc = 0, scale = 1, size = 1000))
        for i in x.index:
            z = x[i]**2
            p = np.sqrt(np.abs(x[i]))
            x[i] = z + p
        x = np.random.normal(loc = 5 * np.random.randint(1, 3), scale = 1) + x
    return x
#
#%% parallel function
#
def parallel_runs(n_trials):
#
    results = []
    for trial in range(n_trials):
           results = results + [slow_function()]
    return results
#
#%% parallel slow function demo
#
start = datetime.now()
parallel_processes = 8
trials_per_process = 3
parallel_results = (Parallel(n_jobs = -1)([delayed(parallel_runs)(trials_per_process)
                                           for _ in range(parallel_processes)]))
print('parallel time required: ', (datetime.now() - start).total_seconds())
#
serial_results = []
start = datetime.now()
for i in range(parallel_processes * trials_per_process):
    serial_results = serial_results + [slow_function()]
print('serial time required: ', (datetime.now() - start).total_seconds())
#
#%% vusualize
#
x_values = np.linspace(0, 25, 1000)
fig, ax = plt.subplots()
for job in parallel_results:
    for run in job:
        kde = gaussian_kde(run)
        density = kde(x_values)
        ax.fill(x_values, density, alpha = 0.3)
plt.show()
#
fig, ax = plt.subplots()
for job in serial_results:
    kde = gaussian_kde(job)
    density = kde(x_values)
    ax.fill(x_values, density, alpha = 0.3)
plt.show()
#
