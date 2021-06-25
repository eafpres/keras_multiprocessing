# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:03:00 2020

@author: bbate
"""
#
#%% libraries
#
import pandas as pd
import numpy as np
from multiprocessing import Pool
#
#%% generate data
#
x = pd.Series(np.arange(1, 100, 0.03))
#
#%% function def
#
def slow_function(x):
    for rep in range(5):
        for i in x.index:
            z = x[i]**2
            p = np.sqrt(np.abs(x[i]))
            x[i] = np.random.normal(loc = 0, scale = 1) + z + p
    return x
#
#%% run processes
#
if __name__ == '__main__':
    with Pool(2) as p:
        p.map(slow_function, [x])
        print(x[0])