# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:46:54 2017

@author: Nam Phung
"""

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

def learning_algorithm_pearsons_r(x, y, data):
    
    # Reindex the data frame, add 2 new columns: z-score of x & y
    col_names = list(data.columns)
    col_names.append('z_x')
    col_names.append('z_y')
    data = data.reindex(columns=col_names)
    
    # Compute the mean and std of x
    m_x = data.mean()[x]
    sd_x = data.std()[x]

    # Compute the mean and std of y
    m_y = data.mean()[y]
    sd_y = data.std()[y]
    
    # Compute the Z-Score for columns z_x and z_y
    y_list = list(data[y])
    for i in range(0, len(y_list)):
        y_list[i] = (y_list[i]-m_y)/sd_y
    data['z_y'] = y_list

    x_list = list(data[x])
    for i in range(0, len(x_list)):
        x_list[i] = (x_list[i]-m_x)/sd_x
    data['z_x'] = x_list
    
    # Compute pearson's r
    r = 0
    for row in data.itertuples():
        r = r + row[3]*row[4]
    r = r/(len(data)-1)
    
    # Compute intercepts and regression coefficients
    b = r*(sd_y/sd_x)
    a = m_y - (b*m_x)
    print('a=',a)
    print('b=',b)
    def hypothesis(n):
        return a+(b*n)
    
    return hypothesis

# Open input data
training_data = DataFrame.from_csv('test3.csv', index_col=None)

# Get hypothesis from training data
hypothesis_func = learning_algorithm_pearsons_r(x='x',y='y', data=training_data)

# Run the hypothesis function on the test set
test_data = DataFrame.from_csv('train2.csv', index_col=None)
col_names = list(test_data.columns)
col_names.append('y_estimated')
test_data = test_data.reindex(columns=col_names)
estimated_list = list(test_data['x'])
for i in range(0, len(estimated_list)):
    estimated_list[i] = hypothesis_func(estimated_list[i])
test_data['y_estimated'] = estimated_list