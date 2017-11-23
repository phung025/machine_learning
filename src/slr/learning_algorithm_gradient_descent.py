# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:34:59 2017

@author: Nam Phung
"""

from pandas import Series, DataFrame

def learning_algorithm(training_set, loss_function, steps):
    
    # Initialize variables
    learning_rate = 0.000001
    theta0 = 0
    theta1 = 0
    
    for i in range(0, steps):
        d_t0 = partial_derivative_se_loss_function(theta0, theta1, training_set, feature=0)
        tmp_t0 = theta0 - learning_rate*d_t0
        
        d_t1 = partial_derivative_se_loss_function(theta0, theta1, training_set, feature=1)
        tmp_t1 = theta1 - learning_rate*d_t1
        
        theta0 = tmp_t0
        theta1 = tmp_t1

    def hypothesis(x):
        return theta0 + theta1*x
    print("h(x)=",theta0,"+",theta1,"x")    
    return hypothesis

# Partial derivative of squared error loss function
def partial_derivative_se_loss_function(theta0, theta1, training_set, feature):
    se = 0
    for row in training_set.itertuples():
        if feature == 0:
            se += theta0 + theta1*row[1] - row[2]
        else:
            se += row[1]*(theta0 + theta1*row[1] - row[2])
    return se/len(training_set)

# Load training data
training_data = DataFrame.from_csv('train2.csv', index_col=None)

# Get hypothesis from training data
print("Starting gradient descent using se loss function at theta0=0, theta1=0, learning_rate=0.000001, steps=6000")
hypothesis = learning_algorithm(training_data, partial_derivative_se_loss_function, steps=6000)

# Run the hypothesis function on the test set
test_data = DataFrame.from_csv('test2.csv', index_col=None)
col_names = list(test_data.columns)
col_names.append('y_estimated')
test_data = test_data.reindex(columns=col_names)
estimated_list = list(test_data['x'])
for i in range(0, len(estimated_list)):
    estimated_list[i] = hypothesis(estimated_list[i])
test_data['y_estimated'] = estimated_list
print(test_data)