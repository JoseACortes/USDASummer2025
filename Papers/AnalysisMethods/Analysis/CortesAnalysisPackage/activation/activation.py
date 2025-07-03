from scipy.stats.mstats import linregress

import numpy as np


def linearActivation(training_x, training_y, testing_x, testing_y, pred_y):
    result = linregress(training_x, training_y)
    analyze = lambda x: result.slope * x + result.intercept
    pred_y = np.array(pred_y)
    mse_test = np.mean(np.square(testing_y - analyze(testing_x)))
    r2 = result.rvalue**2
    act_results = result

    activation_name = 'Linear Activation'
    return activation_name, act_results, mse_test, r2, pred_y 
