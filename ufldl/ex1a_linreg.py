# This exercise uses a data from the UCI repository:
#   Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#   http://archive.ics.uci.edu/ml
#   Irvine, CA: University of California, School of Information and
#     Computer Science.

# Data created by:
#   Harrison, D. and Rubinfeld, D.L.
#   ''Hedonic prices and the demand for clean air''
#   J. Environ. Economics & Management, vol.5, 81-102, 1978.

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs

from lin_reg import linear_regression
from lin_reg import linear_regression_gradient


if __name__ == '__main__':
    data = []
    with open('ufldl/data/housing.csv', newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for line in csvreader:
            data.append(line)

    # convert into numpy ndarray
    array = np.array(data[1:], dtype=float)

    # add intercept
    ones = np.ones((array.shape[0], 1))
    x = np.hstack((ones, array))

    # split into train and test sets
    np.random.shuffle(x)
    split = int(x.shape[0] * 0.8)
    train, test = x[:split, :], x[split:, :]

    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    m, n = train_x.shape

    # initialise the coefficient vector theta to random values
    theta = np.random.random(n)

    # find values of theta that minimise the loss function
    theta = fmin_bfgs(linear_regression, theta, args=(train_x, train_y),
                      fprime=linear_regression_gradient)

    # print root mean squared error (RMSE) for training set
    predicted_train_prices = train_x @ theta
    train_rmse = np.sqrt(np.mean(predicted_train_prices - train_y)**2)

    print('Train RMSE: ', train_rmse)

    # print RMSE on test set
    predicted_test_prices = test_x @ theta
    test_rmse = np.sqrt(np.mean(predicted_test_prices - test_y)**2)

    print('Test RMSE: ', test_rmse)
    print('Theta: ', theta)

    # plot predictions in test data
    prices = sorted(zip(test_y, predicted_test_prices))
    plt.scatter(x=range(len(test_y)), y=[y for y, y_hat in prices],
                marker='x', color='red', label='Actual price')
    plt.scatter(x=range(len(test_y)), y=[y_hat for y, y_hat in prices],
                marker='x', color='blue', label='Predicted price')
    plt.xlabel('House #'); plt.ylabel('House price ($1000s)')
    plt.legend()
    plt.show()
