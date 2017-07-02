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

from ml.linear_model import LinearRegression


if __name__ == '__main__':
    data = []
    with open('exercises/data/housing.csv', newline='\n') as csvfile:
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

    model = LinearRegression()
    model.fit(X=train_x, y=train_y)
    predicted_train_prices = model.predict(X=train_x)

    # print root mean squared error (RMSE) for training set
    train_rmse = np.sqrt(np.mean(predicted_train_prices - train_y)**2)

    print('Train RMSE: ', train_rmse)

    # print RMSE on test set
    predicted_test_prices = model.predict(X=test_x)
    test_rmse = np.sqrt(np.mean(predicted_test_prices - test_y)**2)

    print('Test RMSE: ', test_rmse)

    # plot predictions in test data
    prices = sorted(zip(test_y, predicted_test_prices))
    plt.scatter(x=range(len(test_y)), y=[y for y, y_hat in prices],
                marker='x', color='red', label='Actual price')
    plt.scatter(x=range(len(test_y)), y=[y_hat for y, y_hat in prices],
                marker='x', color='blue', label='Predicted price')
    plt.xlabel('House #'); plt.ylabel('House price ($1000s)')
    plt.legend()
    plt.show()
