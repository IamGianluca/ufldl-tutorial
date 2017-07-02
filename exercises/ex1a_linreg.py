# This exercise uses a data from the UCI repository:
#   Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#   http://archive.ics.uci.edu/ml
#   Irvine, CA: University of California, School of Information and
#     Computer Science.

# Data created by:
#   Harrison, D. and Rubinfeld, D.L.
#   ''Hedonic prices and the demand for clean air''
#   J. Environ. Economics & Management, vol.5, 81-102, 1978.


import matplotlib.pyplot as plt
import numpy as np

from ufldl.linear_model import LinearRegression
from ufldl.datasets import load_housing


if __name__ == '__main__':
    array = load_housing()

    # add intercept
    ones = np.ones((array.shape[0], 1))
    X = np.hstack((ones, array))

    # split into train and test sets
    np.random.shuffle(X)
    split = int(X.shape[0] * 0.8)
    train, test = X[:split, :], X[split:, :]

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    predicted_train_prices = model.predict(X=X_train)

    # print root mean squared error (RMSE) for training set
    train_rmse = np.sqrt(np.mean(predicted_train_prices - y_train) ** 2)

    print('Train RMSE: ', train_rmse)

    # print RMSE on test set
    predicted_test_prices = model.predict(X=X_test)
    test_rmse = np.sqrt(np.mean(predicted_test_prices - y_test) ** 2)

    print('Test RMSE: ', test_rmse)

    # plot predictions in test data
    prices = sorted(zip(y_test, predicted_test_prices))
    plt.scatter(x=range(len(y_test)), y=[y for y, y_hat in prices],
                marker='x', color='red', label='Actual price')
    plt.scatter(x=range(len(y_test)), y=[y_hat for y, y_hat in prices],
                marker='x', color='blue', label='Predicted price')
    plt.xlabel('House #')
    plt.ylabel('House price ($1000s)')
    plt.legend()
    plt.show()
