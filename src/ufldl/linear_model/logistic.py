import numpy as np
from scipy.optimize import minimize


class RegressionError(Exception):
    pass


class LogisticRegression:

    def __init__(self, solver='bfgs'):
        self.solver = solver
        self._coef = None

    def fit(self, X, y):
        m, n = X.shape
        theta = np.random.random(n) * 0.001
        self._coef = minimize(
            method='L-BFGS-B',
            fun=self._least_squares_cost_function, args=(X, y), x0=theta,
            jac=self._least_squares_gradient).x

    def predict(self, X):
        if self._coef is None:
            raise RegressionError
        return self.sigmoid(np.dot(X, self._coef)) > 0.5

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _least_squares_cost_function(self, theta, X, y):
        """Ordinary least squares Linear Regression.

        Args:
            theta [numpy.array]: A vector of size (m, 1) containing the
              parameter values to optimize.
            x [numpy.array]: A matrix of size (m, n) containing the training
              data. x(i, j) is the i'th coordinate of the j'th example.
            y [numpy.array]: A vector of size (m, 1) with the value of the
              independent variable. y(j) is the target for example j.
        Returns:
            The value [float] of the objective function as a function of theta.
        """
        h = self.sigmoid(np.dot(X, theta))
        return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def _least_squares_gradient(self, theta, X, y):
        """Gradient (first derivative) of a linear regression's loss function.

        Args:
            theta [numpy.array]: A vector of size (m, 1) containing the
              parameter values to optimize.
            X [numpy.array]: A matrix of size (m, n) containing the training
              data. x(i, j) is the i'th coordinate of the j'th example.
            y [numpy.array]: A vector of size (m, 1) with the value of the
              independent variable. y(j) is the target for example j.
        Returns:
            The gradient (aka derivative or jacobian) as a function of theta.
            The gradient is a [numpy.array] of size (m, 1).
        """
        errors = self.sigmoid(X.dot(theta)) - y
        return np.dot(errors, X)
