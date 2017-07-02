import numpy as np
from scipy.optimize import minimize


class RegressionError(Exception):
    pass


class LinearRegression:

    def __init__(self, solver='bfgs'):
        self.solver = solver
        self._coef = None

    def fit(self, X, y):
        # initialise the coefficient vector theta
        m, n = X.shape
        theta = np.random.random(n)

        # find the values of theta that minimise the cost function
        self._coef = minimize(
            self._least_squares_cost_function, theta, args=(X, y),
            jac=self._least_squares_gradient, method=self.solver).x

    def predict(self, X):
        if self._coef is None:
            raise RegressionError
        return X @ self._coef

    def _least_squares_cost_function(self, theta, x, y):
        """Ordinary least squares Linear Regression.

        Args:
            theta [numpy.array]: A vector of size (m, 1) containing the
              parameter values to optimize.
            x [numpy.array]: A matrix of size (m, n) containing the trainig
              data. x(i, j) is the i'th coordinate of the j'th example.
            y [numpy.array]: A vector of size (m, 1) with the value of the
              independed variable. y(j) is the target for example j.
        Returns:
            The value [float] of the objective function as a function of
            theta.
        """
        return 1/2 * (x @ theta - y).T @ (x @ theta - y)

    def _least_squares_gradient(self, theta, x, y):
        """Gradient (first derivative) of a linear regression's loss
        function.

        Args:
            theta [numpy.array]: A vector of size (m, 1) containing the
              parameter values to optimize.
            x [numpy.array]: A matrix of size (m, n) containing the trainig
              data. x(i, j) is the i'th coordinate of the j'th example.
            y [numpy.array]: A vector of size (m, 1) with the value of the
              independed variable. y(j) is the target for example j.
        Returns:
            The gradient (aka derivative or jacobian) as a function of theta.
            The gradient is a [numpy.array] of size (m, 1).
        """
        return (x.T @ x @ theta) - (x.T @ y)
