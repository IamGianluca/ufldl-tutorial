import numpy as np
from scipy.optimize import minimize


class RegressionError(Exception):
    pass


class LinearRegression:
    """Ordinary least squares Linear Regression.

    Args:
        fit_intercept: (bool), default=True. whether to calculate the intercept
            for this model. If set to False, no intercept will be used in
            calculations (e.g. data is expected to be already centered).
    Attributes:
        coefficients: (array), shape (n_samples, 1) Estimated coefficients for
            the linear regression problem.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Args:
            X: (array), shape (n_samples, n_features). Training matrix, where
                n_samples is the number of samples and n_features is the number
                of features.
            y: (array), shape (n_samples, ). Target vector relative to X.

        Returns:
            (self).
        """
        # initialise the coefficient vector theta
        m, n = X.shape
        theta = np.random.random(n)

        # find the values of theta that minimise the cost function
        self.coefficients = minimize(
            self._cost_function, theta, args=(X, y),
            jac=self._gradient, method='bfgs').x
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: (array), shape (n_samples, n_features). Samples.

        Returns:
            (array), shape (n_samples, ). Predicted class label per sample.
        """
        if self.coefficients is None:
            raise RegressionError('You need to first fit the model, before '
                                  'being able to make any prediction.')
        return X @ self.coefficients

    @staticmethod
    def _cost_function(theta, X, y):
        """Ordinary least squares Linear Regression.

        Args:
            theta: (array), shape (n_samples, 1). A vector containing the
                parameter values to optimize.
            X: (array), shape (n_samples, n_features) Training matrix, where
                n_samples is the number of samples and n_features is the number
                of features.
            y: (array), shape (n_samples, 1). Target vector relative to X.

        Returns:
            (float) The value of the objective function as a function of theta.
        """
        return 1/2 * (X @ theta - y).T @ (X @ theta - y)

    @staticmethod
    def _gradient(theta, X, y):
        """Gradient (first derivative) of a linear regression's loss
        function.

        Args:
            theta: (array), shape (n_samples, 1). The parameter values to
                optimize.
            X: (array), shape (n_samples, n_features) Training matrix.
            y: (array), shape (n_samples, 1). Target vector relative to X.

        Returns:
            (array), shape (n_samples, 1). The gradient (aka derivative or
            jacobian) as a function of theta.
        """
        return (X.T @ X @ theta) - (X.T @ y)
