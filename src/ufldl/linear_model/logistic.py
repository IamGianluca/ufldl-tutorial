import numpy as np
from scipy.optimize import minimize

from .linear import RegressionError


class LogisticRegression:
    """Logistic Regression (aka logit, MaxEnt) classifier.

    Args:
        fit_intercept: (bool), default=True. Specifies if a constant (aka bias
            or intercept) should be added to the decision function.
        multi_class: (str), default=`ovr`. Multiclass option can be either
            `ovr` or ‘multinomial’. If the option chosen is ‘ovr’, then a
            binary problem is fit for each label. Else the loss minimised is
            the multinomial loss fit across the entire probability distribution.

    Attributes:
        coefficients: (array), shape (n_classes, n_features). Coefficient of
            the features in the decision function.
    """

    def __init__(self, fit_intercept=True, multi_class='ovr'):
        self.fit_intercept = fit_intercept
        self.multi_class = multi_class
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

        Raises:
            NotImplementedError: Softmax regression is not yet implemented.
        """
        if self.multi_class != 'ovr':
            raise NotImplementedError('Multiclass logistic regression is not '
                                      'yet implemented.')
        m, n = X.shape

        if self.fit_intercept:
            X = np.hstack((np.ones((m, 1)), X))
            n += 1

        x0 = np.random.random(n) * 0.001
        self.coefficients = minimize(
            method='L-BFGS-B', fun=self._cost_function,
            args=(X, y), x0=x0, jac=self._gradient).x
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: (array), shape (n_samples, n_features). Samples.

        Returns:
            (array), shape (n_samples, ). Predicted class label per sample.

        Raises:
            RegressionError: `predict` method is called before `fit`.
        """
        if self.coefficients is None:
            raise RegressionError('You need to first fit the model, before '
                                  'being able to make any prediction.')

        start = 1 if self.fit_intercept else 0  # drop intercept coefficient
        predictions = self.sigmoid(np.dot(X, self.coefficients[start:]))
        return [1 if prediction > .5 else 0 for prediction in predictions]

    @staticmethod
    def sigmoid(z):
        """Sigmoid function.

        Args:
            z: (float) The input value in the real domain.

        Returns:
            (float) The output value in the (0, 1) domain.
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, theta, X, y):
        """Loss function for Ordinary Least Squares (OLS) linear regression.

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
        hypothesis = self.sigmoid(np.dot(X, theta))
        return -np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis))

    def _gradient(self, theta, X, y):
        """Gradient (aka first derivative) of the linear regression's loss
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
        errors = self.sigmoid(X.dot(theta)) - y
        return np.dot(errors, X)
