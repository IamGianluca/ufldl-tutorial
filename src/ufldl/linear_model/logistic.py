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
        k: (int) Number of classes.
    """

    def __init__(self, fit_intercept=True, multi_class='ovr'):
        self.fit_intercept = fit_intercept
        self.multi_class = multi_class
        self.coefficients = None
        self.k = None
        self.j_history = []
        self.indicator_mask = None

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
        m, n = X.shape
        self.k = len(np.unique(y))

        if self.fit_intercept:
            X = np.hstack((np.ones((m, 1)), X))
            n += 1

        if self.multi_class == 'ovr':
            x0 = np.random.random(n) * 0.001
        else:
            x0 = np.random.rand(n, self.k) * 0.001
            self.indicator_mask = np.zeros((X.shape[0], x0.shape[1]),
                                           dtype=np.bool)
            for i, idx in enumerate(y):
                self.indicator_mask[i][int(idx)] = True
            X = self._normalise(X)

        self.coefficients = minimize(
            method='L-BFGS-B',
            fun=self._cost_function,
            args=(X, y),
            x0=x0,
            jac=self._gradient,
            options={'maxiter': 100, 'disp': True}
        ).x
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

        start = 1 if self.fit_intercept else 0  # to drop intercept coefficient
        if self.multi_class == 'ovr':
            predictions = self._sigmoid(np.dot(X, self.coefficients[start:]))
            return [1 if prediction > .5 else 0 for prediction in predictions]
        else:
            X = self._normalise(X)
            theta = self.coefficients.reshape(
                (int(self.coefficients.size / self.k), self.k))[start:, :]
            return np.argmax(self.probs(theta, X), axis=1)

    @staticmethod
    def _sigmoid(z):
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
                parameter values to optimize. For the softmax regression case,
                the shape is (n_samples, n_classes).
            X: (array), shape (n_samples, n_features) Training matrix, where
                n_samples is the number of samples and n_features is the number
                of features.
            y: (array), shape (n_samples, 1). Target vector relative to X.

        Returns:
            (array) shape (n_classes, 1). The value of the objective function
            as a function of theta. In case of logistic regression, the return
            value is a float instead of an array.
        """
        if self.multi_class == 'ovr':
            # binary outcome
            h = self._sigmoid(np.dot(X, theta))  # hypothesis
            return - np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        else:
            # multiclass outcome
            log_probs = np.log(self.probs(theta, X))
            cost = 0
            for i in range(X.shape[0]):
                cost -= log_probs[i][int(y[i])]
            return cost

    def probs(self, theta, X):
        if theta.ndim == 1:
            theta = theta.reshape((int(theta.size / self.k), self.k))
        values = np.exp(np.dot(X, theta))
        sums = np.sum(values, axis=1)
        return (values.T / sums).T

    def _normalise(self, X):
        """Normalizes features matrix to a standard normal distribution
        (zero mean and unit variance).

        Args:
            X: (array), shape (n_samples, n_features). Feature matrix.

        Returns:
            (array), shape (n_samples, n_features). Normalised feature matrix.
        """
        X_mean = X.mean(axis=0)
        # +0.1 to avoid division by zero in this specific case
        X_std = X.std(axis=0) + 0.1

        X_normalised = (X - X_mean) / X_std
        return X_normalised

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
        if self.multi_class == 'ovr':
            errors = self._sigmoid(X.dot(theta)) - y
            return np.dot(errors, X)
        else:
            gradient_matrix = - X.T.dot(
                self.indicator_mask * self.probs(theta, X))
            return gradient_matrix.flatten()
