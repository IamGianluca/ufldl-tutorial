import numpy as np


def linear_regression(theta, x, y):
    """Ordinary least squares Linear Regression.

    Args:
        theta [numpy.array]: A vector of size (m, 1) containing the parameter
          values to optimize.
        x [numpy.array]: A matrix of size (m, n) containing the trainig data.
          x(i, j) is the i'th coordinate of the j'th example.
        y [numpy.array]: A vector of size (m, 1) with the value of the
          independed variable. y(j) is the target for example j.
    Returns:
        The value [float] of the objective function as a function of theta.
    """
    return 1/2 * np.sum((x @ theta - y)**2)

def linear_regression_gradient(theta, x, y):
    """Gradient (first derivative) of a linear regression's loss function.

    Args:
        theta [numpy.array]: A vector of size (m, 1) containing the parameter
          values to optimize.
        x [numpy.array]: A matrix of size (m, n) containing the trainig data.
          x(i, j) is the i'th coordinate of the j'th example.
        y [numpy.array]: A vector of size (m, 1) with the value of the
          independed variable. y(j) is the target for example j.
    Returns:
        The gradient (aka derivative or jacobian) as a function of theta. The
        gradient is a [numpy.array] of size (m, 1).
    """
    return (x @ theta - y) @ x
