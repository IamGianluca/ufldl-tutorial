import os
import pickle

import numpy as np

from ufldl.linear_model import LogisticRegression
from ufldl.datasets import load_mnist

DATA_PATH = './exercises/data'
PICKLE_X_TMP = '{}_x_mnist.p'
PICKLE_X_FULLPATH_TMP = os.path.join(DATA_PATH, PICKLE_X_TMP)
PICKLE_Y_TMP = '{}_y_mnist.p'
PICKLE_Y_FULLPATH_TMP = os.path.join(DATA_PATH, PICKLE_Y_TMP)


def load_cached_mnist(type_='train'):
    """Load cached MNIST dataset, if exists, else process raw dataset and
    pickle it.

    Args:
        type_: (str) The type of dataset to load. Accepted values are `train`
        or `test`. Default=`train`.

    Returns:
        The preprocessed MNIST dataset. The feature matrix includes an
        intercept term.
    """

    if type_ not in ['train', 'test']:
        raise ValueError('`type` can only be either `train` or `test`.')

    try:
        # load serialized dataset
        X = pickle.load(open(PICKLE_X_FULLPATH_TMP.format(type_), 'rb'))
        y = pickle.load(open(PICKLE_Y_FULLPATH_TMP.format(type_), 'rb'))
    except FileNotFoundError:
        # load raw dataset
        if type_ == 'train':
            image_fullpath = os.path.join(
                DATA_PATH, 'train-images-idx3-ubyte.gz')
            label_fullpath = os.path.join(
                DATA_PATH, 'train-labels-idx1-ubyte.gz')
        else:
            image_fullpath = os.path.join(
                DATA_PATH, 't10k-images-idx3-ubyte.gz')
            label_fullpath = os.path.join(
                DATA_PATH, 't10k-labels-idx1-ubyte.gz')
        images, labels = load_mnist(
            imagefile=image_fullpath,
            labelfile=label_fullpath)

        # reshape 3D feature tensor to a 2D matrix
        features = images.reshape(len(labels), 28 * 28)

        # add intercept term to feature matrix
        X = np.hstack((np.ones((len(labels), 1)), features))

        # flat outcome vector
        y = labels.ravel()

        # pickle X and y
        pickle.dump(X, open(PICKLE_X_FULLPATH_TMP.format(type_), 'wb'))
        pickle.dump(y, open(PICKLE_Y_FULLPATH_TMP.format(type_), 'wb'))

    return X, y


def consider_only_zero_and_one_examples(X, y):
    """Drop all records from the dataset where the label doesn't have a value
    of either 0 or 1.

    Args:
        X: (array) The feature matrix.
        y: (vector) The outcome vector.

    Returns:
        The feature matrix and outcome vector without the records where the
        outcome value isn't either 0 or 1.
    """
    include_idx = np.where(y < 2)[0]
    return X[include_idx], y[include_idx]


if __name__ == '__main__':
    # load training dataset
    X_train_all, y_train_all = load_cached_mnist(type_='train')

    # NOTE: consider only `0` and `1` images
    X_train, y_train = consider_only_zero_and_one_examples(
        X=X_train_all, y=y_train_all)

    # train logistic regression classifier
    model = LogisticRegression()
    model.fit(X=X_train, y=y_train)

    # compute accuracy on training set
    predicted_train_digit = model.predict(X=X_train)
    correct = np.sum(y_train == predicted_train_digit)
    print('Train accuracy: {:.2%}'.format(correct / len(y_train)))

    # compute accuracy on test set
    X_test_all, y_test_all = load_cached_mnist(type_='test')

    # NOTE: consider only `0` and `1` images
    X_test, y_test = consider_only_zero_and_one_examples(
        X=X_test_all, y=y_test_all)

    predicted_test_prices = model.predict(X=X_test)
    correct = np.sum(y_test == predicted_test_prices)
    print('Test accuracy: {:.2%}'.format(correct / len(y_test)))
