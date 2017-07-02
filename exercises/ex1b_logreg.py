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


def load_cached_data(type_='train'):
    if type_ not in ['train', 'test']:
        raise ValueError

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

        # NOTE: consider only `0` and `1` images
        include_idx = np.where(labels < 2)[0]
        y = labels[include_idx].ravel()
        X = images[include_idx]

        # reshape array from 3D to 2D
        X = X.reshape(len(include_idx), 28 * 28)

        # add row of 1s to the dataset to act as an intercept term
        X = np.hstack((np.ones((len(include_idx), 1)), X))

        # pickle X and y
        pickle.dump(X, open(PICKLE_X_FULLPATH_TMP.format(type_), 'wb'))
        pickle.dump(y, open(PICKLE_Y_FULLPATH_TMP.format(type_), 'wb'))

    return X, y


if __name__ == '__main__':
    # load training dataset
    X_train, y_train = load_cached_data(type_='train')

    # train logistic regression classifier
    model = LogisticRegression(solver='bfgs')
    # from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X=X_train, y=y_train)

    # compute accuracy on training set
    predicted_train_digit = model.predict(X=X_train)
    correct = np.sum(y_train == predicted_train_digit)
    train_accuracy = correct / len(y_train)
    print('Train accuracy: ', train_accuracy)

    # compute accuracy on test set
    X_test, y_test = load_cached_data(type_='test')
    predicted_test_prices = model.predict(X=X_test)
    correct = np.sum(y_test == predicted_test_prices)
    test_accuracy = correct / len(y_test)
    print('Test accuracy: ', test_accuracy)
