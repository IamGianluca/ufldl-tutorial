import os
import pickle

from ufldl.datasets import load_mnist


DATA_PATH = './data'
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
        X = images.reshape(len(labels), 28 * 28)

        # flat outcome vector
        y = labels.ravel()

        # pickle X and y
        pickle.dump(X, open(PICKLE_X_FULLPATH_TMP.format(type_), 'wb'))
        pickle.dump(y, open(PICKLE_Y_FULLPATH_TMP.format(type_), 'wb'))

    return X, y
