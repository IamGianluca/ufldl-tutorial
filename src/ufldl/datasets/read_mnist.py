from struct import unpack
import gzip
from numpy import zeros, uint8, float32


# full credit to Martin Thoma for this utility function
# https://martin-thoma.com/classify-mnist-with-pybrain/


def load_mnist(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.

    Args:
        imagefile [str]: The location of the image file.
        labelfile [str]: The location of the label file.
    Returns:
        Two [numpy.array] objects containing the images and labels. The images
        array contains n (28, 28) [numpy.array] objects. Pixels are organized
        row-wise. Pixel values are between 0 to 255. 0 means background
        (white), 255 means foreground (black).
    """
    # open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # read the binary data

    # we have to get big endian unsigned int. So we need '>I'

    # get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # get the data
    X = zeros((N, rows, cols), dtype=float32)  # initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                X[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return X, y
