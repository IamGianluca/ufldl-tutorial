import csv
from os.path import dirname
from os.path import join

import numpy as np


def load_housing():
    """Load housing dataset (regression)."""
    module_path = dirname(__file__)
    filename = 'data/housing.csv'
    fullpath = join(module_path, filename)

    data = []
    with open(fullpath, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for line in csvreader:
            data.append(line)

    # convert into numpy ndarray
    return np.array(data[1:], dtype=float)
