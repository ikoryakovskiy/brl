"""A set of utils to work with csv files"""

import numpy as np

def load_grid_representation(filename):
  with open(filename, 'rb') as f:
    data = np.fromfile(f, dtype=np.float64)
  return data


def save_grid_representation(data, filename):
  data.tofile(filename)

