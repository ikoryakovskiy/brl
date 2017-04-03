"""A set of utils to work with csv files"""

import numpy as np

def get_header_size(fn):
  with open(fn) as f:
    for idx, line in enumerate(f):
        if "DATA:" in line:
             return idx+1

def csv_read(fn):
  for i, f in enumerate(fn):
    hd_sz = get_header_size(f)
    data = np.loadtxt(f, skiprows=hd_sz, delimiter=',')
  return data
