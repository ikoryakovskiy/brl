"""A set of utils to work with csv files"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def show_grid_representation(data, field_dims, layout):
  if len(data) != np.prod(layout):
    raise Exception("Wrong input size")

  # dimentions of input array alnong which we will choose the best action
  squash = [i for i in range(0,len(layout)) if i not in field_dims]

  m = np.zeros( [layout[i] for i in field_dims] )
  for i in range(0, m.shape[0]):
    for j in range(0, m.shape[1]):
      s = 0
      norm = 1
      for k in range(0, len(squash)): # for every squashing variable
        for l in range(0, layout[squash[k]]): # for every possible value of this variable
          #print i, j, l, i + m.shape[0]*j + np.prod(m.shape)*l
          s += data[i + m.shape[0]*j + np.prod(m.shape)*l]
        norm *= layout[squash[k]]
      m[i][j] = s / norm

  m = np.transpose(m)
  cax = plt.matshow(m, origin='lower')
  plt.colorbar(cax)

######################################################################################
def waitforbuttonpress():
  if (matplotlib.get_backend() != 'agg'):
    plt.waitforbuttonpress()