"""A set of utils to work with csv files"""

import numpy as np

def calc_grid_policy(data, field_dims, layout):
  if len(data) != np.prod(layout):
    raise Exception("Wrong input size")

  # dimentions of input array alnong which we will choose the best action
  squash = [i for i in range(0,len(layout)) if i not in field_dims]

  dim = [layout[i] for i in field_dims]
  print dim
  policy = np.zeros(np.prod(dim))
  for i in range(0, dim[0]):
    for j in range(0, dim[1]):
      for k in range(0, len(squash)): # for every action
        q = np.zeros(layout[squash[k]])
        for l in range(0, layout[squash[k]]): # for every possible value of this variable
          q[l] = data[i + dim[0]*j + np.prod(dim)*l]
        policy[i + dim[0]*j + np.prod(dim)*k] = np.argmax(q)
  return policy
