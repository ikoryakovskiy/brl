# -*- coding: utf-8 -*-

""" Extra """

from __future__ import division, print_function, absolute_import

import numpy as np
import math
from py.my_file.io import *

def load_trajectories(data):
#  print (data.shape)
  xx0 = discretize(data[:, 1], 125, (-math.pi, 2*math.pi))
  xd0 = discretize(data[:, 2], 101, (-12*math.pi, 12*math.pi))
  xx1 = discretize(data[:, 4], 125, (-math.pi, 2*math.pi))
  xd1 = discretize(data[:, 5], 101, (-12*math.pi, 12*math.pi))
  ud  = discretize(data[:, 7], 3, (-3, 3))
  r   = data[:, 8]
  t   = data[:, 9]
  return np.transpose(np.vstack([xx0, xd0, xx1, xd1, ud, r, t]))


def discretize(data, steps, bound):
  delta = (bound[1]-bound[0])/(steps-1);
#  print (data.shape)
#  print (delta)
#  print (np.round((data-bound[0])/delta))
  return np.round((data-bound[0])/delta).astype(np.int64)


def real_targets(tm, tr, gamma):
  dim = (125, 101)
  #print(type(tm))
  #print(tm.shape)
  #print (tr)
  tg = np.empty((0, 4))
  for record in range(0, tr.shape[0]):
    #print(tr[record, :])
    #print (record)
    x0  = tr[record, 0]
    xd0 = tr[record, 1]
    u0  = tr[record, 4]
    r   = tr[record, 5]
    t   = tr[record, 6]
    if (t == 0):
      # normal transition => next action is known
      x1  = tr[record, 2]
      xd1 = tr[record, 3]
      u1  = tr[record+1, 4]
      #print ("{}-{}-{}".format(x1, xd1, u1))
      target = r + gamma*tm[int(x1 + dim[0]*xd1 + np.prod(dim)*u1)]
    elif (t == 1):
      # terminal transition => next action is unknown => skip update
      continue
    elif (t == 2):
      # absorbing transition => next action is not needed => update with negative reward only
      target = r
    else:
      raise Exception('Unknown transition')
    value = tm[int(x0 + dim[0]*xd0 + np.prod(dim)*u0)]
    print(value, " -> ", target)
    tg = np.vstack((tg, [x0, xd0, u0, target]))
  return tg

def ijk2idx(dim, i, j, k):
    return i + dim[0]*j + dim[0]*dim[1]*k

def idx2ijk(dim, li):
    k = li // (dim[0]*dim[1])
    j = (li % (dim[0]*dim[1])) // dim[0]
    i = (li % (dim[0]*dim[1])) %  dim[0]
    return (i, j, k)

def import_data(n = 50, save_mean = 0):
  size  = (125, 101, 3)
  num = np.prod(size)

  data = np.zeros((n, num))
  for i in range(0, n):
    data[i] = load_grid_representation("data/cfg_pendulum_sarsa_grid-{:03d}-mp0-run0-_experiment_agent_policy_representation.dat".format(i))

  data_mean = data.mean(0)
  data_std = data.std(0)
  #tv2 = 2*np.maximum( num * [0.0001], data.var(0))

  if save_mean:
    save_grid_representation(data_mean, "policies/cfg_pendulum_sarsa_grid-init-mp0-run0-_experiment_agent_policy_representation.dat")
  return (data_mean, data_std)