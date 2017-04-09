# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:04:37 2017

@author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

# local
from cmaes import CMAES
from anneal import *
from extra import *
from optimize import *
from py.my_file.io import *
from py.my_plot.plot import *
from py.my_rl.gridworld import *
from py.my_csv.utils import *

def main():

  ######################################################################################
  test_compare_qf("cfg_pendulum_sarsa_grid-it0-mp0-run0-rbf2-test-_experiment_agent_policy_representation.dat")
  return
  ######################################################################################
  #diffq("cfg_pendulum_sarsa_grid-it0-mp0-run0-nrbf-test-_experiment_agent_policy_representation.dat",
  #      "cfg_pendulum_sarsa_grid-it0-mp0-run0-nrbf2-test-_experiment_agent_policy_representation.dat")

  Q = load_grid_representation("policies/q_cfg_pendulum_sarsa_grid-it0-mp0-run0-rbf-test-_experiment_agent_policy_representation.dat")
  Q_mean = import_data()[0]
  err = rmse(Q, Q_mean)

  print("RMSE = {}".format(err))

######################################################################################

def rmse(predictions, targets):
  return np.sqrt(((predictions - targets) ** 2).mean())

######################################################################################
def test_compare_qf(fname):
  size  = (125, 101, 3)
  dsize = (10, 10, 3)
  offset = size[0]*size[1]
  with CMAES(size, dsize, width = 0.4, kind = 'rbf') as cmaes:
    q0 = load_grid_representation("policies/q_{}".format(fname))
    f0 = np.fromfile("policies/f_{}".format(fname))

    q0_ref = cmaes.evaluate(f0)

    csv_data = csv_read(["trajectories/pendulum_sarsa_grid_rand_play-test-0.csv"])
    tr = load_trajectories(csv_data)

    see_by_layers(q0, tr, offset)
    see_by_layers(q0_ref, tr, offset)

    p0 = calc_grid_policy(q0, (0, 1), (125, 101, 3))
    show_grid_representation(p0, (0, 1), (125, 101, 1))
    plt.scatter(tr[:,0], tr[:,1], c='w', s=40, marker='+')
    plt.waitforbuttonpress()

######################################################################################
def diffq(fname0, fname1):
  size  = (125, 101, 3)
  dsize = (10, 10, 3)
  offset = size[0]*size[1]

  q0 = load_grid_representation("policies/q_{}".format(fname0))
  q1 = load_grid_representation("policies/q_{}".format(fname1))

  csv_data = csv_read(["trajectories/pendulum_sarsa_grid_play-test-0.csv"])
  tr = load_trajectories(csv_data)

  see_by_layers(q0-q1, tr, offset)

  p0 = calc_grid_policy(q0, (0, 1), (125, 101, 3))
  p1 = calc_grid_policy(q1, (0, 1), (125, 101, 3))
  show_grid_representation(p0-p1, (0, 1), (125, 101, 1))
  plt.scatter(tr[:,0], tr[:,1], c='w', s=40, marker='+')
  plt.waitforbuttonpress()

######################################################################################
def see_by_layers(q, tr, offset):
  for i in range(0, 3):
    show_grid_representation(q[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
    plt.scatter(tr[:,0], tr[:,1], c='k', s=40, marker='+')
    plt.waitforbuttonpress()

######################################################################################

if __name__ == "__main__":
  main()