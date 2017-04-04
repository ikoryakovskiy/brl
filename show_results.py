# -*- coding: utf-8 -*-

""" Auto Encoder of Value Functions."""

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
  size  = (125, 101, 3)
  dsize = (10, 10, 3)
  offset = size[0]*size[1]

  cmaes = CMAES(size, dsize, width = 0.4, kind = 'nrbf')

  q0 = load_grid_representation("policies/q_cfg_pendulum_sarsa_grid-it0-mp0-run0-nrbf-_experiment_agent_policy_representation.dat")
  f0 = np.fromfile("policies/f_cfg_pendulum_sarsa_grid-it0-mp0-run0-nrbf-_experiment_agent_policy_representation.dat")

  q0_ref = cmaes.evaluate(f0)

  csv_data = csv_read(["trajectories/pendulum_sarsa_grid_play-test-0.csv"])
  tr = load_trajectories(csv_data)


  see_by_layers(q0, tr, offset)
  see_by_layers(q0_ref, tr, offset)

  return

  p0 = calc_grid_policy(q0, (0, 1), (125, 101, 3))
  show_grid_representation(p0, (0, 1), (125, 101, 1))
  plt.scatter(tr[:,0], tr[:,1], c='w', s=40, marker='+')
  plt.waitforbuttonpress()

  p1 = calc_grid_policy(q1, (0, 1), (125, 101, 3))
  show_grid_representation(p1, (0, 1), (125, 101, 1))
  plt.scatter(tr[:,0], tr[:,1], c='w', s=40, marker='+')
  plt.waitforbuttonpress()

  dp = p0-p1
  show_grid_representation(dp, (0, 1), (125, 101, 1))
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