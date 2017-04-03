# -*- coding: utf-8 -*-

""" Auto Encoder of Value Functions."""

from __future__ import division, print_function, absolute_import

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

# local
from anneal import *
from extra import *
from optimize import *
sys.path.append('/home/ivan/work/scripts/py')
from my_file.io import *
from my_plot.plot import *
from my_rl.gridworld import *
from my_csv.utils import *

def main():
  dim = (125, 101, 3)
  offset = dim[0]*dim[1]

  q0 = np.fromfile("policies/cfg_pendulum_sarsa_grid-it0-mp0-run0-_experiment_agent_policy_representation.dat")
  q1 = np.fromfile("policies/cfg_pendulum_sarsa_grid-it0-mp0-run0-v2-_experiment_agent_policy_representation.dat")

  csv_data = csv_read(["trajectories/pendulum_sarsa_grid_play-test-0.csv"])
  tr = load_trajectories(csv_data)


  see_by_layers(q0-q1, tr, offset)


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