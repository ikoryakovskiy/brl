# -*- coding: utf-8 -*-

"""
Created on Tue Mar 28 21:04:37 2017

RBF approximation of Value Functions

@author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
"""

from __future__ import division, print_function, absolute_import

#import tensorflow as tf
import numpy as np
import multiprocessing
import matplotlib as mpl
if (multiprocessing.cpu_count() > 4):
  mpl.use('agg')
import matplotlib.pyplot as plt
#import pickle
import argparse
from functools import partial
import time

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
  # parse arguments
  parser = argparse.ArgumentParser(description="Parser")
  parser.add_argument('-c', '--cores', type=int, help='specify maximum number of cores')
  args = parser.parse_args()
  if args.cores:
      args.cores = min(multiprocessing.cpu_count(), args.cores)
  else:
      args.cores = min(multiprocessing.cpu_count(), 32)
  print('Using {} cores.'.format(args.cores))

  ##############################################
  #rbf_test()
  #cma_test()
  #mp_cma_test(args)
  #return
  ##############################################

  # Import data
  n = 50
  size  = (125, 101, 3)
  offset = size[0]*size[1]
  num = np.prod(size)

  dsize = (3, 3, 3)
  doffset = dsize[0]*dsize[1]

  train = np.zeros((n, num))
  for i in range(0, n):
    train[i] = load_grid_representation("data/cfg_pendulum_sarsa_grid-{:03d}-mp0-run0-_experiment_agent_policy_representation.dat".format(i))

  tm = train.mean(0)
  ts = train.std(0)
  tv2 = 2*np.maximum( num * [0.0001], train.var(0))

  save_grid_representation(tm, "policies/cfg_pendulum_sarsa_grid-init-mp0-run0-_experiment_agent_policy_representation.dat")
  ##############################################

  # Learning representation
  width = 0.4
  kind = 'rbf'
  Q_target = tm
  Q_init = np.zeros(Q_target.size)

  Q_hat, F_hat = mp_cma_run(args, Q_target, Q_init, size, dsize, width, kind)

  fname = "cfg_pendulum_sarsa_grid-it0-mp0-run0-rbf2-test-_experiment_agent_policy_representation.dat"
  Q_hat.tofile("policies/q_{}".format(fname))
  F_hat.tofile("policies/f_{}".format(fname))

  print("Saving complete")

  mp_size = (size[0], size[1], 1)
  mp_dsize = (dsize[0], dsize[1], 1)
  with CMAES(mp_size, mp_dsize, width, kind, name = 'plotting') as cmaes:
    for i in range(0, 1):
      q_init = Q_init[offset*i:offset*(i+1)]
      show_grid_representation(q_init, (0, 1), (125, 101, 1))
      q_target = Q_target[offset*i:offset*(i+1)]
      show_grid_representation(q_target, (0, 1), (125, 101, 1))
      show_grid_representation(Q_hat[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
      q_hat_ff = cmaes.evaluate(F_hat[doffset*i:doffset*(i+1)])
      show_grid_representation(q_hat_ff, (0, 1), (125, 101, 1))

    waitforbuttonpress()

  return

def misc():
  csv_data = csv_read(["trajectories/pendulum_sarsa_grid_play-test-0.csv"])
  tr = load_trajectories(csv_data)

  targets = real_targets(tm, tr, 0.97)
  #targets = np.zeros([1, 4])
  #print("Targets ", targets)
  #print(targets.shape[0])


  #print(ts)
  tsx = np.maximum(ts, 0.0000001)
  init_state = tm#np.random.normal(tm, tsx)
  #show_grid_representation(tm, (0, 1), (125, 101, 3))
  #show_grid_representation(init_state, (0, 1), (125, 101, 3))
  #plt.waitforbuttonpress()
  print("Initial state", init_state)
  print("Min of the state {}".format(np.amin(init_state)))

  optimizer = OptimizerSA(init_state, targets, tm, ts, tv2)
  state, e = optimizer.anneal()
  print("Error {}".format(e))

  show_grid_representation(state, (0, 1), (125, 101, 3))
  plt.waitforbuttonpress()

  save_grid_representation(state, "policies/cfg_pendulum_sarsa_grid-it1-mp0-run0-_experiment_agent_policy_representation.dat")

  #for i in range(0, 3):
  #  show_grid_representation(state[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
  #  plt.waitforbuttonpress()



  #policy = calc_grid_policy(tm, (0, 1), (125, 101, 3))
  #show_grid_representation(policy, (0, 1), (125, 101, 1))
  #plt.waitforbuttonpress()

  #for i in range(0, 3):
  #  show_grid_representation(tm[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
  #  plt.waitforbuttonpress()
  #  show_grid_representation(tv[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
  #  plt.waitforbuttonpress()

######################################################################################
def mp_cma_run(args, Q_target, Q_init, size, dsize, width = 0.4, kind = 'rbf'):
  if (size[2] != dsize[2]):
    raise ValueError('CMAES::init Dimensions are not correct')

  doffset = dsize[0]*dsize[1]
  dnum = np.product(dsize)
  offset = size[0]*size[1]
  actions = size[2]

  q_targets = []
  q_inits = []
  for i in range(actions):
    q_targets.append(Q_target[offset*i:offset*(i+1)])
    q_inits.append(Q_init[offset*i:offset*(i+1)])

  mp_size = (size[0], size[1], 1)
  mp_dsize = (dsize[0], dsize[1], 1)
  res = do_multiprocessing_pool(args, q_targets, q_inits, mp_size, mp_dsize, width, kind)

  Q_hat = np.empty(Q_init.shape)
  F_hat = np.empty(dnum, )
  for i in range(actions):
    Q_hat[offset*i:offset*(i+1)] = res[i][0]
    F_hat[doffset*i:doffset*(i+1)] = res[i][1]
  return (Q_hat, F_hat)

######################################################################################
def rbf_test():
  size  = (125, 101, 3)
  dsize = (10, 10, 3)
  dnum = np.prod(dsize)
  offset = size[0]*size[1]

  cmaes = CMAES(size, dsize, width = 0.4, kind = 'rbf')

  #f_init = 500*np.random.uniform(-1, 1, size=(1, dnum))
  f_init = np.ones([1, dnum]) * 0
  f_init[0, 0] = -500
  #f_init[0, 1] = 500
  q_init_ref = cmaes.evaluate(f_init)
  for i in range(3):
    show_grid_representation(q_init_ref[offset*i:offset*(i+1)], (0, 1), (size[0], size[1], 1))

  q_init_ref.tofile("q_rbf_test.dat")
  f_init.tofile("f_rbf_test.dat")

  waitforbuttonpress()

######################################################################################
def cma_test():
  size  = (125, 101, 1)
  dsize = (3, 2, 1)
  width = 0.4
  kind = 'rbf'

  f_true = np.array([0, 500, 0, 0, 0, -500], dtype='float64')

  with CMAES(size, dsize, width, kind, name='cma_test') as cmaes:
    q_target_ref = cmaes.evaluate(f_true)
    q_target = np.copy(q_target_ref)

    f_init = np.zeros(f_true.shape)
    q_init_ref = cmaes.evaluate(f_init)
    q_init = np.copy(q_init_ref)

    f_hat = cmaes.optimize(q_target, f_init)
    q_hat = cmaes.evaluate(f_hat[0])

    print(np.linalg.norm(q_hat - q_target) + 1*np.linalg.norm(f_hat[0]))
    print(cmaes.objective(f_hat[0], q_target))

    show_grid_representation(q_init, (0, 1), (size[0], size[1], 1))
    show_grid_representation(q_target, (0, 1), (size[0], size[1], 1))
    show_grid_representation(q_hat, (0, 1), (size[0], size[1], 1))

    waitforbuttonpress()

######################################################################################
def mp_cma_test(args):
  size  = (125, 101, 1)
  num = np.prod(size)
  dsize = (3, 2, 1)
  width = 0.4
  pools = 2
  kind = 'rbf'

  q_targets = []
  q_inits = []
  for i in range(pools):
    q_target = np.zeros((num,1))
    for x in range(0, size[0]):
      for y in range(0, size[1]):
        cx = size[0]/2.0
        cy = size[1]/2.0
        q_target[x+y*size[0]] = 6000-(x-cx)*(x-cx) + (y-cy)*(y-cy)
    q_targets.append(q_target)
    q_inits.append(np.zeros((num,1)))

  qf_hats = do_multiprocessing_pool(args, q_targets, q_inits, size, dsize, width, kind)

  for i in range(pools):
    show_grid_representation(q_inits[i], (0, 1), (size[0], size[1], 1))
    show_grid_representation(q_targets[i], (0, 1), (size[0], size[1], 1))
    qf_hat = qf_hats[i]
    q_hat = qf_hat[0]
    #f_hat = qf_hat[1]
    show_grid_representation(q_hat, (0, 1), (size[0], size[1], 1))

  waitforbuttonpress()

######################################################################################
def mp_run(q_targets, q_inits, size, dsize, width, kind, n):
  q_target = q_targets[n]
  q_init = q_inits[n]
  th_name = multiprocessing.current_process().name
  with CMAES(size, dsize, width, kind, name = th_name) as cmaes:
    f_init = cmaes.initial(q_init)
    f_hat = cmaes.optimize(q_target, f_init)
    q_hat_ref = cmaes.evaluate(f_hat[0])
    q_hat = np.copy(q_hat_ref)
    return (q_hat, f_hat[0])

######################################################################################
def do_multiprocessing_pool(args, q_targets, q_inits, size, dsize, width, kind):
  """Do multiprocesing"""
  if (len(q_targets) != len(q_inits)):
    raise ValueError('bayes::do_multiprocessing_pool Input dimensions are not correct')
  pool = multiprocessing.Pool(args.cores)
  func = partial(mp_run, q_targets, q_inits, size, dsize, width, kind)
  res = pool.map(func, range(0, len(q_targets)))
  pool.close()
  pool.join()
  return res
######################################################################################

if __name__ == "__main__":
    main()

