# -*- coding: utf-8 -*-

""" RBF approximation of Value Functions."""

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
  #cma_test()
  #mp_cma_test()
  #return
  ##############################################

  # Import data
  n = 50
  size  = (125, 101, 3)
  dsize = (10, 10, 3)

  offset = size[0]*size[1]
  num = np.prod(size)

  train = np.zeros((n, num))
  for i in range(0, n):
    train[i] = load_grid_representation("data/cfg_pendulum_sarsa_grid-{:03d}-mp0-run0-_experiment_agent_policy_representation.dat".format(i))

  tm = train.mean(0)
  ts = train.std(0)
  tv2 = 2*np.maximum( num * [0.0001], train.var(0))

  save_grid_representation(tm, "policies/cfg_pendulum_sarsa_grid-init-mp0-run0-_experiment_agent_policy_representation.dat")
  ##############################################

  # Learning representation
  Q_init = tm

  Q_hat = mp_cma_run(args, Q_init, size, dsize)
  Q_hat.tofile("q_hat.bin")

  save_grid_representation(Q_hat, "policies/cfg_pendulum_sarsa_grid-it0-mp0-run0-v1-_experiment_agent_policy_representation.dat")

  for i in range(0, 3):
    show_grid_representation(Q_init[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
    show_grid_representation(Q_hat[offset*i:offset*(i+1)], (0, 1), (125, 101, 1))
  waitforbuttonpress()

  return

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
def mp_cma_run(args, Q_init, size, dsize):
  if (size[2] != dsize[2]):
    raise ValueError('CMAES::init Dimensions are not correct')

  offset = size[0]*size[1]
  actions = size[2]

  q_inits = []
  for i in range(actions):
    q_inits.append(Q_init[offset*i:offset*(i+1)])

  mp_size = (size[0], size[1], 1)
  mp_dsize = (dsize[0], dsize[1], 1)
  q_hats = do_multiprocessing_pool(args, q_inits, mp_size, mp_dsize)

  Q_hat = np.empty(Q_init.shape)
  for i in range(actions):
    Q_hat[offset*i:offset*(i+1)] = q_hats[i]
  return Q_hat

######################################################################################
def cma_test():
  size  = (125, 101, 1)
  dsize = (3, 2, 1)

  cmaes = CMAES(size, dsize)

  f_true = np.array([[0, 500, 0, 0, 0, -500]], dtype='float64')
  q_init_ref = cmaes.evaluate(f_true)
  q_init = np.copy(q_init_ref)

  cmaes = CMAES(size, dsize)
  f_init = cmaes.initial(q_init)
  print(f_init)
  f_hat = cmaes.optimize(q_init, f_init)
  q_hat = cmaes.evaluate(f_hat[0])
  print(f_hat[0], f_hat[1])

  print(cmaes.objective(np.array(f_true)))
  show_grid_representation(q_init, (0, 1), (size[0], size[1], 1))
  show_grid_representation(q_hat, (0, 1), (size[0], size[1], 1))

  waitforbuttonpress()

######################################################################################
def mp_cma_test(args):
  size  = (125, 101, 1)
  dsize = (3, 2, 1)

  cmaes = CMAES(size, dsize)

  f_trues = []
  f_trues.append(np.array([[-500, 500, -500, 500, -500, 500]], dtype='float64'))
  f_trues.append(np.array([[-500, 0, 0, 0, 0, 500]], dtype='float64'))

  q_inits = []
  for i in range(2):
    q_inits_ref = cmaes.evaluate(f_trues[i])
    q_inits.append(np.copy(q_inits_ref))

  q_hats = do_multiprocessing_pool(args, q_inits, size, dsize)

  for i in range(2):
    cmaes.q = q_inits[i]
    print(cmaes.objective(np.array(f_trues[i])))
    show_grid_representation(q_inits[i], (0, 1), (size[0], size[1], 1))
    show_grid_representation(q_hats[i], (0, 1), (size[0], size[1], 1))

  waitforbuttonpress()

######################################################################################
def mp_run(size, dsize, q_init):
  print(size, dsize)
  cmaes = CMAES(size, dsize)
  f_init = cmaes.initial(q_init)
  print(f_init)
  f_hat = cmaes.optimize(q_init, f_init)
  q_hat = cmaes.evaluate(f_hat[0])
  print(f_hat[0], f_hat[1])
  return q_hat

######################################################################################
def do_multiprocessing_pool(args, q_initss, size, dsize):
  """Do multiprocesing"""
  pool = multiprocessing.Pool(args.cores)
  func = partial(mp_run, size, dsize)
  res = pool.map(func, q_initss)
  pool.close()
  pool.join()
  return res
######################################################################################

if __name__ == "__main__":
    main()

