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

def main(args):

  ##############################################
  #rbf_test()
  #cma_test()
  #mp_cma_test(args)
  #return
  ##############################################

  if args.output_file is None:
    fname = "it1-rbf-_experiment_agent_policy_representation.dat"
  else:
    fname = args.output_file

  #Q_mean = import_data(save_mean = 1)[0]
  #learn_representation(args, Q_mean, "init-rbf-run0-_experiment_agent_policy_representation.dat")

  Q_current = load_grid_representation("policies/q_init-run0-_experiment_agent_policy_representation.dat")
  TR_targets = prepare_targets(Q_current, "q_init-run0.csv", 0.97)

  learn_representation(args, Q_current, TR_targets, "rbf-run1-_experiment_agent_policy_representation.dat")



def prepare_targets(Q, fname, gamma):
  csv_data = csv_read(["trajectories/{}".format(fname)])
  trajectories = load_trajectories(csv_data)

  targets = real_targets(Q, trajectories, gamma)
  return targets

def learn_representation(args, Q_current, TR_targets = None, fname = "deafult.dat"):
  size  = (125, 101, 3)
  dsize = (3, 3, 3)
  offset = size[0]*size[1]
  doffset = dsize[0]*dsize[1]
  width = 0.4
  if args.rbf:
    kind = 'rbf'
  elif args.nrbf:
    kind = 'nrbf'
  else:
    kind = 'rbf'

  Q_init = np.zeros(Q_current.size)
  start = time.time()
  Q_hat, F_hat = mp_cma_run(args, Q_current, Q_init, TR_targets, size, dsize, width, kind)
  print("Required time: {}".format(time.time() - start))

  # Saving
  Q_hat.tofile("policies/q_{}".format(fname))
  F_hat.tofile("policies/f_{}".format(fname))

  print("Saving complete")

  mp_size = (size[0], size[1], 1)
  mp_dsize = (dsize[0], dsize[1], 1)
  with CMAES(mp_size, mp_dsize, width, kind, name = 'plotting') as cmaes:
    for i in range(0, 1):
      q_init = Q_init[offset*i:offset*(i+1)]
      show_grid_representation(q_init, (0, 1), (125, 101, 1))
      q_current = Q_current[offset*i:offset*(i+1)]
      show_grid_representation(q_current, (0, 1), (125, 101, 1))
      q_hat = Q_hat[offset*i:offset*(i+1)]
      show_grid_representation(q_hat, (0, 1), (125, 101, 1))
      q_hat_ff = cmaes.evaluate(F_hat[doffset*i:doffset*(i+1)])
      show_grid_representation(q_hat_ff, (0, 1), (125, 101, 1))

      assert(q_hat == q_hat_ff).all()

      if TR_targets is not None:
        tr_idxs = np.nonzero(TR_targets[:, 2] == i)[0]
        tr_target = np.copy(TR_targets[tr_idxs, :])
        c_tr = cmaes.tr_cost(q_current, tr_target, size[0])
        print("Current tr cost {}".format(c_tr))
        c_tr = cmaes.tr_cost(q_hat, tr_target, size[0])
        print("Next tr cost {}".format(c_tr))

      waitforbuttonpress()

######################################################################################
def mp_cma_run(args, Q_current, Q_init, TR_targets, size, dsize, width = 0.4, kind = 'rbf'):
  if (size[2] != dsize[2]):
    raise ValueError('CMAES::init Dimensions are not correct')

  doffset = dsize[0]*dsize[1]
  dnum = np.product(dsize)
  offset = size[0]*size[1]
  actions = size[2]

  q_currents = []
  q_inits = []
  tr_targets = []
  for i in range(actions):
    q_currents.append(Q_current[offset*i:offset*(i+1)])
    q_inits.append(Q_init[offset*i:offset*(i+1)])
    if TR_targets is not None:
      tr_idxs = np.nonzero(TR_targets[:, 2] == i)[0]
      tr_targets.append(np.copy(TR_targets[tr_idxs, :]))
    else:
      tr_targets.append(None)

  mp_size = (size[0], size[1], 1)
  mp_dsize = (dsize[0], dsize[1], 1)
  res = do_multiprocessing_pool(args, q_currents, q_inits, tr_targets, mp_size, mp_dsize, width, kind)

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

  with CMAES(size, dsize, width = 0.4, kind = 'rbf') as cmaes:

    #f_init = 500*np.random.uniform(-1, 1, size=(1, dnum))
    f_init = np.ones([dnum,]) * 0
    f_init[0,] = -500
    f_init[1,] = 500
    q_init_ref = cmaes.evaluate(f_init)
    for i in range(1):
      show_grid_representation(q_init_ref[offset*i:offset*(i+1)], (0, 1), (size[0], size[1], 1))

    #q_init_ref.tofile("q_rbf_test.dat")
    #f_init.tofile("f_rbf_test.dat")

    waitforbuttonpress()

######################################################################################
def cma_test():
  size  = (125, 101, 1)
  dsize = (3, 2, 1)
  width = 0.4
  kind = 'rbf'

  f_true = np.array([0, 500, 0, 0, 0, -500], dtype='float64')
  #f_true = np.zeros((dsize[0]*dsize[1],), dtype='float64')

  with CMAES(size, dsize, width, kind, name='cma_test') as cmaes:
    q_current_ref = cmaes.evaluate(f_true)
    q_current = np.copy(q_current_ref)

    Q_current = np.tile(q_current, 3)
    TR_targets = prepare_targets(Q_current, "pendulum_sarsa_grid_rand_play-test-0.csv", 0.97)
    tr_idxs = np.nonzero(TR_targets[:, 2] == 0)[0]
    #tr_idxs = [tr_idxs[0]]
    tr_target = TR_targets[tr_idxs, :]

    f_init = np.zeros(f_true.shape)
    q_init_ref = cmaes.evaluate(f_init)
    q_init = np.copy(q_init_ref)

    c_tr = cmaes.tr_cost(q_init, tr_target, size[0])
    print("Initial tr cost {}".format(c_tr))

    f_hat = cmaes.optimize(q_current, f_init, tr_target)
    q_hat = cmaes.evaluate(f_hat)

    print(np.linalg.norm(q_hat - q_current) + 1*np.linalg.norm(f_hat))
    print(cmaes.objective(f_hat, q_current))

    show_grid_representation(q_init, (0, 1), (size[0], size[1], 1))
    show_grid_representation(q_current, (0, 1), (size[0], size[1], 1))
    show_grid_representation(q_hat, (0, 1), (size[0], size[1], 1))
    plt.scatter(tr_target[:,0], tr_target[:,1], c='k', s=40, marker='+')
    c_tr = cmaes.tr_cost(q_hat, tr_target, size[0])
    print("Final tr cost {}".format(c_tr))

    waitforbuttonpress()

######################################################################################
def mp_cma_test(args):
  size  = (125, 101, 1)
  num = np.prod(size)
  dsize = (3, 2, 1)
  width = 0.4
  pools = 2
  kind = 'rbf'

  Q_currents = []
  q_inits = []
  for i in range(pools):
    Q_current = np.zeros((num,1))
    for x in range(0, size[0]):
      for y in range(0, size[1]):
        cx = size[0]/2.0
        cy = size[1]/2.0
        Q_current[x+y*size[0]] = 6000-(x-cx)*(x-cx) + (y-cy)*(y-cy)
    Q_currents.append(Q_current)
    q_inits.append(np.zeros((num,1)))

  qf_hats = do_multiprocessing_pool(args, Q_currents, q_inits, size, dsize, width, kind)

  for i in range(pools):
    show_grid_representation(q_inits[i], (0, 1), (size[0], size[1], 1))
    show_grid_representation(Q_currents[i], (0, 1), (size[0], size[1], 1))
    qf_hat = qf_hats[i]
    q_hat = qf_hat[0]
    #f_hat = qf_hat[1]
    show_grid_representation(q_hat, (0, 1), (size[0], size[1], 1))

  waitforbuttonpress()

######################################################################################
def mp_run(q_currents, q_inits, tr_targets, size, dsize, width, kind, n):
  q_current = q_currents[n]
  q_init = q_inits[n]
  tr_target = tr_targets[n]
  th_name = multiprocessing.current_process().name
  with CMAES(size, dsize, width, kind, name = th_name) as cmaes:
    f_init = cmaes.initial(q_init)
    f_hat, cost0, cost1 = cmaes.optimize(q_current, f_init, tr_target)
    q_hat_ref = cmaes.evaluate(f_hat)
    q_hat = np.copy(q_hat_ref)
    cost0_ = np.array(cost0)
    cost1_ = np.array(cost1)
  print("\tInitial {} cost: {}, {}, {}".format(th_name, cost0_[0], cost0_[1], cost0_[2]))
  print("\tFinal {} cost: {}, {}, {}".format(th_name, cost1_[0], cost1_[1], cost1_[2]))
  return (q_hat, f_hat)

######################################################################################
def do_multiprocessing_pool(args, q_currents, q_inits, tr_targets, size, dsize, width, kind):
  """Do multiprocesing"""
  if (len(q_currents) != len(q_inits)):
    raise ValueError('bayes::do_multiprocessing_pool Input dimensions are not correct')
  if (len(q_currents) != len(tr_targets)):
    raise ValueError('bayes::do_multiprocessing_pool Input dimensions are not correct {tr_targets}')
  pool = multiprocessing.Pool(args.cores)
  func = partial(mp_run, q_currents, q_inits, tr_targets, size, dsize, width, kind)
  res = pool.map(func, range(0, len(q_currents)))
  pool.close()
  pool.join()
  return res
######################################################################################

if __name__ == "__main__":
  # parse arguments
  parser = argparse.ArgumentParser(description="Parser")
  parser.add_argument('-c', '--cores', type=int,
                      help='Maximum number of cores used by multiprocessing.pool')
  parser.add_argument("-o", "--output_file", help="Output file")
  parser.add_argument("-r", "--rbf", action='store_true', help="Use RBF in representation")
  parser.add_argument("-n", "--nrbf", action='store_true', help="Use NRBF in representation")
  args = parser.parse_args()

  print (args)

  if args.rbf and args.nrbf:
    parser.error("Select one representation type")

  # apply settings
  if args.cores:
      args.cores = min(multiprocessing.cpu_count(), args.cores)
  else:
      args.cores = min(multiprocessing.cpu_count(), 32)
  print('Using {} cores.'.format(args.cores))
  main(args)
