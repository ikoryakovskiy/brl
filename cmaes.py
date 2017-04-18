# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:04:37 2017

@author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
"""
import numpy as np
import ctypes
from ctypes import cdll
import itertools
import cma
from fastloop import get_conditional

lrepc = cdll.LoadLibrary('./librepc.so')

class CMAES(object):
  dnum = 0
  num = 0
  tr_target = None
  wlh = 1.0
  wp = 1.0
  wc = 1.0

  def __init__(self, size, dsize, width = 0.4, kind = 'rbf', name = 'default'):
    if (size[2] != dsize[2]):
      raise ValueError('CMAES::init Input dimensions are not correct')

    self.num = np.prod(size)
    self.size = size
    self.dnum = np.prod(dsize)
    self.dsize = dsize

    xlim = [-np.pi, 2*np.pi]
    ylim = [-12*np.pi, 12*np.pi]
    self.xoffset = xlim[0]
    self.yoffset = ylim[0]
    self.xsquash = 1.0 / (xlim[1] - xlim[0])
    self.ysquash = 1.0 / (ylim[1] - ylim[0])

    # centers are given for [0, 1] scale
    self.cx = np.linspace(1.0/(2.0*dsize[0]), 1.0 - 1.0/(2.0*dsize[0]), dsize[0])
    self.cy = np.linspace(1.0/(2.0*dsize[1]), 1.0 - 1.0/(2.0*dsize[1]), dsize[1])
    self.cz = np.linspace(0, dsize[2]-1, dsize[2])
    locx = []
    locy = []
    locz = []
    for r in itertools.product(self.cz, self.cy, self.cx):
      locx.append(r[2])
      locy.append(r[1])
      locz.append(r[0])
    locx = np.asarray(locx, dtype='float64')
    locy = np.asarray(locy, dtype='float64')
    locz = np.asarray(locz, dtype='float64')

    sigma = width * np.maximum(1.0/np.power(2*dsize[0], 0.5), 1.0/np.power(2*dsize[1], 0.5))
    print("Selected sigma {}".format(sigma))

    csize = (ctypes.c_int * len(size))(*size)
    cdsize = (ctypes.c_int * len(dsize))(*dsize)
    clocx = locx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    clocy = locy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    clocz = locz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if kind == 'rbf':
      self.obj = lrepc.rbf_new(ctypes.c_char_p(name), csize, cdsize, ctypes.c_int(self.dnum), clocx, clocy, clocz, ctypes.c_double(sigma))
    else:
      self.obj = lrepc.nrbf_new(ctypes.c_char_p(name), csize, cdsize, ctypes.c_int(self.dnum), clocx, clocy, clocz, ctypes.c_double(sigma))

  def __enter__(self):
    return self

  def __exit__(self, *args):
    lrepc.clear(self.obj)

  def initial(self, initial_guess):
    # re-convert input arrays
    initial_guess = np.reshape(initial_guess, (initial_guess.size,))
    f_init = self.dnum * [1]
    for r in itertools.product(self.cx, self.cy, self.cz):
      idx_i = int(np.round(r[0]*(self.size[0]-1)))
      idx_j = int(np.round(r[1]*(self.size[1]-1)))
      idx_k = int(r[2])

      click_size = 1
      idx_ii = range(idx_i-click_size, idx_i+click_size+1)
      idx_jj = range(idx_j-click_size, idx_j+click_size+1)
      guess = 0
      for dd in itertools.product(idx_ii, idx_jj):
        idx = int(dd[0] + dd[1]*self.size[0] + idx_k*self.size[0]*self.size[1])
        guess += initial_guess[idx]
      guess = guess / ( (2*click_size+1)**2 )

      f_idx_i = int(np.round(r[0]*(self.dsize[0]-1)))
      f_idx_j = int(np.round(r[1]*(self.dsize[1]-1)))
      f_idx_k = int(r[2])
      f_idx = int(f_idx_i + f_idx_j*self.dsize[0] + f_idx_k*self.dsize[0]*self.dsize[1])
      f_init[f_idx] = guess
    f_init = np.array(f_init, dtype='float64')
    return f_init

  def evaluate(self, feature):
    feature = np.copy(feature)
    cfeature = feature.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # test if provided data is contiguous
    cf = [ cfeature[i] for i in range(0, len(feature))]
    z = [cf[i] != feature[i] for i in range(0, len(feature))]
    if any(z):
      raise ValueError('CMAES::evaluate Provided data is not contiguous')

    output = lrepc.rbf_evaluate(self.obj, cfeature)

    # provide a *reference* to a buffer in C library, no copy is done for speed reasons
    ArrayType = ctypes.c_double*self.num
    array_pointer = ctypes.cast(output, ctypes.POINTER(ArrayType))
    return np.frombuffer(array_pointer.contents)

  def objective(self, f_hat, *args):
    q_data = args[0]

    q_hat = self.evaluate(f_hat)

    likelihood = np.linalg.norm(q_hat - q_data)
    prior = np.linalg.norm(f_hat)
    conditional = get_conditional(q_hat, self.size[0], self.size[1],
                                  self.tr_target_i, self.tr_target_q, 10.0)
    cost = self.wlh * likelihood + self.wp * prior + self.wc * conditional
    return cost

  def tr_cost(self, q, tr_target, height):
    cost = 0
    if tr_target is not None:
      for i in range(tr_target.shape[0]):
        ti = tr_target[i, 0]
        tj = tr_target[i, 1]
        tq = tr_target[i, 3]
        cost += (q[ti + tj*height] - tq)**2
      cost /= float(tr_target.shape[0])
    return cost

  def optimize(self, q_data, f_init, tr_target):

    # re-convert input arrays
    q_data = np.reshape(q_data, (q_data.size,))
    f_init = np.reshape(f_init, (f_init.size,))
    if tr_target is not None:
      self.tr_target_i = np.rint(tr_target[:, 0:3]).astype(int)
      self.tr_target_q = tr_target[:, 3]

    # settings
    opts = cma.CMAOptions()
    opts['verb_log'] = 0

    # actual run with
    es = cma.CMAEvolutionStrategy(f_init, 1, opts)
    es.optimize(self.objective, args = (q_data,)) # , 3, 3

    # finalize
    res = es.result()
    es.stop()

    # Calculate initial
    q_hat = self.evaluate(f_init)
    f_current = self.initial(q_data)

    likelihood = np.linalg.norm(q_hat - q_data)
    prior = np.linalg.norm(f_current)
    conditional = get_conditional(q_data, self.size[0], self.size[1],
                                  self.tr_target_i, self.tr_target_q, 10.0)
    costs0 = (self.wlh * likelihood, self.wp * prior, self.wc * conditional)

    # ... and final costs:
    f_hat = res[0]
    q_hat = self.evaluate(f_hat)

    likelihood = np.linalg.norm(q_hat - q_data)
    prior = np.linalg.norm(f_hat)
    conditional = get_conditional(q_hat, self.size[0], self.size[1],
                                  self.tr_target_i, self.tr_target_q, 10.0)

    costs1 = (self.wlh * likelihood, self.wp * prior, self.wc * conditional)

    np.testing.assert_almost_equal(sum(costs1), res[1], verbose=True)
    return res[0], costs0, costs1
