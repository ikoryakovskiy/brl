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

lrepc = cdll.LoadLibrary('./librepc.so')

class CMAES(object):
    dnum = 0
    num = 0

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

        #print (locz)
        #cmane = (ctypes.c_char_p)
        csize = (ctypes.c_int * len(size))(*size)
        cdsize = (ctypes.c_int * len(dsize))(*dsize)
        #for i in range(0, len(size)): print csize[i]
        #for i in range(0, len(dsize)): print cdsize[i]
        clocx = locx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        clocy = locy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        clocz = locz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #print (np.ctypeslib.as_array((ctypes.c_double * dnum).from_address(ctypes.addressof(clocz.contents))))

        #self.obj = lrepc.tst_new()
        if kind == 'rbf':
          self.obj = lrepc.rbf_new(ctypes.c_char_p(name), csize, cdsize, ctypes.c_int(self.dnum), clocx, clocy, clocz, ctypes.c_double(sigma))
        else:
          self.obj = lrepc.nrbf_new(ctypes.c_char_p(name), csize, cdsize, ctypes.c_int(self.dnum), clocx, clocy, clocz, ctypes.c_double(sigma))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        lrepc.clear(self.obj)

    def initial(self, initial_guess):
        print ("Initial guess shape {}".format(initial_guess.shape))
        f_init = self.dnum * [1]
        for r in itertools.product(self.cx, self.cy, self.cz):
          #print (r)
          idx_i = int(np.round(r[0]*(self.size[0]-1)))
          idx_j = int(np.round(r[1]*(self.size[1]-1)))
          idx_k = int(r[2])

          click_size = 1
          idx_ii = range(idx_i-click_size, idx_i+click_size+1)
          idx_jj = range(idx_j-click_size, idx_j+click_size+1)
          guess = 0
          for dd in itertools.product(idx_ii, idx_jj):
            idx = int(dd[0] + dd[1]*self.size[0] + idx_k*self.size[0]*self.size[1])
            #print(dd[0], dd[1], idx_k, idx)
            guess += initial_guess[idx]
          guess = guess / ( (2*click_size+1)**2 )

          f_idx_i = int(np.round(r[0]*(self.dsize[0]-1)))
          f_idx_j = int(np.round(r[1]*(self.dsize[1]-1)))
          f_idx_k = int(r[2])
          f_idx = int(f_idx_i + f_idx_j*self.dsize[0] + f_idx_k*self.dsize[0]*self.dsize[1])
          #print(f_idx_i, f_idx_j, f_idx_k, f_idx)
          f_init[f_idx] = guess
        f_init = np.array(f_init, dtype='float64')
        return f_init

    def evaluate(self, feature):
        #print(feature.base)
        #print(feature.flags)
        feature = np.copy(feature)
        #print(feature.base)
        cfeature = feature.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #print('Evaluate::feature {}'.format(feature))
        #print('Evaluate::feature shape {}'.format(feature.shape))
        #print('Evaluate::cfeature {}'.format([ cfeature[i] for i in range(0, len(feature))]))
        # test if provided data is contiguous
        cf = [ cfeature[i] for i in range(0, len(feature))]
        z = [cf[i] != feature[i] for i in range(0, len(feature))]
        if any(z):
          raise ValueError('CMAES::evaluate Provided data is not contiguous')

        output = lrepc.rbf_evaluate(self.obj, cfeature)
        #print('Evaluate::feature2 {}'.format(feature))
        # provide a *reference* to a buffer in C library, no copy is done for speed reasons
        ArrayType = ctypes.c_double*self.num
        array_pointer = ctypes.cast(output, ctypes.POINTER(ArrayType))
        return np.frombuffer(array_pointer.contents)
        #return np.zeros([self.num,])

    def objective(self, x, *q_target):
        q_hat = self.evaluate(x)
        cost = np.linalg.norm(q_hat - q_target[0]) + 1*np.linalg.norm(x)
        return cost

    def optimize(self, q_target, f_init):
        #self.q = q

        opts = cma.CMAOptions()
        opts['verb_log'] = 0
        #opts['tolstagnation'] = 0
        #opts['maxiter'] = 3000

        print('Initial feature {}'.format(f_init))
        print('Initial feature shape {}'.format(f_init.shape))
        es = cma.CMAEvolutionStrategy(f_init, 1, opts) #self.dnum * [-500]
        es.optimize(self.objective, args = (q_target,))#, 50, 50)

        #print("\n\n")
        #print('termination by', es.stop())
        res = es.result()

        #q_hat_ref = self.evaluate(res[0])
        #print(res[0])
        #print(q_hat_ref)
        #fc = self.objective(res[0], q_target)
        #rc = np.linalg.norm(q_hat_ref - q_target) + 1*np.linalg.norm(res[0])
        #print("Feature cost {}, representation cost {}".format(fc, rc))
        #print("\n\n")

        es.stop()
        return res
