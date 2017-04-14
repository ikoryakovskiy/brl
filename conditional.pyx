from __future__ import division
import numpy as np
from libc.math cimport fabs
from libc.math cimport isnan
from libc.math cimport exp

from cython.parallel import prange, parallel

#cdef extern from "vfastexp.h":
#  double exp_approx "EXP" (double)

cimport numpy as np

DTYPE = np.double
ITYPE = np.int

ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t

#cimport cython
#cython: boundscheck=False, wraparound=False, nonecheck=False
def get_conditional(np.ndarray[DTYPE_t, ndim=1] q_hat, int height, int width,
              np.ndarray[ITYPE_t, ndim=2] tr_target_i,
              np.ndarray[DTYPE_t, ndim=1] tr_target_q,
              double sigma2):
  assert q_hat.dtype == DTYPE and tr_target_q.dtype == DTYPE
  assert tr_target_i.dtype == ITYPE
  assert tr_target_i.shape[0] == tr_target_q.shape[0]

  # dimention of tr
  cdef unsigned int tr_num = tr_target_i.shape[0]

  cdef double cond = 0, tq, a, e
  cdef unsigned int tr_idx, ti, tj, tk, i, j
  cdef int di, dj

  for tr_idx in prange(tr_num, nogil=True):
    ti = tr_target_i[tr_idx, 0]
    tj = tr_target_i[tr_idx, 1]
    tq = tr_target_q[tr_idx]
    #print(ti, tj, tq)
    for j in range(width):
      for i in range(height):
        di = i-ti
        dj = j-tj
        a = fabs(q_hat[i + j*height] - tq)
        e = exp(-(di**2+dj**2)/sigma2)
        cond += a * e
        #print(cond, i, j, a, e)
        #assert isnan(cond) == 0
  return cond
