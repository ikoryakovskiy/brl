"""A set of utils to work with csv files"""

import sys
import numpy as np
from scipy.interpolate import interp1d
from collections import OrderedDict

sys.path.append('/home/ivan/work/scripts/py')
from my_stat.stat import mean_confidence_dd


###############################################################################
def get_header_size(fn):
  with open(fn) as f:
    for idx, line in enumerate(f):
      if "DATA:" in line:
        return idx+1
  return 0


###############################################################################
def conv(x):
  if x == 'nan':
    return np.nan
  else:
    print('x = '.format(x))
    return float(x)


###############################################################################
def csv_read(fn, verbose = False, **kwargs):
  if not isinstance(fn, (list, tuple)):
    fn = [fn]
  for i, f in enumerate(fn):
    if verbose:
      print('utils.csv_read: loading file {}'.format(f))
    hd_sz = get_header_size(f)
    data = np.loadtxt(f, skiprows=hd_sz, **kwargs)
    if verbose:
      print('utils.csv_read: loaded file {} successfully'.format(f))
  return data


###############################################################################
def subsample(dd, use_cols=None, sub=1, numel=None):
  be = []
  en = []
  times_len = 0
  for d in dd:
    be.append(d['ts'][0])
    en.append(d['ts'][-1])
    times_len += len(d['ts'])

  be = max(be)
  en = min(en)

  # Setup resampling strategy; we resample on average once per timestep.
  if sub is not None:
    t = np.linspace(be, en, times_len / (sub * len(dd)))
  elif numel is not None:
    t = np.linspace(be, en, numel)

  #if not isinstance(use_cols, (list, tuple)):
  #  use_cols = [use_cols]
  new = []
  tmp = {}
  for d in dd:

    # select columns
    if not use_cols:
      use_cols = list(d.keys())
      if 'ts' in use_cols:
        use_cols.remove('ts')
      else:
        raise Exception("'ts' is not found in the input")

    for i, s in enumerate(use_cols):
      finter = interp1d(d['ts'], d[s])
      tmp[s] = finter(t)
    tmp['ts'] = t
    new.append(tmp.copy())
  return new, len(t)


###############################################################################
def listsubsample(series, post='mean_confidence', use_cols=None, sub=1, numel=None, pre=''):

  def process(dd, ret, post, use_cols, pre, key=None):
    if not use_cols and post == 'mean_confidence':
      # ensure that input series all have same columns
      use_cols = set(dd[0].keys())
      for d in dd[1:]:
        if not use_cols == set(d.keys()):
          raise Exception("input data should have same columns for automatic processing")
      if 'ts' in use_cols:
        use_cols.remove('ts')
      else:
        raise Exception("'ts' is not found in the input")

    # actual subsamplong
    new_dd, xlen = subsample(dd, use_cols, sub, numel)

    # pre-processing
    if pre == 'diff':
        nnew_dd = []
        if not use_cols:
            use_cols = list(d.keys())
        for d in new_dd:
            new_d = {}
            dts = np.diff(d['ts'])
            for s in use_cols:
                ds = np.divide(np.diff(d[s]), dts)
                new_d[s] = ds
            new_d['ts'] = d['ts'][1:]
            nnew_dd.append(new_d)
        new_dd = nnew_dd
        xlen = xlen-1

    # post-processing
    if post == 'mean_confidence':
      mc = mean_confidence_dd(new_dd, use_cols, xlen)
      to_ret = mc
    else:
      to_ret = new_dd

    if key:
        ret[key] = to_ret
    else:
        ret.append(to_ret)
    return ret

  if type(series) is list:
    ret = []
    for dd in series:
      ret = process(dd, ret, post, use_cols, pre)
  elif type(series) is dict:
    ret = {}
    for key in series:
      ret = process(series[key], ret, post, use_cols, pre, key=key)
  elif type(series) is OrderedDict:
    ret = OrderedDict()
    for key in series:
      ret = process(series[key], ret, post, use_cols, pre, key=key)
  return ret