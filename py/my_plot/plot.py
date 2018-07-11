"""A set of utils to work with csv files"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

def show_grid_representation(data, field_dims, layout, ax = None):
  if len(data) != np.prod(layout):
    raise Exception("Wrong input size")

  # dimentions of input array alnong which we will choose the best action
  squash = [i for i in range(0,len(layout)) if i not in field_dims]

  m = np.zeros( [layout[i] for i in field_dims] )
  for i in range(0, m.shape[0]):
    for j in range(0, m.shape[1]):
      s = 0
      norm = 1
      for k in range(0, len(squash)): # for every squashing variable
        for l in range(0, layout[squash[k]]): # for every possible value of this variable
          #print i, j, l, i + m.shape[0]*j + np.prod(m.shape)*l
          s += data[i + m.shape[0]*j + np.prod(m.shape)*l]
        norm *= layout[squash[k]]
      m[i][j] = s / norm

  plt_show = 0
  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt_show = 1
  fig = ax.get_figure()

  m = np.transpose(m)
  ms = ax.matshow(m, origin='lower')

  # create an axes on the right side of ax. The width of cax will be 5%
  # of ax and the padding between cax and ax will be fixed at 0.05 inch.
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)

  #np.seterr(divide='ignore', invalid='ignore')
  #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
  fig.colorbar(ms, cax=cax)
  #fig.colorbar(cax)

  numrows, numcols = m.shape
  def format_coord(x, y):
      col = int(x+0.5)
      row = int(y+0.5)
      if col>=0 and col<numcols and row>=0 and row<numrows:
          z = m[row,col]
          return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
          return 'x=%1.4f, y=%1.4f'%(x, y)

  ax.format_coord = format_coord
  if plt_show:
    plt.show()
  return ax


###############################################################################
def waitforbuttonpress():
  if (matplotlib.get_backend() != 'agg'):
    plt.waitforbuttonpress()


###############################################################################
def export_plot(filenamebase, export_area_drawing = True, export_latex = True):

  plt.savefig("{}_l.pdf".format(filenamebase))

  if export_latex:
    ead = ''
    if export_area_drawing:
      ead = '--export-area-drawing'

    el = '--export-latex'
    plt.savefig("{}.svg".format(filenamebase))
    cmd = ''' env -u LD_LIBRARY_PATH inkscape -T -f {0}.svg \
              --export-background-opacity=0 \
              {1} {2} --export-pdf={0}.pdf'''.format(filenamebase, ead, el)
    os.system(cmd)
    os.system('rm {}.svg'.format(filenamebase))


###############################################################################
def limits(df, spacing=0.1):
  space = df.max()-df.min()
  mmin = df.min()-spacing*space
  mmax = df.max()+spacing*space
  return [mmin, mmax]













