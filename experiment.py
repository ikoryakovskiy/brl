# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:01:11 2017

@author: ivan
"""

import os
from bayes import *
from shutil import copyfile
import subprocess
import yaml, collections
from argparse import Namespace

######################################################################################

def remove_viz(conf):
  """Remove everything in conf related to visualization"""
  if "visualizer" in conf:
    del conf["visualizer"]
  if "visualization" in conf:
    del conf["visualization"]
  if "visualization2" in conf:
    del conf["visualization2"]
  return conf

def dict_representer(dumper, data):
  return dumper.represent_dict(data.iteritems())

def dict_constructor(loader, node):
  return collections.OrderedDict(loader.construct_pairs(node))

mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
yaml.add_representer(collections.OrderedDict, dict_representer)
yaml.add_constructor(mapping_tag, dict_constructor)

def read_cfg(cfg):
  """Read configuration file"""
  # check if file exists
  if os.path.isfile(cfg) == False:
    print 'File %s not found' % cfg
    sys.exit()

  # open configuration
  stream = file(cfg, 'r')
  conf = yaml.load(stream)
  stream.close()
  return conf

def write_cfg(outCfg, conf):
  """Write configuration file"""
  # create local yaml configuration file
  outfile = file(outCfg, 'w')
  yaml.dump(conf, outfile)
  outfile.close()

def edit_csv(icfg, ocfg, load_file, exporter):
  conf = read_cfg(icfg)
  conf = remove_viz(conf)
  conf["experiment"]["rate"] = 0
  conf["experiment"]["load_file"] = load_file
  conf["experiment"]["environment"]["exporter"]["file"] = exporter
  write_cfg(ocfg, conf)

######################################################################################

# Main loops

PATH_GRL = "/home/ivan/work/Project/Software/grl/qt-build"
PATH_BRL = "/home/ivan/work/Project/Software/brl"

dat_init = "policies/q_init-run0-_experiment_agent_policy_representation.dat"

TR = []
for i in range(1, 100):
  if i == 1:
    dat_run = dat_init
    TR.append("q_init-run{}.csv".format(i-1))
  else:
    dat_run = "policies/q_rbf-run{}-_experiment_agent_policy_representation.dat".format(i-1)
    TR.append("q_rbf-run{}.csv".format(i-1))

  o_file = "rbf-run{}-_experiment_agent_policy_representation.dat".format(i)
  csv = " ".join(TR)

  os.system("./cbayes --rbf --cores=3 --embed --dat_init={} --dat_run={} --output_file={} --csv {}".format(dat_init, dat_run, o_file, csv))

  # copy obtained value function to GRL folder
  dat_src = "{}/policies/q_rbf-run{}-_experiment_agent_policy_representation.dat".format(PATH_BRL, i)
  dat_dst = "{}/q_rbf-run{}-_experiment_agent_policy_representation.dat".format(PATH_GRL, i)
  copyfile(dat_src, dat_dst)

  # prepare GRL configs
  edit_csv("{}/cfg/pendulum/sarsa_grid_rand_play.yaml".format(PATH_GRL),
           "{}/sarsa_grid_rand_play.yaml".format(PATH_GRL),
           "q_rbf-run{}".format(i), "pendulum_sarsa_grid_rand_play_{}".format(i))

  edit_csv("{}/cfg/pendulum/sarsa_grid_play.yaml".format(PATH_GRL),
           "{}/sarsa_grid_play.yaml".format(PATH_GRL),
           "q_rbf-run{}".format(i), "pendulum_sarsa_grid_play_{}".format(i))

  # run GRL
  p = subprocess.Popen(["{}/grld".format(PATH_GRL), "sarsa_grid_rand_play.yaml"], cwd=PATH_GRL)
  p.wait()
  p = subprocess.Popen(["{}/grld".format(PATH_GRL), "sarsa_grid_play.yaml"], cwd=PATH_GRL)
  p.wait()

  # copy obtained trajectory back to BRL
  tr_src = "{}/pendulum_sarsa_grid_rand_play_{}-test-0.csv".format(PATH_GRL, i)
  tr_dst = "{}/trajectories/q_rbf-run{}.csv".format(PATH_BRL, i)
  copyfile(tr_src, tr_dst)


