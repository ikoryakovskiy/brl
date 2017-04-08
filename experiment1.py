# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:01:11 2017

@author: ivan
"""

import os

os.system("python bayes.py --rbf -o cfg_pendulum_sarsa_grid-it0-mp0-run0-rbf-test-_experiment_agent_policy_representation.dat")
os.system("python bayes.py --rbf -o cfg_pendulum_sarsa_grid-it0-mp0-run0-rbf2-test-_experiment_agent_policy_representation.dat")
os.system("python bayes.py --nrbf -o cfg_pendulum_sarsa_grid-it0-mp0-run0-nrbf-test-_experiment_agent_policy_representation.dat")
os.system("python bayes.py --nrbf -o cfg_pendulum_sarsa_grid-it0-mp0-run0-nrbf2-test-_experiment_agent_policy_representation.dat")