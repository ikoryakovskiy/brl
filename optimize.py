from anneal import *
import numpy as np
from extra import *

class OptimizerSA(Annealer):
    """Test annealer"""
    dim = (125, 101, 3)
    click = [(-1,0), (1,0), (0,-1), (0,1)]
    prior = 0
    prior_delta = 0
    smoothness = 0
    smoothness_delta = 0
    conditional = 0
    conditional_delta = 0
    updated_idx = 0
    updated_val = 0

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, targets, tm, ts, tv2):
        self.targets, self.tm, self.ts, self.tv2 = targets, tm, ts, tv2
        self.Tmax = 100.0     # Max (starting) temperature
        self.Tmin = 0.0000001     # Min (ending) temperature
        self.steps = 2000     # Number of iterations
        self.updates = 1      # Number of updates (by default an update prints to stdout)
        super(OptimizerSA, self).__init__(state)  # important! 
        #(i,j,k) = idx2ijk(np.prod(self.dim)-1)
        #idx = self.ijk2idx(i,j,k)
        #print("{} {} {} {}".format(idx, i, j ,k))
        self.num_targets = targets.shape[0]
        self.prior = self.calc_prior()
        print("Initial prior = {}".format(self.prior))
        self.smoothness = self.calc_smoothness()
        print("Initial smoothness = {}".format(self.smoothness))
        self.conditional = self.calc_conditional()
        print("Initial conditional = {}".format(self.conditional))
        self.updated_val = state[self.updated_idx]


    def move(self, idx):
        """Swaps two cities in the route."""
        #i = np.random.randint(0, len(self.tm))
        self.updated_idx = idx
        self.updated_val = self.state[idx]
        self.state[idx] = np.random.normal(self.state[idx], self.ts[idx])
        #(i,j,k) = idx2ijk(idx)
        #print("Move: {}: ({},{},{}) {} -> {}".format(idx, i, j, k, self.updated_val, self.state[idx]))
        
    def accepted(self):
        self.prior += self.prior_delta
        self.smoothness += self.smoothness_delta
        self.conditional += self.conditional_delta

    def energy(self):
        """Calculates the length of the route."""
        #prior = np.sum(np.divide(np.square(self.state - self.tm), self.tv2))
        s = self.tv2[self.updated_idx]
        old = self.f_prior(self.tm[self.updated_idx], self.updated_val, s)
        new = self.f_prior(self.tm[self.updated_idx], self.state[self.updated_idx], s)
        self.prior_delta = new - old        
        pr = self.prior_delta + self.prior
        #print("Prior difference {}".format(pr - self.calc_prior()))        

        (i,j,k) = idx2ijk(self.dim, self.updated_idx)
        self.smoothness_delta = 0;
        for cl in self.click:
            ni = i+cl[0]
            nj = j+cl[1]
            if (ni >= 0 and ni < self.dim[0] and nj >= 0 and nj < self.dim[1]):
                idx = ijk2idx(self.dim,ni,nj,k)
                old = self.f_smoothness(self.state[idx], self.updated_val)
                new = self.f_smoothness(self.state[idx], self.state[self.updated_idx])
                self.smoothness_delta += 2*(new-old)
        sm = self.smoothness_delta + self.smoothness
        #print("Smoothness difference {}".format(sm - self.calc_smoothness()))        
        '''
        self.conditional_delta = 0;
        t_idxs = np.nonzero(self.targets[:, 2] == k)[0]
        for t_idx in t_idxs:
            ti, tj, tk, tq = self.targets[t_idx, :]
            old = self.f_contitional(self.updated_val, tq, i-ti, j-tj)
            new = self.f_contitional(self.state[self.updated_idx], tq, i-ti, j-tj)
            self.conditional_delta += new-old
        cd = self.conditional_delta / self.num_targets + self.conditional
        #print("Conditional difference {}".format(cd - self.calc_conditional()))  
        '''
        e = pr + 0.001*sm #+ 1.0*cd

        return e
        
    def f_prior(self, a, b, s):
        #print(np.square(a-b), s, np.square(a-b) / s)
        return np.square(a-b) / s

    def f_smoothness(self, a, b):
        return np.minimum(abs(a-b), 20)
        
    def f_contitional(self, a, b, x, y):
        return abs(a-b) * np.exp(-(x*x+y*y)/10.0)
        
    def calc_prior(self):
        prior = np.sum(self.f_prior(self.state, self.tm, self.tv2))
        return prior
        
    def calc_smoothness(self):
        sm = 0
        for k in range(0, self.dim[2]): # for every action
            for i in range(0, self.dim[0]):
                for j in range(0, self.dim[1]):
                    for cl in self.click:
                        ni = i+cl[0]
                        nj = j+cl[1]
                        if (ni >= 0 and ni < self.dim[0] and nj >= 0 and nj < self.dim[1]):
                            v = self.state[ijk2idx(self.dim,i,j,k)]
                            nv = self.state[ijk2idx(self.dim, ni,nj,k)]
                            sm += self.f_smoothness(v, nv)
        return sm / self.num_targets

    def calc_conditional(self):
        cond = 0
        for k in range(0, self.dim[2]): # for every action
            #print("Targets {}".format(self.targets))
            t_idxs = np.nonzero(self.targets[:, 2] == k)[0]
            #print("Selected target idx {}".format(t_idxs))
            for i in range(0, self.dim[0]):
                for j in range(0, self.dim[1]):
                    idx = ijk2idx(self.dim, i,j,k)
                    if self.ts[idx] != 0:
                        for t_idx in t_idxs:
                            ti, tj, tk, tq = self.targets[t_idx, :]
                            v = self.f_contitional(self.state[idx], tq, i-ti, j-tj)
                            cond += v
                            
                            #if v > 1:
                            #    print(self.state[idx], tq, i-ti, j-tj, v)
        return cond   
        