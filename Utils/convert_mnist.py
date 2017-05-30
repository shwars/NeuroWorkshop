#!/usr/bin/python

import numpy as np
import pickle

ftrain = '../Data/MNIST/train.csv'
fpick = '../Data/MNIST/mnist.pkl'

def mnist_lines(fn):
    with open(fn) as f:
        f.readline()
        return f.readlines()

tr_lines = np.array([ [ int(z) for z in x.split(',')] for x in mnist_lines(ftrain)])

MNIST = {}

MNIST['Train'] = {}
MNIST['Train']['Labels'] = tr_lines[:,0]
MNIST['Train']['Features'] = tr_lines[:,1:]

with open(fpick,'wb') as f:
    pickle.dump(MNIST,f)