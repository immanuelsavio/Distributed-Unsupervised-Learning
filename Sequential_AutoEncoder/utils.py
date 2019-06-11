from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys

def make_diagonal(x):
    m = np.zeros((len(x),len(x)))
    for i in range (len(m[0])):
        m[i, i] = x[i]
    return m

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)