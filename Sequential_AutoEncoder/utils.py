from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import progressbar
import math
import sys


def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

def make_diagonal(x):
    m = np.zeros((len(x),len(x)))
    for i in range (len(m[0])):
        m[i, i] = x[i]
    return m

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0)/len(y_true)
    return accuracy