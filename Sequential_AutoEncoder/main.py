from __future__ import print_function, division
from keras.datasets import mnist
import math
import matplotlib.pyplot as plt 
import numpy as np 
import progressbar
from optimizers import Adam
from layers import Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization