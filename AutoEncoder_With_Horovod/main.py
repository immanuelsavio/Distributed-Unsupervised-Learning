from __future__ import print_function
import sys
import argparse

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import tensorflow as tf
import horovod.tensorflow as hvd 
try:
    hvd.init()
except Exception:
    raise
from keras.layers import Input
from keras.models import Model
from keras import backend as K 
from keras.datasets import mnist
from keras.optimizers import SGD
from vae_common import CustomFormatter, make_shared_layers_dict, make_vae, get_encoded, get_decoded
Dataset = tf.data.Dataset
