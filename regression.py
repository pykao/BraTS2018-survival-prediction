import os
import sklearn
import logging
import csv
import argparse

import pandas as pd
import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

import paths
import utils

from classify_using_tractographic_feature import get_weighted_connectivity_feature_vectors_train


# setup logs
log = os.path.join(os.getcwd(), 'log_regression.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

logging.info('loading training set...')

pat_names_train, gt, W_dsi_pass, W_nrm_pass, W_bin_pass, W_dsi_end, W_nrm_end, W_bin_end = get_weighted_connectivity_feature_vectors_train(mode='gt', region='roi')
