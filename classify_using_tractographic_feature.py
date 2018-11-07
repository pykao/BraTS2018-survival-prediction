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

import matplotlib.pyplot as plt

import paths
import utils

from feature_extractor import get_weighted_connectivity_feature_vectors_train, get_weighted_connectivity_feature_vectors_valid, get_weighted_connectivity_feature_vectors_test


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="can be gt or predicted", default='gt', type=str)
args = parser.parse_args()


# setup logs
log = os.path.join(os.getcwd(), 'log_connectivity_ensemble_26_svm.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

# Loading Training
# mode = 'predicted'
mode = args.mode
logging.info('loading training set...')
pat_names_train, gt, W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features = get_weighted_connectivity_feature_vectors_train(mode=mode)


# Validation
logging.info('loading validation set...')
pat_names_valid, W_dsi_pass_histogram_features_valid, W_nrm_pass_histogram_features_valid, W_bin_pass_histogram_features_valid, W_dsi_end_histogram_features_valid, W_nrm_end_histogram_features_valid, W_bin_end_histogram_features_valid = get_weighted_connectivity_feature_vectors_valid()

# Testing 
logging.info('loading testing set...')
pat_names_test, W_dsi_pass_histogram_features_test, W_nrm_pass_histogram_features_test, W_bin_pass_histogram_features_test, W_dsi_end_histogram_features_test, W_nrm_end_histogram_features_test, W_bin_end_histogram_features_test =  get_weighted_connectivity_feature_vectors_test()


# Feature normalizations
logging.info('normalizing features...')
scaler = StandardScaler()
# Normalize Training Features
#normalized_W_dsi_pass_histogram_features = scaler.fit_transform(W_dsi_pass_histogram_features)
#normalized_W_nrm_pass_histogram_features = scaler.fit_transform(W_nrm_pass_histogram_features)
normalized_W_bin_pass_histogram_features = scaler.fit_transform(W_bin_pass_histogram_features)
#normalized_W_dsi_end_histogram_features = scaler.fit_transform(W_dsi_end_histogram_features)
#normalized_W_nrm_end_histogram_features = scaler.fit_transform(W_nrm_end_histogram_features)
#normalized_W_bin_end_histogram_features = scaler.fit_transform(W_bin_end_histogram_features)
# Normalize Validation features 
#normalized_W_dsi_pass_histogram_features_valid = scaler.fit_transform(W_dsi_pass_histogram_features_valid)
#normalized_W_nrm_pass_histogram_features_valid = scaler.fit_transform(W_nrm_pass_histogram_features_valid)
normalized_W_bin_pass_histogram_features_valid = scaler.fit_transform(W_bin_pass_histogram_features_valid)
#normalized_W_dsi_end_histogram_features_valid = scaler.fit_transform(W_dsi_end_histogram_features_valid)
#normalized_W_nrm_end_histogram_features_valid = scaler.fit_transform(W_nrm_end_histogram_features_valid)
#normalized_W_bin_end_histogram_features_valid = scaler.fit_transform(W_bin_end_histogram_features_valid)
# Normalize Testing features
#normalized_W_dsi_pass_histogram_features_test = scaler.fit_transform(W_dsi_pass_histogram_features_test)
#normalized_W_nrm_pass_histogram_features_test = scaler.fit_transform(W_nrm_pass_histogram_features_test)
normalized_W_bin_pass_histogram_features_test = scaler.fit_transform(W_bin_pass_histogram_features_test)
#normalized_W_dsi_end_histogram_features_test = scaler.fit_transform(W_dsi_end_histogram_features_test)
#normalized_W_nrm_end_histogram_features_test = scaler.fit_transform(W_nrm_end_histogram_features_test)
#normalized_W_bin_end_histogram_features_test = scaler.fit_transform(W_bin_end_histogram_features_test)

# Perforamce Feature Selection

# Remove features with low variance
logging.info('Remove features with low variance...')
sel = VarianceThreshold(0)

#sel.fit(normalized_W_dsi_pass_histogram_features)
#selected_normalized_W_dsi_pass_histogram_features = sel.transform(normalized_W_dsi_pass_histogram_features)
#selected_normalized_W_dsi_pass_histogram_features_valid = sel.transform(normalized_W_dsi_pass_histogram_features_valid)

#sel.fit(normalized_W_nrm_pass_histogram_features)
#selected_normalized_W_nrm_pass_histogram_features = sel.transform(normalized_W_nrm_pass_histogram_features)
#selected_normalized_W_nrm_pass_histogram_features_valid = sel.transform(normalized_W_nrm_pass_histogram_features_valid)

# W_bin_pass 
# 12
# 69.65%
sel.fit(normalized_W_bin_pass_histogram_features)
selected_normalized_W_bin_pass_histogram_features = sel.transform(normalized_W_bin_pass_histogram_features)
selected_normalized_W_bin_pass_histogram_features_valid = sel.transform(normalized_W_bin_pass_histogram_features_valid)
selected_normalized_W_bin_pass_histogram_features_test = sel.transform(normalized_W_bin_pass_histogram_features_test)
#aal_list = range(1,117)
#selected_aal_list = [i for idx, i in enumerate(aal_list) if sel.get_support()[idx]]
#print(len(selected_aal_list))


#sel.fit(normalized_W_dsi_end_histogram_features)
#selected_normalized_W_dsi_end_histogram_features = sel.transform(normalized_W_dsi_end_histogram_features)
#selected_normalized_W_dsi_end_histogram_features_valid = sel.transform(normalized_W_dsi_end_histogram_features_valid)

#sel.fit(normalized_W_nrm_end_histogram_features)
#selected_normalized_W_nrm_end_histogram_features = sel.transform(normalized_W_nrm_end_histogram_features)
#selected_normalized_W_nrm_end_histogram_features_valid = sel.transform(normalized_W_nrm_end_histogram_features_valid)

# W_dsi_end
# 3
#sel.fit(normalized_W_bin_end_histogram_features)
#selected_normalized_W_bin_end_histogram_features = sel.transform(normalized_W_bin_end_histogram_features)
#selected_normalized_W_bin_end_histogram_features_valid = sel.transform(normalized_W_bin_end_histogram_features_valid)
#selected_aal_list = [i for idx, i in enumerate(aal_list) if sel.get_support()[idx]]
#print(len(selected_aal_list))


# ============================================= Classification ========================================================= #

# five fold cross-validation
# repeat 1000 times
n_splits = 5
n_repeats = 1000
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=36851234)
scores_rskf_valid = np.zeros(n_splits*n_repeats,dtype=np.float32)
scores_rskf_train = np.zeros(n_splits*n_repeats,dtype=np.float32)

X = selected_normalized_W_bin_pass_histogram_features
# ground truth for classification task
y = np.copy(gt[:,0])

X_valid = selected_normalized_W_bin_pass_histogram_features_valid
y_valid_prob = np.zeros((28, 3), np.float64)
X_test = selected_normalized_W_bin_pass_histogram_features_test
y_test_prob = np.zeros((77, 3), np.float64)

estimator=svm.LinearSVC(max_iter=3000)
rfecv = RFECV(estimator, step=1, cv=rskf, scoring='accuracy', n_jobs = -1)
rfecv.fit(X, y)
X_rfecv = rfecv.transform(X)
X_valid_rfecv = rfecv.transform(X_valid)
X_test_rfecv = rfecv.transform(X_test)
assert(X_rfecv.shape[1] == X_valid_rfecv.shape[1] == X_test_rfecv.shape[1])
assert(X_rfecv.shape[0] == 59)
assert(X_valid_rfecv.shape[0] == 28)
assert(X_test_rfecv.shape[0] == 77)


logging.info('SVM Classifier, Optimal number of features: %d' % X_rfecv.shape[1])

idx = 0
for train_index, test_index in rskf.split(X_rfecv, y):

	X_train, X_test = X_rfecv[train_index], X_rfecv[test_index]
	y_train, y_test = y[train_index], y[test_index]
	# SVM classifier
	clf = svm.LinearSVC(max_iter=3000)
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	scores_rskf_valid[idx] = accuracy
	scores_rskf_train[idx] = clf.score(X_train, y_train)

	idx += 1

	prob_clf = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
	prob_clf.fit(X_train, y_train)
	y_v_prob = prob_clf.predict_proba(X_valid_rfecv)
	y_valid_prob += y_v_prob
	y_t_prob = prob_clf.predict_proba(X_test_rfecv)
	y_test_prob += y_t_prob

# ======= Plot ======== #

t = np.arange(0, n_splits*n_repeats)

plt.plot(t, scores_rskf_train, 'r-', scores_rskf_valid, 'b-')
plt.show()

svm_accuracy_train, svm_std_train = np.mean(scores_rskf_train), np.std(scores_rskf_train)
svm_accuracy_valid, svm_std_valid = np.mean(scores_rskf_valid), np.std(scores_rskf_valid)
logging.info("Best Scores of weighted tractographic features  - Using SVM - Training Accuracy: %0.4f (+/- %0.4f)" %(svm_accuracy_train, svm_std_train))
logging.info("Best Scores of weighted tractographic features  - Using SVM - Validation Accuracy: %0.4f (+/- %0.4f)" %(svm_accuracy_valid, svm_std_valid))


y_valid_pred = np.argmax(y_valid_prob, axis=1)
y_valid_pred_days = np.zeros(y_valid_pred.shape)
y_valid_pred_days[y_valid_pred==0] = 150
y_valid_pred_days[y_valid_pred==1] = 380
y_valid_pred_days[y_valid_pred==2] = 520

raw_data_valid = {}
raw_data_valid['name'] = pat_names_valid
raw_data_valid['days'] = y_valid_pred_days
df_valid = pd.DataFrame(raw_data_valid, columns=['name', 'days'])
df_valid.to_csv('survival_W_bin_pass_linearscv_prob_valid.csv', header=False, index=False)


y_test_pred = np.argmax(y_test_prob, axis=1)
y_test_pred_days = np.zeros(y_test_pred.shape)
y_test_pred_days[y_test_pred==0] = 150
y_test_pred_days[y_test_pred==1] = 380
y_test_pred_days[y_test_pred==2] = 520

raw_data_test = {}
raw_data_test['name'] = pat_names_test
raw_data_test['days'] = y_test_pred_days
df_test = pd.DataFrame(raw_data_test, columns=['name', 'days'])
df_test.to_csv('survival_W_bin_pass_linearscv_prob_test.csv', header=False, index=False)