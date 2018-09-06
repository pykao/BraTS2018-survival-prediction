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


def binarize_connectivity_matrix(connectivity_matrix, threshold=0.01):
    ''' binarize the matrix '''
    binary_connectivity_matrix = np.zeros(connectivity_matrix.shape, dtype=np.float)
    #print threshold*np.amax(connectivity_matrix)
    binary_connectivity_matrix[connectivity_matrix >= threshold*np.amax(connectivity_matrix)] = 1
    return binary_connectivity_matrix

def normalize_conncetivity_matrix(connectivity_matrix):
    ''' normalize the connectivity matrix'''
    normalized_connectivity_matrix = np.copy(connectivity_matrix)
    return normalized_connectivity_matrix/np.amax(connectivity_matrix)

def threshold_connectivity_matrix(connectivity_matrix, threshold=0.01):
    ''' threshold the connectiivty matrix in order to remove the noise'''
    thresholded_connectivity_matrix= np.copy(connectivity_matrix)
    thresholded_connectivity_matrix[connectivity_matrix <= threshold*np.amax(connectivity_matrix)] = 0
    return thresholded_connectivity_matrix

def weight_conversion(W):
    ''' convert to the normalized version and binary version'''
    W_bin = np.copy(W)
    W_bin[W!=0]=1
    W_nrm = np.copy(W)
    W_nrm = W_nrm/np.amax(np.absolute(W))
    return W_nrm, W_bin

def get_pat_name(pat_dir):
    ''' get the patient's name'''
    temp = os.path.split(pat_dir)[1]
    return temp[:temp.find('_whole_tumor')]

def get_lesion_weights(whole_tumor_mni_path):
    ''' get the weight vector'''
    #print(whole_tumor_mni_path)
    aal_path = os.path.join(paths.dsi_studio_path, 'atlas', 'aal.nii.gz')
    aal_nda = utils.ReadImage(aal_path)
    aal_182_218_182 = utils.reshape_by_padding_upper_coords(aal_nda, (182,218,182), 0)
    whole_tumor_mni_nda = utils.ReadImage(whole_tumor_mni_path)
    weights = np.zeros(int(np.amax(aal_182_218_182)), dtype=float)
    for bp_number in range(int(np.amax(aal_182_218_182))):
	    mask = np.zeros(aal_182_218_182.shape, aal_182_218_182.dtype)
	    mask[aal_182_218_182==(bp_number+1)]=1
	    bp_size = float(np.count_nonzero(mask))
	    whole_tumor_in_bp = np.multiply(mask, whole_tumor_mni_nda)
	    whole_tumor_in_bp_size = float(np.count_nonzero(whole_tumor_in_bp))
	    weights[bp_number] = whole_tumor_in_bp_size/bp_size
    return weights

def get_weighted_connectivity_feature_vectors_test(dsi_studio_path=paths.dsi_studio_path):
	connectivity_testing_dir = os.path.join(dsi_studio_path, 'connectivity', 'testing')
	whole_tumor_mni_testing_dir = os.path.join(dsi_studio_path, 'predicted_whole_tumor', 'testing')

	connectivity_pass_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_testing_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'pass' in name and name.endswith('.mat')]
	connectivity_pass_files.sort()

	connectivity_end_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_testing_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'end' in name and name.endswith('.mat')]
	connectivity_end_files.sort()

	whole_tumor_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(whole_tumor_mni_testing_dir) for name in files if 'whole_tumor' in name and 'MNI152_1mm' in name and name.endswith('nii.gz')]
	whole_tumor_mni_paths.sort()

	assert(len(connectivity_pass_files) == len(connectivity_end_files) == len(whole_tumor_mni_paths)==77)

	W_dsi_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_nrm_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_bin_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)

	W_dsi_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_nrm_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_bin_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	pat_names=[]
	for idx, (connectivity_pass_file, connectivity_end_file, whole_tumor_mni_path) in enumerate(zip(connectivity_pass_files, connectivity_end_files, whole_tumor_mni_paths)):

	    assert(get_pat_name(connectivity_pass_file)==get_pat_name(connectivity_end_file))
	    assert(get_pat_name(connectivity_pass_file) in whole_tumor_mni_path)
	    pat_name = get_pat_name(connectivity_pass_file)
	    pat_names.append(pat_name)
	    #lesion weights
	    lesion_weights = get_lesion_weights(whole_tumor_mni_path)

	    connectivity_matrix_pass_obj = loadmat(connectivity_pass_file)
	    weighted_connectivity_matrix_pass_temp = connectivity_matrix_pass_obj['connectivity']
	    weighted_connectivity_matrix_pass = threshold_connectivity_matrix(weighted_connectivity_matrix_pass_temp, 0)
	    W_nrm_pass, W_bin_pass = weight_conversion(weighted_connectivity_matrix_pass)

	    connectivity_matrix_end_obj = loadmat(connectivity_end_file)
	    weighted_connectivity_matrix_end_temp = connectivity_matrix_end_obj['connectivity']
	    weighted_connectivity_matrix_end = threshold_connectivity_matrix(weighted_connectivity_matrix_end_temp, 0)
	    W_nrm_end, W_bin_end = weight_conversion(weighted_connectivity_matrix_end)

	    # weighted connectivity histogram
	    W_dsi_pass_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_matrix_pass, axis=0), lesion_weights)
	    W_nrm_pass_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_pass, axis=0), lesion_weights)
	    W_bin_pass_histogram_features[idx, :] = np.multiply(np.sum(W_bin_pass, axis=0), lesion_weights)

	    W_dsi_end_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_matrix_end, axis=0), lesion_weights)
	    W_nrm_end_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_end, axis=0), lesion_weights)
	    W_bin_end_histogram_features[idx, :] = np.multiply(np.sum(W_bin_end, axis=0), lesion_weights)
	return pat_names , W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features


def get_weighted_connectivity_feature_vectors_valid(dsi_studio_path=paths.dsi_studio_path):
	connectivity_valid_dir = os.path.join(dsi_studio_path, 'connectivity', 'validation')
	whole_tumor_mni_valid_dir = os.path.join(dsi_studio_path, 'predicted_whole_tumor', 'validation')

	connectivity_pass_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_valid_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'pass' in name and name.endswith('.mat')]
	connectivity_pass_files.sort()

	connectivity_end_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_valid_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'end' in name and name.endswith('.mat')]
	connectivity_end_files.sort()

	whole_tumor_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(whole_tumor_mni_valid_dir) for name in files if 'whole_tumor' in name and 'MNI152_1mm' in name and name.endswith('nii.gz')]
	whole_tumor_mni_paths.sort()

	assert(len(connectivity_pass_files) == len(connectivity_end_files) == len(whole_tumor_mni_paths)==28)

	W_dsi_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_nrm_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_bin_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)

	W_dsi_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_nrm_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_bin_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	pat_names=[]
	for idx, (connectivity_pass_file, connectivity_end_file, whole_tumor_mni_path) in enumerate(zip(connectivity_pass_files, connectivity_end_files, whole_tumor_mni_paths)):

	    assert(get_pat_name(connectivity_pass_file)==get_pat_name(connectivity_end_file))
	    assert(get_pat_name(connectivity_pass_file) in whole_tumor_mni_path)
	    pat_name = get_pat_name(connectivity_pass_file)
	    pat_names.append(pat_name)
	    #lesion weights
	    lesion_weights = get_lesion_weights(whole_tumor_mni_path)

	    connectivity_matrix_pass_obj = loadmat(connectivity_pass_file)
	    weighted_connectivity_matrix_pass_temp = connectivity_matrix_pass_obj['connectivity']
	    weighted_connectivity_matrix_pass = threshold_connectivity_matrix(weighted_connectivity_matrix_pass_temp, 0)
	    W_nrm_pass, W_bin_pass = weight_conversion(weighted_connectivity_matrix_pass)

	    connectivity_matrix_end_obj = loadmat(connectivity_end_file)
	    weighted_connectivity_matrix_end_temp = connectivity_matrix_end_obj['connectivity']
	    weighted_connectivity_matrix_end = threshold_connectivity_matrix(weighted_connectivity_matrix_end_temp, 0)
	    W_nrm_end, W_bin_end = weight_conversion(weighted_connectivity_matrix_end)

	    # weighted connectivity histogram
	    W_dsi_pass_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_matrix_pass, axis=0), lesion_weights)
	    W_nrm_pass_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_pass, axis=0), lesion_weights)
	    W_bin_pass_histogram_features[idx, :] = np.multiply(np.sum(W_bin_pass, axis=0), lesion_weights)

	    W_dsi_end_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_matrix_end, axis=0), lesion_weights)
	    W_nrm_end_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_end, axis=0), lesion_weights)
	    W_bin_end_histogram_features[idx, :] = np.multiply(np.sum(W_bin_end, axis=0), lesion_weights)
	return pat_names , W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features

def get_weighted_connectivity_feature_vectors_train(dsi_studio_path=paths.dsi_studio_path, mode='gt'):
	# Loading the survival dataset
	survival_dataset = utils.load_survival_training_dataset()
	if mode == 'gt':
		connectivity_train_dir = os.path.join(dsi_studio_path, 'connectivity', 'gt')
		whole_tumor_mni_train_dir = os.path.join(dsi_studio_path, 'gt_whole_tumor')
	if mode == 'predicted':
		connectivity_train_dir = os.path.join(dsi_studio_path, 'connectivity', 'training')
		whole_tumor_mni_train_dir = os.path.join(dsi_studio_path, 'predicted_whole_tumor', 'training')


	connectivity_pass_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_train_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'pass' in name and name.endswith('.mat')]
	connectivity_pass_files.sort()

	connectivity_end_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_train_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'end' in name and name.endswith('.mat')]
	connectivity_end_files.sort()

	whole_tumor_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(whole_tumor_mni_train_dir) for name in files if 'whole_tumor' in name and 'MNI152_1mm' in name and name.endswith('nii.gz')]
	whole_tumor_mni_paths.sort()

	assert(len(connectivity_pass_files) == len(connectivity_end_files) == len(whole_tumor_mni_paths)==59)
	pat_names = []
	gt = np.zeros((len(connectivity_pass_files),2), dtype = np.float32)

	W_dsi_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_nrm_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_bin_pass_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)

	W_dsi_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_nrm_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)
	W_bin_end_histogram_features = np.zeros((len(connectivity_pass_files), 116), dtype=np.float32)

	for idx, (connectivity_pass_file, connectivity_end_file, whole_tumor_mni_path) in enumerate(zip(connectivity_pass_files, connectivity_end_files, whole_tumor_mni_paths)):

	    assert(get_pat_name(connectivity_pass_file)==get_pat_name(connectivity_end_file))
	    assert(get_pat_name(connectivity_pass_file) in whole_tumor_mni_path)
	    pat_name = get_pat_name(connectivity_pass_file)
	    pat_names.append(pat_name)

	    # short 
	    if int(survival_dataset[pat_name]['survival']) < 305:
	        gt[idx, 0] = 0
	        gt[idx, 1] = int(survival_dataset[pat_name]['survival'])
	        #short_period += 1 
	    # long should be 454 or 456.25
	    elif int(survival_dataset[pat_name]['survival']) > 456:
	        gt[idx, 0] = 2
	        gt[idx, 1] = int(survival_dataset[pat_name]['survival'])
	        #long_period += 1
	    # mid
	    else:
	        gt[idx, 0] = 1
	        gt[idx, 1] = int(survival_dataset[pat_name]['survival'])

	    lesion_weights = get_lesion_weights(whole_tumor_mni_path)

	    connectivity_matrix_pass_obj = loadmat(connectivity_pass_file)
	    weighted_connectivity_matrix_pass_temp = connectivity_matrix_pass_obj['connectivity']
	    weighted_connectivity_matrix_pass = threshold_connectivity_matrix(weighted_connectivity_matrix_pass_temp, 0)
	    W_nrm_pass, W_bin_pass = weight_conversion(weighted_connectivity_matrix_pass)

	    connectivity_matrix_end_obj = loadmat(connectivity_end_file)
	    weighted_connectivity_matrix_end_temp = connectivity_matrix_end_obj['connectivity']
	    weighted_connectivity_matrix_end = threshold_connectivity_matrix(weighted_connectivity_matrix_end_temp, 0)
	    W_nrm_end, W_bin_end = weight_conversion(weighted_connectivity_matrix_end)


	    # weighted connectivity histogram
	    W_dsi_pass_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_matrix_pass, axis=0), lesion_weights)
	    W_nrm_pass_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_pass, axis=0), lesion_weights)
	    W_bin_pass_histogram_features[idx, :] = np.multiply(np.sum(W_bin_pass, axis=0), lesion_weights)

	    W_dsi_end_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_matrix_end, axis=0), lesion_weights)
	    W_nrm_end_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_end, axis=0), lesion_weights)
	    W_bin_end_histogram_features[idx, :] = np.multiply(np.sum(W_bin_end, axis=0), lesion_weights)
	return pat_names, gt, W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features

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
scores_rskf = np.zeros(n_splits*n_repeats,dtype=np.float32)


X = selected_normalized_W_bin_pass_histogram_features
# ground truth for classification task
y = np.copy(gt[:,0])

X_valid = selected_normalized_W_bin_pass_histogram_features_valid
y_valid_prob = np.zeros((28, 3), np.float64)
X_test = selected_normalized_W_bin_pass_histogram_features_test
y_test_prob = np.zeros((77, 3), np.float64)

estimator=svm.LinearSVC()
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
	clf = svm.LinearSVC()
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	scores_rskf[idx] = accuracy
	idx += 1

	prob_clf = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
	prob_clf.fit(X_train, y_train)
	y_v_prob = prob_clf.predict_proba(X_valid_rfecv)
	y_valid_prob += y_v_prob
	y_t_prob = prob_clf.predict_proba(X_test_rfecv)
	y_test_prob += y_t_prob



svm_accuracy, svm_std = np.mean(scores_rskf), np.std(scores_rskf)
logging.info("Best Scores of Weighted connectivity matrix  - Using SVM - Accuracy: %0.4f (+/- %0.4f)" %(svm_accuracy, svm_std))

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