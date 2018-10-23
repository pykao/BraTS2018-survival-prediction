import numpy as np
import os
import SimpleITK as sitk

from scipy.io import loadmat
from skimage.measure import regionprops

import paths
import utils

def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


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



def get_weighted_connectivity_feature_vectors_test(dsi_studio_path=paths.dsi_studio_path, region='seed'):
    connectivity_testing_dir = os.path.join(dsi_studio_path, 'connectivity', region, 'testing')
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

def get_weighted_connectivity_feature_vectors_valid(dsi_studio_path=paths.dsi_studio_path, region='seed'):
    connectivity_valid_dir = os.path.join(dsi_studio_path, 'connectivity', region, 'validation')
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

def get_weighted_connectivity_feature_vectors_train(dsi_studio_path=paths.dsi_studio_path, mode='gt', region='seed'):
    ''' Loading the survival dataset '''
    survival_dataset = utils.load_survival_training_dataset()
    if mode == 'gt':
        connectivity_train_dir = os.path.join(dsi_studio_path, 'connectivity', region, 'gt')
        whole_tumor_mni_train_dir = os.path.join(dsi_studio_path, 'gt_whole_tumor')
    if mode == 'predicted':
        connectivity_train_dir = os.path.join(dsi_studio_path, 'connectivity', region, 'training')
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
'''
def extract_gt_age():
    ''' extract the subject's survival day and age'''
    gt = np.zeros((59,2), dtype = np.float32)
    age_features = np.zeros((59,1), dtype = np.float32)
    survival_dataset = utils.load_survival_training_dataset()
    for idx, pat_name in enumerate(survival_dataset.keys()):
        subject_age = float(survival_dataset[pat_name]['age'])
        survival_days = int(survival_dataset[pat_name]['survival'])
    

        # get the number of subjects in different survival period
        if int(survival_dataset[pat_name]['survival']) < 305:
            #period = 'Short'
            gt[idx, 0] = 0
            gt[idx, 1] = survival_days
            #short_period += 1 
        # should be 454 or 456.25
        elif int(survival_dataset[pat_name]['survival']) > 456:
            #period = 'Long'
            gt[idx, 0] = 2
            gt[idx, 1] = survival_days
            #long_period += 1
        else:
            #period = 'Medium'
            gt[idx, 0] = 1
            gt[idx, 1] = survival_days
            #medium_period += 1
        age_features[idx] = subject_age
    return gt, age_features, ['age']
'''