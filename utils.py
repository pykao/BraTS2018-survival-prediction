import os
import paths
import SimpleITK as sitk
import numpy as np
import csv

def Brats2018TestingN4ITKFilePaths(brats_testing_path = paths.brats2018_testing_dir):
	''' This fucntion gives the filepathes of all MR images with N4ITK and ground truth'''
	t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_testing_path)
					for name in files if 't1' in name and 'ce' not in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1_filepaths.sort()

	t1c_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_testing_path) 
					for name in files if 't1' in name and 'ce' in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1c_filepaths.sort()

	t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_testing_path)
					for name in files if 't2' in name and 'corrected' in name and 'normalized' not in name 
					and 'MNI152' not in name and name.endswith('.nii.gz')]
	t2_filepaths.sort()

	flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_testing_path)
						for name in files if 'flair' in name and 'corrected' in name and 'normalized' not in name 
						and 'MNI152' not in name and name.endswith('.nii.gz')]
	flair_filepaths.sort()

	assert(len(t1_filepaths)==len(t1c_filepaths)==len(t2_filepaths)==len(flair_filepaths))

	return t1_filepaths, t1c_filepaths, t2_filepaths, flair_filepaths


def Brats2018ValidationN4ITKFilePaths(brats_validation_path = paths.brats2018_validation_dir):
	''' This fucntion gives the filepathes of all MR images with N4ITK and ground truth'''
	t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_validation_path)
					for name in files if 't1' in name and 'ce' not in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1_filepaths.sort()

	t1c_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_validation_path) 
					for name in files if 't1' in name and 'ce' in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1c_filepaths.sort()

	t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_validation_path)
					for name in files if 't2' in name and 'corrected' in name and 'normalized' not in name 
					and 'MNI152' not in name and name.endswith('.nii.gz')]
	t2_filepaths.sort()

	flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_validation_path)
						for name in files if 'flair' in name and 'corrected' in name and 'normalized' not in name 
						and 'MNI152' not in name and name.endswith('.nii.gz')]
	flair_filepaths.sort()

	assert(len(t1_filepaths)==len(t1c_filepaths)==len(t2_filepaths)==len(flair_filepaths))

	return t1_filepaths, t1c_filepaths, t2_filepaths, flair_filepaths

def Brats2018TrainingN4ITKFilePaths(brats_training_path = paths.brats2018_training_dir):
	''' This fucntion gives the filepathes of all MR images with N4ITK and ground truth'''
	t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_training_path)
					for name in files if 't1' in name and 'ce' not in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1_filepaths.sort()

	t1c_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_training_path) 
					for name in files if 't1' in name and 'ce' in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1c_filepaths.sort()

	t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_training_path)
					for name in files if 't2' in name and 'corrected' in name and 'normalized' not in name 
					and 'MNI152' not in name and name.endswith('.nii.gz')]
	t2_filepaths.sort()

	flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_training_path)
						for name in files if 'flair' in name and 'corrected' in name and 'normalized' not in name 
						and 'MNI152' not in name and name.endswith('.nii.gz')]
	flair_filepaths.sort()

	seg_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats_training_path)
					for name in files if 'seg' in name and 'FAST' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	seg_filepaths.sort()

	assert(len(t1_filepaths)==len(t1c_filepaths)==len(t2_filepaths)==len(flair_filepaths)==len(seg_filepaths))

	return t1_filepaths, t1c_filepaths, t2_filepaths, flair_filepaths, seg_filepaths

def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

def FindOneElement(s, ch):
	''' This function gives the indexs of one element ch on the string s'''
	return [i for i, ltr in enumerate(s) if ltr == ch]

def SubjectID(bratsPath):
	''' This function gives you the subject ID'''
	tmp = os.path.split(bratsPath)[1]
	if '_t1' in tmp:
		return tmp[:tmp.find('_t1')]
	if '_t2' in tmp:
		return tmp[:tmp.find('_t2')]
	if '_flair' in tmp:
		return tmp[:tmp.find('_flair')]
	if '_seg' in tmp:
		return tmp[:tmp.find('_seg')]

def AllSubjectID(dataset_type='training'):
	''' This function gives you all subject IDs'''
	if 'train' in dataset_type:
		t1_filepaths, _, _, _, _ = Brats2018TrainingN4ITKFilePaths()
	if 'valid' in dataset_type:
		t1_filepaths, _, _, _, = Brats2018ValidationN4ITKFilePaths()
	if 'testing' in dataset_type:
		t1_filepaths, _, _, _, = Brats2018TestingN4ITKFilePaths()	
	#subject_dirs = [os.path.split(seg_path)[1] for seg_path in seg_filepaths]
	all_subject_ID = [SubjectID(t1_filepath) for t1_filepath in t1_filepaths]
	all_subject_ID.sort()
	return all_subject_ID

def PredictedLesionMaskPath(predicted_path):
	predicted_lesions_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path)
					for name in files if 'necrosis' not in name and 'edema' not in name and 'enhancing' not in name 
					and 'MNI152' not in name
					and name.endswith('.nii.gz')]
	predicted_lesions_paths.sort()
	print(len(predicted_lesions_paths))
	return predicted_lesions_paths

def Brats2018PredictedLesionsPaths(predicted_path):
	''' This function gives you the paths for evey lesion files'''
	necrosis_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path) 
	for name in files if 'MNI' not in name and 'necrosis' in name and name.endswith('.nii.gz')]
	necrosis_paths.sort()
	edema_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path) 
	for name in files if 'MNI' not in name and 'edema' in name and name.endswith('.nii.gz')]
	edema_paths.sort()
	enhancing_tumor_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path) 
	for name in files if 'MNI' not in name and 'enhancing' in name and name.endswith('.nii.gz')]
	enhancing_tumor_paths.sort()

	assert len(necrosis_paths) == len(edema_paths) == len(enhancing_tumor_paths)
	return necrosis_paths, edema_paths, enhancing_tumor_paths

def Brats2018PredictedLesionsProbMapMNI152Paths(predicted_path):
	''' This function gives you the paths for evey lesion in MNI 152 space files'''
	necrosis_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path) 
	for name in files if 'MNI' in name and 'necrosis' in name and 'prob' in name and name.endswith('.nii.gz')]
	necrosis_paths.sort()
	edema_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path) 
	for name in files if 'MNI' in name and 'edema' in name and 'prob' in name and name.endswith('.nii.gz')]
	edema_paths.sort()
	enhancing_tumor_paths = [os.path.join(root, name) for root, dirs, files in os.walk(predicted_path) 
	for name in files if 'MNI' in name and 'enhancing' in name and 'prob' in name and name.endswith('.nii.gz')]
	enhancing_tumor_paths.sort()
	assert len(necrosis_paths) == len(edema_paths) == len(enhancing_tumor_paths)
	return necrosis_paths, edema_paths, enhancing_tumor_paths

def Brats2018GTLesionsPaths(bratsPath=paths.brats2018_training_dir):
	'''This function gives you the paths for every ground truth lesions for training dataset'''
	necrosis_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
	for name in files if 'MNI' not in name and 'necrosis' in name and name.endswith('.nii.gz')]
	necrosis_paths.sort()
	edema_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
	for name in files if 'MNI' not in name and 'edema' in name and name.endswith('.nii.gz')]
	edema_paths.sort()
	enhancing_tumor_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
	for name in files if 'MNI' not in name and 'enhancing' in name and name.endswith('.nii.gz')]
	enhancing_tumor_paths.sort()

	assert len(necrosis_paths) == len(edema_paths) == len(enhancing_tumor_paths)
	return necrosis_paths, edema_paths, enhancing_tumor_paths

def Brats2018GTLesionsProbMapMNI152Paths(bratsPath=paths.brats2018_training_dir):
	'''This function gives you the probability maps for every ground truth lesions in MNI152 for training dataset'''
	necrosis_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
	for name in files if 'MNI' in name and 'necrosis' in name and 'prob' in name and name.endswith('.nii.gz')]
	necrosis_paths.sort()
	edema_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
	for name in files if 'MNI' in name and 'edema' in name and 'prob' in name and name.endswith('.nii.gz')]
	edema_paths.sort()
	enhancing_tumor_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
	for name in files if 'MNI' in name and 'enhancing' in name and 'prob' in name and name.endswith('.nii.gz')]
	assert len(necrosis_paths) == len(edema_paths) == len(enhancing_tumor_paths)
	return necrosis_paths, edema_paths, enhancing_tumor_paths

def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape)==2:
            pad_value = image[0,0]
        elif len(shape)==3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res

def load_survival_training_dataset(brats2018_training_dir = paths.brats2018_training_dir):
	survival_dataset = {}
	survival_file = 'survival_data.csv'
	survival_path = os.path.join(brats2018_training_dir, survival_file)
	with open(survival_path, 'rt') as csv_file:
		csv_reader = csv.reader(csv_file)
		for line in csv_reader:
			if line[3] =='GTR':
				pat_age = line[1]
				survival_days = line[2]
				pat_name = line[0]
				survival_dataset[pat_name] = {}
				survival_dataset[pat_name]['age'] = pat_age
				survival_dataset[pat_name]['survival'] = survival_days		
	return survival_dataset
