'''
@author: pkao

This code has three funtions:
1. Converting a combined lesion label to three individual lesion labels
2. Mapping the individual lesions to MNI152 space
3. Mergeing the individual lesion label to segmentation mask in MNI152 space
'''

from utils import Brats2018ValidationN4ITKFilePaths, AllSubjectID, FindOneElement, ReadImage
from utils import PredictedLesionMaskPath, Brats2018PredictedLesionsPaths, Brats2018PredictedLesionsProbMapMNI152Paths
import paths

import argparse
import os
import subprocess

from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np


def Seg2Lesions(seg_path):
	''' This code converts a combined lesion label to three individual lesion labels'''
	subject_dir, seg_file = os.path.split(seg_path)

	lesion_dir = os.path.join(subject_dir, 'LesionLabels')

	gt_img = sitk.ReadImage(seg_path)
	gt_nda = sitk.GetArrayFromImage(gt_img)


	necrosis_nda = np.zeros(gt_nda.shape, gt_nda.dtype)
	necrosis_nda[gt_nda==1] = 1
	necrosis_img = sitk.GetImageFromArray(necrosis_nda)
	necrosis_img.CopyInformation(gt_img)
	necrosis_name = seg_file[:seg_file.index('.nii.gz')] + '_necrosis.nii.gz'
	sitk.WriteImage(necrosis_img, os.path.join(lesion_dir, necrosis_name))

	edema_nda = np.zeros(gt_nda.shape, gt_nda.dtype)
	edema_nda[gt_nda==2] = 1
	edema_img = sitk.GetImageFromArray(edema_nda)
	edema_img.CopyInformation(gt_img)
	edema_name = seg_file[:seg_file.index('.nii.gz')] + '_edema.nii.gz'
	sitk.WriteImage(edema_img, os.path.join(lesion_dir, edema_name))

	enhancing_nda = np.zeros(gt_nda.shape, gt_nda.dtype)
	enhancing_nda[gt_nda==4] = 1
	enhancing_img = sitk.GetImageFromArray(enhancing_nda)
	enhancing_img.CopyInformation(gt_img)
	enhancing_name = seg_file[:seg_file.index('.nii.gz')] + '_enhancing.nii.gz'
	sitk.WriteImage(enhancing_img, os.path.join(lesion_dir, enhancing_name))

	print('Complete subject %s' %seg_file)

def Lesions2MNI152(necrosisInVol, edemaInVol, enhancingTumorInVol):
	'''This code maps the individual lesions to MNI152 space'''
	new_name_append = "_prob_MNI152_T1_1mm.nii.gz"
	assert SubjectID(necrosisInVol) == SubjectID(edemaInVol) == SubjectID(enhancingTumorInVol) 
	print('Working on %s' %os.path.split(necrosisInVol)[1])
	omats = [os.path.join(root,name) for root, dirs, files in os.walk(brats_path) for name in files if "invol2refvol" in name and name.endswith(".mat")]
	omat_temp = [f for f in omats if SubjectID(necrosisInVol) in f]
	omat = omat_temp[0]
	assert SubjectID(necrosisInVol) == SubjectID(edemaInVol) == SubjectID(enhancingTumorInVol) == SubjectID(omat)
	subprocess.call(["flirt", "-in", necrosisInVol, "-ref", refVol, "-out", necrosisInVol[:-7] + new_name_append, "-init", omat, "-applyxfm"])
	subprocess.call(["flirt", "-in", edemaInVol, "-ref", refVol, "-out", edemaInVol[:-7] + new_name_append, "-init", omat, "-applyxfm"])
	subprocess.call(["flirt", "-in", enhancingTumorInVol, "-ref", refVol, "-out", enhancingTumorInVol[:-7] + new_name_append, "-init", omat, "-applyxfm"])
	print('Finished %s' %os.path.split(necrosisInVol)[1])

def Lesions2MNI152_star(lesion_dirs):
	return Lesions2MNI152(*lesion_dirs)

def Lesions2SegMNI152(necrosis_mni_path, edema_mni_path, enhancing_tumor_path, subject_id):
	'''This function maps the lesions in MNI152 space to tumor compartments'''
	print(SubjectID(necrosis_mni_path), SubjectID(edema_mni_path), SubjectID(enhancing_tumor_path), subject_id)
	assert (SubjectID(necrosis_mni_path) == SubjectID(edema_mni_path) == SubjectID(enhancing_tumor_path) == subject_id), 'Subject Mismatch!!!'
	mni152_path = os.path.join(necrosis_mni_path[:FindOneElement(necrosis_mni_path, '/')[-2]], 'MNI152')
	necrosis_mni_img = sitk.ReadImage(necrosis_mni_path)
	necrosis_mask_nda = sitk.GetArrayFromImage(necrosis_mni_img)
	edema_mask_nda = ReadImage(edema_mni_path)
	enhancing_tumor_mask_nda = ReadImage(enhancing_tumor_path)

	# seg in MNI 152 space
	seg_mni = np.zeros((5, necrosis_mask_nda.shape[0], necrosis_mask_nda.shape[1], necrosis_mask_nda.shape[2]), dtype=necrosis_mask_nda.dtype)
	seg_mni[1, :] = necrosis_mask_nda
	seg_mni[2, :] = edema_mask_nda
	seg_mni[4, :] = enhancing_tumor_mask_nda
	seg_mask_mni = np.argmax(seg_mni, axis=0).astype(np.int16)
	seg_name = os.path.join(mni152_path, subject_id+'_seg_MNI152_1mm.nii.gz')
	print('Working on %s' %seg_name)
	seg_mask_mni_img = sitk.GetImageFromArray(seg_mask_mni)
	seg_mask_mni_img.CopyInformation(necrosis_mni_img)
	sitk.WriteImage(seg_mask_mni_img, seg_name)

	necrosis_mni_nda = np.zeros(seg_mask_mni.shape, seg_mask_mni.dtype)
	edema_mni_nda = np.zeros(seg_mask_mni.shape, seg_mask_mni.dtype)
	enhancing_mni_nda = np.zeros(seg_mask_mni.shape, seg_mask_mni.dtype)
	necrosis_mni_nda[seg_mask_mni==1] = 1
	edema_mni_nda[seg_mask_mni==2] = 1
	enhancing_mni_nda[seg_mask_mni==4] = 1

	# whole tumor binary mask
	whole_tumor_mask_mni = necrosis_mni_nda + edema_mni_nda + enhancing_mni_nda
	whole_tumor_mask_mni_nda = whole_tumor_mask_mni.astype(np.int16)
	whole_tumor_mask_mni_name = os.path.join(mni152_path, subject_id+'_whole_tumor_MNI152_1mm.nii.gz')
	whole_tumor_mask_mni_img = sitk.GetImageFromArray(whole_tumor_mask_mni_nda)
	whole_tumor_mask_mni_img.CopyInformation(necrosis_mni_img)
	assert (np.amax(whole_tumor_mask_mni_nda) <= 1), 'Maximum of whole tumor mask not equal to 1'
	sitk.WriteImage(whole_tumor_mask_mni_img, whole_tumor_mask_mni_name)

	# tumor core binary mask
	tumor_core_mask_mni = necrosis_mni_nda + enhancing_mni_nda
	tumor_core_mask_mni_nda = tumor_core_mask_mni.astype(np.int16)
	tumor_core_mask_mni_name = os.path.join(mni152_path, subject_id+'_tumor_core_MNI152_1mm.nii.gz')
	tumor_core_mask_mni_img = sitk.GetImageFromArray(tumor_core_mask_mni_nda)
	tumor_core_mask_mni_img.CopyInformation(necrosis_mni_img)
	assert (np.amax(tumor_core_mask_mni_nda) <= 1), 'Maximum of tumor core mask not equal to 1'
	sitk.WriteImage(tumor_core_mask_mni_img, tumor_core_mask_mni_name)

	# enhancing tumor binary mask
	enhancing_tumor_mask_mni = enhancing_mni_nda
	enhancing_tumor_mask_mni_nda = enhancing_tumor_mask_mni.astype(np.int16)
	enhancing_tumor_mask_mni_name = os.path.join(mni152_path, subject_id+'_enhancing_tumor_MNI152_1mm.nii.gz')
	enhancing_tumor_mask_mni_img = sitk.GetImageFromArray(enhancing_tumor_mask_mni_nda)
	enhancing_tumor_mask_mni_img.CopyInformation(necrosis_mni_img)
	assert (np.amax(enhancing_tumor_mask_mni_nda) <= 1), 'Maximum of enhancing tumor mask not equal to 1'
	sitk.WriteImage(enhancing_tumor_mask_mni_img, enhancing_tumor_mask_mni_name)

def Lesions2SegMNI152_star(dirs):
	return Lesions2SegMNI152(*dirs)


def SubjectID(sub_dir):
	subject_file = os.path.split(sub_dir)[1]
	if 'necrosis' in subject_file:
		return subject_file[:subject_file.find('_necrosis')]
	if 'edema' in subject_file:
		return subject_file[:subject_file.find('_edema')]
	if 'enhancing' in subject_file:
		return subject_file[:subject_file.find('_enhancing')]
	if 'MNI152' in subject_file:
		return subject_file[:subject_file.find('_MNI152')]


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="can be train, valid or test", default='train', type=str)
parser.add_argument("-t", "--thread", help="the number of thread you want to use ", default=8, type=int)
args = parser.parse_args()

if args.mode == "train":
    brats_path = paths.brats2018_training_dir
    predicted_path = paths.brats2018_training_predicted_lesions_dir
elif args.mode == "valid":
    brats_path = paths.brats2018_validation_dir
    predicted_path = paths.brats2018_validation_predicted_lesions_dir
elif args.mode == "test":
    brats_path = paths.brats2018_testing_dir
    predicted_path = paths.brats2018_testing_predicted_lesions_dir
else:
    raise ValueError("Unknown value for --mode. Use \"train\", \"valid\" or \"test\"")

refVol = paths.mni152_1mm_path

pool = Pool(args.thread)

# The following lines work on spliting seg into individual lesion label
predicted_lesions_paths = PredictedLesionMaskPath(predicted_path=predicted_path)
assert(len(predicted_lesions_paths)==191 or len(predicted_lesions_paths)==285 or len(predicted_lesions_paths)==66)
predicted_lesions_dir = os.path.split(predicted_lesions_paths[0])[0]
if not os.path.exists(os.path.join(predicted_lesions_dir, 'LesionLabels')):
	os.mkdir(os.path.join(predicted_lesions_dir, 'LesionLabels'))
pool.map(Seg2Lesions, predicted_lesions_paths)


# The following lines work on mapping these individual lesion label into probability maps in MNI152 space 
necrosis_paths, edema_paths, enhancing_tumor_paths = Brats2018PredictedLesionsPaths(predicted_path=predicted_path)
assert(len(necrosis_paths) == len(edema_paths) == len(enhancing_tumor_paths))
pool.map(Lesions2MNI152_star, zip(necrosis_paths, edema_paths, enhancing_tumor_paths))


# The following lines merge the individual lesion label to segmentation mask in MNI152 space
necrosis_mni_paths, edema_mni_paths, enhancing_tumor_paths = Brats2018PredictedLesionsProbMapMNI152Paths(predicted_path=predicted_path)
all_ids = AllSubjectID(brats_path)
assert(len(all_ids) == len(necrosis_mni_paths) == len(edema_mni_paths) == len(enhancing_tumor_paths)), 'brats_path and mode mismatch!!!'
if not os.path.exists(os.path.join(predicted_lesions_dir, 'MNI152')):
	os.mkdir(os.path.join(predicted_lesions_dir, 'MNI152'))
pool.map(Lesions2SegMNI152_star, zip(necrosis_mni_paths, edema_mni_paths, enhancing_tumor_paths, all_ids))