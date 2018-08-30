'''
@author: pkao

This code has three funtions:
1. Converting a combined lesion label to three individual lesion labels
2. Mapping the individual lesions to MNI152 space
3. Mergeing the individual lesion label to segmentation mask in MNI152 space
'''

from utils import Brats2018TrainingN4ITKFilePaths, Brats2018GTLesionsPaths, Brats2018GTLesionsProbMapMNI152Paths, SubjectID, AllSubjectID, FindOneElement, ReadImage
import paths

import os
import subprocess
import argparse

from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np


def Seg2Lesions(seg_path):
	''' This code converts a combined lesion label to three individual lesion labels'''
	subject_dir, seg_file = os.path.split(seg_path)

	lesion_dir = os.path.join(subject_dir, 'LesionLabels')

	if not os.path.exists(lesion_dir):
		os.makedirs(lesion_dir)

	gt_img = sitk.ReadImage(seg_path)
	gt_nda = sitk.GetArrayFromImage(gt_img)

	necrosis_nda = np.zeros(gt_nda.shape, gt_nda.dtype)
	necrosis_nda[gt_nda==1] = 1
	necrosis_img = sitk.GetImageFromArray(necrosis_nda)
	necrosis_img.CopyInformation(gt_img)
	necrosis_name = seg_file[:seg_file.index('seg.nii.gz')] + 'necrosis.nii.gz'
	sitk.WriteImage(necrosis_img, os.path.join(lesion_dir, necrosis_name))

	edema_nda = np.zeros(gt_nda.shape, gt_nda.dtype)
	edema_nda[gt_nda==2] = 1
	edema_img = sitk.GetImageFromArray(edema_nda)
	edema_img.CopyInformation(gt_img)
	edema_name = seg_file[:seg_file.index('seg.nii.gz')] + 'edema.nii.gz'
	sitk.WriteImage(edema_img, os.path.join(lesion_dir, edema_name))

	enhancing_nda = np.zeros(gt_nda.shape, gt_nda.dtype)
	enhancing_nda[gt_nda==4] = 1
	enhancing_img = sitk.GetImageFromArray(enhancing_nda)
	enhancing_img.CopyInformation(gt_img)
	enhancing_name = seg_file[:seg_file.index('seg.nii.gz')] + 'enhancing.nii.gz'
	sitk.WriteImage(enhancing_img, os.path.join(lesion_dir, enhancing_name))

	print('Complete subject %s' %seg_file)

def Lesions2MNI152(necrosisInVol, edemaInVol, enhancingTumorInVol, omat):
	'''This code maps the individual lesions to the probability maps in MNI152 space'''
	new_name_append = "_prob_MNI152_T1_1mm.nii.gz"
	assert SubjectID(necrosisInVol) == SubjectID(edemaInVol) ==SubjectID(enhancingTumorInVol) ==SubjectID(omat)
	print('Working on %s' %os.path.split(necrosisInVol)[1])
	subprocess.call(["flirt", "-in", necrosisInVol, "-ref", refVol, "-out", necrosisInVol[:-7] + new_name_append, "-init", omat, "-applyxfm"])
	subprocess.call(["flirt", "-in", edemaInVol, "-ref", refVol, "-out", edemaInVol[:-7] + new_name_append, "-init", omat, "-applyxfm"])
	subprocess.call(["flirt", "-in", enhancingTumorInVol, "-ref", refVol, "-out", enhancingTumorInVol[:-7] + new_name_append, "-init", omat, "-applyxfm"])
	print('Finished %s' %os.path.split(necrosisInVol)[1])

def Lesions2MNI152_star(lesion_dirs):
	return Lesions2MNI152(*lesion_dirs)

def Lesions2SegMNI152(necrosis_mni_path, edema_mni_path, enhancing_tumor_path, subject_id):
	'''This code maps the lesion probability maps into lesion masks in MNI152'''
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

global refVol

refVol = paths.mni152_1mm_path

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--thread", help="the number of thread you want to use ", default=8, type=int)
args = parser.parse_args()


bratsPath=paths.brats2018_training_dir

pool = Pool(args.thread)

# The following two lines work on spliting seg into individual lesion label
t1_filepaths, t1c_filepaths, t2_filepaths, flair_filepaths, seg_filepaths = Brats2018TrainingN4ITKFilePaths(bratsPath)
pool.map(Seg2Lesions, seg_filepaths)

# The following four lines work on mapping these individual lesion label into MNI152 space 
necrosis_paths, edema_paths, enhancing_tumor_paths = Brats2018GTLesionsPaths(bratsPath)
omat = [os.path.join(root,name) for root, dirs, files in os.walk(bratsPath) for name in files if "invol2refvol" in name and name.endswith(".mat")]
omat.sort()
pool.map(Lesions2MNI152_star, zip(necrosis_paths, edema_paths, enhancing_tumor_paths, omat))

# The following three lines merge the individual lesion label to segmentation mask in MNI152 space
necrosis_prob_mni_paths, edema_prob_mni_paths, enhancing_tumor_prob_paths = Brats2018GTLesionsProbMapMNI152Paths(bratsPath)
all_training_ids = AllSubjectID(dataset_type='training')
pool.map(Lesions2SegMNI152_star, zip(necrosis_prob_mni_paths, edema_prob_mni_paths, enhancing_tumor_prob_paths, all_training_ids))