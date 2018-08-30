# This is the file to create the connectivity matrix and the network measures

import os
import shutil
import subprocess
import paths
import csv

def survival_subject_name(data_path):
	survival_file = [os.path.join(root, name) for root, dirs, files in os.walk(data_path) for name in files if name.endswith('csv')]

	survival_name = []
	with  open(survival_file[0], 'rt') as csv_file:
		csv_reader = csv.reader(csv_file)
		for line in csv_reader:
			if 'test' in data_path or 'valid' in data_path:
				if line[2] =='GTR':
					survival_name.append(line[0])
			if 'train' in data_path:
				if line[3] =='GTR':
					survival_name.append(line[0])

	return survival_name

def MoveLesionsMNIMask(brats_root, dst_dir, subject_names):
    ''' This function moves the whole tumor mask in MNI152 to the working directory for dsi studio'''

    for subject_name in subject_names:
        whole_tumor_mni_path = [os.path.join(root, name) for root, dirs, files in os.walk(brats_root) for name in files if subject_name in name and 'whole_tumor' in name and 'MNI152_1mm' in name and name.endswith('nii.gz')]
        src = whole_tumor_mni_path[0]
        whole_tumor_mni_file = os.path.split(whole_tumor_mni_path[0])[1]
        dst = os.path.join(dst_dir, whole_tumor_mni_file)
        shutil.copy(src, dst)

work_dir = paths.dsi_studio_path

dst_dir = os.path.join(work_dir, 'gt_whole_tumor')

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

survival_names = survival_subject_name(paths.brats2018_training_dir)

MoveLesionsMNIMask(paths.brats2018_training_dir, dst_dir, survival_names)

network_measures_dir = os.path.join(work_dir, 'network_measures')

if not os.path.exists(network_measures_dir):
    os.mkdir(network_measures_dir)

if not os.path.exists(os.path.join(network_measures_dir, 'gt')):
    os.mkdir(os.path.join(network_measures_dir, 'gt'))

connectogram_dir = os.path.join(work_dir, 'connectogram')

if not os.path.exists(connectogram_dir):
    os.mkdir(connectogram_dir)

if not os.path.exists(os.path.join(connectogram_dir, 'gt')):
    os.mkdir(os.path.join(connectogram_dir, 'gt'))

connectivity_dir = os.path.join(work_dir, 'connectivity')

if not os.path.exists(connectivity_dir):
    os.mkdir(connectivity_dir)

if not os.path.exists(os.path.join(connectivity_dir, 'gt')):
    os.mkdir(os.path.join(connectivity_dir, 'gt'))

source ='HCP1021.1mm.fib.gz'

assert (os.path.exists(os.path.join(work_dir, source))), 'HCP1021 template is not in the dsi studio directory'

parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4340420Fca01dcba'

whole_tumor_dir = dst_dir
whole_tumor_files_dir = os.listdir(dst_dir)
whole_tumor_files_dir.sort()
assert(len(whole_tumor_files_dir)==59)

os.chdir(work_dir)

# end type of connectivity matrices
for idx, whole_tumor_file in enumerate(whole_tumor_files_dir):

    pat_name = whole_tumor_file[:whole_tumor_file.find('.nii.gz')]
    print(idx, pat_name)
    roi = '--roi='+os.path.join(whole_tumor_dir, whole_tumor_file)
    seed = '--seed='+os.path.join(whole_tumor_dir, whole_tumor_file)
    connectivity_type = '--connectivity_type=end'
    connectivity_value = '--connectivity_value=count'
    connectivity_threshold = '--connectivity_threshold=0'
    subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, seed, roi, parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])

    network_measure_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'network_measures' in name and name.endswith('.txt')]
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', 'gt', os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', 'gt', os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', 'gt', os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)

# pass type of connectivity matrices
for idx, whole_tumor_file in enumerate(whole_tumor_files_dir):

    pat_name = whole_tumor_file[:whole_tumor_file.find('.nii.gz')]
    print(idx, pat_name)
    roi = '--roi='+os.path.join(whole_tumor_dir, whole_tumor_file)
    seed = '--seed='+os.path.join(whole_tumor_dir, whole_tumor_file)
    connectivity_type = '--connectivity_type=pass'
    connectivity_value = '--connectivity_value=count'
    connectivity_threshold = '--connectivity_threshold=0'
    subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, seed, roi, parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])

    network_measure_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'network_measures' in name and name.endswith('.txt')]
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', 'gt', os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', 'gt', os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', 'gt', os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)