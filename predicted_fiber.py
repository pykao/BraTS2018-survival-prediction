import paths
import os
import csv
import shutil
import subprocess
import argparse

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

    for subject_name in subject_names:
        whole_tumor_mni_path = [os.path.join(root, name) for root, dirs, files in os.walk(brats_root) for name in files if subject_name in name and 'whole_tumor' in name and 'MNI152_1mm' in name and name.endswith('nii.gz')]
        src = whole_tumor_mni_path[0]
        whole_tumor_mni_file = os.path.split(whole_tumor_mni_path[0])[1]
        dst = os.path.join(dst_dir, whole_tumor_mni_file)
        shutil.copy(src, dst)


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="can be train, valid or test", default='train', type=str)
parser.add_argument("-t", "--thread", help="the number of thread you want to use ", default=8, type=int)
args = parser.parse_args()

work_dir = paths.dsi_studio_path

if args.mode == "train":
    mode = 'training'
    survival_names = survival_subject_name(paths.brats2018_training_dir)
    predicted_path = paths.brats2018_training_predicted_lesions_dir
elif args.mode == "valid":
    mode = 'validation'
    survival_names = survival_subject_name(paths.brats2018_validation_dir)
    predicted_path = paths.brats2018_validation_predicted_lesions_dir
elif args.mode == "test":
    mode = 'testing'    
    survival_names = survival_subject_name(paths.brats2018_testing_dir)
    predicted_path = paths.brats2018_testing_predicted_lesions_dir
else:
    raise ValueError("Unknown value for --mode. Use \"train\", \"valid\" or \"test\"")


assert(len(survival_names)==59 or len(survival_names)==28 or len(survival_names)== 77)

predicted_whole_tumor_path = os.path.join(work_dir, 'predicted_whole_tumor')
if not os.path.exists(predicted_whole_tumor_path):
    os.mkdir(predicted_whole_tumor_path)

dst_dir = os.path.join(predicted_whole_tumor_path, mode)
if not os.path.exists(dst_dir):
	os.mkdir(dst_dir)

MoveLesionsMNIMask(predicted_path, dst_dir, survival_names)

if not os.path.exists(os.path.join(work_dir, 'network_measures', mode)):
	os.mkdir(os.path.join(work_dir, 'network_measures', mode))
if not os.path.exists(os.path.join(work_dir, 'connectogram', mode)):
	os.mkdir(os.path.join(work_dir, 'connectogram', mode))
if not os.path.exists(os.path.join(work_dir, 'connectivity', mode)):
	os.mkdir(os.path.join(work_dir, 'connectivity', mode))

source ='HCP1021.1mm.fib.gz'

assert (os.path.exists(os.path.join(work_dir, source))), 'HCP1021 template is not in the dsi studio directory'

parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4340420Fca01dcba'

whole_tumor_dir = dst_dir

whole_tumor_files_dir = os.listdir(whole_tumor_dir)


os.chdir(work_dir)

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
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', mode, os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', mode, os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', mode, os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)

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
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', mode, os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', mode, os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', mode, os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)