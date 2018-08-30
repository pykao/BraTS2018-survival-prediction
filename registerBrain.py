"""
@ author: pkao

This code applies registration on all MR images that map the images from subject space to MNI152 1mm sapce

This code will create a folder MNI152 under subject's directory

The outputs for this code should be two affine transformation matrices containing one matrix mapping subject to MNI152 1mm space and its inverse matrix
"""
import argparse
from utils import Brats2018TrainingN4ITKFilePaths, Brats2018ValidationN4ITKFilePaths, Brats2018TestingN4ITKFilePaths
import os
import subprocess
from multiprocessing import Pool
import paths

def RegisterBrain(t1_path):
    path, t1_file = os.path.split(t1_path)
    mni_path = os.path.join(path, 'MNI152')
    if not os.path.exists(mni_path):
        os.makedirs(mni_path)
    subject2mni_mat = os.path.join(mni_path, t1_file[:t1_file.index('_t1_')]+'_MNI152_1mm_invol2refvol.mat')
    mni2subject_mat = os.path.join(mni_path, t1_file[:t1_file.index('_t1_')]+'_MNI152_1mm_refvol2invol.mat')
    print('Working on subject: %s' %t1_file[:t1_file.index('_t1_')])
    print(t1_path, paths.mni152_1mm_path, subject2mni_mat, mni2subject_mat)
    # Create the affine transformation matrix from subject space to MNI152 1mm space
    subprocess.call(["flirt", "-in", t1_path, "-ref", paths.mni152_1mm_path, "-omat", subject2mni_mat])
    subprocess.call(["convert_xfm", "-omat", mni2subject_mat, "-inverse", subject2mni_mat])
    print('Finish this subject: %s'  %t1_file[:t1_file.index('_t1_')])


def main():

    if args.mode == "train":
        t1_filepaths, _, _, _, _ = Brats2018TrainingN4ITKFilePaths()
    elif args.mode == "valid":
        t1_filepaths, _, _, _ = Brats2018ValidationN4ITKFilePaths()
    elif args.mode == "test":
        t1_filepaths, _, _, _ = Brats2018TestingN4ITKFilePaths()
    else:
        raise ValueError("Unknown value for --mode. Use \"train\", \"valid\" or \"test\"")

    pool = Pool(args.thread)

    pool.map(RegisterBrain, t1_filepaths)


if __name__ == "__main__":

    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="can be train, valid or test", default='train', type=str)
    parser.add_argument("-t", "--thread", help="the number of thread you want to use ", default=8, type=int)
    args = parser.parse_args()
    main()
