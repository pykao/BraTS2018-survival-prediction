#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 18:54:21 2018

@author: pkao

This code applies N4ITK for BRATS2018 database

The output for this code should be name_of_mri_corrected.nii.gz
"""
import argparse
import os
from nipype.interfaces.ants import N4BiasFieldCorrection
from multiprocessing import Pool
import paths

def N4ITK(filepath):
    print('Working on: %s' %filepath) 
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = filepath
    outputPath = filepath[:filepath.find(".nii.gz")]+'_N4ITK_corrected.nii.gz'
    n4.inputs.output_image = outputPath
    n4.run()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="can be train, valid or test", type=str)
parser.add_argument("-t", "--thread", help="the number of thread you want to use ", default=8, type=int)
args = parser.parse_args()

if args.mode == "train":
    brats2018_path = paths.brats2018_training_dir
elif args.mode == "valid":
    brats2018_path = paths.brats2018_valid_dir
elif args.mode == "test":
    brats2018_path = paths.brats2018_test_dir
else:
    raise ValueError("Unknown value for --mode. Use \"train\", \"valid\" or \"test\"")

t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2018_path)
for name in files if 't1' in name and 'ce' not in name and 'N4ITK' not in name and name.endswith('.nii.gz')]
t1_filepaths.sort()

t1ce_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2018_path)
for name in files if 't1ce' in name and 'N4ITK' not in name and name.endswith('.nii.gz')]
t1ce_filepaths.sort()

t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2018_path)
for name in files if 't2' in name and 'N4ITK' not in name and name.endswith('.nii.gz')]
t2_filepaths.sort()

flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2018_path)
for name in files if 'flair' in name and 'N4ITK' not in name  and name.endswith('.nii.gz')]
flair_filepaths.sort()

#file_paths = t1_filepaths + t1ce_filepaths + t2_filepaths + flair_filepaths
file_paths = t1_filepaths

pool = Pool(args.thread)

pool.map(N4ITK, file_paths)
