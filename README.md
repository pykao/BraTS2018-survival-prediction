# Overall Survival Prediction for BraTS2018 Competition
This repository implements the survival prediction task of the paper [Brain Tumor Segmentation and Tractographic Feature Extraction from Structural MR Images for Overall Survival Prediction](https://www.researchgate.net/publication/326549702_Brain_Tumor_Segmentation_and_Tractographic_Feature_Extraction_from_Structural_MR_Images_for_Overall_Survival_Prediction) which participates in BraTS2018 survival prediction challenge.

## Dependencies

Python 3.6

## Required Softwares

FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)

DSI studio (http://dsi-studio.labsolver.org/dsi-studio-download)

## Required Atlases

1. MNI152 T1 1mm brain ('/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')

2. HCP1021 1mm template (https://pitt.app.box.com/v/HCP1021-1mm)

3. AAL Atlas ('~/dsistudio/dsi_studio_64/atlas/aal.nii.gz)

## Required python libraries

nipype, SimpleITK, sklearn, scipy, pandas, csv

```
pip install nipype,SimpleITK,sklearn,scipy,pandas,csv
```
## How to Run

First, you need to change the directories in paths.py

### Pre-processing

Then, you run the N4ITK bias correction on training, validation and testing dataset.
```
python N4ITKforBraTS2018.py --mode train
python N4ITKforBraTS2018.py --mode valid
python N4ITKforBraTS2018.py --mode test
```
You are able to find the corrected files in the same directory named USERID_modality_N4ITK_corredted.nii.gz

### Get the affine matrices mapping subject to MNI152 1mm space and MNI152 T1 1mm to subject space

```
python registerBrain.py --mode train
python registerBrain.py --mode valid
python registerBrain.py --mode test
```

## For Training Dataset with Ground Truth Lesion Labels

### Map the ground truth lesion from subject space to MNI152 space (seg, whole tumor, tumor core, and enhancing tumor in MNI152 space)

```
python gt_lesions.py
```
### Create the fiber tracts for the ground truth whole tumor region

Please refer to params.txt for the parameter setting.

Move HCP1021 1mm template (HCP1021.1mm.fib.gz) to the paths.dsi_studio_path

Change the parameter_id in gt_fiber.py to your own id

```
python gt_fiber.py
```
After you run the above command, you are able to get three folders named connectivity, connectogram and network_measures in the paths.dsi_studio_path. Within these three folders, you can find a folder named gt which contains all information.


## For Dataset with Predicted Lesion Labels

### Map the predicted lesion from subject space to MNI152 space (predicted_seg, predicted whole tumor, predicted tumor core, and predicted enhancing tumor in MNI152 space)

```
python predicted_lesions.py --mode train
python predicted_lesions.py --mode valid
python predicted_lesions.py --mode test
```

### Create the fiber tracts for the predicted whole tumor region

Please refer to params.txt for the parameter setting.

Move HCP1021 1mm template (HCP1021.1mm.fib.gz) to the paths.dsi_studio_path

Change the parameter_id in gt_fiber.py to your own id

```
python predicted_fiber.py --mode train
python predicted_fiber.py --mode valid
python predicted_fiber.py --mode test
```
After you run the above three commands, you are able to get three folders named connectivity, connectogram and network_measures in the paths.dsi_studio_path. Within these three folders, you can find three folder named training, validation, and testing which contains all information.

## Overall Survival Prediction (Classification)
