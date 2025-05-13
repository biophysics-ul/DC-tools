# dc_tools

Some additional info for the module `dc_tools.py`.

## Overview
The functions in `dc_tools.py` are designed to streamline the process of image classification in flow cytometry and other biological research. The key functionalities include:
1. Image Processing: Image processing and augmentation techniques before training
2. Model Training: Train a modified ResNet18 model with user-defined hyperparameters
3. Classification: Use the trained model to classify images and save the results.

### Dependencies

To use this project, you need the following Python packages:  
`OpenCV` >=4.0  
`dclab` (only if you work with RTDC files)
`numpy` >=1.21.0
`pandas`
`matplotlib` >=3.7.1
`torch` >=2.0.1
`torchvision` >=0.8.0
`opencv-python` >=4.8.0
`tqdm`
`Pillow`
`ipykernel` >=6.25.0
`scikit-learn`
`seaborn`
`h5py`

### Main functions

## Model Training
`train_model(image_folder_path, results_folder)`

**Description:** The `train_model` function in `dc_tools.py` handles the training of the cell classification model. It processes images, trains the model, and saves the trained model along with performance metrics. 
A detailed tutorial for model training is included in the /docs subdirectory.
Parameters:
The `train_model` function includes several key parameters that can be modified for fine-tuning:
- `Validation Split (val_split)`: Defines the percentage of data used for validation.
- `Batch Size (batch_size)`: Refers to the number of training examples processed before the model updates its parameters.
- `Epochs (epochs)`: Number of complete passes through the training dataset.
- `Early Stopping Patience (patience_total)`: Halts training if validation loss does not improve.
- `Learning Rate (lr)`: Determines the step size at each iteration while optimizing the loss function.
- `Weight Decay (weight_decay)`: Reduces overfitting by preventing complex models.
- `Dropout Probability (dropout_prob)`: Regularizes the model to prevent overfitting.

## Inference
`inference(image_folder, results_folder, model, labels, device)`
The `inference` function allows users to perform image classification using a pre-trained model. It processes images, runs inference to predict class labels, and saves classified images and performance metrics.<br>
Parameters:
- `doublechannel`: Set to True if processing doublechannel images. This activates background subtraction during segmentation.
- `extra_pixels`: Defines additional padding around objects during segmentation.

## RTDC handling
`extract_events_from_raw_rtdc_to_zip(rtdc_path, zip_path, mode)`

**Description:** Extracts images with events and their timestamps from a raw (non-filtered) RTDC file into a ZIP archive. Images with events are those that, after background subtraction, have a signal-to-noise ratio (SNR) above a predefined threshold.

The user can choose if the extracted images are a) original images with events, b) images with subtracted background or c) concatenated images, with the original image on top and the background image below (mode=original,subtracted,concatenated).


