# ml_tools

Some additional info for the module `ml_tools.py`.

### Dependencies and 

To use this project, you need the following Python packages:  
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
  

pip install opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install dclab
pip install tqdm
pip install matplotlib
pip install scikit-learn
pip install seaborn



pip install torch==2.0.1 torchvision==0.8.0 --index-url https://download.pytorch.org/whl/cu121

https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


### Overview
The `ml_tools.py` module is designed to streamline the process of image classification in flow cytometry and other biological research. The key functionalities include:
1. Image Processing: Image processing and augmentation techniques before training
2. Model Training: Train a modified ResNet18 model with user-defined hyperparameters
3. Classification: Use the trained model to classify images and save the results. 

TrainingSet.zip
├── class1/
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
├── class2/
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
├── class3/
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
└── ...


### Functions

## Model Training
`train_model(image_folder_path, results_folder)`

**Description:** The `train_model` function in `ml_tools.py` handles the training of the cell classification model. It processes images, trains the model, and saves the trained model along with performance metrics. 
A detailed tutorial for model training is included in the /docs subdirectory.
Parameters:
The train_model function includes several key parameters that can be modified for fine-tuning:
- `Validation Split (val_split)`: Defines the percentage of data used for validation.
- `Batch Size (batch_size`: Refers to the number of training examples processed before the model updates its parameters.
- `Epochs (epochs)`: Number of complete passes through the training dataset.
- `Early Stopping Patience (patience_total)`: Halts training if validation loss does not improve.
- `Learning Rate (lr)`: Determines the step size at each iteration while optimizing the loss function.
- `Weight Decay (weight_decay)`: Reduces overfitting by preventing complex models.
- `Dropout Probability (dropout_prob)`: Regularizes the model to prevent overfitting.

## Inference
`inference(image_folder, results_folder, model, labels, device)`
**Description:** The `inference` function in `ml_tools.py` allow users to perform image classification using a pre-trained model. It processes images, runs inference to predict class labels, and saves classified images and performance metrics.<br>
Parameters:
- `doublechannel`: Set to True if processing doublechannel images. This activates background subtraction during segmentation.
- `extra_pixels`: Defines additional padding around objects during segmentation.




## Conclusion
The `ml_tools.py` module provides a comprehensive framework for cell image classification. By following the provided instructions, users can effectively train models and classify images to support their research needs. Adjusting the parameters allows for fine-tuning the model to specific datasets, enhancing performance and accuracy.
