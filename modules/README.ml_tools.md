# ml_tools

Some additional info for the module `ml_tools.py`.

### Overview
The `ml_tools.py` module is designed to streamline the process of image classification in flow cytometry and other biological research. The key two functionalities are:
1. Model Training: Train a modified ResNet18 model with user provided training set.
2. Classification: Use the trained model to classify images and save the results. 

**Notes:** 
- The images in the training and sample sets don't need to be of the same size, but if you preprocess the images in the training set, you need to apply the same preprocess to the images in the sample set you want to classify.
- See `ml_tools_example_script.py` in the `scripts` folder for the usage example.

## 1. Model Training
`train_model(training_set_path, output_model_folder)`

The `train_model` function in `ml_tools.py` handles the training of the cell classification model. It processes training set images stored in a ZIP file, trains the model, and saves the trained model along with performance metrics into `output_model_folder`. 

The function runs fine as it is, but the user can also change the following parameters to fine-tune the training:

- `EPOCHS`: Number of complete passes (epochs) through the training dataset.
- `BATCH_SIZE`: Refers to the number of training examples processed before the model updates its parameters.
- `VALIDATION_SPLIT`: Defines the percentage of data used for validation.
- `PATIENCE_TOTAL`: Halts training if validation loss does not improve.
- `LEARNING_RATE`: Determines the step size at each iteration while optimizing the loss function.
- `WEIGHT_DECAY`: Reduces overfitting by preventing complex models.
- `DROPOUT_PROB`: Dropout probability, regularizes the model to prevent overfitting.


**Notes:** 
- It is highly recommended that the training is run on a GPU. In theory you can run it on CPU, but it will be very slow. 

 ### Input data: Training set 
The training set has to be stored in a single ZIP file, with the images sorted into folders having names of the classes:

```
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
```

### Output data:
The trained model and some performance metrics are stored into `output_model_folder`.

- The filename of the final model (e.g., the model trained after the last training pass - epoch ) will have the structure "ml_model_{date}_classes_{classes}.pt", where {date} is the current date, and {classes} are the names of the classes inferred from the ZIP file, e.g.,   
    `ml_model_2025_05_25_classes_class1_class2_class3_class4.pt` 
- The perfomache metrics include
    - the data and figure of the confusion matrix
    - the data and figures for the performance of each class as well as overal performance
    - the models and performance after each epoch are also included in separate folders

**Notes:** 
- You can later change the first part of the model filename, but you should keep the classes part as it is. For example:  
`my_model_classes_class1_class2_class3_class4.pt`  
This is because the subsequent classification then inferes the classes names and their order from the model filename. 

## 2. Classification (inference)
`classify_images(sample_images_path, model_path, output_data_file_path)`

The `classify_images` function in `ml_tools.py` allows users to perform classification of the experimental sample images using a pre-trained model. It reads the images from the ZIP file specified in the `sample_images_path`, runs inference using the pre-trained model specified in the `model_path` and saves the result of classification into a `tsv` `txt` file specified in the `output_data_file_path`. The result file is a table with three columns:  "img_name", "img_class" and "confidence". 


**Notes:**
- Classification runs much faster on a GPU, but with some patience you can do it also on a CPU.
- Images in the `sample_images_path` have to be preprocessed in the same way as the images from the training set. 

## 3. Additional useful functions
- Once you classify your images using `classify_images` you might want to see the result of classification, that is to extract and sort the images from the experimental sample according to their classes. For this you can use function `sort_class_images_from_zip(sample_images_path, classification_df, output_folder)` from  `ml_tools.py`.

    It reads the images from the ZIP file specified in the `sample_images_path`, and saves them in separate ZIP files into `output_folder` according to their classes. The classification data has to be provided as a dataframe in `classification_df`. See `ml_tools_example_script.py` in the `scripts` folder for the usage example.
    

## Conclusion
The `ml_tools.py` module provides a comprehensive framework for cell image classification. By following the provided instructions, users can effectively train models and classify images to support their research needs. Adjusting the parameters allows for fine-tuning the model to specific datasets, enhancing performance and accuracy.
