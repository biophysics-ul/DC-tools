import ml_tools
import pandas as pd


ml_tools.IMG_EXT='.tif'
ml_tools.EPOCHS=30

########################################
# First we will train a model

# set the training set folder/ZIP file with manually sorted training set
# training set is a zip file containing folders with classes of manually sorted images
training_set_folder = r"C:\Jure\Dropbox\temp\IBF-DC\TrainingSet.zip"

# set the folder where the trained model will be saved (along with some additional data)
# the model name will be inferred from the structure of the training set - the class names will be in the model name
results_folder= r'C:\Jure\Dropbox\temp\IBF-DC\TrainingResults'

# now start the training
ml_tools.train_model(training_set_folder, results_folder)

########################################
#Next we will classify images from an experiment

# set the folder/ZIP file with images you want to classify
input_image_folder=r'C:\Jure\Dropbox\temp\IBF-DC\Experiment20000.zip'

# set the path to the ml model you want to use
model_path=r'C:\Jure\Dropbox\temp\IBF-DC\TrainingResults\Models\ml_model_2025_05_21_classes_DoubleCells_NiceCells_DeadCells_Debris_DefectedCells\ml_model_2025_05_21                                                                                                        _classes_DoubleCells_NiceCells_DeadCells_Debris_DefectedCells.pt'

#set the path where classification data will be saved
classification_data_path= r'C:\Jure\Dropbox\temp\IBF-DC\classification_data.txt'

# now start the classification
ml_tools.classify_images(input_image_folder,model_path,classification_data_path)

########################################
#Finally we will sort images from the experiment according to their classes and save them to zip files

#Now read the classification data into dataframe df
df = pd.read_csv(classification_data_path, sep='\t')
#Next clean df of extra spaces and NaN values
df=df.dropna()
df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

#set the folder where to save sorted images in zip files
output_folder=r"C:\Jure\Dropbox\temp\IBF-DC\ClassifiedImages"

#sort the images from the experiment (by default it will save just up to 1000 random images per class)
ml_tools.sort_class_images_from_zip(input_image_folder, df, output_folder)