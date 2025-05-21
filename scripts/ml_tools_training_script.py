import ml_tools

# set the training set folder/ZIP file with manually sorted training set
training_set_folder = r"C:\Jure\Dropbox\temp\IBF-DC\TrainingSet.zip"

# set the folder where the trained model will be saved (along with some additional data)
# the model name will be inferred from the structure of the training set - the class names will be in the model name
results_folder= r'C:\Jure\Dropbox\temp\IBF-DC\TrainingResults'

ml_tools.train_model(training_set_folder, results_folder)


print("Done!")
