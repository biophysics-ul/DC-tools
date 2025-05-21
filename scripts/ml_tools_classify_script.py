import ml_tools

# set the folder/ZIP file with images you want to classify
input_image_folder=r'C:\Jure\Dropbox\temp\IBF-DC\Experiment1000.zip'

# set the path to the model you want to use
model_path=r'C:\Jure\Dropbox\temp\IBF-DC\2025_05_14_M1_RN_DeadCells_Debris_DefectedCells_DoubleCells_NiceCells\2025_05_14_M1_RN_DeadCells_Debris_DefectedCells_DoubleCells_NiceCells'

#set the path where classification data will be saved
results_path= r'C:\Jure\Dropbox\temp\IBF-DC\classification_data.txt'

ml_tools.classify_images(input_image_folder,model_path,results_path)

print("Done!")
