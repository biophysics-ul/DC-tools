import ml_tools

results_folder= 'F:/Results'
model_path='C:/Users/User/Documents/GitHub/cell_detector_final/Models/2025_05_13_M1_RN_DeadCells_Debris_DefectedCells_DoubleCells_NiceCells/2025_05_13_M1_RN_DeadCells_Debris_DefectedCells_DoubleCells_NiceCells'

doublechannel = True
extra_pixels = 40

# If the model_path isn’t specified, the load_model function will open a filedialog window to choose the model
model, labels, device = ml_tools.load_model(model_path)

# You can also switch this with the path to the folder
substracted_image_folder = ml_tools.choose_image_folder()

ml_tools.inference(substracted_image_folder, results_folder, model, labels, device)

print("Done!")
