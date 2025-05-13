import ml_tools

results_folder= 'F:/Results'


image_folder = ml_tools.choose_image_folder()
ml_tools.train_model(image_folder, results_folder)


print("Done!")
