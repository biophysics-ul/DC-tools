# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:19:07 2025

@author: jderganc
"""

import cv2 as cv2 # shape detection functions
import dclab
import tkinter as tk
from tkinter import filedialog
import os
import sys

N_COLS=5
N_ROWS=10
FONT_SCALE = 0.5  # Small font size for text
FONT_THICKNESS = 1  # Font thickness
BORDER_SIZE=10
TEXT_POSITION = (BORDER_SIZE , BORDER_SIZE + 10)
PLOT_CONTOURS = True

# Open a file dialog to select an RTDC file
root = tk.Tk()
root.withdraw()  # Hide the root window
rtdc_path = filedialog.askopenfilename(title="Select RTDC File", filetypes=[("RTDC File", "*.rtdc")])
if not rtdc_path:
    print("No file selected. Exiting...")
    sys.exit()


rtdc_filename = os.path.basename(rtdc_path)
window_name="RTDC: "+rtdc_filename+" Press any key to exit."

ds = dclab.new_dataset(rtdc_path)
print("Number of images in RTDC: ",len(ds))

def read_and_concatenate(start_idx, n_cols, n_rows):

    images = []    
    
    if start_idx + (n_cols * n_rows) > len(ds):
        print("Not enough images in the list to form the grid.")
        return None
    
    for i in range(start_idx, start_idx + (n_cols * n_rows)):
        img=ds["image"][i]
        img_frame=ds["frame"][i] # index of the image
        cv2.putText(img, str(img_frame), TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 
                    FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        if PLOT_CONTOURS:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img[ds["contour"][i][:,1], ds["contour"][i][:,0]] = [0, 0, 255] # Red in BGR
        
        images.append(cv2.copyMakeBorder(img, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    grid = []
    for i in range(n_rows):
        row = images[i * n_cols:(i + 1) * n_cols]  # Extract row
        row_concat = cv2.hconcat(row)  # Concatenate row images horizontally
        grid.append(row_concat)
    final_image = cv2.vconcat(grid)  # Concatenate all rows vertically
    return final_image
        

start_idx=0
max_idx=int(len(ds)/(N_COLS*N_ROWS))
# Initialize OpenCV Window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("Concatenated Images", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Callback function for trackbar
def update_display(start_idx):
    start_idx = max(0, start_idx)  # Ensure valid index
    concatenated_image = read_and_concatenate(start_idx*N_COLS*N_ROWS, N_COLS, N_ROWS)
    if concatenated_image is not None:
        cv2.imshow(window_name, concatenated_image)

# Create trackbar
cv2.createTrackbar("Start Index", window_name, 1, max_idx, update_display)

# Initial display
update_display(1)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()