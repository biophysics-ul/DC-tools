# -*- coding: utf-8 -*-
"""
Script: extract_events_from_raw_rtdc_to_zip.py
Description: Extracts images with events and their timestamps from a raw (non-filtered) RTDC format to ZIP archive.
Author: Jure Derganc
Date: 2025-05-07

Input data: 
  rtdc file with raw images ()
Outup data:
  ZIP archive with images with detected events and a txt file with timestamps of the images

Parameters:
  rtdc_path
  zip_path
  
Constants:  
  SUBTRACT_BACKGROUND=True/False (if subtract background for extracted images)
  BACKGROUND_SUBTRACTION_SHIFT=100 (intensity shift after subtraction)
  SOMETHING_TRESHOLD=3 (SNR treshold for positive events)
  
"""


import dclab
import dc_tools
import numpy as np
import cv2 as cv # shape detection functions
import zipfile
import pandas as pd
from tqdm import tqdm


rtdc_path=r"D:\RTDC\2025-03-05_14.03_M001_Sara_1.rtdc"
zip_path=rtdc_path[:-5]+".zip"

ds = dclab.new_dataset(rtdc_path)

# set parameters that influence outcome
# parameters for image sorting (based od background subtration separates images with somethin on image and nothing on image)
background_subtraction_shift = 100
something_treshold = 3 # meja SNR slike, nad katero rečemo, da je nekaj na sliki. Varianca slike = imgMaxMin/img_bg_std
cell_threshold_factor = 1 # threshold = background_subtraction_shift-cell_threshold_factor*img_bg_std 
# sizes of kernels for blurring 
gauss_blur_k_vector_size = 5
gauss_blur_k_vector_spread = int(round(0.75*gauss_blur_k_vector_size))

def img_diff(img1,img2):
    img_subtracted=(img1.astype(np.int32) + background_subtraction_shift - img2).astype(np.uint8)   
    img_subtracted_blurred=cv.GaussianBlur(img_subtracted,(gauss_blur_k_vector_size,gauss_blur_k_vector_size),gauss_blur_k_vector_spread)
    img_std=np.std(img_subtracted)
    img_min=np.min(img_subtracted_blurred)
    img_max=np.max(img_subtracted_blurred)      
    img_MaxMin= img_max-img_min  #max difference of pixels on img
    return img_subtracted, img_subtracted_blurred, img_min, img_MaxMin/img_std



image_stack = []
for i in range(50):
    image = ds["image"][i]
    img_array = np.array(image)
    image_stack.append(img_array)
stacked_images = np.stack(image_stack, axis=0)
pixelwise_median = np.median(stacked_images, axis=0).astype('uint8')
img_background0=pixelwise_median

image_stack = []
for i in range(51,100):
    image = ds["image"][i]
    img_array = np.array(image)
    image_stack.append(img_array)
stacked_images = np.stack(image_stack, axis=0)
pixelwise_median = np.median(stacked_images, axis=0).astype('uint8')
img_background1=pixelwise_median

img_subtracted,img_subtracted_blurred,img_bg_min,img_bg_SNR = img_diff(img_background0,img_background1)
img_previous=img_background1

df = pd.DataFrame(columns=["frame", "timestamp"])

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED, allowZip64=True) as zipf:

    for i in tqdm(range(len(ds))):
        img=ds["image"][i]
        img_index=ds["frame"][i]
        timestamp = ds["time"][i]
        img_subtracted,img_subtracted_blurred,img_min,img_SNR = img_diff(img,img_previous)               
        
        if img_SNR<1.5*img_bg_SNR:
        # if yes, set previous image as background
            img_background=img_previous
            img_bg_min=img_min
            img_bg_SNR = img_SNR
            img_bg_index=img_index-1         
            threshold_delta=background_subtraction_shift-img_bg_min
            img_tresh=background_subtraction_shift-threshold_delta*cell_threshold_factor
            
        #next we check if something is on image / if the image is similar to the background
        img_subtracted,img_subtracted_blurred,img_min,img_SNR = img_diff(img,img_background)
        something_on_img = img_SNR>something_treshold*img_bg_SNR # če je kaj na sliki bo to 1, drugače 0
        if something_on_img:   
          new_row = pd.DataFrame({"frame": [img_index], "timestamp": [timestamp]})
          if not df.empty:
              df = pd.concat([df, new_row], ignore_index=True)
          else:
              df = new_row 
          img_original_with_bg = np.concatenate((img,img_background), axis=0)   
          _, img_buffer = cv.imencode('.png', img_original_with_bg)
          png_filename=f"{img_index:06}.png"
          zipf.writestr(png_filename, img_buffer.tobytes())
          
                    
        key_pressed=cv.waitKeyEx(1)
        if key_pressed == ord('q'): 
            break
        
        #if i_files>=img_index_to_break:break  
           
        img_previous=img
    with zipf.open("timestamps.txt", 'w') as tsv_file:
        df.to_csv(tsv_file, sep='\t', index=False, encoding='utf-8', lineterminator='\n')

    
    