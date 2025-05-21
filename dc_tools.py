# -*- coding: utf-8 -*-
"""
Module Name: dc_tools.py
Description: useful functions for working with Deformability Cytometry (DC) data
Authors: Darin Lah & Jure Derganc
Institute: Institute of Biophysics, University of Ljubljana
Last modified: 2025-05-07
License: GPL 3.0

Notes:
some functions work with images and data stored in a RTDC format
other functions work with images and data stored in a ZIP archive
"""

import cv2 as cv 
import numpy as np
import dclab
import pandas as pd
import zipfile
from tqdm import tqdm
import os

IMG_EXT=".png"

def get_img_diff(img1, img2, background_subtraction_shift = 100):
    img_subtracted=(img1.astype(np.int32) + background_subtraction_shift - img2).astype(np.uint8)      
    return img_subtracted


def extract_features_from_analyzed_rtdc_to_tsv(rtdc_path,tsv_path):
# reads a rtdc file and saves all scalar features in a tsv file    
    ds = dclab.new_dataset(rtdc_path)
    features = ds.features
    print("All features: ",features)    
    scalar_features = ds.features_scalar
    print("Scalar features: ",scalar_features)
    data = {f: np.asarray(ds[f]) for f in scalar_features}    
    df = pd.DataFrame(data)
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Scalar features saved to {tsv_path}")
    
    
def extract_images_from_analyzed_rtdc_to_zip(rtdc_path,zip_path,subtract=True,extra_pixels=20,img_index_to_break=1000000):
# extracts images from a rtdc file - image filenames are frame-event_index.IMG_EXT
# the rtdc file has to contain data on images, their contours and backgrounds 
    ds = dclab.new_dataset(rtdc_path)
    
    frame_index_previous=0 #these two are used to handle multiple events in an image
    event_index=1     
        
    (img_h,img_w)=ds["image"][0].shape
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED, allowZip64=True) as zipf:

        for i in tqdm(range(len(ds))):
            
            img=ds["image"][i]
            img_bg=ds["image_bg"][i]
            if subtract:
                img=get_img_diff(img, img_bg)
            
            x_contour_points=ds["contour"][i][:,0]  
            x_min=max(0,min(x_contour_points)-extra_pixels)
            x_max=min(img_w,max(x_contour_points)+extra_pixels)  
            img=img[:, x_min:x_max]
            
            frame_index=ds["frame"][i]            
            if frame_index==frame_index_previous:
                event_index=event_index+1 
            else:
                event_index=1
                frame_index_previous=frame_index
                  
            _, img_buffer = cv.imencode(IMG_EXT, img)
            png_filename=f"{frame_index:06}-{event_index}.png"
            zipf.writestr(png_filename, img_buffer.tobytes())            
            
            if i>=img_index_to_break:break  
               
