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


IMG_EXT=".png" # the extension of the images in the zip file

def get_img_diff(img1, img2, background_subtraction_shift =128):
    img_subtracted=(img1.astype(np.int32) + background_subtraction_shift - img2).astype(np.uint8)      
    return img_subtracted


def extract_features_from_rtdc_to_tsv(rtdc_path,tsv_path):
# reads a rtdc file and saves all scalar features into a tsv file    
    with dclab.new_dataset(rtdc_path) as ds:
        ds = dclab.new_dataset(rtdc_path)
        features = ds.features
        print("All features: ",features)    
        scalar_features = ds.features_scalar
        print("Scalar features: ",scalar_features)
        print("Extraction started...")
        data = {f: np.asarray(ds[f]) for f in scalar_features}    
        df = pd.DataFrame(data)
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"Scalar features saved to {tsv_path}")
        
    
def extract_images_from_rtdc_to_zip(rtdc_path,zip_path,subtract=True,extra_pixels=20,img_index_to_break=1000000):
# extracts images from a rtdc file - image filenames will be of the form: frame-event_index.IMG_EXT
# the rtdc file has to contain images
# if rtdc contains event contours, it will extract only events, otherwise it will extract the whole images
# subtract flag: if the background image should be subtracted from the image
# extra_pixels: how many pixels to add to the left and right of the contour
# img_index_to_break: how many images to extract (for testing purposes) 
    with dclab.new_dataset(rtdc_path) as ds:
            
        frame_index_previous=0 #these two are used to handle multiple events in an image
        event_index=1     
            
        (img_h,img_w)=ds["image"][0].shape
        
        contour_flag = "contour" in ds.features
        if "image" not in ds.features:
            print("rtdc file does not contain images! I did nothing.")
            return

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED, allowZip64=True) as zipf:
            print(f"Extracting images from rtdc to zip, countours={contour_flag}.")
            for i in tqdm(range(len(ds))):
                
                img=ds["image"][i]                
                
                if contour_flag:
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
               
def add_class_data_to_rtdc(input_rtdc_path,df_classes):
    # df classes has to be one column pandas dataframe with classes denoted wiht integer values (1,2,3...)
    # the length of df should be equal to the lenght of rtdc
    foldername = os.path.dirname(input_rtdc_path)
    basefilename, extension = os.path.splitext(os.path.basename(input_rtdc_path))
    output_rtdc_path = os.path.join(foldername, basefilename + "_with_classes" + extension)
    class_array= df_classes.to_numpy().flatten()
    
    with (dclab.new_dataset(input_rtdc_path) as ds,
          dclab.RTDCWriter(output_rtdc_path,mode='reset') as hw):
        if len(ds)!=len(class_array):
            print("Class info length is not the same as the RTDC lengt")
        # `ds` is the basin
        # `hw` is the referrer
    
        # First of all, we have to copy the metadata from the input file
        # to the output file. If we forget to do this, then dclab will
        # not be able to open the output file.
        hw.store_metadata(ds.config.as_dict(pop_filtering=True))
    
        # Next, we can compute and write the new feature to the output file.
        hw.store_feature("userdef1", class_array)
    
        # Finally, we write the basin information to the output file.
        hw.store_basin(
            basin_name="class data",
            basin_type="file",
            basin_format="hdf5",
            basin_locs=[input_rtdc_path],
        )
    
    # You can now open the output file and verify that everything worked.
    with dclab.new_dataset(output_rtdc_path) as ds_out:
        assert "userdef1" in ds_out, "check that the feature we wrote is there"
        assert "image" in ds_out, "check that we can access basin features"
        print(f"New RTDC file saved with {len(ds_out)} events: {output_rtdc_path}")