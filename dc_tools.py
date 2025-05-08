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

def img_diff(img1, img2, background_subtraction_shift = 100):
    img_subtracted=(img1.astype(np.int32) + background_subtraction_shift - img2).astype(np.uint8)      
    return img_subtracted

def img_blurr(img, gauss_blur_k_vector_size = 5, gauss_blur_k_vector_spread = 4):
    img_blurred=cv.GaussianBlur(img,(gauss_blur_k_vector_size,gauss_blur_k_vector_size),gauss_blur_k_vector_spread)
    return img_blurred

def img_SNR(img):
    img_std=np.std(img) #calculate standard deviation
    img_ptp=np.ptp(img) #calculate peak-to-peak value (max-min)
    if img_std != 0.0:
        return img_ptp/img_std
    else:
        return 0
