# dc_tools

Some additional info for the module `dc_tools.py`.

### Dependencies

To use this project, you need the following Python packages:  
`OpenCV` >=4.0  
`dclab` (only if you work with RTDC files)

### Functions

`extract_events_from_raw_rtdc_to_zip(rtdc_path, zip_path, mode)`

**Description:** Extracts images with events and their timestamps from a raw (non-filtered) RTDC file into a ZIP archive. Images with events are those that, after background subtraction, have a signal-to-noise ratio (SNR) above a predefined threshold.

The user can choose if the extracted images are a) original images with events, b) images with subtracted background or c) concatenated images, with the original image on top and the background image below (mode=original,subtracted,concatenated).


