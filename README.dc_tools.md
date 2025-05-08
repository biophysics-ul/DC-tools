# dc_tools

Some additional info for the module `dc_tools.py`.

### Dependencies

To use this project, you need the following Python packages:  
`OpenCV` >=4.0  
`dclab` (only if you work with RTDC files)

### Functions

`extract_events_from_raw_rtdc_to_zip(rtdc_path, zip_path)`

**Description:** Extracts images with events and their timestamps from a raw (non-filtered) RTDC format to ZIP archive.

Images with events are those which, upon background subtraction, have signal-to-noise (SNR) ratio above predefined threshols. 
