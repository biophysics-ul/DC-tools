# dc_tools

Some additional info for the module `dc_tools.py`.

### Dependencies

To use this project, you need the following Python packages:  
`OpenCV` >=4.0  
`dclab` (only if you work with RTDC files)

### Functions

`extract_events_from_raw_rtdc_to_zip(rtdc_path, zip_path)`

**Description:** Extracts images with events and their timestamps from a raw (non-filtered) RTDC file into a ZIP archive.

Images with events are those that, after background subtraction, have a signal-to-noise ratio (SNR) above a predefined threshold.
