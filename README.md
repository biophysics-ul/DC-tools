# DC-tools
Custom Python tools for Deformability Cytometry (DC).  
These tools are designed for **reusability** rather than speed. 
The repository contains two main modules:
### `dc_tools.py`
This module provides functions for working with and analyzing DC images. It supports images stored in a ZIP file or images stored in a RTDC format. For the latter, it uses [**dclab**](https://github.com/DC-analysis/dclab).

### `ml_tools.py`
This module offers functions for ML-based model training and image classification.
It is independent of DC (and `dc_tools.py`) and works with images stored in ZIP files.
Machine learning is based on PyTorch and can utilize a GPU.

## Usage  

Just download the modules, make sure they're in the same folder as your script, and you're ready to go. 


- If you want to work with RTDC files, you need to install [**dclab**](https://github.com/DC-analysis/dclab).

- To use GPU acceleration with ml_tools.py, make sure CUDA is installed and properly configured.

More detailed information is available in the module-specific README files: 
- [README.dc_tools](README.dc_tools.md)
- [README.ml_tools](README.ml_tools.md)




---
**Authors:**  
Darin Lah  
Bor Ivanuš  
Jure Derganc  
Institute of Biophysics, Faculty of Medicine, University of Ljubljana  
