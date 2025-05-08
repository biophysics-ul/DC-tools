# DC-tools
Custom Python tools for Deformability Cytometry (DC)  
The tools are not optimized for speed but for reusability.  
The repository contains two main modules:
### dc_tools.py
This module comprises functions for working and analyzing DC images. It works with images stored in a ZIP file or images stored in a RTDC format. For the latter in uses [**dclab**](https://github.com/DC-analysis/dclab).

### ml_tools.py
This module comprises functions for ML-based model traning and image classification. It is idependent of DC and works with images stored in ZIP files. ML is based PyTorch and can make use of a GPU.

---
Authors:  
Darin Lah  
Bor Ivanuš  
Jure Derganc  
Institute of Biophysics, Faculty of Medicine, University of Ljubljana  
