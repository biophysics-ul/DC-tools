# DC-tools

Custom Python tools for Deformability Cytometry (DC), developed at the Institute of Biophysics, Faculty of Medicine, University of Ljubljana.  
These tools are designed for reusability rather than speed.  
The repository contains two main modules and some useful scripts:

### 1. Module `dc_tools.py`
This module provides functions for working with and analyzing DC images. It supports images stored in ZIP files or in the RTDC format. For the latter, it uses [**dclab**](https://github.com/DC-analysis/dclab).

### 2. Module `ml_tools.py`
This module offers functions for ML-based model training and image classification.  
It is **independent of DC** (and `dc_tools.py`) and works with any images stored in ZIP files.  
Machine learning is based on PyTorch and can utilize a GPU. To use GPU acceleration, make sure CUDA is installed and properly configured.

### 3. Useful scripts
Some useful scripts and examples are in the `Scripts` folder:
- `rtdc_ImageViewer`: Script for previewing images in RTDC files
- `zip_ImageViewer`: Script for previewing images in ZIP files
- `ml_tools_example_script`: Script with a complete ML workflow example (model training & image classification)

## Usage

- Install the required packages (see below)
- Download the files, make sure the modules are in the same folder as your script, and you're ready to go!

Additional information about the modules is available in the module-specific README files:
- [README.dc_tools](README.dc_tools.md)
- [README.ml_tools](README.ml_tools.md)

Additional information about the scripts is included in the scripts themselves.



## Installation

For us, this process worked well:

- Create a new Python environment  
- Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), select your system configuration, copy the installation command, and run it in your console  
- For our configuration, the correct command was:

```
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- Then install the remaining packages:

```
    pip install numpy opencv-python dclab tqdm matplotlib scikit-learn seaborn
```

If you won't use `dc_tools.py`, you can skip installing `dclab`.  
If you won't use `ml_tools.py`, you can skip `torch` and `torchvision`.  

The versions that worked for us:
```
numpy         2.2.6
opencv-python 4.11.0.86
dclab         0.64.0
tqdm          4.67.1
matplotlib    3.10.3
scikit-learn  1.6.1
seaborn       0.13.2
torch         2.7.0+cu126
torchvision   0.22.0+cu126
```

---

**Authors:**  
Darin Lah  
Bor Ivanuš  
Jure Derganc  
Institute of Biophysics, Faculty of Medicine, University of Ljubljana
