# Clay Foundation Model
This model has been modified. The original version is in the repository. https://github.com/Clay-foundation/model.

## Installation

### Basic

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>
    cd model

Then we recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

    mamba env create --file environment.yml

> [!NOTE]
> The command above has been tested on Linux devices with CUDA GPUs.

Activate the virtual environment first.

    mamba activate claymodel

Finally, double-check that the libraries have been installed.

    mamba list

## Dataset structure
```
dataset/
├── raw_train/ # Raw file train
├── raw_val/   # Raw file val
├── train/
│   ├── img/  # Training images
│   └── gt/   # Ground truths corresponding to training images
├── val/
│   ├── img/  # Training images
│   └── gt/   # Ground truths corresponding to training images
```

### Files (`raw_train/`)
- **Format:** `.npz`
- **Resolution:** 256x256.

### Files (`raw_train/`)
- **Format:** `.npz`
- **Resolution:** 256x256. 

### Images (`img/`)
- **Format:** `.npy`
- **Resolution:** 256x256. 

### Ground Truths (`gt/`)
- **Format:** `.npy`
- **Resolution:** 256x256. 

