# OPTIMIST-25 Presentation Archive
## Installation
```
python -m venv ./.venv
source .venv/bin/activate
pythom -m pip install -r requirements.txt
```

## Outline
### Examples / Figures
All figures displayed in our presentation were generated using the code in the first
block of the jupyter notebook (`main.ipynb`).

### Datasets
Datasets are held in the `data` directory, and are SCARR formatted datasets. The 
datasets used were recorded with a ChipWhisperer Husky on a SAM4S target running 
firmware AES. Each dataset was recorded with the target on an enclosed hot plate set
to the temperature as described in the dataset name. Each dataset contains 100,000
traces recorded with random keys and plaintexts (identical between datasets).

### Source
The `src` directory contains 3 important items.

#### Oracle (`aes_intf/`)
Here, the modified source code for the AES implementation (tiny-aes) is compiled
as a shared library. This module also contains a python-facing interface to use
the shared library.

#### Analysis Operations (`generic_operation.py`)
Contains the base class for all analysis operations, as well as the implementation of
the NICV, SNR, and AES Oracle operations. The AES Oracle operation is simply a wrapper
around the interface implemented in `aes_intf/`

#### Dataset Handler (`zarr_handler.py`)
A compatability layer allowing access to datasets formatted for SCARR such that they
can be used in our custom analysis workflows.

