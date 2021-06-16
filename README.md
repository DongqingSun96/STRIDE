# STRIDE
STRIDE (Spatial TRanscrIptomics DEconvolution by topic modelling) is a cell-type deconvolution tool for spatial transcriptomics by using single-cell transcriptomics data. 

## Change Log
### v0.0.1
* Build STRIDE.

## Install STRIDE
```bash
git clone https://github.com/DongqingSun96/STRIDE.git
cd STRIDE
python setup.py install
```

## Usage
```bash
STRIDE --help
usage: STRIDE [-h] [-v] {deconvolve,plot} ...

STRIDE (Spatial TRanscRIptomics DEconvolution by topic modelling) is a cell-
type deconvolution tool for spatial transcriptomics by using single-cell
transcriptomics data.

positional arguments:
  {deconvolve,plot}
    deconvolve       Decompose celltype proportion for spatial
                     transcriptomics.
    plot             Visualize the deconvolution result.

optional arguments:
  -h, --help         show this help message and exit
  -v, --version      Print version info.
```
